#!/usr/bin/env python3
"""
V3 模型訓練程式

改進:
1. 更好的特徵工程 (20-25個精簡特徵)
2. 更好的標籤定義 (基於實際 TP/SL)
3. 雙階段訓練 (分類 + 機率校準)
4. 更好的機率分佈
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent))

from utils.feature_engineering_v3 import FeatureEngineerV3
from utils.logger import setup_logger
from huggingface_hub import hf_hub_download
from config import Config

# ML
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    precision_recall_curve,
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

logger = setup_logger('train_v3', 'logs/train_v3.log')

class V3ModelTrainer:
    def __init__(self, 
                 tp_pct: float = 0.02,
                 sl_pct: float = 0.01,
                 quick_test: bool = False):
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.quick_test = quick_test
        self.feature_engineer = FeatureEngineerV3()
        
        logger.info("="*80)
        logger.info("V3 MODEL TRAINER INITIALIZED")
        logger.info("="*80)
        logger.info(f"TP/SL: {tp_pct*100:.1f}% / {sl_pct*100:.1f}%")
        logger.info(f"Quick test mode: {quick_test}")
    
    def load_data(self) -> pd.DataFrame:
        """載入數據"""
        logger.info("\nLoading data from HuggingFace...")
        
        try:
            local_path = hf_hub_download(
                repo_id=Config.HF_REPO_ID,
                filename="klines/BTCUSDT/BTC_1m.parquet",
                repo_type="dataset",
                token=Config.HF_TOKEN
            )
            df = pd.read_parquet(local_path)
            
            if 'open_time' in df.columns:
                df['open_time'] = pd.to_datetime(df['open_time'])
                df.set_index('open_time', inplace=True)
            
            logger.info(f"Loaded {len(df):,} rows")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            
            # Quick test: 只用最近 30 天
            if self.quick_test:
                df = df.tail(30 * 1440)
                logger.info(f"Quick test mode: using last {len(df):,} rows")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """生成特徵"""
        logger.info("\nGenerating V3 features...")
        
        df_features = self.feature_engineer.create_features_from_1m(
            df,
            tp_pct=self.tp_pct,
            sl_pct=self.sl_pct,
            label_type='both'
        )
        
        # 分離特徵和標籤
        feature_cols = self.feature_engineer.get_feature_list()
        
        X = df_features[feature_cols].copy()
        y_long = df_features['label_long'].copy()
        y_short = df_features['label_short'].copy()
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Long labels: {y_long.sum()}/{len(y_long)} ({y_long.mean()*100:.2f}% positive)")
        logger.info(f"Short labels: {y_short.sum()}/{len(y_short)} ({y_short.mean()*100:.2f}% positive)")
        
        return X, y_long, y_short
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, direction: str) -> dict:
        """
        訓練單個模型
        
        雙階段訓練:
        1. CatBoost 基礎模型
        2. Isotonic 機率校準
        """
        logger.info("="*80)
        logger.info(f"TRAINING {direction.upper()} MODEL")
        logger.info("="*80)
        
        # Train/Val 切分 (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Train: {len(X_train):,} | Val: {len(X_val):,}")
        
        # 階段 1: CatBoost
        logger.info("\nStage 1: Training CatBoost classifier...")
        
        if self.quick_test:
            iterations = 200
        else:
            iterations = 1000
        
        model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=42,
            verbose=100,
            early_stopping_rounds=50
        )
        
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )
        
        # 評估基礎模型
        y_pred_proba_train = model.predict_proba(X_train)[:, 1]
        y_pred_proba_val = model.predict_proba(X_val)[:, 1]
        
        train_auc = roc_auc_score(y_train, y_pred_proba_train)
        val_auc = roc_auc_score(y_val, y_pred_proba_val)
        
        logger.info(f"\nBase model performance:")
        logger.info(f"  Train AUC: {train_auc:.4f}")
        logger.info(f"  Val AUC: {val_auc:.4f}")
        
        # 階段 2: 機率校準
        logger.info("\nStage 2: Calibrating probabilities...")
        
        calibrated_model = CalibratedClassifierCV(
            model,
            method='isotonic',
            cv='prefit'
        )
        
        calibrated_model.fit(X_val, y_val)
        
        # 評估校準後模型
        y_pred_proba_cal = calibrated_model.predict_proba(X_val)[:, 1]
        
        # 分析機率分佈
        self._analyze_probability_distribution(y_pred_proba_cal, y_val, direction)
        
        # 特徵重要性
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 features:")
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # 儲存模型和 metadata
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_data = {
            'model': calibrated_model,
            'feature_names': list(X.columns),
            'version': 'v3',
            'direction': direction,
            'timestamp': timestamp,
            'tp_pct': self.tp_pct,
            'sl_pct': self.sl_pct,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'feature_importance': feature_importance.to_dict('records')
        }
        
        # 儲存
        models_dir = Path('models_output')
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / f"catboost_{direction}_v3_{timestamp}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"\nModel saved: {model_path}")
        
        return {
            'model_path': model_path,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'feature_importance': feature_importance
        }
    
    def _analyze_probability_distribution(self, y_pred_proba: np.ndarray, y_true: pd.Series, direction: str):
        """分析機率分佈"""
        logger.info("\n" + "="*80)
        logger.info(f"PROBABILITY DISTRIBUTION ANALYSIS - {direction.upper()}")
        logger.info("="*80)
        
        # 基本統計
        logger.info(f"Min:  {y_pred_proba.min():.4f}")
        logger.info(f"25%:  {np.percentile(y_pred_proba, 25):.4f}")
        logger.info(f"50%:  {np.percentile(y_pred_proba, 50):.4f}")
        logger.info(f"75%:  {np.percentile(y_pred_proba, 75):.4f}")
        logger.info(f"95%:  {np.percentile(y_pred_proba, 95):.4f}")
        logger.info(f"Max:  {y_pred_proba.max():.4f}")
        
        # 不同閾值的表現
        logger.info("\nPerformance at different thresholds:")
        
        for threshold in [0.10, 0.15, 0.20, 0.25, 0.30]:
            above_threshold = y_pred_proba >= threshold
            if above_threshold.sum() > 0:
                precision = y_true[above_threshold].mean()
                count = above_threshold.sum()
                pct = count / len(y_pred_proba) * 100
                logger.info(f"  >= {threshold:.2f}: Precision={precision*100:.1f}%, Count={count:,} ({pct:.2f}%)")
        
        # 推薦閾值
        logger.info("\nRecommended thresholds:")
        
        # 找到 precision > 55% 的最低閾值
        for threshold in np.arange(0.05, 0.50, 0.01):
            above = y_pred_proba >= threshold
            if above.sum() > 50:  # 至少 50 個樣本
                precision = y_true[above].mean()
                if precision >= 0.55:
                    logger.info(f"  For 55%+ win rate: >= {threshold:.2f}")
                    break
        
        # 找到 precision > 60% 的最低閾值
        for threshold in np.arange(0.05, 0.50, 0.01):
            above = y_pred_proba >= threshold
            if above.sum() > 30:
                precision = y_true[above].mean()
                if precision >= 0.60:
                    logger.info(f"  For 60%+ win rate: >= {threshold:.2f}")
                    break
        
        logger.info("="*80)
    
    def run(self):
        """執行完整訓練流程"""
        logger.info("\n" + "="*80)
        logger.info("STARTING V3 TRAINING PIPELINE")
        logger.info("="*80)
        
        try:
            # 1. 載入數據
            df = self.load_data()
            
            # 2. 生成特徵
            X, y_long, y_short = self.prepare_features(df)
            
            # 3. 訓練 Long 模型
            long_results = self.train_model(X, y_long, 'long')
            
            # 4. 訓練 Short 模型
            short_results = self.train_model(X, y_short, 'short')
            
            # 5. 報告
            logger.info("\n" + "="*80)
            logger.info("TRAINING COMPLETED")
            logger.info("="*80)
            logger.info(f"Long model:  {long_results['model_path']}")
            logger.info(f"  Train AUC: {long_results['train_auc']:.4f}")
            logger.info(f"  Val AUC:   {long_results['val_auc']:.4f}")
            logger.info(f"\nShort model: {short_results['model_path']}")
            logger.info(f"  Train AUC: {short_results['train_auc']:.4f}")
            logger.info(f"  Val AUC:   {short_results['val_auc']:.4f}")
            logger.info("="*80)
            
            # 儲存訓練報告
            report = {
                'version': 'v3',
                'timestamp': datetime.now().isoformat(),
                'tp_pct': self.tp_pct,
                'sl_pct': self.sl_pct,
                'quick_test': self.quick_test,
                'long_model': {
                    'path': str(long_results['model_path']),
                    'train_auc': long_results['train_auc'],
                    'val_auc': long_results['val_auc']
                },
                'short_model': {
                    'path': str(short_results['model_path']),
                    'train_auc': short_results['train_auc'],
                    'val_auc': short_results['val_auc']
                }
            }
            
            report_path = Path('training_reports') / f"v3_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path.parent.mkdir(exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"\nTraining report saved: {report_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train V3 models')
    parser.add_argument('--tp', type=float, default=0.02, help='Take profit %')
    parser.add_argument('--sl', type=float, default=0.01, help='Stop loss %')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    
    args = parser.parse_args()
    
    trainer = V3ModelTrainer(
        tp_pct=args.tp,
        sl_pct=args.sl,
        quick_test=args.quick
    )
    
    trainer.run()