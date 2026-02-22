import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from pathlib import Path
import joblib
import warnings
import sys

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger('oracle_trainer', 'logs/oracle_trainer.log')

class OraclePredictor:
    """
    Oracle 預測器 - 量化世界模型的核心大腦
    
    三道防線:
    1. 帶淨空期的時間序列切分 (Purged Time-Series Split)
    2. CatBoost 對稱樹引擎 + L2 正則化
    3. Isotonic Regression 機率校準
    """
    
    def __init__(self, lookahead_bars: int = 16):
        """
        Args:
            lookahead_bars: 預測未來幾根 K 線 (16 = 4小時)
                           用於設定淨空期避免資料洩漏
        """
        logger.info(f"Initializing OraclePredictor (lookahead_bars={lookahead_bars})")
        
        self.lookahead_bars = lookahead_bars
        self.tscv = TimeSeriesSplit(n_splits=5, gap=lookahead_bars)
        self.calibrated_model = None
        self.base_model = None
        self.feature_names = None
        
        logger.info("Oracle initialized with 3-layer defense:")
        logger.info(f"  1. Purged gap: {lookahead_bars} bars")
        logger.info("  2. CatBoost symmetric trees + L2 regularization")
        logger.info("  3. Isotonic probability calibration")
    
    def train_and_calibrate(self, X: pd.DataFrame, y: pd.Series, 
                           iterations: int = 800,
                           learning_rate: float = 0.03,
                           depth: int = 5,
                           l2_leaf_reg: float = 5.0) -> dict:
        """
        執行嚴格的 Walk-Forward 訓練與機率校準
        
        Args:
            X: 特徵矩陣
            y: 目標標籤
            iterations: CatBoost 樹的數量
            learning_rate: 學習率
            depth: 樹深度
            l2_leaf_reg: L2 正則化強度
        
        Returns:
            results: 訓練結果與指標
        """
        logger.info("="*80)
        logger.info("ORACLE TRAINING PIPELINE")
        logger.info("="*80)
        
        self.feature_names = X.columns.tolist()
        
        # 計算樣本不平衡權重
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        logger.info(f"Total samples: {len(X)}")
        logger.info(f"Positive samples: {pos_count} ({pos_count/len(y)*100:.2f}%)")
        logger.info(f"Negative samples: {neg_count} ({neg_count/len(y)*100:.2f}%)")
        logger.info(f"Auto-calculated scale_pos_weight: {scale_pos_weight:.2f}")
        
        # 實例化 CatBoost 基礎模型
        logger.info("\nCatBoost configuration:")
        logger.info(f"  - iterations: {iterations}")
        logger.info(f"  - learning_rate: {learning_rate}")
        logger.info(f"  - depth: {depth}")
        logger.info(f"  - l2_leaf_reg: {l2_leaf_reg}")
        logger.info(f"  - scale_pos_weight: {scale_pos_weight}")
        
        self.base_model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            scale_pos_weight=scale_pos_weight,
            eval_metric='AUC',
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1
        )
        
        # 包裝機率校準器
        logger.info("\nExecuting time-series cross-validation with Isotonic calibration...")
        logger.info(f"CV splits: {self.tscv.n_splits}, Gap: {self.lookahead_bars} bars")
        
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.base_model,
            method='isotonic',
            cv=self.tscv,
            n_jobs=-1
        )
        
        # 正式擬合模型
        logger.info("\nFitting Oracle model...")
        self.calibrated_model.fit(X, y)
        
        logger.info("Oracle model training and calibration completed!")
        
        # 訓練集評估
        train_proba = self.predict_real_probability(X)
        train_preds = (train_proba >= 0.5).astype(int)
        
        train_auc = roc_auc_score(y, train_proba)
        train_precision = precision_score(y, train_preds, zero_division=0)
        train_recall = recall_score(y, train_preds, zero_division=0)
        train_brier = brier_score_loss(y, train_proba)
        
        logger.info("\n=== Training Set Metrics ===")
        logger.info(f"AUC: {train_auc:.4f}")
        logger.info(f"Precision: {train_precision:.4f}")
        logger.info(f"Recall: {train_recall:.4f}")
        logger.info(f"Brier Score: {train_brier:.4f} (lower is better)")
        
        results = {
            'train_auc': train_auc,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_brier': train_brier,
            'feature_names': self.feature_names,
            'scale_pos_weight': scale_pos_weight
        }
        
        return results
    
    def predict_real_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        輸出校準後的真實統計機率
        
        Returns:
            機率陣列 (範圍 0-1)
            當輸出 0.7 時，代表歷史上 70% 的相似狀況最終獲利
        """
        if self.calibrated_model is None:
            raise ValueError("模型尚未訓練")
        
        # predict_proba 返回 [負樣本機率, 正樣本機率]
        return self.calibrated_model.predict_proba(X)[:, 1]
    
    def evaluate_oos(self, X_test: pd.DataFrame, y_test: pd.Series, 
                    threshold: float = 0.60) -> dict:
        """
        樣本外 (OOS) 嚴格測試
        
        Args:
            X_test: 測試集特徵
            y_test: 測試集標籤
            threshold: 決策閉值
        
        Returns:
            results: OOS 指標
        """
        logger.info("="*80)
        logger.info("ORACLE OOS EVALUATION")
        logger.info("="*80)
        
        proba = self.predict_real_probability(X_test)
        preds = (proba >= threshold).astype(int)
        
        auc = roc_auc_score(y_test, proba)
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        brier = brier_score_loss(y_test, proba)
        
        triggered_trades = preds.sum()
        
        logger.info(f"\nOOS Test Set: {len(X_test)} samples")
        logger.info(f"Threshold: {threshold}")
        logger.info("\n=== Oracle Brain OOS Report ===")
        logger.info(f"OOS AUC       : {auc:.4f} (模型對特徵的排序能力)")
        logger.info(f"OOS Precision : {precision:.2%} (閉值 {threshold} 下的真實勝率)")
        logger.info(f"OOS Recall    : {recall:.2%} (捕捉到的市場機會比例)")
        logger.info(f"Brier Score   : {brier:.4f} (機率校準質量, 越低越好)")
        logger.info(f"觸發交易次數  : {triggered_trades}")
        logger.info("==================================\n")
        
        # 機率分布分析
        logger.info("Probability Distribution:")
        prob_bins = [0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        prob_dist = pd.cut(proba, bins=prob_bins).value_counts().sort_index()
        for interval, count in prob_dist.items():
            logger.info(f"  {interval}: {count} samples")
        
        results = {
            'oos_auc': auc,
            'oos_precision': precision,
            'oos_recall': recall,
            'oos_brier': brier,
            'triggered_trades': triggered_trades,
            'threshold': threshold,
            'probabilities': proba,
            'predictions': preds
        }
        
        return results
    
    def save_model(self, path: str):
        """保存模型"""
        if self.calibrated_model is None:
            raise ValueError("模型尚未訓練")
        
        model_data = {
            'calibrated_model': self.calibrated_model,
            'feature_names': self.feature_names,
            'lookahead_bars': self.lookahead_bars
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Oracle model saved to {path}")
    
    def load_model(self, path: str):
        """載入模型"""
        model_data = joblib.load(path)
        
        self.calibrated_model = model_data['calibrated_model']
        self.feature_names = model_data['feature_names']
        self.lookahead_bars = model_data['lookahead_bars']
        
        logger.info(f"Oracle model loaded from {path}")
        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"Lookahead bars: {self.lookahead_bars}")