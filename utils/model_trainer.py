import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, 
    confusion_matrix, roc_curve, brier_score_loss
)
import joblib
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger('model_trainer', 'logs/model_trainer.log')

class ModelTrainer:
    """模型訓練器 - 支援 LightGBM 與 CatBoost (with Isotonic Calibration)"""
    
    def __init__(self, model_type='catboost'):
        """
        Args:
            model_type: 'lightgbm' or 'catboost'
        """
        logger.info(f"Initialized ModelTrainer with model_type={model_type}")
        self.model_type = model_type
        self.feature_cols = [
            # 1m 微觀特徵
            '1m_bull_sweep', '1m_bear_sweep', '1m_bull_bos', '1m_bear_bos', '1m_dist_to_poc',
            # 15m 戰術特徵
            '15m_z_score', '15m_bb_width_pct',
            # 1h/1d 宏觀特徵
            '1h_z_score', '1d_atr_pct',
            # 時間特徵
            'hour', 'day_of_week', 'session_asia', 'session_europe', 'session_us',
            'session_overlap', 'is_weekend', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            # Feature Engineering 2.0
            'rvol', 'sweep_with_high_rvol',
            'volatility_squeeze_ratio',
            'is_bullish_divergence', 'is_macd_bullish_divergence'
        ]
    
    def time_series_split(self, df: pd.DataFrame, train_ratio: float = 0.8) -> tuple:
        """嚴格時間序列切分"""
        logger.info(f"Performing time series split with ratio {train_ratio}")
        
        n = len(df)
        split_idx = int(n * train_ratio)
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"Train set: {len(train_df)} records")
        logger.info(f"Test set: {len(test_df)} records")
        
        return train_df, test_df
    
    def calculate_scale_pos_weight(self, y: pd.Series) -> float:
        """計算樣本權重"""
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        scale_pos_weight = neg_count / pos_count
        
        logger.info(f"Negative samples: {neg_count}, Positive samples: {pos_count}")
        logger.info(f"Scale_pos_weight: {scale_pos_weight:.4f}")
        
        return scale_pos_weight
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """準備特徵與目標"""
        logger.info("Preparing features and target")
        
        available_features = [col for col in self.feature_cols if col in df.columns]
        logger.info(f"Available features ({len(available_features)}): {available_features}")
        
        X = df[available_features].copy()
        y = df['target'].copy()
        
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, available_features
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                              target_recall: float = 0.55) -> dict:
        """決策閉值最佳化"""
        logger.info(f"Finding optimal threshold with target recall >= {target_recall}")
        
        thresholds = np.arange(0.3, 0.8, 0.05)
        results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            recall = recall_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'recall': recall,
                'precision': precision,
                'f1': f1
            })
            
            logger.info(f"Threshold {threshold:.2f}: Recall={recall:.4f}, Precision={precision:.4f}, F1={f1:.4f}")
        
        valid_results = [r for r in results if r['recall'] >= target_recall]
        
        if valid_results:
            optimal = max(valid_results, key=lambda x: x['precision'])
            logger.info(f"Optimal threshold: {optimal['threshold']:.2f} (Recall={optimal['recall']:.4f}, Precision={optimal['precision']:.4f})")
        else:
            optimal = max(results, key=lambda x: x['f1'])
            logger.warning(f"No threshold meets target recall, using best F1: {optimal['threshold']:.2f}")
        
        return {
            'optimal': optimal,
            'all_results': results
        }
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test, params: dict) -> dict:
        """訓練 LightGBM 模型"""
        logger.info("Training LightGBM model...")
        
        scale_pos_weight = self.calculate_scale_pos_weight(y_train)
        
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': params.get('learning_rate', 0.01),
            'max_depth': params.get('max_depth', 6),
            'num_leaves': params.get('num_leaves', 31),
            'min_child_samples': params.get('min_child_samples', 50),
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'scale_pos_weight': scale_pos_weight,
            'verbose': -1,
            'random_state': 42
        }
        
        logger.info(f"LightGBM parameters: {lgb_params}")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=params.get('n_estimators', 1000),
            valid_sets=[train_data, test_data],
            valid_names=['train', 'test'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=params.get('early_stopping_rounds', 50)),
                lgb.log_evaluation(period=100)
            ]
        )
        
        logger.info(f"Training completed at iteration {model.best_iteration}")
        
        y_train_pred_proba = model.predict(X_train, num_iteration=model.best_iteration)
        y_test_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
        
        return model, y_train_pred_proba, y_test_pred_proba, lgb_params
    
    def train_catboost(self, X_train, y_train, X_test, y_test, params: dict) -> dict:
        """訓練 CatBoost 模型 (with Isotonic Calibration)"""
        logger.info("Training CatBoost model with Isotonic calibration...")
        
        scale_pos_weight = self.calculate_scale_pos_weight(y_train)
        
        # CatBoost 基礎模型
        base_model = CatBoostClassifier(
            iterations=params.get('iterations', 1000),
            learning_rate=params.get('learning_rate', 0.03),
            depth=params.get('depth', 5),
            l2_leaf_reg=params.get('l2_leaf_reg', 5.0),
            scale_pos_weight=scale_pos_weight,
            eval_metric='AUC',
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1
        )
        
        logger.info(f"CatBoost base model parameters:")
        logger.info(f"  iterations: {params.get('iterations', 1000)}")
        logger.info(f"  learning_rate: {params.get('learning_rate', 0.03)}")
        logger.info(f"  depth: {params.get('depth', 5)}")
        logger.info(f"  l2_leaf_reg: {params.get('l2_leaf_reg', 5.0)}")
        logger.info(f"  scale_pos_weight: {scale_pos_weight:.4f}")
        
        # Isotonic 機率校準
        lookahead_bars = params.get('lookahead_bars', 16)
        tscv = TimeSeriesSplit(n_splits=5, gap=lookahead_bars)
        
        logger.info(f"Isotonic calibration with TimeSeriesSplit (gap={lookahead_bars})")
        
        calibrated_model = CalibratedClassifierCV(
            estimator=base_model,
            method='isotonic',
            cv=tscv,
            n_jobs=-1
        )
        
        # 訓練校準模型
        calibrated_model.fit(X_train, y_train)
        
        logger.info("CatBoost training and calibration completed")
        
        # 預測
        y_train_pred_proba = calibrated_model.predict_proba(X_train)[:, 1]
        y_test_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
        
        return calibrated_model, y_train_pred_proba, y_test_pred_proba, base_model.get_params()
    
    def train(self, features_df: pd.DataFrame, params: dict) -> dict:
        """完整訓練流程"""
        logger.info("="*80)
        logger.info(f"Starting training pipeline with {self.model_type}")
        logger.info(f"Input shape: {features_df.shape}")
        logger.info("="*80)
        
        try:
            # 1. 時間序列切分
            train_df, test_df = self.time_series_split(features_df)
            
            # 2. 準備特徵
            X_train, y_train, feature_names = self.prepare_features(train_df)
            X_test, y_test, _ = self.prepare_features(test_df)
            
            # 3. 訓練模型
            if self.model_type == 'catboost':
                model, y_train_pred_proba, y_test_pred_proba, model_params = self.train_catboost(
                    X_train, y_train, X_test, y_test, params
                )
            else:
                model, y_train_pred_proba, y_test_pred_proba, model_params = self.train_lightgbm(
                    X_train, y_train, X_test, y_test, params
                )
            
            # 4. 計算指標
            train_auc = roc_auc_score(y_train, y_train_pred_proba)
            test_auc = roc_auc_score(y_test, y_test_pred_proba)
            
            # Brier Score (機率校準質量)
            train_brier = brier_score_loss(y_train, y_train_pred_proba)
            test_brier = brier_score_loss(y_test, y_test_pred_proba)
            
            # 閉值最佳化
            threshold_results = self.find_optimal_threshold(y_test, y_test_pred_proba, target_recall=0.55)
            optimal_threshold = threshold_results['optimal']['threshold']
            
            y_test_pred_optimal = (y_test_pred_proba >= optimal_threshold).astype(int)
            test_recall_optimal = recall_score(y_test, y_test_pred_optimal)
            test_precision_optimal = precision_score(y_test, y_test_pred_optimal)
            
            cm = confusion_matrix(y_test, y_test_pred_optimal)
            tn, fp, fn, tp = cm.ravel()
            
            logger.info("="*80)
            logger.info("TRAINING RESULTS")
            logger.info("="*80)
            logger.info(f"Train AUC: {train_auc:.4f}")
            logger.info(f"Test AUC: {test_auc:.4f}")
            logger.info(f"Train Brier Score: {train_brier:.4f}")
            logger.info(f"Test Brier Score: {test_brier:.4f} (lower is better)")
            logger.info(f"Test Recall (optimal): {test_recall_optimal:.4f}")
            logger.info(f"Test Precision (optimal): {test_precision_optimal:.4f}")
            logger.info(f"Optimal threshold: {optimal_threshold:.2f}")
            logger.info(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            logger.info("="*80)
            
            # 5. 特徵重要性
            if self.model_type == 'lightgbm':
                feature_importance = model.feature_importance(importance_type='gain')
            else:
                # CatBoost 校準後的模型，取基礎模型的特徵重要性
                base_estimators = model.calibrated_classifiers_
                feature_importance = base_estimators[0].estimator.feature_importances_
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            logger.info("Top 15 features:")
            for idx, row in importance_df.head(15).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.2f}")
            
            # 6. 保存模型
            model_dir = Path("models_output")
            model_dir.mkdir(exist_ok=True)
            
            model_suffix = 'catboost' if self.model_type == 'catboost' else 'lgb'
            model_path = model_dir / f"{model_suffix}_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            
            if self.model_type == 'catboost':
                joblib.dump(model, str(model_path))
            else:
                model.save_model(str(model_path).replace('.pkl', '.txt'))
                model_path = model_path.parent / (model_path.stem + '.txt')
            
            logger.info(f"Model saved to {model_path}")
            
            # 7. 返回結果
            results = {
                'model_path': str(model_path),
                'model_type': self.model_type,
                'metrics': {
                    'train_auc': float(train_auc),
                    'test_auc': float(test_auc),
                    'train_brier': float(train_brier),
                    'test_brier': float(test_brier),
                    'test_recall': float(test_recall_optimal),
                    'test_precision': float(test_precision_optimal),
                    'optimal_threshold': float(optimal_threshold)
                },
                'confusion_matrix': {
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn),
                    'tp': int(tp)
                },
                'threshold_analysis': threshold_results,
                'feature_importance': {
                    'features': importance_df['feature'].tolist(),
                    'importances': importance_df['importance'].tolist()
                },
                'params': model_params
            }
            
            return results
        
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            return None