import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, 
    confusion_matrix, roc_curve
)
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger('model_trainer', 'logs/model_trainer.log')

class ModelTrainer:
    """模型訓練器 - 嚴格時間序列切分與防過擬合"""
    
    def __init__(self):
        logger.info("Initialized ModelTrainer")
        self.feature_cols = [
            # 1m 微觀特徵
            '1m_bull_sweep', '1m_bear_sweep', '1m_bull_bos', '1m_bear_bos', '1m_dist_to_poc',
            # 15m 戰術特徵
            '15m_z_score', '15m_bb_width_pct',
            # 1h/1d 宏觀特徵
            '1h_z_score', '1d_atr_pct',
            # 時間特徵 (策略 1)
            'hour', 'day_of_week', 'session_asia', 'session_europe', 'session_us',
            'session_overlap', 'is_weekend', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
        ]
    
    def time_series_split(self, df: pd.DataFrame, train_ratio: float = 0.8) -> tuple:
        """嚴格時間序列切分 - 絕不打亂"""
        logger.info(f"Performing time series split with ratio {train_ratio}")
        
        n = len(df)
        split_idx = int(n * train_ratio)
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"Train set: {len(train_df)} records")
        logger.info(f"Test set: {len(test_df)} records")
        
        return train_df, test_df
    
    def calculate_scale_pos_weight(self, y: pd.Series) -> float:
        """計算樣本權重 - 處理不平衡"""
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        scale_pos_weight = neg_count / pos_count
        
        logger.info(f"Negative samples: {neg_count}, Positive samples: {pos_count}")
        logger.info(f"Scale_pos_weight: {scale_pos_weight:.4f}")
        
        return scale_pos_weight
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """準備特徵與目標"""
        logger.info("Preparing features and target")
        
        # 築選存在的特徵
        available_features = [col for col in self.feature_cols if col in df.columns]
        logger.info(f"Available features ({len(available_features)}): {available_features}")
        
        X = df[available_features].copy()
        y = df['target'].copy()
        
        # 處理無限值與 NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, available_features
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                              target_recall: float = 0.55) -> dict:
        """策略 3: 決策閉值最佳化"""
        logger.info(f"Finding optimal threshold with target recall >= {target_recall}")
        
        # 計算不同閉值下的指標
        thresholds = np.arange(0.3, 0.8, 0.05)
        results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            recall = recall_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            
            # F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'recall': recall,
                'precision': precision,
                'f1': f1
            })
            
            logger.info(f"Threshold {threshold:.2f}: Recall={recall:.4f}, Precision={precision:.4f}, F1={f1:.4f}")
        
        # 找到滿足 recall 要求且 precision 最高的閉值
        valid_results = [r for r in results if r['recall'] >= target_recall]
        
        if valid_results:
            optimal = max(valid_results, key=lambda x: x['precision'])
            logger.info(f"Optimal threshold: {optimal['threshold']:.2f} (Recall={optimal['recall']:.4f}, Precision={optimal['precision']:.4f})")
        else:
            # 如果沒有滿足 recall 的，選擇 F1 最高的
            optimal = max(results, key=lambda x: x['f1'])
            logger.warning(f"No threshold meets target recall, using best F1: {optimal['threshold']:.2f}")
        
        return {
            'optimal': optimal,
            'all_results': results
        }
    
    def train(self, features_df: pd.DataFrame, params: dict) -> dict:
        """完整訓練流程"""
        logger.info("Starting training pipeline")
        logger.info(f"Input shape: {features_df.shape}")
        
        try:
            # 1. 時間序列切分
            train_df, test_df = self.time_series_split(features_df)
            
            # 2. 準備特徵
            X_train, y_train, feature_names = self.prepare_features(train_df)
            X_test, y_test, _ = self.prepare_features(test_df)
            
            # 3. 計算樣本權重
            scale_pos_weight = self.calculate_scale_pos_weight(y_train)
            
            # 4. 設定 LightGBM 參數 (策略 2: 進階超參數)
            lgb_params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'learning_rate': params.get('learning_rate', 0.01),  # 降低學習率
                'max_depth': params.get('max_depth', 6),  # 增加深度
                'num_leaves': params.get('num_leaves', 31),
                'min_child_samples': params.get('min_child_samples', 50),
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,  # L1 正則化
                'reg_lambda': 0.1,  # L2 正則化
                'scale_pos_weight': scale_pos_weight,
                'verbose': -1,
                'random_state': 42
            }
            
            logger.info(f"LightGBM parameters: {lgb_params}")
            
            # 5. 訓練模型
            logger.info("Training LightGBM model...")
            
            train_data = lgb.Dataset(X_train, label=y_train)
            test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=params.get('n_estimators', 1000),  # 增加樹數量
                valid_sets=[train_data, test_data],
                valid_names=['train', 'test'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=params.get('early_stopping_rounds', 50)),
                    lgb.log_evaluation(period=100)
                ]
            )
            
            logger.info(f"Training completed at iteration {model.best_iteration}")
            
            # 6. 預測
            y_train_pred_proba = model.predict(X_train, num_iteration=model.best_iteration)
            y_test_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
            
            # 7. 計算指標 (預設閉值 0.5)
            train_auc = roc_auc_score(y_train, y_train_pred_proba)
            test_auc = roc_auc_score(y_test, y_test_pred_proba)
            
            y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
            test_recall = recall_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred)
            
            # 8. 閉值最佳化 (策略 3)
            threshold_results = self.find_optimal_threshold(y_test, y_test_pred_proba, target_recall=0.55)
            optimal_threshold = threshold_results['optimal']['threshold']
            
            # 使用最佳閉值重新計算
            y_test_pred_optimal = (y_test_pred_proba >= optimal_threshold).astype(int)
            test_recall_optimal = recall_score(y_test, y_test_pred_optimal)
            test_precision_optimal = precision_score(y_test, y_test_pred_optimal)
            
            # 混淆矩陣 (最佳閉值)
            cm = confusion_matrix(y_test, y_test_pred_optimal)
            tn, fp, fn, tp = cm.ravel()
            
            logger.info(f"Train AUC: {train_auc:.4f}")
            logger.info(f"Test AUC: {test_auc:.4f}")
            logger.info(f"Test Recall (0.5): {test_recall:.4f}, Precision (0.5): {test_precision:.4f}")
            logger.info(f"Test Recall (optimal): {test_recall_optimal:.4f}, Precision (optimal): {test_precision_optimal:.4f}")
            logger.info(f"Optimal threshold: {optimal_threshold:.2f}")
            
            # 9. 特徵重要性
            feature_importance = model.feature_importance(importance_type='gain')
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            logger.info("Top 10 features:")
            for idx, row in importance_df.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.2f}")
            
            # 10. 保存模型
            model_dir = Path("models_output")
            model_dir.mkdir(exist_ok=True)
            
            model_path = model_dir / f"lgb_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
            model.save_model(str(model_path))
            logger.info(f"Model saved to {model_path}")
            
            # 11. 返回結果
            results = {
                'model_path': str(model_path),
                'metrics': {
                    'train_auc': float(train_auc),
                    'test_auc': float(test_auc),
                    'test_recall': float(test_recall_optimal),
                    'test_precision': float(test_precision_optimal),
                    'optimal_threshold': float(optimal_threshold)
                },
                'metrics_default': {
                    'test_recall': float(test_recall),
                    'test_precision': float(test_precision)
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
                'params': lgb_params,
                'best_iteration': model.best_iteration
            }
            
            return results
        
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            return None