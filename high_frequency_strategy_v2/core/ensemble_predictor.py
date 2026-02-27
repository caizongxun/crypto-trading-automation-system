"""
Ensemble Predictor combining Transformer, LightGBM, and other models
集成預測器
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import lightgbm as lgb
from pathlib import Path
import joblib

class EnsemblePredictor:
    """集成多個模型的預測器"""
    def __init__(self, config: Dict):
        self.config = config
        self.use_transformer = config.get('use_transformer', True)
        self.use_lgb = config.get('use_lgb', True)
        self.ensemble_method = config.get('ensemble_method', 'weighted_avg')
        
        # 模型權重
        self.weights = config.get('weights', {
            'transformer': 0.5,
            'lgb': 0.5
        })
        
        self.transformer_model = None
        self.lgb_model = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             X_train_seq: np.ndarray = None,
             X_val_seq: np.ndarray = None) -> Dict:
        """訓練所有子模型"""
        results = {}
        
        # 訓練LightGBM
        if self.use_lgb:
            print("\n訓練 LightGBM...")
            lgb_params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
            
            train_data = lgb.Dataset(X_train, label=y_train + 1)
            val_data = lgb.Dataset(X_val, label=y_val + 1, reference=train_data)
            
            self.lgb_model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=500,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'val'],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
            )
            
            results['lgb_trained'] = True
        
        # 訓練Transformer (如果提供了序列數據)
        if self.use_transformer and X_train_seq is not None:
            print("\n訓練 Transformer...")
            from .transformer_model import TransformerTrainer
            
            transformer_config = {
                'feature_dim': X_train_seq.shape[2],
                'd_model': 128,
                'nhead': 8,
                'num_layers': 4,
                'dim_feedforward': 512,
                'dropout': 0.1,
                'learning_rate': 0.001
            }
            
            self.transformer_model = TransformerTrainer(transformer_config)
            history = self.transformer_model.train(
                X_train_seq, y_train,
                X_val_seq, y_val,
                epochs=50, batch_size=64
            )
            
            results['transformer_trained'] = True
            results['transformer_history'] = history
        
        return results
    
    def predict(self, X: np.ndarray, X_seq: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """集成預測"""
        predictions = []
        confidences = []
        
        # LightGBM預測
        if self.lgb_model is not None:
            lgb_probs = self.lgb_model.predict(X)
            lgb_pred = np.argmax(lgb_probs, axis=1) - 1
            lgb_conf = np.max(lgb_probs, axis=1)
            predictions.append(lgb_pred * self.weights['lgb'])
            confidences.append(lgb_conf * self.weights['lgb'])
        
        # Transformer預測
        if self.transformer_model is not None and X_seq is not None:
            trans_pred, trans_probs = self.transformer_model.predict(X_seq)
            trans_conf = np.max(trans_probs, axis=1)
            predictions.append(trans_pred * self.weights['transformer'])
            confidences.append(trans_conf * self.weights['transformer'])
        
        # 集成結果
        if self.ensemble_method == 'weighted_avg':
            final_pred = np.round(np.sum(predictions, axis=0)).astype(int)
            final_conf = np.sum(confidences, axis=0) / len(confidences)
        elif self.ensemble_method == 'voting':
            final_pred = np.round(np.median(predictions, axis=0)).astype(int)
            final_conf = np.mean(confidences, axis=0)
        else:
            final_pred = predictions[0]
            final_conf = confidences[0]
        
        return final_pred, final_conf
    
    def save(self, path: Path):
        """保存模型"""
        path.mkdir(parents=True, exist_ok=True)
        
        if self.lgb_model is not None:
            self.lgb_model.save_model(str(path / 'lgb_model.txt'))
        
        if self.transformer_model is not None:
            self.transformer_model.save(path)
        
        joblib.dump(self.config, path / 'ensemble_config.pkl')
    
    def load(self, path: Path):
        """加載模型"""
        self.config = joblib.load(path / 'ensemble_config.pkl')
        
        if (path / 'lgb_model.txt').exists():
            self.lgb_model = lgb.Booster(model_file=str(path / 'lgb_model.txt'))
        
        if (path / 'transformer_model.pt').exists():
            from .transformer_model import TransformerTrainer
            import torch
            checkpoint = torch.load(path / 'transformer_model.pt')
            self.transformer_model = TransformerTrainer(checkpoint['config'])
            self.transformer_model.load(path)
