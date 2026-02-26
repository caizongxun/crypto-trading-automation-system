"""
Ensemble Predictor
集成預測器 - 結合Transformer + LSTM + LightGBM
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from pathlib import Path
import lightgbm as lgb
import joblib

class EnsemblePredictor:
    """
    集成三個模型：
    1. Transformer - 時序注意力
    2. LSTM - 長期記憶
    3. LightGBM - 快速決策
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.transformer = None
        self.lstm = None
        self.lgb_model = None
        self.voting_weights = config.get('voting_weights', [0.4, 0.3, 0.3])
        self.min_confidence = config.get('min_confidence', 0.6)
        
    def train(self, X_seq: np.ndarray, X_flat: np.ndarray, y: np.ndarray,
              val_size: float = 0.2) -> Dict:
        """
        訓練所有模型
        X_seq: 序列特徵 (samples, seq_len, features) - 用於Transformer/LSTM
        X_flat: 平面特徵 (samples, features) - 用於LightGBM
        """
        from sklearn.model_selection import train_test_split
        
        # 分割訓練/驗證集
        indices = np.arange(len(y))
        train_idx, val_idx = train_test_split(indices, test_size=val_size, stratify=y)
        
        X_seq_train, X_seq_val = X_seq[train_idx], X_seq[val_idx]
        X_flat_train, X_flat_val = X_flat[train_idx], X_flat[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        results = {}
        
        # 1. 訓練Transformer
        print("\n=== 訓練Transformer ===")
        from .transformer_model import TransformerPredictor
        self.transformer = TransformerPredictor(self.config.get('transformer', {}))
        self.transformer.build_model(X_seq.shape[-1])
        trans_history = self.transformer.train(X_seq_train, y_train, X_seq_val, y_val)
        results['transformer'] = trans_history
        
        # 2. 訓練LSTM
        print("\n=== 訓練LSTM ===")
        from .lstm_model import LSTMPredictor
        self.lstm = LSTMPredictor(self.config.get('lstm', {}))
        self.lstm.build_model(X_seq.shape[-1])
        lstm_history = self.lstm.train(X_seq_train, y_train, X_seq_val, y_val)
        results['lstm'] = lstm_history
        
        # 3. 訓練LightGBM
        print("\n=== 訓練LightGBM ===")
        lgb_train = lgb.Dataset(X_flat_train, y_train)
        lgb_val = lgb.Dataset(X_flat_val, y_val, reference=lgb_train)
        
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        self.lgb_model = lgb.train(
            params,
            lgb_train,
            num_boost_round=200,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(10)]
        )
        results['lightgbm'] = {'best_iteration': self.lgb_model.best_iteration}
        
        # 驗證集評估
        print("\n=== 集成模型評估 ===")
        ensemble_preds, ensemble_probs = self.predict(X_seq_val, X_flat_val)
        accuracy = (ensemble_preds == y_val).mean()
        print(f"集成模型驗證集準確率: {accuracy:.4f}")
        results['ensemble_accuracy'] = accuracy
        
        return results
    
    def predict(self, X_seq: np.ndarray, X_flat: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        集成預測
        返回: 預測類別, 詳細機率
        """
        # Transformer預測
        _, trans_probs = self.transformer.predict(X_seq)
        
        # LSTM預測
        _, lstm_probs = self.lstm.predict(X_seq)
        
        # LightGBM預測
        lgb_probs = self.lgb_model.predict(X_flat)
        
        # 加權投票
        ensemble_probs = (
            trans_probs * self.voting_weights[0] +
            lstm_probs * self.voting_weights[1] +
            lgb_probs * self.voting_weights[2]
        )
        
        # 最終預測
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        # 置信度過濾
        max_probs = np.max(ensemble_probs, axis=1)
        low_confidence_mask = max_probs < self.min_confidence
        ensemble_preds[low_confidence_mask] = 1  # 低置信度設為NEUTRAL
        
        result = {
            'ensemble_probs': ensemble_probs,
            'transformer_probs': trans_probs,
            'lstm_probs': lstm_probs,
            'lgb_probs': lgb_probs,
            'confidence': max_probs
        }
        
        return ensemble_preds, result
    
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.transformer.save(path)
        self.lstm.save(path)
        joblib.dump(self.lgb_model, path / 'lgb_model.pkl')
        joblib.dump(self.config, path / 'ensemble_config.pkl')
        print(f"集成模型已保存至 {path}")
    
    def load(self, path: Path):
        self.config = joblib.load(path / 'ensemble_config.pkl')
        
        from .transformer_model import TransformerPredictor
        from .lstm_model import LSTMPredictor
        
        self.transformer = TransformerPredictor(self.config.get('transformer', {}))
        self.transformer.load(path)
        
        self.lstm = LSTMPredictor(self.config.get('lstm', {}))
        self.lstm.load(path)
        
        self.lgb_model = joblib.load(path / 'lgb_model.pkl')
        print(f"集成模型已加載從 {path}")
