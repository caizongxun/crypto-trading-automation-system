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
import importlib.util

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
        
        # 信心度閾值 (可從外部調整)
        self.confidence_threshold = config.get('confidence_threshold', 0.35)
        
        self.transformer_model = None
        self.lgb_model = None
        
        # 診斷模式
        self.debug_mode = False
    
    def _load_transformer_module(self):
        """動態加載Transformer模組"""
        try:
            from .transformer_model import TransformerTrainer
            return TransformerTrainer
        except ImportError:
            transformer_path = Path(__file__).parent / 'transformer_model.py'
            if transformer_path.exists():
                spec = importlib.util.spec_from_file_location('transformer_model', transformer_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module.TransformerTrainer
            else:
                raise ImportError(f"Cannot find transformer_model.py at {transformer_path}")
    
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
            
            # 標籤轉換: -1,0,1 -> 0,1,2
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
            print("[OK] LightGBM訓練完成")
        
        # 訓練Transformer (如果提供了序列數據)
        if self.use_transformer and X_train_seq is not None:
            print("\n訓練 Transformer...")
            try:
                TransformerTrainer = self._load_transformer_module()
                
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
                print("[OK] Transformer訓練完成")
            except Exception as e:
                print(f"[警告] Transformer訓練失敗: {str(e)}")
                print("繼續使用僅LightGBM模式")
                results['transformer_trained'] = False
        
        return results
    
    def predict(self, X: np.ndarray, X_seq: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        集成預測
        
        Returns:
            predictions: -1 (做空), 0 (中立), 1 (做多)
            confidences: 信心度 (0-1)
        """
        predictions = []
        all_probs = []  # 保存原始機率
        
        # LightGBM預測
        if self.lgb_model is not None:
            lgb_probs = self.lgb_model.predict(X)  # shape: (n_samples, 3) 類別機率
            
            if self.debug_mode:
                print(f"[DEBUG] LGB probs shape: {lgb_probs.shape}")
                print(f"[DEBUG] LGB probs sample: {lgb_probs[:5]}")
                print(f"[DEBUG] LGB probs min/max: {lgb_probs.min():.4f}/{lgb_probs.max():.4f}")
            
            # 類別 0,1,2 對應 -1,0,1
            lgb_pred_class = np.argmax(lgb_probs, axis=1) - 1  # 轉換回 -1,0,1
            lgb_conf = np.max(lgb_probs, axis=1)  # 每個樣本的最高類別機率
            
            if self.debug_mode:
                print(f"[DEBUG] LGB pred before filter: {lgb_pred_class[:20]}")
                print(f"[DEBUG] LGB conf: min={lgb_conf.min():.4f}, max={lgb_conf.max():.4f}, mean={lgb_conf.mean():.4f}")
                print(f"[DEBUG] Confidence threshold: {self.confidence_threshold}")
                print(f"[DEBUG] Samples above threshold: {(lgb_conf >= self.confidence_threshold).sum()} / {len(lgb_conf)}")
            
            # 信心度過濾: 只有高信心度才交易
            lgb_pred_filtered = np.where(lgb_conf >= self.confidence_threshold, lgb_pred_class, 0)
            
            if self.debug_mode:
                print(f"[DEBUG] LGB pred after filter: {lgb_pred_filtered[:20]}")
                print(f"[DEBUG] Long signals: {(lgb_pred_filtered == 1).sum()}")
                print(f"[DEBUG] Short signals: {(lgb_pred_filtered == -1).sum()}")
                print(f"[DEBUG] Neutral signals: {(lgb_pred_filtered == 0).sum()}")
            
            predictions.append(lgb_pred_filtered)
            all_probs.append(lgb_conf)
        
        # Transformer預測
        if self.transformer_model is not None and X_seq is not None:
            trans_pred, trans_probs = self.transformer_model.predict(X_seq)
            trans_conf = np.max(trans_probs, axis=1)
            predictions.append(trans_pred)
            all_probs.append(trans_conf)
        
        # 集成結果
        if len(predictions) == 1:
            # 單一模型
            final_pred = predictions[0]
            final_conf = all_probs[0]
        elif self.ensemble_method == 'weighted_avg':
            # 加權平均 (但不影響信心度)
            weighted_preds = []
            for i, pred in enumerate(predictions):
                model_name = 'lgb' if i == 0 else 'transformer'
                weight = self.weights.get(model_name, 0.5)
                weighted_preds.append(pred * weight)
            final_pred = np.round(np.sum(weighted_preds, axis=0)).astype(int)
            # 信心度取平均，不加權
            final_conf = np.mean(all_probs, axis=0)
        elif self.ensemble_method == 'voting':
            final_pred = np.round(np.median(predictions, axis=0)).astype(int)
            final_conf = np.mean(all_probs, axis=0)
        else:
            final_pred = predictions[0]
            final_conf = all_probs[0]
        
        # 確保輸出範圍在 -1, 0, 1
        final_pred = np.clip(final_pred, -1, 1)
        
        if self.debug_mode:
            print(f"[DEBUG] Final pred: {final_pred[:20]}")
            print(f"[DEBUG] Final conf: min={final_conf.min():.4f}, max={final_conf.max():.4f}, mean={final_conf.mean():.4f}")
        
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
            try:
                TransformerTrainer = self._load_transformer_module()
                import torch
                checkpoint = torch.load(path / 'transformer_model.pt')
                self.transformer_model = TransformerTrainer(checkpoint['config'])
                self.transformer_model.load(path)
            except Exception as e:
                print(f"[警告] Transformer模型加載失敗: {str(e)}")
