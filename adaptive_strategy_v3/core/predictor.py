"""
V3 Predictor - LightGBM with Improved Parameters
LightGBM預測器 (防止過擬合)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Tuple
from pathlib import Path
import joblib
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

class Predictor:
    """
    LightGBM預測器
    
    核心改進:
    1. 降低學習率
    2. 增加最小葉節點樣本
    3. 特徵抽樣
    4. 類別權重平衡
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.feature_importance = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        訓練LightGBM模型
        
        Returns:
            training_results
        """
        print("\n[LightGBM訓練] 開始...")
        
        # 標籤轉換: -1,0,1 -> 0,1,2
        y_train_lgb = y_train + 1
        y_val_lgb = y_val + 1
        
        # 計算類別權重
        class_weights = self._calculate_class_weights(y_train_lgb)
        
        # 生成樣本權重陣列
        sample_weights_train = np.array([class_weights[int(label)] for label in y_train_lgb])
        
        # LightGBM參數 (防止過擬合)
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            
            # 核心參數調整
            'learning_rate': 0.03,       # 降低學習率
            'num_leaves': 31,
            'max_depth': 6,              # 限制深度
            'min_data_in_leaf': 100,     # 增加最小葉節點樣本
            
            # 正則化
            'feature_fraction': 0.7,     # 特徵抽樣
            'bagging_fraction': 0.7,     # 數據抽樣
            'bagging_freq': 5,
            'lambda_l1': 0.1,            # L1正則化
            'lambda_l2': 0.1,            # L2正則化
            
            # 其他
            'verbose': -1,
            'seed': 42
        }
        
        # 創建數據集
        train_data = lgb.Dataset(
            X_train, 
            label=y_train_lgb,
            weight=sample_weights_train
        )
        val_data = lgb.Dataset(
            X_val,
            label=y_val_lgb,
            reference=train_data
        )
        
        # 訓練
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )
        
        print("[OK] 訓練完成")
        
        # 特徵重要性
        self.feature_importance = self.model.feature_importance(importance_type='gain')
        
        # 評估
        results = self._evaluate(X_train, y_train, X_val, y_val)
        
        return results
    
    def _calculate_class_weights(self, y: np.ndarray) -> Dict:
        """
        計算類別權重
        """
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        weights = {}
        for cls, count in zip(unique, counts):
            # 反比例加權: 樣本少的類別獲得更高權重
            weights[int(cls)] = total / (len(unique) * count)
        
        # 歸一化
        max_weight = max(weights.values())
        for cls in weights:
            weights[cls] /= max_weight
        
        return weights
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        預測
        
        Returns:
            predictions: -1, 0, 1
            confidences: 每個樣本的最高類別機率
        """
        if self.model is None:
            raise ValueError("模型尚未訓練")
        
        # 預測機率 (n_samples, 3)
        probs = self.model.predict(X)
        
        # 轉換回 -1, 0, 1
        pred_class = np.argmax(probs, axis=1) - 1
        
        # 信心度 = 最高類別機率
        confidences = np.max(probs, axis=1)
        
        return pred_class, confidences
    
    def _evaluate(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        評估模型
        """
        print("\n[評估] 開始...")
        
        # 預測
        y_pred_train, conf_train = self.predict(X_train)
        y_pred_val, conf_val = self.predict(X_val)
        
        # 訓練集指標
        train_report = classification_report(
            y_train, y_pred_train,
            target_names=['Short', 'Neutral', 'Long'],
            output_dict=True,
            zero_division=0
        )
        
        # 驗證集指標
        val_report = classification_report(
            y_val, y_pred_val,
            target_names=['Short', 'Neutral', 'Long'],
            output_dict=True,
            zero_division=0
        )
        
        # AUC
        try:
            y_val_bin = label_binarize(y_val, classes=[-1, 0, 1])
            y_pred_val_bin = label_binarize(y_pred_val, classes=[-1, 0, 1])
            val_auc = roc_auc_score(y_val_bin, y_pred_val_bin, average='macro', multi_class='ovr')
        except:
            val_auc = 0
        
        # 信心度統計
        avg_conf_long = conf_val[y_pred_val == 1].mean() if (y_pred_val == 1).sum() > 0 else 0
        avg_conf_short = conf_val[y_pred_val == -1].mean() if (y_pred_val == -1).sum() > 0 else 0
        avg_conf_neutral = conf_val[y_pred_val == 0].mean() if (y_pred_val == 0).sum() > 0 else 0
        
        print("\n[訓練集績效]")
        print(f"準確率: {train_report['accuracy']:.3f}")
        print(f"Macro Precision: {train_report['macro avg']['precision']:.3f}")
        print(f"Macro Recall: {train_report['macro avg']['recall']:.3f}")
        print(f"Macro F1: {train_report['macro avg']['f1-score']:.3f}")
        
        print("\n[驗證集績效]")
        print(f"準確率: {val_report['accuracy']:.3f}")
        print(f"Macro Precision: {val_report['macro avg']['precision']:.3f}")
        print(f"Macro Recall: {val_report['macro avg']['recall']:.3f}")
        print(f"Macro F1: {val_report['macro avg']['f1-score']:.3f}")
        print(f"AUC (OvR): {val_auc:.3f}")
        
        print("\n[信心度統計]")
        print(f"做多平均: {avg_conf_long:.1%}")
        print(f"做空平均: {avg_conf_short:.1%}")
        print(f"中立平均: {avg_conf_neutral:.1%}")
        
        # 警告
        if val_report['accuracy'] - train_report['accuracy'] < -0.05:
            print("\n[警告] 驗證集準確率明顯低於訓練集,可能過擬合")
        
        if avg_conf_long < 0.3 or avg_conf_short < 0.3:
            print("[警告] 交易信號平均信心度<30%,可能產生太少交易")
        
        return {
            'train_accuracy': train_report['accuracy'],
            'val_accuracy': val_report['accuracy'],
            'val_auc': val_auc,
            'val_precision': val_report['macro avg']['precision'],
            'val_recall': val_report['macro avg']['recall'],
            'val_f1': val_report['macro avg']['f1-score'],
            'avg_conf_long': float(avg_conf_long),
            'avg_conf_short': float(avg_conf_short),
            'avg_conf_neutral': float(avg_conf_neutral),
            'train_report': train_report,
            'val_report': val_report
        }
    
    def get_feature_importance(self, feature_names: list = None, top_k: int = 20) -> pd.DataFrame:
        """
        獲取特徵重要性
        """
        if self.feature_importance is None:
            return None
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_k)
    
    def save(self, path: Path):
        """
        保存模型
        """
        path.mkdir(parents=True, exist_ok=True)
        
        if self.model is not None:
            self.model.save_model(str(path / 'lgb_model.txt'))
        
        joblib.dump(self.config, path / 'predictor_config.pkl')
        
        if self.feature_importance is not None:
            np.save(path / 'feature_importance.npy', self.feature_importance)
    
    def load(self, path: Path):
        """
        加載模型
        """
        self.config = joblib.load(path / 'predictor_config.pkl')
        
        if (path / 'lgb_model.txt').exists():
            self.model = lgb.Booster(model_file=str(path / 'lgb_model.txt'))
        
        if (path / 'feature_importance.npy').exists():
            self.feature_importance = np.load(path / 'feature_importance.npy')
