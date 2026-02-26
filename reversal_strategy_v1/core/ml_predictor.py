"""
ML Predictor using XGBoost
使用XGBoost進行信號驗證
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
import joblib
import json
from typing import Dict, Tuple, Optional

class MLPredictor:
    def __init__(self, config: dict):
        self.n_estimators = config.get('n_estimators', 200)
        self.max_depth = config.get('max_depth', 5)
        self.learning_rate = config.get('learning_rate', 0.05)
        
        self.model_long = None
        self.model_short = None
        self.feature_names = None
        
    def train(self, df: pd.DataFrame, feature_cols: list, 
             test_size: float = 0.2, oos_size: float = 0.1) -> Dict:
        """訓練做多和做空兩個模型"""
        
        df_clean = df[feature_cols + ['label']].dropna()
        
        total_samples = len(df_clean)
        oos_samples = int(total_samples * oos_size)
        train_val_samples = total_samples - oos_samples
        
        df_train_val = df_clean.iloc[:train_val_samples]
        df_oos = df_clean.iloc[train_val_samples:]
        
        X = df_train_val[feature_cols]
        y = df_train_val['label']
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        print(f"訓練集: {len(X_train)} | 驗證集: {len(X_val)} | OOS測試集: {len(df_oos)}")
        
        y_train_long = (y_train == 1).astype(int)
        y_train_short = (y_train == -1).astype(int)
        y_val_long = (y_val == 1).astype(int)
        y_val_short = (y_val == -1).astype(int)
        
        print("訓練做多模型...")
        self.model_long = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42
        )
        self.model_long.fit(X_train, y_train_long)
        
        print("訓練做空模型...")
        self.model_short = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42
        )
        self.model_short.fit(X_train, y_train_short)
        
        self.feature_names = feature_cols
        
        train_acc_long = accuracy_score(y_train_long, self.model_long.predict(X_train))
        val_acc_long = accuracy_score(y_val_long, self.model_long.predict(X_val))
        
        train_acc_short = accuracy_score(y_train_short, self.model_short.predict(X_train))
        val_acc_short = accuracy_score(y_val_short, self.model_short.predict(X_val))
        
        if len(df_oos) > 0:
            X_oos = df_oos[feature_cols]
            y_oos = df_oos['label']
            y_oos_long = (y_oos == 1).astype(int)
            y_oos_short = (y_oos == -1).astype(int)
            
            oos_acc_long = accuracy_score(y_oos_long, self.model_long.predict(X_oos))
            oos_acc_short = accuracy_score(y_oos_short, self.model_short.predict(X_oos))
        else:
            oos_acc_long = 0
            oos_acc_short = 0
        
        results = {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'oos_samples': len(df_oos),
            'train_acc_long': train_acc_long,
            'val_acc_long': val_acc_long,
            'oos_acc_long': oos_acc_long,
            'train_acc_short': train_acc_short,
            'val_acc_short': val_acc_short,
            'oos_acc_short': oos_acc_short
        }
        
        print(f"做多模型 - 訓練準確率: {train_acc_long:.4f} | 驗證準確率: {val_acc_long:.4f} | OOS準確率: {oos_acc_long:.4f}")
        print(f"做空模型 - 訓練準確率: {train_acc_short:.4f} | 驗證準確率: {val_acc_short:.4f} | OOS準確率: {oos_acc_short:.4f}")
        
        return results
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """預測信號是否有效"""
        df = df.copy()
        
        if self.model_long is None or self.model_short is None:
            raise ValueError("模型尚未訓練")
        
        X = df[self.feature_names].fillna(0)
        
        pred_long = self.model_long.predict(X)
        pred_short = self.model_short.predict(X)
        
        pred_long_proba = self.model_long.predict_proba(X)[:, 1]
        pred_short_proba = self.model_short.predict_proba(X)[:, 1]
        
        df['pred_long_valid'] = pred_long
        df['pred_short_valid'] = pred_short
        df['pred_long_confidence'] = pred_long_proba
        df['pred_short_confidence'] = pred_short_proba
        
        return df
    
    def save(self, save_dir: Path):
        """保存模型"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model_long, save_dir / 'model_long.pkl')
        joblib.dump(self.model_short, save_dir / 'model_short.pkl')
        
        with open(save_dir / 'feature_names.json', 'w') as f:
            json.dump(self.feature_names, f)
        
        print(f"模型已保存至: {save_dir}")
    
    def load(self, load_dir: Path):
        """加載模型"""
        load_dir = Path(load_dir)
        
        self.model_long = joblib.load(load_dir / 'model_long.pkl')
        self.model_short = joblib.load(load_dir / 'model_short.pkl')
        
        with open(load_dir / 'feature_names.json', 'r') as f:
            self.feature_names = json.load(f)
        
        print(f"模型已加載: {load_dir}")
