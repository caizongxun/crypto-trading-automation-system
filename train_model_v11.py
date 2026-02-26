#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11 模型訓練腳本
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def train_v11_model(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    train_split: float = 0.8,
    model_type: str = 'CatBoost',
    use_class_weights: bool = True,
    train_long: bool = True,
    train_short: bool = True
) -> dict:
    """
    V11 模型訓練
    """
    
    results = {}
    
    # 準備特徵
    feature_cols = df.attrs.get('feature_columns', [])
    
    if not feature_cols:
        raise ValueError("未找到特徵欄位")
    
    print(f"\n使用 {len(feature_cols)} 個特徵")
    
    # 移除 NaN
    df_clean = df[feature_cols + ['label']].dropna()
    
    print(f"清理後數據: {len(df_clean)} 根K線")
    
    X = df_clean[feature_cols]
    y = df_clean['label']
    
    # 檢查標籤分佈
    long_count = (y == 1).sum()
    short_count = (y == -1).sum()
    neutral_count = (y == 0).sum()
    
    print(f"\n標籤分佈:")
    print(f"- Long: {long_count} ({long_count/len(y)*100:.2f}%)")
    print(f"- Short: {short_count} ({short_count/len(y)*100:.2f}%)")
    print(f"- Neutral: {neutral_count} ({neutral_count/len(y)*100:.2f}%)")
    
    # 切分訓練/測試集
    split_idx = int(len(df_clean) * train_split)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 訓練 Long 模型
    if train_long:
        print("\n[Long] 開始訓練...")
        
        y_train_long = (y_train == 1).astype(int)
        y_test_long = (y_test == 1).astype(int)
        
        # 檢查訓練集是否有足夠的正樣本
        train_pos = y_train_long.sum()
        train_neg = (y_train_long == 0).sum()
        
        print(f"訓練集: Positive={train_pos}, Negative={train_neg}")
        
        if train_pos < 10:
            print(f"⚠️ 警告: Long 正樣本太少 ({train_pos}),跳過訓練")
            print("建議: 降低 ZigZag 門檻或增加訓練天數")
        elif train_pos < 2 or train_neg < 2:
            print("❌ 錯誤: 訓練集標籤不足,無法訓練")
        else:
            try:
                model_long, metrics_long = train_single_model(
                    X_train, y_train_long,
                    X_test, y_test_long,
                    model_type, use_class_weights
                )
                
                # 保存模型
                model_path = Path('models_output') / f'v11_long_{symbol}_{timeframe}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'model': model_long,
                        'feature_cols': feature_cols,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'version': 'v11'
                    }, f)
                
                results['long'] = {
                    'model': model_long,
                    'model_path': str(model_path),
                    'metrics': metrics_long
                }
                
                print(f"✅ [Long] AUC-ROC: {metrics_long['auc_roc']:.3f}")
                print(f"✅ [Long] 標籤率: {metrics_long['label_rate']*100:.2f}%")
            except Exception as e:
                print(f"❌ [Long] 訓練失敗: {str(e)}")
    
    # 訓練 Short 模型
    if train_short:
        print("\n[Short] 開始訓練...")
        
        y_train_short = (y_train == -1).astype(int)
        y_test_short = (y_test == -1).astype(int)
        
        # 檢查訓練集是否有足夠的正樣本
        train_pos = y_train_short.sum()
        train_neg = (y_train_short == 0).sum()
        
        print(f"訓練集: Positive={train_pos}, Negative={train_neg}")
        
        if train_pos < 10:
            print(f"⚠️ 警告: Short 正樣本太少 ({train_pos}),跳過訓練")
            print("建議: 降低 ZigZag 門檻或增加訓練天數")
        elif train_pos < 2 or train_neg < 2:
            print("❌ 錯誤: 訓練集標籤不足,無法訓練")
        else:
            try:
                model_short, metrics_short = train_single_model(
                    X_train, y_train_short,
                    X_test, y_test_short,
                    model_type, use_class_weights
                )
                
                # 保存模型
                model_path = Path('models_output') / f'v11_short_{symbol}_{timeframe}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'model': model_short,
                        'feature_cols': feature_cols,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'version': 'v11'
                    }, f)
                
                results['short'] = {
                    'model': model_short,
                    'model_path': str(model_path),
                    'metrics': metrics_short
                }
                
                print(f"✅ [Short] AUC-ROC: {metrics_short['auc_roc']:.3f}")
                print(f"✅ [Short] 標籤率: {metrics_short['label_rate']*100:.2f}%")
            except Exception as e:
                print(f"❌ [Short] 訓練失敗: {str(e)}")
    
    # 保存報告
    if results:
        save_training_report(results, symbol, timeframe)
    else:
        print("\n⚠️ 沒有成功訓練任何模型")
    
    return results


def train_single_model(X_train, y_train, X_test, y_test, model_type, use_class_weights):
    """
    訓練單個模型
    """
    
    # 再次檢查標籤
    unique_labels = np.unique(y_train)
    if len(unique_labels) < 2:
        raise ValueError(f"訓練集只有一個類別: {unique_labels}")
    
    if model_type == 'CatBoost':
        from catboost import CatBoostClassifier
        
        class_weight = None
        if use_class_weights:
            pos_count = y_train.sum()
            if pos_count > 0:
                pos_weight = len(y_train) / (2 * pos_count)
                class_weight = {0: 1.0, 1: pos_weight}
        
        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function='Logloss',
            class_weights=class_weight,
            random_seed=42,
            verbose=False
        )
    
    elif model_type == 'XGBoost':
        from xgboost import XGBClassifier
        
        scale_pos_weight = 1.0
        if use_class_weights:
            pos_count = y_train.sum()
            if pos_count > 0:
                scale_pos_weight = (len(y_train) - pos_count) / pos_count
        
        model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )
    
    else:  # LightGBM
        from lightgbm import LGBMClassifier
        
        class_weight = None
        if use_class_weights:
            class_weight = 'balanced'
        
        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            class_weight=class_weight,
            random_state=42,
            verbose=-1
        )
    
    # 訓練
    model.fit(X_train, y_train)
    
    # 評估
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    metrics = {
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'auc_pr': average_precision_score(y_test, y_pred_proba),
        'f1': f1_score(y_test, y_pred),
        'label_rate': y_train.mean(),
        'n_samples': len(y_train),
        'n_positive': y_train.sum()
    }
    
    return model, metrics


def save_training_report(results, symbol, timeframe):
    """
    保存訓練報告
    """
    
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'symbol': symbol,
        'timeframe': timeframe,
        'version': 'v11',
        'results': {},
        'config': {}
    }
    
    for direction in ['long', 'short']:
        if direction in results:
            report['results'][direction] = {
                'model_path': results[direction]['model_path'],
                'metrics': results[direction]['metrics']
            }
    
    report_dir = Path('training_reports/v11')
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = report_dir / f'v11_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n報告已保存: {report_file}")
