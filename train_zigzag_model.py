#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZigZag 模型訓練腳本 (獨立版本)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def train_zigzag_model(
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
    訓練 ZigZag 模型
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
    
    print(f"\n標籤分佈:")
    print(f"- Long: {long_count} ({long_count/len(y)*100:.2f}%)")
    print(f"- Short: {short_count} ({short_count/len(y)*100:.2f}%)")
    
    # 切分訓練/測試集
    split_idx = int(len(df_clean) * train_split)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 訓練 Long 模型
    if train_long:
        print("\n[Long] 開始訓練...")
        
        y_train_long = (y_train == 1).astype(int)
        y_test_long = (y_test == 1).astype(int)
        
        train_pos = y_train_long.sum()
        train_neg = (y_train_long == 0).sum()
        
        print(f"訓練集: Positive={train_pos}, Negative={train_neg}")
        
        if train_pos >= 10 and train_neg >= 10:
            try:
                from train_model_v11 import train_single_model
                
                model_long, metrics_long = train_single_model(
                    X_train, y_train_long,
                    X_test, y_test_long,
                    model_type, use_class_weights
                )
                
                # 保存模型
                model_dir = Path('models_output/zigzag')
                model_dir.mkdir(parents=True, exist_ok=True)
                
                model_path = model_dir / f'zigzag_long_{symbol}_{timeframe}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'model': model_long,
                        'feature_cols': feature_cols,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'version': 'zigzag'
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
        else:
            print(f"⚠️ Long 樣本不足,跳過訓練")
    
    # 訓練 Short 模型
    if train_short:
        print("\n[Short] 開始訓練...")
        
        y_train_short = (y_train == -1).astype(int)
        y_test_short = (y_test == -1).astype(int)
        
        train_pos = y_train_short.sum()
        train_neg = (y_train_short == 0).sum()
        
        print(f"訓練集: Positive={train_pos}, Negative={train_neg}")
        
        if train_pos >= 10 and train_neg >= 10:
            try:
                from train_model_v11 import train_single_model
                
                model_short, metrics_short = train_single_model(
                    X_train, y_train_short,
                    X_test, y_test_short,
                    model_type, use_class_weights
                )
                
                # 保存模型
                model_dir = Path('models_output/zigzag')
                model_dir.mkdir(parents=True, exist_ok=True)
                
                model_path = model_dir / f'zigzag_short_{symbol}_{timeframe}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'model': model_short,
                        'feature_cols': feature_cols,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'version': 'zigzag'
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
        else:
            print(f"⚠️ Short 樣本不足,跳過訓練")
    
    # 保存報告
    if results:
        save_zigzag_training_report(results, symbol, timeframe)
    
    return results


def save_zigzag_training_report(results, symbol, timeframe):
    """保存 ZigZag 訓練報告"""
    
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'symbol': symbol,
        'timeframe': timeframe,
        'version': 'zigzag',
        'results': {}
    }
    
    for direction in ['long', 'short']:
        if direction in results:
            report['results'][direction] = {
                'model_path': results[direction]['model_path'],
                'metrics': results[direction]['metrics']
            }
    
    report_dir = Path('training_reports/zigzag')
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = report_dir / f'zigzag_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n報告已保存: {report_file}")
