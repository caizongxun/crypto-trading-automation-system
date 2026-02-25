#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v7 模型閾值優化

目標: 找到最佳預測閾值,使 Precision >= 0.50
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def load_model(model_path: str):
    """載入模型"""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data

def optimize_threshold_for_precision(y_true, y_pred_proba, target_precision=0.5, min_recall=0.1):
    """
    找到使 Precision >= target_precision 的最低閾值
    
    Args:
        y_true: 真實標籤
        y_pred_proba: 預測機率
        target_precision: 目標 Precision
        min_recall: 最低 Recall 要求 (避免太保守)
    
    Returns:
        optimal_threshold, metrics
    """
    results = []
    
    for threshold in np.arange(0.3, 0.95, 0.02):
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if y_pred.sum() == 0:  # 沒有預測為正
            continue
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        pred_rate = y_pred.mean()
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pred_rate': pred_rate,
            'signals_per_day': pred_rate * 96  # 15m = 96 bars per day
        })
    
    df_results = pd.DataFrame(results)
    
    # 找到 Precision >= target 且 Recall >= min_recall 的最低閾值
    valid = df_results[
        (df_results['precision'] >= target_precision) & 
        (df_results['recall'] >= min_recall)
    ]
    
    if len(valid) == 0:
        print(f"警告: 無法達到 Precision >= {target_precision} 且 Recall >= {min_recall}")
        print(f"嘗試放寬條件...")
        # 放寬條件
        valid = df_results[df_results['precision'] >= target_precision * 0.8]
        if len(valid) == 0:
            # 還是沒有,返回 F1 最高的
            return df_results.loc[df_results['f1'].idxmax()], df_results
    
    optimal = valid.iloc[0]  # 最低閾值
    
    return optimal, df_results

def analyze_model(model_path: str, X_test, y_test, model_type: str):
    """
    分析模型並找到最佳閾值
    
    Args:
        model_path: 模型路徑
        X_test: 測試特徵
        y_test: 測試標籤
        model_type: 'upper' or 'lower'
    """
    print("="*80)
    print(f"[{model_type.upper()}] 模型分析")
    print("="*80)
    
    # 載入模型
    data = load_model(model_path)
    model = data['model']
    
    # 預測
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nAUC: {auc:.4f}")
    print(f"實際正樣本率: {y_test.mean():.2%}")
    print(f"")
    
    # 優化閾值
    print("尋找最佳閾值...")
    optimal, all_results = optimize_threshold_for_precision(
        y_test.values, y_pred_proba, 
        target_precision=0.50,
        min_recall=0.10
    )
    
    print(f"\n最佳閾值: {optimal['threshold']:.3f}")
    print(f"  Precision: {optimal['precision']:.4f}")
    print(f"  Recall: {optimal['recall']:.4f}")
    print(f"  F1: {optimal['f1']:.4f}")
    print(f"  預測正率: {optimal['pred_rate']:.2%}")
    print(f"  每日訊號數: {optimal['signals_per_day']:.1f}")
    
    # 顯示不同閾值的效果
    print(f"\n不同閾值對比:")
    print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'訊號/天':<10}")
    print("-" * 60)
    
    for _, row in all_results.iloc[::5].iterrows():  # 每5個顯示一個
        print(f"{row['threshold']:<10.2f} {row['precision']:<10.3f} {row['recall']:<10.3f} {row['f1']:<10.3f} {row['signals_per_day']:<10.1f}")
    
    # 保存優化後的模型
    data['optimal_threshold'] = optimal['threshold']
    data['threshold_metrics'] = {
        'precision': float(optimal['precision']),
        'recall': float(optimal['recall']),
        'f1': float(optimal['f1']),
        'pred_rate': float(optimal['pred_rate']),
        'signals_per_day': float(optimal['signals_per_day'])
    }
    
    # 保存
    optimized_path = model_path.replace('.pkl', '_optimized.pkl')
    with open(optimized_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\n已保存優化模型: {optimized_path}")
    
    return optimal, all_results

if __name__ == '__main__':
    print("\n載入最新的 v7 模型...")
    
    models_dir = Path('models_output')
    
    # 找最新的模型
    upper_models = sorted(models_dir.glob('keltner_upper_15m_v7_*.pkl'))
    lower_models = sorted(models_dir.glob('keltner_lower_15m_v7_*.pkl'))
    
    if not upper_models or not lower_models:
        print("錯誤: 找不到 v7 模型文件")
        print("請先運行 train_v7_keltner_bounce.py 訓練模型")
        sys.exit(1)
    
    upper_model_path = str(upper_models[-1])
    lower_model_path = str(lower_models[-1])
    
    print(f"\nUpper model: {Path(upper_model_path).name}")
    print(f"Lower model: {Path(lower_model_path).name}")
    
    # 重新載入測試數據
    print("\n重新載入數據進行閾值優化...")
    
    from datetime import datetime, timedelta
    from utils.hf_data_loader import load_klines
    from train_v7_keltner_bounce import (
        calculate_keltner_channels, 
        identify_touch_events,
        calculate_bounce_features,
        create_bounce_labels
    )
    
    # 載入數據
    df = load_klines(
        symbol='BTCUSDT',
        timeframe='15m',
        start_date=(datetime.now() - timedelta(days=9999)).strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    print(f"載入 {len(df)} 根K線")
    
    # 計算通道和特徵
    upper, middle, lower = calculate_keltner_channels(df, 20, 14, 2.0)
    upper_touch, lower_touch = identify_touch_events(df, upper, lower)
    features = calculate_bounce_features(df, upper, middle, lower)
    upper_bounce_labels, lower_bounce_labels = create_bounce_labels(df, upper_touch, lower_touch, 8)
    
    # 清理
    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    
    # 取測試集 (最後 10%)
    X_upper = features[upper_touch].copy()
    y_upper = upper_bounce_labels[upper_touch].copy()
    X_lower = features[lower_touch].copy()
    y_lower = lower_bounce_labels[lower_touch].copy()
    
    # 取最後 10% 作為測試集
    test_start_upper = int(len(X_upper) * 0.9)
    test_start_lower = int(len(X_lower) * 0.9)
    
    X_test_upper = X_upper.iloc[test_start_upper:]
    y_test_upper = y_upper.iloc[test_start_upper:]
    X_test_lower = X_lower.iloc[test_start_lower:]
    y_test_lower = y_lower.iloc[test_start_lower:]
    
    print(f"\nUpper 測試樣本: {len(X_test_upper)}")
    print(f"Lower 測試樣本: {len(X_test_lower)}")
    
    # 分析並優化
    upper_optimal, upper_results = analyze_model(
        upper_model_path, X_test_upper, y_test_upper, 'upper'
    )
    
    print("\n" + "="*80 + "\n")
    
    lower_optimal, lower_results = analyze_model(
        lower_model_path, X_test_lower, y_test_lower, 'lower'
    )
    
    # 總結
    print("\n" + "="*80)
    print("[總結] 優化後的交易策略")
    print("="*80)
    print(f"\nShort 訊號 (上軌反彈):")
    print(f"  閾值: {upper_optimal['threshold']:.3f}")
    print(f"  勝率: {upper_optimal['precision']:.1%}")
    print(f"  每日訊號: {upper_optimal['signals_per_day']:.1f} 次")
    
    print(f"\nLong 訊號 (下軌反彈):")
    print(f"  閾值: {lower_optimal['threshold']:.3f}")
    print(f"  勝率: {lower_optimal['precision']:.1%}")
    print(f"  每日訊號: {lower_optimal['signals_per_day']:.1f} 次")
    
    total_signals = upper_optimal['signals_per_day'] + lower_optimal['signals_per_day']
    avg_precision = (upper_optimal['precision'] + lower_optimal['precision']) / 2
    
    print(f"\n總計:")
    print(f"  每日總訊號: {total_signals:.1f} 次")
    print(f"  平均勝率: {avg_precision:.1%}")
    print(f"\n建議: 使用優化後的 *_optimized.pkl 模型進行回測和實盤交易")
