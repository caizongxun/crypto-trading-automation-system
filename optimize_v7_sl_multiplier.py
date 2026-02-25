#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v7 止損倍數優化

目標: 找到最佳 SL multiplier 使盈虧比 >= 1.5
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Import backtester
from backtest_v7_mean_reversion import MeanReversionBacktester


def test_sl_multiplier(
    upper_model_path: str,
    lower_model_path: str,
    df: pd.DataFrame,
    start_idx: int,
    sl_multiplier: float,
    threshold: float = 0.5
) -> dict:
    """測試特定 SL 倍數"""
    
    # 暫時修改 metadata
    import pickle
    with open(upper_model_path, 'rb') as f:
        upper_data = pickle.load(f)
    
    # 修改 SL multiplier
    upper_data['metadata']['sl_multiplier'] = sl_multiplier
    
    # 寫回暫存檔
    temp_upper = 'temp_upper.pkl'
    with open(temp_upper, 'wb') as f:
        pickle.dump(upper_data, f)
    
    # 同樣處理 lower
    with open(lower_model_path, 'rb') as f:
        lower_data = pickle.load(f)
    lower_data['metadata']['sl_multiplier'] = sl_multiplier
    
    temp_lower = 'temp_lower.pkl'
    with open(temp_lower, 'wb') as f:
        pickle.dump(lower_data, f)
    
    # 回測
    backtester = MeanReversionBacktester(
        upper_model_path=temp_upper,
        lower_model_path=temp_lower,
        initial_capital=10000,
        position_size=0.02,
        leverage=10,
        threshold=threshold
    )
    
    report = backtester.run_backtest(df, start_idx=start_idx)
    
    # 清理暫存檔
    Path(temp_upper).unlink(missing_ok=True)
    Path(temp_lower).unlink(missing_ok=True)
    
    return report


if __name__ == '__main__':
    print("\n" + "="*80)
    print("[SL MULTIPLIER OPTIMIZATION]")
    print("="*80)
    
    # 找模型
    models_dir = Path('models_output')
    upper_models = sorted(models_dir.glob('keltner_upper_*_v7mr_*.pkl'))
    lower_models = sorted(models_dir.glob('keltner_lower_*_v7mr_*.pkl'))
    
    if not upper_models or not lower_models:
        print("錯誤: 找不到 v7mr 模型")
        sys.exit(1)
    
    upper_path = str(upper_models[-1])
    lower_path = str(lower_models[-1])
    
    print(f"\nUsing models:")
    print(f"  {Path(upper_path).name}")
    print(f"  {Path(lower_path).name}")
    
    # 載入數據
    print(f"\nLoading data...")
    from utils.hf_data_loader import load_klines
    
    df = load_klines(
        symbol='BTCUSDT',
        timeframe='15m',
        start_date=(datetime.now() - timedelta(days=9999)).strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    oos_start = int(len(df) * 0.9)
    print(f"OOS period: {len(df) - oos_start} bars")
    
    # 測試不同 SL multiplier
    print(f"\n{'='*80}")
    print("Testing different SL multipliers...")
    print(f"{'='*80}\n")
    
    results = []
    
    # 測試範圍: 0.2 到 0.6 (更小的 SL)
    test_multipliers = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    
    for sl_mult in test_multipliers:
        print(f"\n[Testing SL Multiplier = {sl_mult}]")
        print("-" * 80)
        
        try:
            report = test_sl_multiplier(
                upper_path, lower_path, df, oos_start, sl_mult
            )
            
            s = report['summary']
            
            result = {
                'sl_multiplier': sl_mult,
                'total_trades': s['total_trades'],
                'win_rate': s['win_rate'],
                'total_return_pct': s['total_return_pct'],
                'profit_factor': s['profit_factor'],
                'max_drawdown': s['max_drawdown'],
                'sharpe_ratio': s['sharpe_ratio'],
                'avg_win': s['avg_win'],
                'avg_loss': s['avg_loss'],
                'final_capital': s['final_capital']
            }
            
            results.append(result)
            
            print(f"  交易數: {s['total_trades']}")
            print(f"  勝率: {s['win_rate']:.2%}")
            print(f"  總報酬: {s['total_return_pct']:.2%}")
            print(f"  盈虧比: {s['profit_factor']:.2f}")
            print(f"  平均獲利: ${s['avg_win']:.2f}")
            print(f"  平均虧損: ${s['avg_loss']:.2f}")
            print(f"  最大回撤: {s['max_drawdown']:.2%}")
            print(f"  Sharpe: {s['sharpe_ratio']:.2f}")
            
        except Exception as e:
            print(f"  錯誤: {e}")
            continue
    
    # 結果比較
    if results:
        print(f"\n\n{'='*80}")
        print("[RESULTS COMPARISON]")
        print(f"{'='*80}\n")
        
        df_results = pd.DataFrame(results)
        
        # 排序依 Sharpe Ratio
        df_results = df_results.sort_values('sharpe_ratio', ascending=False)
        
        print(f"{'SL Mult':<10} {'Trades':<8} {'Win%':<8} {'Return%':<10} {'PF':<8} {'Sharpe':<8} {'MaxDD%':<10}")
        print("-" * 80)
        
        for _, row in df_results.iterrows():
            print(f"{row['sl_multiplier']:<10.2f} {row['total_trades']:<8.0f} {row['win_rate']*100:<8.1f} {row['total_return_pct']*100:<10.1f} {row['profit_factor']:<8.2f} {row['sharpe_ratio']:<8.2f} {row['max_drawdown']*100:<10.1f}")
        
        # 找最佳參數
        print(f"\n\n{'='*80}")
        print("[BEST CONFIGURATION]")
        print(f"{'='*80}\n")
        
        # 篩選: 報酬 > 0 且 Sharpe > 0
        valid = df_results[(df_results['total_return_pct'] > 0) & (df_results['sharpe_ratio'] > 0)]
        
        if len(valid) > 0:
            best = valid.iloc[0]
            print(f"\u6700佳 SL Multiplier: {best['sl_multiplier']:.2f}")
            print(f"")
            print(f"  總報酬: {best['total_return_pct']:.2%}")
            print(f"  勝率: {best['win_rate']:.2%}")
            print(f"  盈虧比: {best['profit_factor']:.2f}")
            print(f"  Sharpe Ratio: {best['sharpe_ratio']:.2f}")
            print(f"  最大回撤: {best['max_drawdown']:.2%}")
            print(f"  最終資金: ${best['final_capital']:,.2f}")
            print(f"")
            print(f"建議: 使用 SL multiplier = {best['sl_multiplier']:.2f} 重新訓練模型")
            print(f"")
            print(f"python train_v7_channel_mean_reversion.py \\")
            print(f"  --symbol BTCUSDT \\")
            print(f"  --timeframe 15m \\")
            print(f"  --days 9999 \\")
            print(f"  --sl-multiplier {best['sl_multiplier']:.2f} \\")
            print(f"  --train-ratio 0.8 \\")
            print(f"  --val-ratio 0.1")
        else:
            print("警告: 所有配置都無法獲利")
            print("可能原因:")
            print("  1. 策略本身不適合這個市場")
            print("  2. 需要調整預測閾值 (threshold)")
            print("  3. 需要修改通道參數")
            
            # 顯示最佳不虧配置
            best_by_dd = df_results.loc[df_results['max_drawdown'].idxmax()]
            print(f"\n最小虧損配置: SL multiplier = {best_by_dd['sl_multiplier']:.2f}")
            print(f"  總報酬: {best_by_dd['total_return_pct']:.2%}")
            print(f"  最大回撤: {best_by_dd['max_drawdown']:.2%}")
        
        # 保存結果
        output_dir = Path('optimization_results')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df_results.to_csv(output_dir / f'sl_optimization_{timestamp}.csv', index=False)
        
        print(f"\n結果已保存: {output_dir / f'sl_optimization_{timestamp}.csv'}")
        print("="*80)
    
    else:
        print("錯誤: 沒有成功的測試")
