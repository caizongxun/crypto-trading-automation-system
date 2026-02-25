#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v7 預測閾值優化

目標: 找到最佳 prediction threshold 使策略獲利
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

from backtest_v7_mean_reversion import MeanReversionBacktester


def test_threshold(
    upper_model_path: str,
    lower_model_path: str,
    df: pd.DataFrame,
    start_idx: int,
    threshold: float
) -> dict:
    """測試特定預測閾值"""
    
    backtester = MeanReversionBacktester(
        upper_model_path=upper_model_path,
        lower_model_path=lower_model_path,
        initial_capital=10000,
        position_size=0.02,
        leverage=10,
        threshold=threshold
    )
    
    report = backtester.run_backtest(df, start_idx=start_idx)
    
    return report


if __name__ == '__main__':
    print("\n" + "="*80)
    print("[PREDICTION THRESHOLD OPTIMIZATION]")
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
    
    # 測試不同 threshold
    print(f"\n{'='*80}")
    print("Testing different prediction thresholds...")
    print(f"{'='*80}\n")
    
    results = []
    
    # 測試範圍: 0.5 到 0.9
    test_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    
    for threshold in test_thresholds:
        print(f"\n[Testing Threshold = {threshold:.2f}]")
        print("-" * 80)
        
        try:
            report = test_threshold(
                upper_path, lower_path, df, oos_start, threshold
            )
            
            s = report['summary']
            
            result = {
                'threshold': threshold,
                'total_trades': s['total_trades'],
                'win_rate': s['win_rate'],
                'total_return_pct': s['total_return_pct'],
                'profit_factor': s['profit_factor'],
                'max_drawdown': s['max_drawdown'],
                'sharpe_ratio': s['sharpe_ratio'],
                'avg_win': s['avg_win'],
                'avg_loss': s['avg_loss'],
                'final_capital': s['final_capital'],
                'avg_bars_held': s['avg_bars_held']
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
            import traceback
            traceback.print_exc()
            continue
    
    # 結果比較
    if results:
        print(f"\n\n{'='*80}")
        print("[RESULTS COMPARISON]")
        print(f"{'='*80}\n")
        
        df_results = pd.DataFrame(results)
        
        # 排序依總報酬
        df_results = df_results.sort_values('total_return_pct', ascending=False)
        
        print(f"{'Thresh':<8} {'Trades':<8} {'Win%':<8} {'Return%':<10} {'PF':<8} {'Sharpe':<8} {'AvgWin':<10} {'AvgLoss':<10}")
        print("-" * 90)
        
        for _, row in df_results.iterrows():
            print(f"{row['threshold']:<8.2f} {row['total_trades']:<8.0f} {row['win_rate']*100:<8.1f} {row['total_return_pct']*100:<10.1f} {row['profit_factor']:<8.2f} {row['sharpe_ratio']:<8.2f} ${row['avg_win']:<9.2f} ${row['avg_loss']:<9.2f}")
        
        # 找最佳參數
        print(f"\n\n{'='*80}")
        print("[BEST CONFIGURATION]")
        print(f"{'='*80}\n")
        
        # 篩選: 報酬 > 0
        valid = df_results[df_results['total_return_pct'] > 0]
        
        if len(valid) > 0:
            best = valid.iloc[0]
            print(f"最佳 Threshold: {best['threshold']:.2f}")
            print(f"")
            print(f"  交易數: {best['total_trades']:.0f}")
            print(f"  總報酬: {best['total_return_pct']:.2%}")
            print(f"  勝率: {best['win_rate']:.2%}")
            print(f"  盈虧比: {best['profit_factor']:.2f}")
            print(f"  Sharpe Ratio: {best['sharpe_ratio']:.2f}")
            print(f"  最大回撤: {best['max_drawdown']:.2%}")
            print(f"  最終資金: ${best['final_capital']:,.2f}")
            print(f"  平均獲利: ${best['avg_win']:.2f}")
            print(f"  平均虧損: ${best['avg_loss']:.2f}")
            print(f"  平均持有: {best['avg_bars_held']:.1f} bars")
            print(f"")
            print(f"建議: 使用 threshold = {best['threshold']:.2f} 進行交易")
            
            # 計算每日交易頻率
            total_days = (len(df) - oos_start) / 96  # 15m = 96 bars per day
            trades_per_day = best['total_trades'] / total_days
            print(f"  每日交易數: {trades_per_day:.1f}")
            
        else:
            print("警告: 所有配置都無法獲利")
            print("")
            print("可能原因:")
            print("  1. TP/SL 設置不當 (平均獲利 < 平均虧損)")
            print("  2. 策略本身不適合通道交易")
            print("  3. 需要使用固定 RR 比例 (不是中軌)")
            print("")
            
            # 顯示最佳不虧配置
            best = df_results.iloc[0]
            print(f"最佳不虧配置: Threshold = {best['threshold']:.2f}")
            print(f"  交易數: {best['total_trades']:.0f}")
            print(f"  總報酬: {best['total_return_pct']:.2%}")
            print(f"  勝率: {best['win_rate']:.2%}")
            print(f"  盈虧比: {best['profit_factor']:.2f}")
            print(f"  平均獲利: ${best['avg_win']:.2f}")
            print(f"  平均虧損: ${best['avg_loss']:.2f}")
            print("")
            print("分析: 即使最低虧損配置仍然虧損,說明策略核心邏輯有問題")
            print("      需要改變 TP/SL 設定方式")
        
        # 保存結果
        output_dir = Path('optimization_results')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df_results.to_csv(output_dir / f'threshold_optimization_{timestamp}.csv', index=False)
        
        print(f"\n結果已保存: {output_dir / f'threshold_optimization_{timestamp}.csv'}")
        print("="*80)
    
    else:
        print("錯誤: 沒有成功的測試")
