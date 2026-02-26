#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
驗證優化結果 - 檢查為何 GUI 結果與優化結果不一致
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def compare_results():
    print("="*80)
    print("[驗證] 優化結果 vs GUI 結果")
    print("="*80)
    
    # 讀取最佳配置
    opt_dir = Path('backtest_results/v10_optimization')
    config_files = sorted(opt_dir.glob('best_config_*.json'))
    
    if not config_files:
        print("未找到優化結果")
        return
    
    with open(config_files[-1], 'r', encoding='utf-8') as f:
        best_config = json.load(f)
    
    print("\n[1] 優化配置:")
    print(f"  時間戳: {best_config['timestamp']}")
    print(f"  測試天數: {best_config['days']}")
    print(f"  交易對: {best_config['symbol']}")
    print(f"  時間框架: {best_config['timeframe']}")
    
    params = best_config['best_params']
    print("\n[2] 最佳參數:")
    print(f"  Threshold: {params['threshold']:.2f}")
    print(f"  TP: {params['tp_pct']*100:.1f}%")
    print(f"  SL: {params['sl_pct']*100:.1f}%")
    print(f"  倉位: {params['position_size']*100:.1f}%")
    
    perf = best_config['performance']
    print("\n[3] 優化績效:")
    print(f"  報酬: {perf['total_return_pct']*100:.2f}%")
    print(f"  Sharpe: {perf['sharpe_ratio']:.2f}")
    print(f"  勝率: {perf['win_rate']*100:.1f}%")
    print(f"  交易數: {perf['total_trades']}")
    print(f"  盈虧比: {perf['profit_factor']:.2f}")
    print(f"  最大回撤: {perf['max_drawdown']*100:.2f}%")
    
    # 讀取 GUI 結果
    results_dir = Path('backtest_results/v10_detailed')
    summary_files = sorted(results_dir.glob('summary_*.json'))
    
    if not summary_files:
        print("\n⚠️  未找到 GUI 回測結果")
        print("\n請在 GUI 中執行回測後再次執行此腳本")
        return
    
    with open(summary_files[-1], 'r', encoding='utf-8') as f:
        gui_data = json.load(f)
    
    gui_config = gui_data['config']
    gui_summary = gui_data['summary']
    
    print("\n" + "="*80)
    print("[4] GUI 配置:")
    print("="*80)
    print(f"  時間戳: {gui_data['timestamp']}")
    print(f"  測試天數: {gui_config.get('backtest_days', 'N/A')}")
    print(f"  交易對: {gui_config.get('symbol', 'N/A')}")
    print(f"  時間框架: {gui_config.get('timeframe', 'N/A')}")
    print(f"  訓練集比例: 未記錄 (可能預設 0.9)")
    
    print("\n[5] GUI 參數:")
    print(f"  Threshold: {gui_config.get('threshold', 'N/A')}")
    print(f"  TP: {gui_config.get('tp_pct', 'N/A')}%")
    print(f"  SL: {gui_config.get('sl_pct', 'N/A')}%")
    print(f"  倉位: {gui_config.get('position_size', 'N/A')}%")
    
    print("\n[6] GUI 績效:")
    print(f"  報酬: {gui_summary['total_return_pct']*100:.2f}%")
    print(f"  Sharpe: {gui_summary['sharpe_ratio']:.2f}")
    print(f"  勝率: {gui_summary['win_rate']*100:.1f}%")
    print(f"  交易數: {gui_summary['total_trades']}")
    print(f"  盈虧比: {gui_summary['profit_factor']:.2f}")
    print(f"  最大回撤: {gui_summary['max_drawdown']*100:.2f}%")
    
    # 對比分析
    print("\n" + "="*80)
    print("[7] 差異分析")
    print("="*80)
    
    # 參數對比
    print("\n參數對比:")
    param_match = True
    
    if gui_config.get('threshold') != params['threshold']:
        print(f"  ⚠️  Threshold 不一致: {gui_config.get('threshold')} vs {params['threshold']}")
        param_match = False
    
    if abs(gui_config.get('tp_pct', 0) - params['tp_pct']*100) > 0.01:
        print(f"  ⚠️  TP 不一致: {gui_config.get('tp_pct')}% vs {params['tp_pct']*100:.1f}%")
        param_match = False
    
    if abs(gui_config.get('sl_pct', 0) - params['sl_pct']*100) > 0.01:
        print(f"  ⚠️  SL 不一致: {gui_config.get('sl_pct')}% vs {params['sl_pct']*100:.1f}%")
        param_match = False
    
    if abs(gui_config.get('position_size', 0) - params['position_size']*100) > 0.01:
        print(f"  ⚠️  倉位不一致: {gui_config.get('position_size')}% vs {params['position_size']*100:.1f}%")
        param_match = False
    
    if param_match:
        print("  ✅ 所有參數匹配")
    
    # 績效對比
    print("\n績效對比:")
    
    return_diff = (gui_summary['total_return_pct'] - perf['total_return_pct']) * 100
    print(f"  報酬差異: {return_diff:+.2f}% ", end="")
    if abs(return_diff) < 0.5:
        print("(幾乎一致)")
    elif abs(return_diff) < 1.5:
        print("(正常波動)")
    else:
        print("(差異較大)")
    
    trades_diff = gui_summary['total_trades'] - perf['total_trades']
    print(f"  交易數差異: {trades_diff:+d} ", end="")
    if abs(trades_diff) < 10:
        print("(幾乎一致)")
    elif abs(trades_diff) < 50:
        print("(正常波動)")
    else:
        print("(差異較大)")
    
    wr_diff = (gui_summary['win_rate'] - perf['win_rate']) * 100
    print(f"  勝率差異: {wr_diff:+.1f}% ", end="")
    if abs(wr_diff) < 2:
        print("(幾乎一致)")
    elif abs(wr_diff) < 5:
        print("(正常波動)")
    else:
        print("(差異較大)")
    
    # 原因分析
    print("\n" + "="*80)
    print("[8] 可能原因")
    print("="*80)
    
    print("\n常見原因:")
    
    # 原因1: 測試期間不同
    opt_days = best_config['days']
    gui_days = gui_config.get('backtest_days', 'N/A')
    
    if gui_days != opt_days:
        print(f"\n1. ✅ 測試期間不同")
        print(f"   優化: {opt_days} 天")
        print(f"   GUI: {gui_days} 天")
        print(f"   影響: 不同市場條件導致績效差異")
    else:
        print(f"\n1. 測試期間: 一致 ({opt_days} 天)")
    
    # 原因2: 訓練/測試集切分
    print(f"\n2. 訓練集比例")
    print(f"   優化: 80% (固定)")
    print(f"   GUI: 可能不同 (請檢查 GUI 設定)")
    print(f"   影響: 不同時段的市場特性不同")
    
    # 原因3: 數據時間範圍
    print(f"\n3. 數據時間範圍")
    print(f"   優化時間: 2026-02-26 11:39")
    print(f"   GUI 時間: {gui_data['timestamp'][:8]}_{gui_data['timestamp'][9:15]}")
    
    time_diff_minutes = (datetime.strptime(gui_data['timestamp'], '%Y%m%d_%H%M%S') - 
                         datetime.strptime(best_config['timestamp'], '%Y%m%d_%H%M%S')).total_seconds() / 60
    
    if abs(time_diff_minutes) > 5:
        print(f"   差異: {abs(time_diff_minutes):.0f} 分鐘")
        print(f"   影響: 數據範圍可能略有不同")
    else:
        print(f"   差異: < 5 分鐘 (幾乎相同)")
    
    # 原因4: 隨機性
    print(f"\n4. 模型隨機性")
    print(f"   機器學習模型有內建隨機性")
    print(f"   影響: 每次預測可能略有差異 (通常 < 1%)")
    
    print("\n" + "="*80)
    print("[9] 建議")
    print("="*80)
    
    if abs(return_diff) < 1.5:
        print("""
✅ 績效差異在正常範圍內 (< 1.5%)

這是正常現象,原因:
1. 不同時間段的市場波動不同
2. 訓練/測試集切分位置不同
3. 機器學習的隨機性

建議:
- 如果報酬 > 0: 可以使用
- 如果 Sharpe > 2: 績效優秀
- 如果勝率 > 45%: 策略穩定
        """)
    elif abs(return_diff) < 3:
        print("""
⚠️  績效差異較大 (1.5-3%)

可能原因:
1. 測試天數不同 (優化 30 天 vs GUI 90 天)
2. 訓練集比例不同
3. 過擬合 (overfitting)

建議:
1. 確認 GUI 設定的測試天數 = 30
2. 確認訓練集比例 = 0.8 (80%)
3. 執行更長期優化 (90 天):
   python optimize_v10_parameters.py --days 90
        """)
    else:
        print("""
❌ 績效差異很大 (> 3%)

可能原因:
1. 參數未正確應用
2. 數據範圍完全不同
3. 優化方案誤啟用

建議:
1. 確認所有參數匹配
2. 確認所有優化方案未勾選
3. 執行 debug 腳本:
   python debug_v10_params.py --run-backtest
        """)
    
    # 精確復現檢查
    print("\n" + "="*80)
    print("[10] 精確復現檢查清單")
    print("="*80)
    
    print("""
請確認以下項目:

☐ 回測天數 = 30
☐ 訓練集比例 = 0.8 (80%)
☐ 交易對 = BTCUSDT
☐ 時間框架 = 15m
☐ Threshold = 0.65
☐ TP = 0.8%
☐ SL = 0.4%
☐ 倉位 = 3.0%
☐ 槓桿 = 10x
☐ 所有優化方案未勾選

如果全部匹配,但結果仍然不同:
- 這是正常的市場波動
- 只要報酬 > 0 就是有效的
    """)


if __name__ == '__main__':
    compare_results()
