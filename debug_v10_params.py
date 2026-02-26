#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug Script - 檢查 v10 回測實際使用的參數
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime, timedelta


def debug_backtest_params():
    print("="*80)
    print("[DEBUG] v10 回測參數檢查")
    print("="*80)
    
    # 模擬 GUI 傳參
    print("\n[1] GUI 滑桿設定:")
    tp_slider = 0.6  # GUI 顯示 0.6%
    sl_slider = 0.3  # GUI 顯示 0.3%
    print(f"  TP 滑桿: {tp_slider}%")
    print(f"  SL 滑桿: {sl_slider}%")
    
    # GUI 轉換
    print("\n[2] GUI 轉換 (除以 100):")
    tp_pct = tp_slider / 100
    sl_pct = sl_slider / 100
    print(f"  tp_pct = {tp_pct} ({tp_pct*100:.1f}%)")
    print(f"  sl_pct = {sl_pct} ({sl_pct*100:.1f}%)")
    
    # 檢查回測引擎是否再次轉換
    print("\n[3] 回測引擎接收:")
    print(f"  __init__(tp_pct={tp_pct}, sl_pct={sl_pct})")
    print(f"  self.base_tp_pct = {tp_pct}")
    print(f"  self.base_sl_pct = {sl_pct}")
    
    # 計算實際 TP/SL 價格
    print("\n[4] 實際交易計算:")
    entry_price = 50000  # BTC 價格
    
    tp_price_long = entry_price * (1 + tp_pct)
    sl_price_long = entry_price * (1 - sl_pct)
    
    print(f"  進場價: ${entry_price:,.2f}")
    print(f"  Long TP: ${tp_price_long:,.2f} (+{(tp_price_long-entry_price)/entry_price*100:.2f}%)")
    print(f"  Long SL: ${sl_price_long:,.2f} (-{(entry_price-sl_price_long)/entry_price*100:.2f}%)")
    
    # 計算 PnL
    print("\n[5] 盈虧計算:")
    initial_capital = 10000
    position_size = 0.02
    leverage = 10
    notional = initial_capital * position_size * leverage
    
    print(f"  本金: ${initial_capital:,.2f}")
    print(f"  倉位: {position_size*100:.1f}%")
    print(f"  槓桿: {leverage}x")
    print(f"  名目價值: ${notional:,.2f}")
    
    # TP 盈利
    tp_pnl = (tp_price_long - entry_price) / entry_price * notional
    print(f"\n  TP 盈利: ${tp_pnl:.2f}")
    
    # SL 虧損
    sl_pnl = (sl_price_long - entry_price) / entry_price * notional
    print(f"  SL 虧損: ${sl_pnl:.2f}")
    
    # 風險報酬比
    rr_ratio = abs(tp_pnl / sl_pnl)
    print(f"  RR 比: {rr_ratio:.2f}:1")
    
    # 盈虧平衡點
    breakeven_wr = 1 / (1 + rr_ratio)
    print(f"  盈虧平衡點勝率: {breakeven_wr*100:.1f}%")
    
    print("\n" + "="*80)
    print("[6] 問題診斷:")
    print("="*80)
    
    # 檢查實際結果
    print("\n你的實際結果:")
    actual_avg_loss = -5.98
    actual_avg_win = 11.68
    actual_win_rate = 0.3314
    
    print(f"  平均虧損: ${actual_avg_loss:.2f}")
    print(f"  平均獲利: ${actual_avg_win:.2f}")
    print(f"  實際勝率: {actual_win_rate*100:.1f}%")
    
    # 比對期望值
    print(f"\n期望值計算:")
    expected_loss = sl_pnl
    expected_win = tp_pnl
    
    print(f"  理論 SL 虧損: ${expected_loss:.2f}")
    print(f"  實際平均虧損: ${actual_avg_loss:.2f}")
    print(f"  差異: {abs(actual_avg_loss) / abs(expected_loss) * 100:.1f}%")
    
    if abs(actual_avg_loss) < abs(expected_loss) * 0.5:
        print("\n  ⚠️  警告: SL 虧損明顯偏小,可能原因:")
        print("     1. SL 百分比又被除以 100 (變成 0.003%)")
        print("     2. 倉位計算錯誤")
        print("     3. 槓桿未生效")
    
    # 期望報酬
    actual_expected_value = actual_win_rate * actual_avg_win + (1 - actual_win_rate) * actual_avg_loss
    theoretical_wr_needed = abs(expected_loss) / (actual_avg_win + abs(expected_loss))
    
    print(f"\n  實際期望值: ${actual_expected_value:.2f} (per trade)")
    print(f"  需要的勝率: {theoretical_wr_needed*100:.1f}% (使期望值 > 0)")
    print(f"  實際勝率: {actual_win_rate*100:.1f}%")
    
    if actual_win_rate < theoretical_wr_needed:
        print(f"\n  ⚠️  勝率不足! 短缺 {(theoretical_wr_needed - actual_win_rate)*100:.1f}%")
    
    print("\n" + "="*80)
    print("[7] 解決方案:")
    print("="*80)
    
    print("""
1. 檢查 GUI 傳參是否正確:
   ▶ 打開 tabs/v10_scalping_tab.py
   ▶ 搜尋: backtester = AdvancedScalpingBacktester(
   ▶ 確認: tp_pct=tp_pct (不是 tp_pct/100)

2. 降低信號閾值增加交易數:
   ▶ 當前: 0.55 (過高)
   ▶ 建議: 0.50-0.52

3. 調整 TP/SL 比例:
   當前 RR = {rr_ratio:.2f}:1, 但勝率太低
   ▶ 方案1: 降低 TP 到 0.4-0.5% (提高勝率)
   ▶ 方案2: 增大 SL 到 0.35-0.4% (容忍波動)
   ▶ 方案3: 啟用動態 TP/SL

4. 執行參數優化:
   python optimize_v10_parameters.py --days 90
    """)
    
    print("\n" + "="*80)
    print("[8] 快速測試:")
    print("="*80)
    print("""
執行以下命令查看實際參數:

python debug_v10_params.py --run-backtest
    """)


def run_actual_backtest():
    """執行實際回測並顯示第一筆交易詳情"""
    print("\n" + "="*80)
    print("[BACKTEST] 執行實際回測")
    print("="*80)
    
    from utils.hf_data_loader import load_klines
    from backtest_v10_scalping_advanced import AdvancedScalpingBacktester
    
    # 載入數據
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"\n載入數據: BTCUSDT 15m, {start_date.date()} ~ {end_date.date()}")
    df = load_klines(
        symbol='BTCUSDT',
        timeframe='15m',
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    if df is None or len(df) == 0:
        print("數據載入失敗")
        return
    
    print(f"成功載入 {len(df)} 根K線")
    
    # 找模型
    models_dir = Path('models_output')
    long_models = sorted(models_dir.glob('scalping_long_*_v10_*.pkl'))
    short_models = sorted(models_dir.glob('scalping_short_*_v10_*.pkl'))
    
    if not long_models or not short_models:
        print("未找到 v10 模型")
        return
    
    # 模擬 GUI 參數
    print("\n回測參數:")
    tp_pct = 0.6 / 100  # 0.006
    sl_pct = 0.3 / 100  # 0.003
    threshold = 0.55
    position_size = 0.02
    leverage = 10
    initial_capital = 10000
    
    print(f"  Threshold: {threshold}")
    print(f"  TP: {tp_pct*100:.1f}% ({tp_pct})")
    print(f"  SL: {sl_pct*100:.1f}% ({sl_pct})")
    print(f"  倉位: {position_size*100:.1f}%")
    print(f"  槓桿: {leverage}x")
    print(f"  本金: ${initial_capital:,.2f}")
    
    # 執行回測
    backtester = AdvancedScalpingBacktester(
        long_model_path=str(long_models[-1]),
        short_model_path=str(short_models[-1]),
        initial_capital=initial_capital,
        position_size=position_size,
        leverage=leverage,
        threshold=threshold,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        enable_dynamic_tpsl=False,
        enable_quality_sizing=False,
        enable_trailing_stop=False,
        enable_time_filter=False,
        enable_strict_filter=False
    )
    
    print("\n確認回測引擎參數:")
    print(f"  backtester.base_tp_pct = {backtester.base_tp_pct} ({backtester.base_tp_pct*100:.4f}%)")
    print(f"  backtester.base_sl_pct = {backtester.base_sl_pct} ({backtester.base_sl_pct*100:.4f}%)")
    print(f"  backtester.threshold = {backtester.threshold}")
    
    oos_start = int(len(df) * 0.8)
    
    print(f"\n執行回測: 訓練集 0-{oos_start}, 測試集 {oos_start}-{len(df)}")
    
    report = backtester.run_backtest(
        df,
        start_idx=oos_start,
        long_enabled=True,
        short_enabled=True
    )
    
    if not report:
        print("回測失敗或無交易")
        return
    
    trades_df = report['trades']
    summary = report['summary']
    
    print("\n" + "="*80)
    print("[結果] 回測統計")
    print("="*80)
    
    print(f"\n總交易數: {summary['total_trades']}")
    print(f"勝率: {summary['win_rate']*100:.2f}%")
    print(f"總報酬: {summary['total_return_pct']*100:.2f}%")
    print(f"Sharpe: {summary['sharpe_ratio']:.2f}")
    print(f"盈虧比: {summary['profit_factor']:.2f}")
    print(f"最大回撤: {summary['max_drawdown']*100:.2f}%")
    print(f"平均獲利: ${summary['avg_win']:.2f}")
    print(f"平均虧損: ${summary['avg_loss']:.2f}")
    
    # 顯示前 5 筆交易
    print("\n" + "="*80)
    print("[第一筆交易詳情]")
    print("="*80)
    
    first_trade = trades_df.iloc[0]
    
    print(f"\n進場時間: {first_trade['entry_time']}")
    print(f"出場時間: {first_trade['exit_time']}")
    print(f"方向: {first_trade['side'].upper()}")
    print(f"進場價: ${first_trade['entry_price']:,.2f}")
    print(f"出場價: ${first_trade['exit_price']:,.2f}")
    print(f"出場原因: {first_trade['exit_reason']}")
    print(f"PnL: ${first_trade['pnl']:.2f}")
    print(f"報酬%: {first_trade['return_pct']*100:.2f}%")
    print(f"持有K線: {first_trade['bars_held']}")
    print(f"結果: {'WIN' if first_trade['win'] else 'LOSS'}")
    print(f"信心度: {first_trade['confidence']:.3f}")
    
    # 計算期望 TP/SL
    entry_price = first_trade['entry_price']
    if first_trade['side'] == 'long':
        expected_tp = entry_price * (1 + tp_pct)
        expected_sl = entry_price * (1 - sl_pct)
    else:
        expected_tp = entry_price * (1 - tp_pct)
        expected_sl = entry_price * (1 + sl_pct)
    
    print(f"\n期望 TP: ${expected_tp:,.2f} ({tp_pct*100:.2f}%)")
    print(f"期望 SL: ${expected_sl:,.2f} ({sl_pct*100:.2f}%)")
    
    # 計算期望 PnL
    notional = initial_capital * position_size * leverage
    
    if first_trade['side'] == 'long':
        expected_tp_pnl = (expected_tp - entry_price) / entry_price * notional
        expected_sl_pnl = (expected_sl - entry_price) / entry_price * notional
    else:
        expected_tp_pnl = (entry_price - expected_tp) / entry_price * notional
        expected_sl_pnl = (entry_price - expected_sl) / entry_price * notional
    
    print(f"\n期望 TP PnL: ${expected_tp_pnl:.2f}")
    print(f"期望 SL PnL: ${expected_sl_pnl:.2f}")
    print(f"實際 PnL: ${first_trade['pnl']:.2f}")
    
    if first_trade['exit_reason'] == 'sl':
        diff_pct = (first_trade['pnl'] - expected_sl_pnl) / expected_sl_pnl * 100
        print(f"\nSL PnL 差異: {diff_pct:.1f}%")
        if abs(diff_pct) > 10:
            print("⚠️  警告: SL PnL 與預期差異過大!")


if __name__ == '__main__':
    import sys
    
    if '--run-backtest' in sys.argv:
        run_actual_backtest()
    else:
        debug_backtest_params()
