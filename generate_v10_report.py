#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 v10 詳細報告

配置: Both, threshold=0.6 (平衡型)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
rcParams['axes.unicode_minus'] = False

from backtest_v10_scalping import ScalpingBacktester


def generate_detailed_report():
    print("\n" + "="*80)
    print("[V10 DETAILED REPORT - BALANCED CONFIGURATION]")
    print("="*80)
    
    # 找模型
    models_dir = Path('models_output')
    long_models = sorted(models_dir.glob('scalping_long_*_v10_*.pkl'))
    short_models = sorted(models_dir.glob('scalping_short_*_v10_*.pkl'))
    
    if not long_models or not short_models:
        print("錯誤: 找不到 v10 模型")
        sys.exit(1)
    
    long_path = str(long_models[-1])
    short_path = str(short_models[-1])
    
    print(f"\nModels:")
    print(f"  {Path(long_path).name}")
    print(f"  {Path(short_path).name}")
    
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
    
    # 回測
    print(f"\nRunning backtest...")
    backtester = ScalpingBacktester(
        long_model_path=long_path,
        short_model_path=short_path,
        initial_capital=10000,
        position_size=0.02,
        leverage=10,
        threshold=0.6,
        tp_pct=0.004,
        sl_pct=0.0025
    )
    
    report = backtester.run_backtest(df, start_idx=oos_start, long_enabled=True, short_enabled=True)
    
    if not report:
        print("錯誤: 回測失敗")
        sys.exit(1)
    
    trades_df = report['trades']
    equity_df = report['equity']
    summary = report['summary']
    
    # 創建輸出目錄
    output_dir = Path('backtest_results/v10_detailed')
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. 保存交易明細
    trades_export = trades_df.copy()
    trades_export['entry_time'] = pd.to_datetime(trades_export['entry_time'])
    trades_export['exit_time'] = pd.to_datetime(trades_export['exit_time'])
    trades_export.to_csv(output_dir / f'trades_{timestamp}.csv', index=False, encoding='utf-8-sig')
    print(f"\n[SAVED] Trades: {output_dir / f'trades_{timestamp}.csv'}")
    
    # 2. 保存資金曲線
    equity_df.to_csv(output_dir / f'equity_curve_{timestamp}.csv', index=False, encoding='utf-8-sig')
    print(f"[SAVED] Equity: {output_dir / f'equity_curve_{timestamp}.csv'}")
    
    # 3. 統計報告
    print(f"\n\n{'='*80}")
    print("[DETAILED STATISTICS]")
    print(f"{'='*80}\n")
    
    # 按時間分析
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['hour'] = trades_df['entry_time'].dt.hour
    trades_df['weekday'] = trades_df['entry_time'].dt.dayofweek
    
    print("■ 按小時統計 (TOP 5):")
    hour_stats = trades_df.groupby('hour').agg({
        'win': ['count', 'mean', 'sum'],
        'pnl': 'sum'
    }).round(3)
    hour_stats.columns = ['trades', 'win_rate', 'wins', 'pnl']
    hour_stats = hour_stats.sort_values('pnl', ascending=False)
    print(hour_stats.head())
    
    print(f"\n■ 按星期統計:")
    weekday_names = ['一', '二', '三', '四', '五', '六', '日']
    weekday_stats = trades_df.groupby('weekday').agg({
        'win': ['count', 'mean'],
        'pnl': 'sum'
    }).round(3)
    weekday_stats.columns = ['trades', 'win_rate', 'pnl']
    weekday_stats.index = [weekday_names[i] for i in weekday_stats.index]
    print(weekday_stats)
    
    print(f"\n■ 按方向統計:")
    side_stats = trades_df.groupby('side').agg({
        'win': ['count', 'mean'],
        'pnl': ['sum', 'mean'],
        'bars_held': 'mean'
    }).round(3)
    side_stats.columns = ['trades', 'win_rate', 'total_pnl', 'avg_pnl', 'avg_bars']
    print(side_stats)
    
    print(f"\n■ 出場原因統計:")
    exit_stats = trades_df.groupby('exit_reason').agg({
        'win': 'count',
        'pnl': ['sum', 'mean']
    }).round(2)
    exit_stats.columns = ['count', 'total_pnl', 'avg_pnl']
    print(exit_stats)
    
    # 連勝/連敗分析
    trades_df['win_int'] = trades_df['win'].astype(int)
    trades_df['streak'] = (trades_df['win_int'] != trades_df['win_int'].shift()).cumsum()
    streaks = trades_df.groupby(['streak', 'win']).size()
    
    win_streaks = streaks[streaks.index.get_level_values(1) == True]
    loss_streaks = streaks[streaks.index.get_level_values(1) == False]
    
    print(f"\n■ 連勝/連敗:")
    print(f"  最長連勝: {win_streaks.max() if len(win_streaks) > 0 else 0} 筆")
    print(f"  最長連敗: {loss_streaks.max() if len(loss_streaks) > 0 else 0} 筆")
    print(f"  平均連勝: {win_streaks.mean():.1f} 筆")
    print(f"  平均連敗: {loss_streaks.mean():.1f} 筆")
    
    # 持有時間分析
    print(f"\n■ 持有時間分析 (bars):")
    print(f"  中位數: {trades_df['bars_held'].median():.1f}")
    print(f"  平均: {trades_df['bars_held'].mean():.1f}")
    print(f"  最小: {trades_df['bars_held'].min():.0f}")
    print(f"  最大: {trades_df['bars_held'].max():.0f}")
    
    # 獲利分佈
    print(f"\n■ PnL 分佈:")
    print(trades_df['pnl'].describe())
    
    # 4. 繪製圖表
    print(f"\n\nGenerating charts...")
    fig = plt.figure(figsize=(20, 12))
    
    # 4.1 資金曲線
    ax1 = plt.subplot(3, 3, 1)
    equity_df['time'] = pd.to_datetime(equity_df['time'])
    ax1.plot(equity_df['time'], equity_df['equity'], linewidth=1.5, color='#2E86DE')
    ax1.fill_between(equity_df['time'], 10000, equity_df['equity'], alpha=0.3, color='#2E86DE')
    ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_title('資金曲線', fontsize=14, fontweight='bold')
    ax1.set_xlabel('時間')
    ax1.set_ylabel('資金 (USD)')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # 4.2 回撤曲線
    ax2 = plt.subplot(3, 3, 2)
    drawdown_pct = equity_df['drawdown'] * 100
    ax2.fill_between(equity_df['time'], 0, drawdown_pct, color='#EE5A6F', alpha=0.6)
    ax2.set_title('回撤曲線', fontsize=14, fontweight='bold')
    ax2.set_xlabel('時間')
    ax2.set_ylabel('回撤 (%)')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    
    # 4.3 每筆交易 PnL
    ax3 = plt.subplot(3, 3, 3)
    colors = ['#26de81' if w else '#fc5c65' for w in trades_df['win']]
    ax3.bar(range(len(trades_df)), trades_df['pnl'], color=colors, alpha=0.6, width=1)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_title('每筆交易 PnL', fontsize=14, fontweight='bold')
    ax3.set_xlabel('交易編號')
    ax3.set_ylabel('PnL (USD)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4.4 累積 PnL
    ax4 = plt.subplot(3, 3, 4)
    cumulative_pnl = trades_df['pnl'].cumsum()
    ax4.plot(cumulative_pnl.values, linewidth=2, color='#2E86DE')
    ax4.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl.values, alpha=0.3, color='#2E86DE')
    ax4.set_title('累積 PnL', fontsize=14, fontweight='bold')
    ax4.set_xlabel('交易編號')
    ax4.set_ylabel('累積 PnL (USD)')
    ax4.grid(True, alpha=0.3)
    
    # 4.5 勝率越勢 (100筆移動平均)
    ax5 = plt.subplot(3, 3, 5)
    win_rate_ma = trades_df['win'].rolling(100).mean() * 100
    ax5.plot(win_rate_ma.values, linewidth=2, color='#26de81')
    ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax5.axhline(y=summary['win_rate']*100, color='red', linestyle='--', alpha=0.5, label=f'總勝率: {summary["win_rate"]*100:.1f}%')
    ax5.set_title('勝率趋勢 (100MA)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('交易編號')
    ax5.set_ylabel('勝率 (%)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 4.6 PnL 分佈
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(trades_df['pnl'], bins=50, color='#2E86DE', alpha=0.7, edgecolor='black')
    ax6.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax6.axvline(x=trades_df['pnl'].mean(), color='green', linestyle='--', linewidth=2, label=f'平均: ${trades_df["pnl"].mean():.2f}')
    ax6.set_title('PnL 分佈', fontsize=14, fontweight='bold')
    ax6.set_xlabel('PnL (USD)')
    ax6.set_ylabel('次數')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 4.7 按小時 PnL
    ax7 = plt.subplot(3, 3, 7)
    hour_pnl = trades_df.groupby('hour')['pnl'].sum()
    colors = ['#26de81' if x > 0 else '#fc5c65' for x in hour_pnl]
    ax7.bar(hour_pnl.index, hour_pnl.values, color=colors, alpha=0.7)
    ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax7.set_title('按小時 PnL', fontsize=14, fontweight='bold')
    ax7.set_xlabel('小時')
    ax7.set_ylabel('PnL (USD)')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 4.8 按星期 PnL
    ax8 = plt.subplot(3, 3, 8)
    weekday_pnl = trades_df.groupby('weekday')['pnl'].sum()
    weekday_pnl.index = [weekday_names[i] for i in weekday_pnl.index]
    colors = ['#26de81' if x > 0 else '#fc5c65' for x in weekday_pnl]
    ax8.bar(weekday_pnl.index, weekday_pnl.values, color=colors, alpha=0.7)
    ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax8.set_title('按星期 PnL', fontsize=14, fontweight='bold')
    ax8.set_xlabel('星期')
    ax8.set_ylabel('PnL (USD)')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 4.9 Long vs Short
    ax9 = plt.subplot(3, 3, 9)
    side_data = trades_df.groupby('side').agg({
        'pnl': 'sum',
        'win': lambda x: (x.sum() / len(x) * 100)
    })
    x = np.arange(len(side_data))
    width = 0.35
    ax9_twin = ax9.twinx()
    bars1 = ax9.bar(x - width/2, side_data['pnl'], width, label='PnL', color='#2E86DE', alpha=0.7)
    bars2 = ax9_twin.bar(x + width/2, side_data['win'], width, label='勝率', color='#26de81', alpha=0.7)
    ax9.set_title('Long vs Short', fontsize=14, fontweight='bold')
    ax9.set_xlabel('方向')
    ax9.set_ylabel('PnL (USD)', color='#2E86DE')
    ax9_twin.set_ylabel('勝率 (%)', color='#26de81')
    ax9.set_xticks(x)
    ax9.set_xticklabels(side_data.index.str.upper())
    ax9.legend(loc='upper left')
    ax9_twin.legend(loc='upper right')
    ax9.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    chart_path = output_dir / f'analysis_{timestamp}.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] Charts: {chart_path}")
    plt.close()
    
    # 5. 生成 JSON 摘要
    report_json = {
        'config': {
            'strategy': 'v10_scalping',
            'threshold': 0.6,
            'tp_pct': 0.4,
            'sl_pct': 0.25,
            'timeframe': '15m'
        },
        'summary': {
            'total_trades': int(summary['total_trades']),
            'win_rate': float(summary['win_rate']),
            'total_return_pct': float(summary['total_return_pct']),
            'total_pnl': float(summary['total_pnl']),
            'profit_factor': float(summary['profit_factor']),
            'sharpe_ratio': float(summary['sharpe_ratio']),
            'max_drawdown': float(summary['max_drawdown']),
            'avg_win': float(summary['avg_win']),
            'avg_loss': float(summary['avg_loss'])
        },
        'by_side': side_stats.to_dict(),
        'by_hour': hour_stats.head(5).to_dict(),
        'by_weekday': weekday_stats.to_dict()
    }
    
    import json
    with open(output_dir / f'summary_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(report_json, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] Summary: {output_dir / f'summary_{timestamp}.json'}")
    
    print(f"\n\n{'='*80}")
    print("[COMPLETE] All reports generated successfully!")
    print(f"{'='*80}")
    print(f"\n輸出目錄: {output_dir.absolute()}")
    print(f"\n檔案:")
    print(f"  - trades_{timestamp}.csv (交易明細)")
    print(f"  - equity_curve_{timestamp}.csv (資金曲線)")
    print(f"  - analysis_{timestamp}.png (圖表分析)")
    print(f"  - summary_{timestamp}.json (摘要報告)")
    
    return output_dir, timestamp


if __name__ == '__main__':
    output_dir, timestamp = generate_detailed_report()
