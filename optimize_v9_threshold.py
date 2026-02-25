#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v9 策略閾值優化

測試不同的:
1. prediction threshold (0.5 - 0.9)
2. TP/SL multiplier
3. 只做 Short
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

from backtest_v9_reversal import ReversalBacktester


def test_configuration(
    bearish_path: str,
    bullish_path: str,
    df: pd.DataFrame,
    start_idx: int,
    threshold: float,
    tp_multiplier: float,
    sl_multiplier: float,
    long_enabled: bool = True,
    short_enabled: bool = True
) -> dict:
    """測試特定配置"""
    
    # 修改回測器類別以支持禁用某個方向
    class SelectiveBacktester(ReversalBacktester):
        def __init__(self, *args, long_enabled=True, short_enabled=True, **kwargs):
            super().__init__(*args, **kwargs)
            self.long_enabled = long_enabled
            self.short_enabled = short_enabled
        
        def run_backtest(self, df, start_idx=0):
            print(f"\n{'='*80}")
            print(f"[BACKTEST] v9 Momentum Reversal Strategy")
            print(f"{'='*80}")
            print(f"Period: {df.index[start_idx]} to {df.index[-1]}")
            print(f"Initial capital: ${self.initial_capital:,.2f}")
            print(f"Position size: {self.position_size:.1%}")
            print(f"Leverage: {self.leverage}x")
            print(f"Threshold: {self.threshold}")
            print(f"TP: ATR × {self.tp_atr_multiplier}")
            print(f"SL: ATR × {self.sl_atr_multiplier}")
            print(f"Long enabled: {self.long_enabled}")
            print(f"Short enabled: {self.short_enabled}")
            print("")
            
            # 計算指標
            print("Calculating indicators...")
            indicators = self.calculate_indicators(df)
            
            # 識別衰竭點
            print("Identifying exhaustion points...")
            bullish_exhaustion, bearish_exhaustion = self.identify_exhaustion_points(df, indicators)
            
            # 計算特徵
            print("Calculating features...")
            features = self.calculate_features(df, indicators)
            
            # 預測
            print("Making predictions...")
            X_bearish = features[bullish_exhaustion]
            X_bullish = features[bearish_exhaustion]
            
            bearish_proba = self.bearish_model.predict_proba(X_bearish)[:, 1] if len(X_bearish) > 0 else []
            bullish_proba = self.bullish_model.predict_proba(X_bullish)[:, 1] if len(X_bullish) > 0 else []
            
            # 模擬交易
            print(f"\nSimulating trades...")
            capital = self.initial_capital
            self.trades = []
            self.equity_curve = [(df.index[start_idx], capital)]
            
            max_horizon = 24
            
            # Short signals
            if self.short_enabled:
                bullish_exhaustion_indices = bullish_exhaustion[bullish_exhaustion].index
                for i, (idx, prob) in enumerate(zip(bullish_exhaustion_indices, bearish_proba)):
                    if idx < start_idx or prob < self.threshold:
                        continue
                    
                    idx_pos = df.index.get_loc(idx)
                    if idx_pos >= len(df) - max_horizon:
                        continue
                    
                    entry_price = df.loc[idx, 'close']
                    atr_value = indicators['atr'].loc[idx]
                    
                    tp_price = entry_price - (atr_value * self.tp_atr_multiplier)
                    sl_price = entry_price + (atr_value * self.sl_atr_multiplier)
                    
                    future_bars = df.iloc[idx_pos+1:idx_pos+1+max_horizon]
                    result = self.simulate_trade(entry_price, tp_price, sl_price, 'short', future_bars)
                    
                    capital += result['pnl']
                    
                    self.trades.append({
                        'entry_time': idx,
                        'entry_price': entry_price,
                        'side': 'short',
                        'tp_price': tp_price,
                        'sl_price': sl_price,
                        'prediction': prob,
                        **result
                    })
                    
                    self.equity_curve.append((idx, capital))
            
            # Long signals
            if self.long_enabled:
                bearish_exhaustion_indices = bearish_exhaustion[bearish_exhaustion].index
                for i, (idx, prob) in enumerate(zip(bearish_exhaustion_indices, bullish_proba)):
                    if idx < start_idx or prob < self.threshold:
                        continue
                    
                    idx_pos = df.index.get_loc(idx)
                    if idx_pos >= len(df) - max_horizon:
                        continue
                    
                    entry_price = df.loc[idx, 'close']
                    atr_value = indicators['atr'].loc[idx]
                    
                    tp_price = entry_price + (atr_value * self.tp_atr_multiplier)
                    sl_price = entry_price - (atr_value * self.sl_atr_multiplier)
                    
                    future_bars = df.iloc[idx_pos+1:idx_pos+1+max_horizon]
                    result = self.simulate_trade(entry_price, tp_price, sl_price, 'long', future_bars)
                    
                    capital += result['pnl']
                    
                    self.trades.append({
                        'entry_time': idx,
                        'entry_price': entry_price,
                        'side': 'long',
                        'tp_price': tp_price,
                        'sl_price': sl_price,
                        'prediction': prob,
                        **result
                    })
                    
                    self.equity_curve.append((idx, capital))
            
            self.trades = sorted(self.trades, key=lambda x: x['entry_time'])
            
            print(f"\nTotal trades: {len(self.trades)}")
            
            return self.generate_report()
    
    backtester = SelectiveBacktester(
        bearish_model_path=bearish_path,
        bullish_model_path=bullish_path,
        initial_capital=10000,
        position_size=0.02,
        leverage=10,
        threshold=threshold,
        tp_atr_multiplier=tp_multiplier,
        sl_atr_multiplier=sl_multiplier,
        long_enabled=long_enabled,
        short_enabled=short_enabled
    )
    
    report = backtester.run_backtest(df, start_idx=start_idx)
    return report


if __name__ == '__main__':
    print("\n" + "="*80)
    print("[V9 STRATEGY OPTIMIZATION]")
    print("="*80)
    
    # 找模型
    models_dir = Path('models_output')
    bearish_models = sorted(models_dir.glob('reversal_bearish_*_v9_*.pkl'))
    bullish_models = sorted(models_dir.glob('reversal_bullish_*_v9_*.pkl'))
    
    if not bearish_models or not bullish_models:
        print("錯誤: 找不到 v9 模型")
        sys.exit(1)
    
    bearish_path = str(bearish_models[-1])
    bullish_path = str(bullish_models[-1])
    
    print(f"\nUsing models:")
    print(f"  {Path(bearish_path).name}")
    print(f"  {Path(bullish_path).name}")
    
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
    
    # 優化結果
    results = []
    
    print(f"\n\n{'='*80}")
    print("[TEST 1] Threshold optimization (both sides)")
    print(f"{'='*80}")
    
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        print(f"\n[Threshold = {threshold:.2f}]")
        print("-" * 80)
        
        try:
            report = test_configuration(
                bearish_path, bullish_path, df, oos_start,
                threshold=threshold,
                tp_multiplier=1.5,
                sl_multiplier=1.0,
                long_enabled=True,
                short_enabled=True
            )
            
            s = report['summary']
            results.append({
                'config': 'both',
                'threshold': threshold,
                'tp_mult': 1.5,
                'sl_mult': 1.0,
                'trades': s['total_trades'],
                'win_rate': s['win_rate'],
                'return_pct': s['total_return_pct'],
                'profit_factor': s['profit_factor'],
                'sharpe': s['sharpe_ratio'],
                'max_dd': s['max_drawdown']
            })
            
            print(f"  交易: {s['total_trades']}, 勝率: {s['win_rate']:.1%}, 報酬: {s['total_return_pct']:.2%}")
        
        except Exception as e:
            print(f"  錯誤: {e}")
            continue
    
    print(f"\n\n{'='*80}")
    print("[TEST 2] Short only with different thresholds")
    print(f"{'='*80}")
    
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        print(f"\n[Short Only, Threshold = {threshold:.2f}]")
        print("-" * 80)
        
        try:
            report = test_configuration(
                bearish_path, bullish_path, df, oos_start,
                threshold=threshold,
                tp_multiplier=1.5,
                sl_multiplier=1.0,
                long_enabled=False,
                short_enabled=True
            )
            
            s = report['summary']
            results.append({
                'config': 'short_only',
                'threshold': threshold,
                'tp_mult': 1.5,
                'sl_mult': 1.0,
                'trades': s['total_trades'],
                'win_rate': s['win_rate'],
                'return_pct': s['total_return_pct'],
                'profit_factor': s['profit_factor'],
                'sharpe': s['sharpe_ratio'],
                'max_dd': s['max_drawdown']
            })
            
            print(f"  交易: {s['total_trades']}, 勝率: {s['win_rate']:.1%}, 報酬: {s['total_return_pct']:.2%}")
        
        except Exception as e:
            print(f"  錯誤: {e}")
            continue
    
    print(f"\n\n{'='*80}")
    print("[TEST 3] TP/SL multiplier optimization (Short only, threshold=0.6)")
    print(f"{'='*80}")
    
    for tp_mult in [1.5, 2.0, 2.5]:
        for sl_mult in [0.8, 1.0, 1.2]:
            print(f"\n[Short Only, TP={tp_mult}, SL={sl_mult}, RR={tp_mult/sl_mult:.2f}]")
            print("-" * 80)
            
            try:
                report = test_configuration(
                    bearish_path, bullish_path, df, oos_start,
                    threshold=0.6,
                    tp_multiplier=tp_mult,
                    sl_multiplier=sl_mult,
                    long_enabled=False,
                    short_enabled=True
                )
                
                s = report['summary']
                results.append({
                    'config': 'short_only',
                    'threshold': 0.6,
                    'tp_mult': tp_mult,
                    'sl_mult': sl_mult,
                    'trades': s['total_trades'],
                    'win_rate': s['win_rate'],
                    'return_pct': s['total_return_pct'],
                    'profit_factor': s['profit_factor'],
                    'sharpe': s['sharpe_ratio'],
                    'max_dd': s['max_drawdown']
                })
                
                print(f"  交易: {s['total_trades']}, 勝率: {s['win_rate']:.1%}, 報酬: {s['total_return_pct']:.2%}")
            
            except Exception as e:
                print(f"  錯誤: {e}")
                continue
    
    # 結果比較
    if results:
        print(f"\n\n{'='*80}")
        print("[RESULTS COMPARISON]")
        print(f"{'='*80}\n")
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('return_pct', ascending=False)
        
        print(f"{'Config':<12} {'Thresh':<8} {'TP':<6} {'SL':<6} {'Trades':<8} {'Win%':<8} {'Return%':<10} {'PF':<8} {'Sharpe':<8}")
        print("-" * 90)
        
        for _, row in df_results.head(10).iterrows():
            print(f"{row['config']:<12} {row['threshold']:<8.2f} {row['tp_mult']:<6.1f} {row['sl_mult']:<6.1f} {row['trades']:<8.0f} {row['win_rate']*100:<8.1f} {row['return_pct']*100:<10.2f} {row['profit_factor']:<8.2f} {row['sharpe']:<8.2f}")
        
        # 最佳配置
        best = df_results.iloc[0]
        
        print(f"\n\n{'='*80}")
        print("[BEST CONFIGURATION]")
        print(f"{'='*80}\n")
        
        print(f"配置: {best['config']}")
        print(f"Threshold: {best['threshold']:.2f}")
        print(f"TP multiplier: {best['tp_mult']:.1f}")
        print(f"SL multiplier: {best['sl_mult']:.1f}")
        print(f"RR ratio: {best['tp_mult'] / best['sl_mult']:.2f}")
        print(f"")
        print(f"交易數: {best['trades']:.0f}")
        print(f"勝率: {best['win_rate']:.2%}")
        print(f"總報酬: {best['return_pct']:.2%}")
        print(f"盈虧比: {best['profit_factor']:.2f}")
        print(f"Sharpe: {best['sharpe']:.2f}")
        print(f"最大回撤: {best['max_dd']:.2%}")
        print(f"")
        
        # 每日交易
        total_days = (len(df) - oos_start) / 96
        trades_per_day = best['trades'] / total_days
        print(f"每日交易: {trades_per_day:.1f}")
        
        # 保存
        output_dir = Path('optimization_results')
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        df_results.to_csv(output_dir / f'v9_optimization_{timestamp}.csv', index=False)
        
        print(f"\n結果已保存: {output_dir / f'v9_optimization_{timestamp}.csv'}")
        print("="*80)
    
    else:
        print("錯誤: 沒有成功的測試")
