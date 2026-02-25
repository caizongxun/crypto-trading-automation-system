#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v9 激進版回測

測試更嚴格條件下的模型表現
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


def test_config(bearish_path, bullish_path, df, start_idx, config):
    """測試特定配置"""
    
    class SelectiveBacktester(ReversalBacktester):
        def __init__(self, *args, long_enabled=True, short_enabled=True, **kwargs):
            super().__init__(*args, **kwargs)
            self.long_enabled = long_enabled
            self.short_enabled = short_enabled
        
        def run_backtest(self, df, start_idx=0):
            print(f"\n{'='*80}")
            print(f"[BACKTEST] v9 Aggressive Strategy")
            print(f"{'='*80}")
            print(f"Period: {df.index[start_idx]} to {df.index[-1]}")
            print(f"Config: {config['name']}")
            print(f"Threshold: {self.threshold}")
            print(f"TP: ATR × {self.tp_atr_multiplier}")
            print(f"SL: ATR × {self.sl_atr_multiplier}")
            print(f"Long: {self.long_enabled}, Short: {self.short_enabled}")
            print("")
            
            indicators = self.calculate_indicators(df)
            bullish_exhaustion, bearish_exhaustion = self.identify_exhaustion_points(df, indicators)
            features = self.calculate_features(df, indicators)
            
            X_bearish = features[bullish_exhaustion]
            X_bullish = features[bearish_exhaustion]
            
            bearish_proba = self.bearish_model.predict_proba(X_bearish)[:, 1] if len(X_bearish) > 0 else []
            bullish_proba = self.bullish_model.predict_proba(X_bullish)[:, 1] if len(X_bullish) > 0 else []
            
            capital = self.initial_capital
            self.trades = []
            self.equity_curve = [(df.index[start_idx], capital)]
            
            max_horizon = 24
            
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
            
            print(f"Total trades: {len(self.trades)}")
            
            return self.generate_report() if self.trades else None
    
    backtester = SelectiveBacktester(
        bearish_model_path=bearish_path,
        bullish_model_path=bullish_path,
        initial_capital=10000,
        position_size=0.02,
        leverage=10,
        threshold=config['threshold'],
        tp_atr_multiplier=config['tp_mult'],
        sl_atr_multiplier=config['sl_mult'],
        long_enabled=config['long_enabled'],
        short_enabled=config['short_enabled']
    )
    
    report = backtester.run_backtest(df, start_idx=start_idx)
    return report, backtester.trades


if __name__ == '__main__':
    print("\n" + "="*80)
    print("[V9 AGGRESSIVE MODEL BACKTEST]")
    print("="*80)
    
    # 找最新模型
    models_dir = Path('models_output')
    bearish_models = sorted(models_dir.glob('reversal_bearish_*_v9_*.pkl'))
    bullish_models = sorted(models_dir.glob('reversal_bullish_*_v9_*.pkl'))
    
    if len(bearish_models) < 2 or len(bullish_models) < 2:
        print("錯誤: 需要至少 2 組模型")
        sys.exit(1)
    
    bearish_path = str(bearish_models[-1])
    bullish_path = str(bullish_models[-1])
    
    print(f"\nUsing aggressive models:")
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
    
    # 測試配置
    configs = [
        {
            'name': 'Short Only, threshold=0.5, TP=2.0, SL=1.0',
            'threshold': 0.5,
            'tp_mult': 2.0,
            'sl_mult': 1.0,
            'long_enabled': False,
            'short_enabled': True
        },
        {
            'name': 'Short Only, threshold=0.6, TP=2.0, SL=1.0',
            'threshold': 0.6,
            'tp_mult': 2.0,
            'sl_mult': 1.0,
            'long_enabled': False,
            'short_enabled': True
        },
        {
            'name': 'Short Only, threshold=0.7, TP=2.0, SL=1.0',
            'threshold': 0.7,
            'tp_mult': 2.0,
            'sl_mult': 1.0,
            'long_enabled': False,
            'short_enabled': True
        },
        {
            'name': 'Short Only, threshold=0.5, TP=2.5, SL=0.8',
            'threshold': 0.5,
            'tp_mult': 2.5,
            'sl_mult': 0.8,
            'long_enabled': False,
            'short_enabled': True
        },
        {
            'name': 'Short Only, threshold=0.5, TP=1.5, SL=1.0',
            'threshold': 0.5,
            'tp_mult': 1.5,
            'sl_mult': 1.0,
            'long_enabled': False,
            'short_enabled': True
        },
        {
            'name': 'Both sides, threshold=0.5, TP=2.0, SL=1.0',
            'threshold': 0.5,
            'tp_mult': 2.0,
            'sl_mult': 1.0,
            'long_enabled': True,
            'short_enabled': True
        },
    ]
    
    results = []
    
    for config in configs:
        print(f"\n\n{'='*80}")
        print(f"Testing: {config['name']}")
        print(f"{'='*80}")
        
        try:
            report, trades = test_config(bearish_path, bullish_path, df, oos_start, config)
            
            if report:
                s = report['summary']
                
                results.append({
                    'name': config['name'],
                    'threshold': config['threshold'],
                    'tp_mult': config['tp_mult'],
                    'sl_mult': config['sl_mult'],
                    'rr_ratio': config['tp_mult'] / config['sl_mult'],
                    'trades': s['total_trades'],
                    'win_rate': s['win_rate'],
                    'return_pct': s['total_return_pct'],
                    'profit_factor': s['profit_factor'],
                    'sharpe': s['sharpe_ratio'],
                    'max_dd': s['max_drawdown'],
                    'avg_win': s['avg_win'],
                    'avg_loss': s['avg_loss']
                })
                
                print(f"\n[SUMMARY]")
                print(f"  交易: {s['total_trades']}")
                print(f"  勝率: {s['win_rate']:.1%}")
                print(f"  報酬: {s['total_return_pct']:.2%}")
                print(f"  盈虧比: {s['profit_factor']:.2f}")
                print(f"  Sharpe: {s['sharpe_ratio']:.2f}")
                print(f"  平均獲利: ${s['avg_win']:.2f}")
                print(f"  平均虧損: ${s['avg_loss']:.2f}")
            else:
                print("  無交易")
        
        except Exception as e:
            print(f"  錯誤: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 比較結果
    if results:
        print(f"\n\n{'='*80}")
        print("[RESULTS COMPARISON]")
        print(f"{'='*80}\n")
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('return_pct', ascending=False)
        
        print(f"{'Config':<50} {'Trades':<8} {'Win%':<8} {'Return%':<10} {'PF':<8} {'Sharpe':<8}")
        print("-" * 100)
        
        for _, row in df_results.iterrows():
            print(f"{row['name']:<50} {row['trades']:<8.0f} {row['win_rate']*100:<8.1f} {row['return_pct']*100:<10.2f} {row['profit_factor']:<8.2f} {row['sharpe']:<8.2f}")
        
        # 最佳配置
        best = df_results.iloc[0]
        
        print(f"\n\n{'='*80}")
        print("[BEST CONFIGURATION - V9 AGGRESSIVE]")
        print(f"{'='*80}\n")
        
        print(f"配置: {best['name']}")
        print(f"")
        print(f"Threshold: {best['threshold']:.2f}")
        print(f"TP multiplier: {best['tp_mult']:.1f}")
        print(f"SL multiplier: {best['sl_mult']:.1f}")
        print(f"RR ratio: {best['rr_ratio']:.2f}")
        print(f"")
        print(f"交易數: {best['trades']:.0f}")
        print(f"勝率: {best['win_rate']:.2%}")
        print(f"總報酬: {best['return_pct']:.2%}")
        print(f"盈虧比: {best['profit_factor']:.2f}")
        print(f"Sharpe: {best['sharpe']:.2f}")
        print(f"最大回撤: {best['max_dd']:.2%}")
        print(f"平均獲利: ${best['avg_win']:.2f}")
        print(f"平均虧損: ${best['avg_loss']:.2f}")
        print(f"")
        
        total_days = (len(df) - oos_start) / 96
        trades_per_day = best['trades'] / total_days
        print(f"每日交易: {trades_per_day:.1f}")
        
        # 年化
        annual_return = best['return_pct'] * (365 / (total_days))
        print(f"年化報酬: {annual_return:.1%}")
        
        # 保存
        output_dir = Path('backtest_results')
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        df_results.to_csv(output_dir / f'v9_aggressive_comparison_{timestamp}.csv', index=False)
        
        print(f"\n結果已保存: {output_dir / f'v9_aggressive_comparison_{timestamp}.csv'}")
        print("="*80)
    
    else:
        print("錯誤: 沒有成功的測試")
