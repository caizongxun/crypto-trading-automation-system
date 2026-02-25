#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v10 剝頭皮策略回測

測試不同 threshold 找到最佳配置
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import json

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from train_v10_high_frequency import calculate_microstructure_features


class ScalpingBacktester:
    def __init__(
        self,
        long_model_path: str,
        short_model_path: str,
        initial_capital: float = 10000,
        position_size: float = 0.02,
        leverage: int = 10,
        threshold: float = 0.5,
        tp_pct: float = 0.004,
        sl_pct: float = 0.0025
    ):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.leverage = leverage
        self.threshold = threshold
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        
        # 載入模型
        with open(long_model_path, 'rb') as f:
            data = pickle.load(f)
            self.long_model = data['model']
            self.feature_names = data['features']
        
        with open(short_model_path, 'rb') as f:
            data = pickle.load(f)
            self.short_model = data['model']
        
        self.trades = []
        self.equity_curve = []
    
    def simulate_trade(self, entry_price, tp_price, sl_price, side, future_bars):
        """模擬單筆交易"""
        position_value = self.initial_capital * self.position_size * self.leverage
        
        for i, (idx, row) in enumerate(future_bars.iterrows()):
            if side == 'long':
                if row['high'] >= tp_price:
                    pnl = position_value * self.tp_pct
                    return {
                        'exit_time': idx,
                        'exit_price': tp_price,
                        'exit_reason': 'TP',
                        'pnl': pnl,
                        'return_pct': self.tp_pct,
                        'bars_held': i + 1,
                        'win': True
                    }
                elif row['low'] <= sl_price:
                    pnl = -position_value * self.sl_pct
                    return {
                        'exit_time': idx,
                        'exit_price': sl_price,
                        'exit_reason': 'SL',
                        'pnl': pnl,
                        'return_pct': -self.sl_pct,
                        'bars_held': i + 1,
                        'win': False
                    }
            
            else:  # short
                if row['low'] <= tp_price:
                    pnl = position_value * self.tp_pct
                    return {
                        'exit_time': idx,
                        'exit_price': tp_price,
                        'exit_reason': 'TP',
                        'pnl': pnl,
                        'return_pct': self.tp_pct,
                        'bars_held': i + 1,
                        'win': True
                    }
                elif row['high'] >= sl_price:
                    pnl = -position_value * self.sl_pct
                    return {
                        'exit_time': idx,
                        'exit_price': sl_price,
                        'exit_reason': 'SL',
                        'pnl': pnl,
                        'return_pct': -self.sl_pct,
                        'bars_held': i + 1,
                        'win': False
                    }
        
        # 超時出場
        exit_price = future_bars.iloc[-1]['close']
        if side == 'long':
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price
        
        pnl = position_value * pnl_pct
        
        return {
            'exit_time': future_bars.index[-1],
            'exit_price': exit_price,
            'exit_reason': 'Timeout',
            'pnl': pnl,
            'return_pct': pnl_pct,
            'bars_held': len(future_bars),
            'win': pnl > 0
        }
    
    def run_backtest(self, df, start_idx=0, long_enabled=True, short_enabled=True):
        print(f"\n{'='*80}")
        print(f"[BACKTEST] v10 Scalping Strategy")
        print(f"{'='*80}")
        print(f"Period: {df.index[start_idx]} to {df.index[-1]}")
        print(f"Threshold: {self.threshold}")
        print(f"TP: {self.tp_pct:.2%}, SL: {self.sl_pct:.2%}, RR: {self.tp_pct/self.sl_pct:.2f}")
        print(f"Long: {long_enabled}, Short: {short_enabled}")
        print("")
        
        # 計算特徵
        features = calculate_microstructure_features(df)
        features = features.fillna(0).replace([np.inf, -np.inf], 0)
        features = features[self.feature_names]
        
        # 預測
        long_proba = self.long_model.predict_proba(features)[:, 1]
        short_proba = self.short_model.predict_proba(features)[:, 1]
        
        # 模擬交易
        capital = self.initial_capital
        self.trades = []
        self.equity_curve = [(df.index[start_idx], capital)]
        
        max_horizon = 5
        
        for i in range(start_idx, len(df) - max_horizon):
            idx = df.index[i]
            entry_price = df.loc[idx, 'close']
            
            # Long
            if long_enabled and long_proba[i] >= self.threshold:
                tp_price = entry_price * (1 + self.tp_pct)
                sl_price = entry_price * (1 - self.sl_pct)
                
                future_bars = df.iloc[i+1:i+1+max_horizon]
                result = self.simulate_trade(entry_price, tp_price, sl_price, 'long', future_bars)
                
                capital += result['pnl']
                
                self.trades.append({
                    'entry_time': idx,
                    'entry_price': entry_price,
                    'side': 'long',
                    'prediction': long_proba[i],
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    **result
                })
                
                self.equity_curve.append((idx, capital))
            
            # Short
            elif short_enabled and short_proba[i] >= self.threshold:
                tp_price = entry_price * (1 - self.tp_pct)
                sl_price = entry_price * (1 + self.sl_pct)
                
                future_bars = df.iloc[i+1:i+1+max_horizon]
                result = self.simulate_trade(entry_price, tp_price, sl_price, 'short', future_bars)
                
                capital += result['pnl']
                
                self.trades.append({
                    'entry_time': idx,
                    'entry_price': entry_price,
                    'side': 'short',
                    'prediction': short_proba[i],
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    **result
                })
                
                self.equity_curve.append((idx, capital))
        
        print(f"Total trades: {len(self.trades)}")
        
        return self.generate_report() if self.trades else None
    
    def generate_report(self):
        trades_df = pd.DataFrame(self.trades)
        
        wins = trades_df[trades_df['win']]
        losses = trades_df[~trades_df['win']]
        
        total_pnl = trades_df['pnl'].sum()
        total_return = total_pnl / self.initial_capital
        
        win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0
        
        equity_df = pd.DataFrame(self.equity_curve, columns=['time', 'equity'])
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['equity'].cummax()) / equity_df['equity'].cummax()
        max_drawdown = equity_df['drawdown'].min()
        
        returns = equity_df['equity'].pct_change().dropna()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        print(f"\n{'='*80}")
        print(f"[REPORT]")
        print(f"{'='*80}\n")
        
        print(f"總交易: {len(trades_df)}")
        print(f"勝/敗: {len(wins)} / {len(losses)}")
        print(f"勝率: {win_rate:.2%}")
        print(f"")
        print(f"總 PnL: ${total_pnl:.2f}")
        print(f"總報酬: {total_return:.2%}")
        print(f"最終資金: ${self.initial_capital + total_pnl:,.2f}")
        print(f"")
        print(f"平均獲利: ${avg_win:.2f}")
        print(f"平均虧損: ${avg_loss:.2f}")
        print(f"盈虧比: {profit_factor:.2f}")
        print(f"最大回撤: {max_drawdown:.2%}")
        print(f"Sharpe: {sharpe:.2f}")
        print(f"平均持有: {trades_df['bars_held'].mean():.1f} bars")
        print(f"")
        
        # 分邊
        for side in ['long', 'short']:
            side_trades = trades_df[trades_df['side'] == side]
            if len(side_trades) > 0:
                side_wins = side_trades[side_trades['win']]
                side_wr = len(side_wins) / len(side_trades)
                side_pnl = side_trades['pnl'].sum()
                print(f"  {side.upper()}: {len(side_trades)} trades, {side_wr:.1%} win, ${side_pnl:.2f}")
        
        print(f"\n{'='*80}\n")
        
        return {
            'summary': {
                'total_trades': len(trades_df),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'total_return_pct': total_return,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe,
                'avg_bars_held': trades_df['bars_held'].mean()
            },
            'trades': trades_df,
            'equity': equity_df
        }


if __name__ == '__main__':
    print("\n" + "="*80)
    print("[V10 SCALPING BACKTEST & OPTIMIZATION]")
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
    
    print(f"\nUsing models:")
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
    print(f"OOS period: {len(df) - oos_start} bars\n")
    
    # 測試配置
    configs = [
        {'name': 'Both, threshold=0.5', 'threshold': 0.5, 'long': True, 'short': True},
        {'name': 'Both, threshold=0.6', 'threshold': 0.6, 'long': True, 'short': True},
        {'name': 'Both, threshold=0.7', 'threshold': 0.7, 'long': True, 'short': True},
        {'name': 'Both, threshold=0.8', 'threshold': 0.8, 'long': True, 'short': True},
        {'name': 'Long only, threshold=0.5', 'threshold': 0.5, 'long': True, 'short': False},
        {'name': 'Long only, threshold=0.6', 'threshold': 0.6, 'long': True, 'short': False},
        {'name': 'Long only, threshold=0.7', 'threshold': 0.7, 'long': True, 'short': False},
        {'name': 'Short only, threshold=0.5', 'threshold': 0.5, 'long': False, 'short': True},
        {'name': 'Short only, threshold=0.6', 'threshold': 0.6, 'long': False, 'short': True},
        {'name': 'Short only, threshold=0.7', 'threshold': 0.7, 'long': False, 'short': True},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f"Testing: {config['name']}")
        print(f"{'='*80}")
        
        try:
            backtester = ScalpingBacktester(
                long_model_path=long_path,
                short_model_path=short_path,
                initial_capital=10000,
                position_size=0.02,
                leverage=10,
                threshold=config['threshold'],
                tp_pct=0.004,
                sl_pct=0.0025
            )
            
            report = backtester.run_backtest(
                df, 
                start_idx=oos_start,
                long_enabled=config['long'],
                short_enabled=config['short']
            )
            
            if report:
                s = report['summary']
                results.append({
                    'config': config['name'],
                    'threshold': config['threshold'],
                    'trades': s['total_trades'],
                    'win_rate': s['win_rate'],
                    'return_pct': s['total_return_pct'],
                    'profit_factor': s['profit_factor'],
                    'sharpe': s['sharpe_ratio'],
                    'max_dd': s['max_drawdown'],
                    'avg_bars': s['avg_bars_held']
                })
        
        except Exception as e:
            print(f"  錯誤: {e}")
            import traceback
            traceback.print_exc()
    
    # 比較
    if results:
        print(f"\n\n{'='*80}")
        print("[RESULTS COMPARISON]")
        print(f"{'='*80}\n")
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('return_pct', ascending=False)
        
        print(f"{'Config':<30} {'Thresh':<8} {'Trades':<8} {'Win%':<8} {'Return%':<10} {'PF':<8} {'Sharpe':<8}")
        print("-" * 90)
        
        for _, row in df_results.iterrows():
            print(f"{row['config']:<30} {row['threshold']:<8.2f} {row['trades']:<8.0f} {row['win_rate']*100:<8.1f} {row['return_pct']*100:<10.2f} {row['profit_factor']:<8.2f} {row['sharpe']:<8.2f}")
        
        # 最佳
        best = df_results.iloc[0]
        
        print(f"\n\n{'='*80}")
        print("[BEST CONFIGURATION - V10 SCALPING]")
        print(f"{'='*80}\n")
        
        print(f"配置: {best['config']}")
        print(f"Threshold: {best['threshold']:.2f}")
        print(f"")
        print(f"交易數: {best['trades']:.0f}")
        print(f"勝率: {best['win_rate']:.2%}")
        print(f"總報酬: {best['return_pct']:.2%}")
        print(f"盈虧比: {best['profit_factor']:.2f}")
        print(f"Sharpe: {best['sharpe']:.2f}")
        print(f"最大回撤: {best['max_dd']:.2%}")
        print(f"平均持有: {best['avg_bars']:.1f} bars")
        print(f"")
        
        total_days = (len(df) - oos_start) / 96
        trades_per_day = best['trades'] / total_days
        annual_return = best['return_pct'] * (365 / total_days)
        
        print(f"每日交易: {trades_per_day:.1f}")
        print(f"年化報酬: {annual_return:.1%}")
        
        # 保存
        output_dir = Path('backtest_results')
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        df_results.to_csv(output_dir / f'v10_scalping_optimization_{timestamp}.csv', index=False)
        
        print(f"\n結果已保存: {output_dir / f'v10_scalping_optimization_{timestamp}.csv'}")
        print("="*80)
