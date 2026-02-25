#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v7 均值回歸模型回測系統

功能:
1. 加載 v7 模型
2. 模擬真實交易 (TP/SL/手續費)
3. 生成詳細報告
4. 視覺化結果
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


class MeanReversionBacktester:
    """均值回歸策略回測器"""
    
    def __init__(
        self,
        upper_model_path: str,
        lower_model_path: str,
        initial_capital: float = 10000.0,
        position_size: float = 0.02,  # 2% per trade
        leverage: int = 10,
        maker_fee: float = 0.0002,  # 0.02%
        taker_fee: float = 0.0004,  # 0.04%
        threshold: float = 0.5
    ):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.leverage = leverage
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.threshold = threshold
        
        # 加載模型
        print(f"Loading models...")
        with open(upper_model_path, 'rb') as f:
            upper_data = pickle.load(f)
            self.upper_model = upper_data['model']
            self.features = upper_data['features']
            self.metadata = upper_data['metadata']
        
        with open(lower_model_path, 'rb') as f:
            lower_data = pickle.load(f)
            self.lower_model = lower_data['model']
        
        print(f"Models loaded successfully")
        print(f"Strategy: {self.metadata.get('strategy', 'unknown')}")
        print(f"SL Multiplier: {self.metadata.get('sl_multiplier', 0.5)}")
        
        # 回測結果
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
    
    def calculate_keltner_channels(self, df: pd.DataFrame):
        """計算肯特納通道"""
        params = self.metadata['keltner_params']
        ema_period = params['ema_period']
        atr_period = params['atr_period']
        multiplier = params['multiplier']
        
        middle = df['close'].ewm(span=ema_period, adjust=False).mean()
        
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(atr_period).mean()
        upper = middle + (atr * multiplier)
        lower = middle - (atr * multiplier)
        
        return upper, middle, lower
    
    def identify_touch_events(self, df: pd.DataFrame, upper: pd.Series, lower: pd.Series):
        """識別觸碰事件"""
        threshold = 0.003
        dist_to_upper = (upper - df['close']) / df['close']
        dist_to_lower = (df['close'] - lower) / df['close']
        
        upper_touch = (dist_to_upper <= threshold) & (dist_to_upper >= -threshold)
        lower_touch = (dist_to_lower <= threshold) & (dist_to_lower >= -threshold)
        
        return upper_touch, lower_touch
    
    def calculate_features(self, df: pd.DataFrame, upper: pd.Series, middle: pd.Series, lower: pd.Series):
        """計算特徵 (複製 train 的邏輯)"""
        features = pd.DataFrame(index=df.index)
        
        channel_width = upper - lower
        features['position_in_channel'] = (df['close'] - lower) / (channel_width + 1e-8)
        features['dist_to_upper'] = (upper - df['close']) / df['close']
        features['dist_to_lower'] = (df['close'] - lower) / df['close']
        features['dist_to_middle'] = (df['close'] - middle) / middle
        
        features['momentum_1'] = df['close'].pct_change(1)
        features['momentum_2'] = df['close'].pct_change(2)
        features['momentum_3'] = df['close'].pct_change(3)
        features['acceleration'] = features['momentum_1'] - features['momentum_1'].shift(1)
        
        high_low_range = df['high'] - df['low'] + 1e-8
        body = abs(df['close'] - df['open'])
        features['body_ratio'] = body / high_low_range
        
        upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
        lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
        features['upper_shadow_ratio'] = upper_shadow / high_low_range
        features['lower_shadow_ratio'] = lower_shadow / high_low_range
        features['buy_pressure'] = (df['close'] - df['low']) / high_low_range
        
        vol_ma6 = df['volume'].rolling(6).mean()
        vol_ma24 = df['volume'].rolling(24).mean()
        features['volume_ratio_6'] = df['volume'] / (vol_ma6 + 1e-8)
        features['volume_ratio_24'] = df['volume'] / (vol_ma24 + 1e-8)
        
        price_change = df['close'].pct_change()
        features['volume_price_corr'] = price_change.rolling(6).corr(df['volume'].pct_change())
        
        features['volatility'] = df['close'].pct_change().rolling(6).std()
        features['volatility_24h'] = df['close'].pct_change().rolling(24).std()
        features['volatility_expanding'] = (features['volatility'] > features['volatility_24h']).astype(int)
        
        features['channel_width'] = (upper - lower) / middle
        features['channel_width_change'] = features['channel_width'].pct_change(3)
        features['middle_slope'] = middle.pct_change(3)
        features['deviation_from_middle'] = abs(df['close'] - middle) / middle
        
        dist_to_upper_abs = abs((upper - df['close']) / df['close'])
        dist_to_lower_abs = abs((df['close'] - lower) / df['close'])
        near_upper = (dist_to_upper_abs < 0.005).astype(int)
        near_lower = (dist_to_lower_abs < 0.005).astype(int)
        features['upper_touches_24h'] = near_upper.rolling(24).sum()
        features['lower_touches_24h'] = near_lower.rolling(24).sum()
        
        price_changes = df['close'].diff()
        gains = price_changes.where(price_changes > 0, 0).rolling(14).mean()
        losses = -price_changes.where(price_changes < 0, 0).rolling(14).mean()
        features['rsi_like'] = 100 - (100 / (1 + gains / (losses + 1e-8)))
        
        if 'open_time' in df.columns:
            hour = pd.to_datetime(df['open_time']).dt.hour
        elif isinstance(df.index, pd.DatetimeIndex):
            hour = df.index.hour
        else:
            hour = pd.Series(12, index=df.index)
        features['is_high_vol_time'] = ((hour >= 8) & (hour < 16)).astype(int)
        
        return features.fillna(0).replace([np.inf, -np.inf], 0)
    
    def simulate_trade(
        self,
        entry_price: float,
        tp_price: float,
        sl_price: float,
        side: str,
        future_bars: pd.DataFrame
    ) -> Dict:
        """
        模擬交易執行
        
        Args:
            entry_price: 開倉價
            tp_price: 止盈價
            sl_price: 止損價
            side: 'long' or 'short'
            future_bars: 未來的K線數據
        
        Returns:
            交易結果
        """
        position_value = self.initial_capital * self.position_size * self.leverage
        entry_fee = position_value * self.taker_fee
        
        for i, (idx, bar) in enumerate(future_bars.iterrows()):
            if side == 'long':
                # Long: 看是否先觸及 SL 或 TP
                if bar['low'] <= sl_price:
                    # 觸及止損
                    exit_price = sl_price
                    pnl_pct = (exit_price - entry_price) / entry_price
                    exit_fee = position_value * self.taker_fee
                    net_pnl = (position_value * pnl_pct) - entry_fee - exit_fee
                    
                    return {
                        'result': 'loss',
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'pnl_pct': net_pnl / (self.initial_capital * self.position_size),
                        'bars_held': i + 1,
                        'exit_reason': 'stop_loss'
                    }
                
                if bar['high'] >= tp_price:
                    # 觸及止盈
                    exit_price = tp_price
                    pnl_pct = (exit_price - entry_price) / entry_price
                    exit_fee = position_value * self.maker_fee  # TP 用 maker fee
                    net_pnl = (position_value * pnl_pct) - entry_fee - exit_fee
                    
                    return {
                        'result': 'win',
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'pnl_pct': net_pnl / (self.initial_capital * self.position_size),
                        'bars_held': i + 1,
                        'exit_reason': 'take_profit'
                    }
            
            else:  # short
                if bar['high'] >= sl_price:
                    exit_price = sl_price
                    pnl_pct = (entry_price - exit_price) / entry_price
                    exit_fee = position_value * self.taker_fee
                    net_pnl = (position_value * pnl_pct) - entry_fee - exit_fee
                    
                    return {
                        'result': 'loss',
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'pnl_pct': net_pnl / (self.initial_capital * self.position_size),
                        'bars_held': i + 1,
                        'exit_reason': 'stop_loss'
                    }
                
                if bar['low'] <= tp_price:
                    exit_price = tp_price
                    pnl_pct = (entry_price - exit_price) / entry_price
                    exit_fee = position_value * self.maker_fee
                    net_pnl = (position_value * pnl_pct) - entry_fee - exit_fee
                    
                    return {
                        'result': 'win',
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'pnl_pct': net_pnl / (self.initial_capital * self.position_size),
                        'bars_held': i + 1,
                        'exit_reason': 'take_profit'
                    }
        
        # 超時未出場
        exit_price = future_bars.iloc[-1]['close']
        if side == 'long':
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price
        
        exit_fee = position_value * self.taker_fee
        net_pnl = (position_value * pnl_pct) - entry_fee - exit_fee
        
        return {
            'result': 'timeout',
            'exit_price': exit_price,
            'pnl': net_pnl,
            'pnl_pct': net_pnl / (self.initial_capital * self.position_size),
            'bars_held': len(future_bars),
            'exit_reason': 'timeout'
        }
    
    def run_backtest(self, df: pd.DataFrame, start_idx: int = 0) -> Dict:
        """
        執行回測
        
        Args:
            df: K線數據
            start_idx: 開始索引 (跳過訓練集)
        """
        print(f"\n{'='*80}")
        print(f"[BACKTEST] Starting...")
        print(f"{'='*80}")
        print(f"Period: {df.index[start_idx]} to {df.index[-1]}")
        print(f"Total bars: {len(df) - start_idx}")
        print(f"Initial capital: ${self.initial_capital:,.2f}")
        print(f"Position size: {self.position_size:.1%} per trade")
        print(f"Leverage: {self.leverage}x")
        print(f"Prediction threshold: {self.threshold}")
        print(f"")
        
        # 計算通道和特徵
        print("Calculating indicators...")
        upper, middle, lower = self.calculate_keltner_channels(df)
        upper_touch, lower_touch = self.identify_touch_events(df, upper, lower)
        features = self.calculate_features(df, upper, middle, lower)
        
        # 預測
        print("Making predictions...")
        X_upper = features[upper_touch]
        X_lower = features[lower_touch]
        
        upper_proba = self.upper_model.predict_proba(X_upper)[:, 1] if len(X_upper) > 0 else []
        lower_proba = self.lower_model.predict_proba(X_lower)[:, 1] if len(X_lower) > 0 else []
        
        # 模擬交易
        print(f"\nSimulating trades...")
        capital = self.initial_capital
        self.trades = []
        self.equity_curve = [(df.index[start_idx], capital)]
        
        sl_multiplier = self.metadata.get('sl_multiplier', 0.5)
        max_horizon = 24  # 最多看 24 根K線 (6小時)
        
        # 處理上軌觸碰 (Short)
        upper_indices = upper_touch[upper_touch].index
        for i, (idx, prob) in enumerate(zip(upper_indices, upper_proba)):
            if idx < start_idx:
                continue
            
            if prob < self.threshold:
                continue
            
            idx_pos = df.index.get_loc(idx)
            if idx_pos >= len(df) - max_horizon:
                continue
            
            entry_price = df.loc[idx, 'close']
            tp_price = middle.loc[idx]
            channel_width = upper.loc[idx] - lower.loc[idx]
            sl_price = upper.loc[idx] + (channel_width * sl_multiplier)
            
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
        
        # 處理下軌觸碰 (Long)
        lower_indices = lower_touch[lower_touch].index
        for i, (idx, prob) in enumerate(zip(lower_indices, lower_proba)):
            if idx < start_idx:
                continue
            
            if prob < self.threshold:
                continue
            
            idx_pos = df.index.get_loc(idx)
            if idx_pos >= len(df) - max_horizon:
                continue
            
            entry_price = df.loc[idx, 'close']
            tp_price = middle.loc[idx]
            channel_width = upper.loc[idx] - lower.loc[idx]
            sl_price = lower.loc[idx] - (channel_width * sl_multiplier)
            
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
        
        # 排序交易 (依時間)
        self.trades = sorted(self.trades, key=lambda x: x['entry_time'])
        
        print(f"\nBacktest complete!")
        print(f"Total trades: {len(self.trades)}")
        
        return self.generate_report()
    
    def generate_report(self) -> Dict:
        """生成回測報告"""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        df_trades = pd.DataFrame(self.trades)
        
        # 基本統計
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['result'] == 'win'])
        losing_trades = len(df_trades[df_trades['result'] == 'loss'])
        timeout_trades = len(df_trades[df_trades['result'] == 'timeout'])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL
        total_pnl = df_trades['pnl'].sum()
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # 回報
        final_capital = self.initial_capital + total_pnl
        total_return_pct = (final_capital - self.initial_capital) / self.initial_capital
        
        # 最大回撤
        equity_series = pd.Series([e[1] for e in self.equity_curve])
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio (簡化版)
        returns = df_trades['pnl_pct'].values
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0
        
        report = {
            'summary': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'timeout_trades': timeout_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'total_return_pct': total_return_pct,
                'final_capital': final_capital,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe,
                'avg_bars_held': df_trades['bars_held'].mean()
            },
            'by_side': {
                'long': {
                    'trades': len(df_trades[df_trades['side'] == 'long']),
                    'win_rate': len(df_trades[(df_trades['side'] == 'long') & (df_trades['result'] == 'win')]) / len(df_trades[df_trades['side'] == 'long']) if len(df_trades[df_trades['side'] == 'long']) > 0 else 0,
                    'pnl': df_trades[df_trades['side'] == 'long']['pnl'].sum()
                },
                'short': {
                    'trades': len(df_trades[df_trades['side'] == 'short']),
                    'win_rate': len(df_trades[(df_trades['side'] == 'short') & (df_trades['result'] == 'win')]) / len(df_trades[df_trades['side'] == 'short']) if len(df_trades[df_trades['side'] == 'short']) > 0 else 0,
                    'pnl': df_trades[df_trades['side'] == 'short']['pnl'].sum()
                }
            }
        }
        
        return report
    
    def print_report(self, report: Dict):
        """列印報告"""
        print(f"\n{'='*80}")
        print(f"[BACKTEST REPORT]")
        print(f"{'='*80}")
        
        s = report['summary']
        print(f"\n總覽:")
        print(f"  總交易數: {s['total_trades']}")
        print(f"  勝交易: {s['winning_trades']} | 敗交易: {s['losing_trades']} | 超時: {s['timeout_trades']}")
        print(f"  勝率: {s['win_rate']:.2%}")
        print(f"")
        print(f"績效:")
        print(f"  總 PnL: ${s['total_pnl']:,.2f}")
        print(f"  總報酬: {s['total_return_pct']:.2%}")
        print(f"  最終資金: ${s['final_capital']:,.2f}")
        print(f"  平均獲利: ${s['avg_win']:,.2f}")
        print(f"  平均虧損: ${s['avg_loss']:,.2f}")
        print(f"  盈虧比: {s['profit_factor']:.2f}")
        print(f"  最大回撤: {s['max_drawdown']:.2%}")
        print(f"  Sharpe Ratio: {s['sharpe_ratio']:.2f}")
        print(f"  平均持有: {s['avg_bars_held']:.1f} bars")
        
        print(f"\n分邊統計:")
        for side, data in report['by_side'].items():
            print(f"  {side.upper()}:")
            print(f"    交易數: {data['trades']}")
            print(f"    勝率: {data['win_rate']:.2%}")
            print(f"    PnL: ${data['pnl']:,.2f}")
        
        print(f"\n{'='*80}")


if __name__ == '__main__':
    # 找最新的 v7mr 模型
    models_dir = Path('models_output')
    upper_models = sorted(models_dir.glob('keltner_upper_*_v7mr_*.pkl'))
    lower_models = sorted(models_dir.glob('keltner_lower_*_v7mr_*.pkl'))
    
    if not upper_models or not lower_models:
        print("錯誤: 找不到 v7mr 模型")
        print("請先執行: python train_v7_channel_mean_reversion.py")
        sys.exit(1)
    
    upper_path = str(upper_models[-1])
    lower_path = str(lower_models[-1])
    
    print(f"Using models:")
    print(f"  Upper: {Path(upper_path).name}")
    print(f"  Lower: {Path(lower_path).name}")
    
    # 初始化回測器
    backtester = MeanReversionBacktester(
        upper_model_path=upper_path,
        lower_model_path=lower_path,
        initial_capital=10000,
        position_size=0.02,
        leverage=10,
        threshold=0.5
    )
    
    # 載入數據
    print(f"\nLoading data...")
    from utils.hf_data_loader import load_klines
    
    df = load_klines(
        symbol='BTCUSDT',
        timeframe='15m',
        start_date=(datetime.now() - timedelta(days=9999)).strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    # 只回測 OOS 時段 (最後 10%)
    oos_start = int(len(df) * 0.9)
    
    # 執行回測
    report = backtester.run_backtest(df, start_idx=oos_start)
    
    # 顯示結果
    backtester.print_report(report)
    
    # 保存結果
    output_dir = Path('backtest_results')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存報告
    with open(output_dir / f'report_v7mr_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 保存交易詳情
    pd.DataFrame(backtester.trades).to_csv(output_dir / f'trades_v7mr_{timestamp}.csv', index=False)
    
    # 保存權益曲線
    pd.DataFrame(backtester.equity_curve, columns=['time', 'equity']).to_csv(
        output_dir / f'equity_v7mr_{timestamp}.csv', index=False
    )
    
    print(f"\n結果已保存到 {output_dir}")
