#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v9 動能反轉模型回測系統

策略:
1. 識別動能衰竭點
2. 模型預測反轉
3. 動態 TP/SL (基於 ATR)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from datetime import datetime, timedelta
import json
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


class ReversalBacktester:
    """動能反轉策略回測器"""
    
    def __init__(
        self,
        bearish_model_path: str,
        bullish_model_path: str,
        initial_capital: float = 10000.0,
        position_size: float = 0.02,
        leverage: int = 10,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0004,
        threshold: float = 0.5,
        tp_atr_multiplier: float = 1.5,
        sl_atr_multiplier: float = 1.0
    ):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.leverage = leverage
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.threshold = threshold
        self.tp_atr_multiplier = tp_atr_multiplier
        self.sl_atr_multiplier = sl_atr_multiplier
        
        # 載入模型
        print("Loading models...")
        with open(bearish_model_path, 'rb') as f:
            data = pickle.load(f)
            self.bearish_model = data['model']
            self.features = data['features']
            self.metadata = data['metadata']
        
        with open(bullish_model_path, 'rb') as f:
            data = pickle.load(f)
            self.bullish_model = data['model']
        
        print("Models loaded successfully")
        
        self.trades = []
        self.equity_curve = []
    
    def calculate_indicators(self, df: pd.DataFrame) -> dict:
        """計算技術指標"""
        indicators = {}
        
        indicators['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        indicators['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        indicators['atr'] = tr.rolling(14).mean()
        
        middle = indicators['ema_20']
        indicators['keltner_upper'] = middle + (indicators['atr'] * 2.0)
        indicators['keltner_lower'] = middle - (indicators['atr'] * 2.0)
        
        price_changes = df['close'].diff()
        gains = price_changes.where(price_changes > 0, 0).rolling(14).mean()
        losses = -price_changes.where(price_changes < 0, 0).rolling(14).mean()
        rs = gains / (losses + 1e-8)
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        indicators['macd'] = ema_12 - ema_26
        indicators['macd_signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
        indicators['macd_hist'] = indicators['macd'] - indicators['macd_signal']
        
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        indicators['bb_upper'] = sma_20 + (std_20 * 2)
        indicators['bb_lower'] = sma_20 - (std_20 * 2)
        indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / sma_20
        
        return indicators
    
    def identify_exhaustion_points(self, df: pd.DataFrame, indicators: dict) -> tuple:
        """識別動能衰竭點"""
        params = self.metadata['params']
        consecutive_bars = params['consecutive_bars']
        min_move_pct = params['min_move_pct']
        
        close = df['close']
        rsi = indicators['rsi']
        keltner_upper = indicators['keltner_upper']
        keltner_lower = indicators['keltner_lower']
        
        returns = close.pct_change()
        
        bullish_exhaustion = pd.Series(False, index=df.index)
        bearish_exhaustion = pd.Series(False, index=df.index)
        
        for i in range(consecutive_bars, len(df)):
            recent_returns = returns.iloc[i-consecutive_bars+1:i+1]
            cumulative_return = (1 + recent_returns).prod() - 1
            
            is_consecutive_up = (recent_returns > 0).all()
            is_consecutive_down = (recent_returns < 0).all()
            
            if is_consecutive_up and cumulative_return >= min_move_pct:
                near_upper = (close.iloc[i] >= keltner_upper.iloc[i] * 0.997)
                rsi_overbought = (rsi.iloc[i] >= 70)
                
                if near_upper or rsi_overbought:
                    bullish_exhaustion.iloc[i] = True
            
            if is_consecutive_down and abs(cumulative_return) >= min_move_pct:
                near_lower = (close.iloc[i] <= keltner_lower.iloc[i] * 1.003)
                rsi_oversold = (rsi.iloc[i] <= 30)
                
                if near_lower or rsi_oversold:
                    bearish_exhaustion.iloc[i] = True
        
        return bullish_exhaustion, bearish_exhaustion
    
    def calculate_features(self, df: pd.DataFrame, indicators: dict) -> pd.DataFrame:
        """計算特徵"""
        features = pd.DataFrame(index=df.index)
        
        close = df['close']
        volume = df['volume']
        atr = indicators['atr']
        rsi = indicators['rsi']
        macd = indicators['macd']
        macd_hist = indicators['macd_hist']
        
        features['momentum_1'] = close.pct_change(1)
        features['momentum_2'] = close.pct_change(2)
        features['momentum_3'] = close.pct_change(3)
        features['momentum_5'] = close.pct_change(5)
        
        features['acceleration_1'] = features['momentum_1'].diff()
        features['acceleration_2'] = features['momentum_2'].diff()
        
        features['momentum_weakening'] = (
            (features['momentum_1'].abs() < features['momentum_2'].abs()) &
            (features['momentum_2'].abs() < features['momentum_3'].abs())
        ).astype(int)
        
        features['rsi'] = rsi
        features['rsi_overbought'] = (rsi >= 70).astype(int)
        features['rsi_oversold'] = (rsi <= 30).astype(int)
        features['rsi_extreme'] = ((rsi >= 75) | (rsi <= 25)).astype(int)
        features['rsi_change'] = rsi.diff()
        features['rsi_divergence'] = ((close > close.shift(5)) & (rsi < rsi.shift(5))).astype(int)
        
        features['macd'] = macd
        features['macd_hist'] = macd_hist
        features['macd_hist_decreasing'] = (macd_hist < macd_hist.shift(1)).astype(int)
        features['macd_cross'] = ((macd > 0) & (macd.shift(1) <= 0)).astype(int) - ((macd < 0) & (macd.shift(1) >= 0)).astype(int)
        
        keltner_upper = indicators['keltner_upper']
        keltner_lower = indicators['keltner_lower']
        keltner_middle = indicators['ema_20']
        
        features['dist_to_upper'] = (keltner_upper - close) / close
        features['dist_to_lower'] = (close - keltner_lower) / close
        features['dist_to_middle'] = (close - keltner_middle) / close
        features['at_upper_band'] = (close >= keltner_upper * 0.997).astype(int)
        features['at_lower_band'] = (close <= keltner_lower * 1.003).astype(int)
        
        features['atr_pct'] = atr / close
        features['volatility_5'] = close.pct_change().rolling(5).std()
        features['volatility_10'] = close.pct_change().rolling(10).std()
        features['volatility_expanding'] = (features['volatility_5'] > features['volatility_10']).astype(int)
        
        features['bb_width'] = indicators['bb_width']
        features['bb_squeeze'] = (features['bb_width'] < features['bb_width'].rolling(20).mean()).astype(int)
        
        vol_ma_5 = volume.rolling(5).mean()
        vol_ma_20 = volume.rolling(20).mean()
        features['volume_ratio'] = volume / (vol_ma_20 + 1e-8)
        features['volume_spike'] = (volume > vol_ma_5 * 1.5).astype(int)
        features['volume_declining'] = (vol_ma_5 < vol_ma_20).astype(int)
        
        high_low_range = df['high'] - df['low'] + 1e-8
        body = abs(close - df['open'])
        features['body_ratio'] = body / high_low_range
        features['candle_size'] = high_low_range / close
        
        upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
        lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
        features['upper_shadow_ratio'] = upper_shadow / high_low_range
        features['lower_shadow_ratio'] = lower_shadow / high_low_range
        
        features['hammer_like'] = ((features['lower_shadow_ratio'] > 0.6) & (features['body_ratio'] < 0.3)).astype(int)
        features['shooting_star_like'] = ((features['upper_shadow_ratio'] > 0.6) & (features['body_ratio'] < 0.3)).astype(int)
        
        ema_20 = indicators['ema_20']
        ema_50 = indicators['ema_50']
        features['ema_trend'] = ((ema_20 > ema_50).astype(int) * 2 - 1)
        features['price_vs_ema20'] = (close - ema_20) / ema_20
        features['ema_slope'] = ema_20.pct_change(3)
        
        returns = close.pct_change()
        features['consecutive_up'] = (returns > 0).rolling(3).sum()
        features['consecutive_down'] = (returns < 0).rolling(3).sum()
        
        return features.fillna(0).replace([np.inf, -np.inf], 0)
    
    def simulate_trade(
        self,
        entry_price: float,
        tp_price: float,
        sl_price: float,
        side: str,
        future_bars: pd.DataFrame
    ) -> Dict:
        """模擬交易"""
        position_value = self.initial_capital * self.position_size * self.leverage
        entry_fee = position_value * self.taker_fee
        
        for i, (idx, bar) in enumerate(future_bars.iterrows()):
            if side == 'long':
                if bar['low'] <= sl_price:
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
                    exit_price = tp_price
                    pnl_pct = (exit_price - entry_price) / entry_price
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
        
        # 超時
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
        """執行回測"""
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
    
    def generate_report(self) -> Dict:
        """生成報告"""
        if not self.trades:
            return {'error': 'No trades'}
        
        df_trades = pd.DataFrame(self.trades)
        
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['result'] == 'win'])
        losing_trades = len(df_trades[df_trades['result'] == 'loss'])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = df_trades['pnl'].sum()
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        final_capital = self.initial_capital + total_pnl
        total_return_pct = (final_capital - self.initial_capital) / self.initial_capital
        
        equity_series = pd.Series([e[1] for e in self.equity_curve])
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        returns = df_trades['pnl_pct'].values
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0
        
        report = {
            'summary': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
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
        print(f"[REPORT]")
        print(f"{'='*80}")
        
        s = report['summary']
        print(f"\n總交易: {s['total_trades']}")
        print(f"勝/敗: {s['winning_trades']} / {s['losing_trades']}")
        print(f"勝率: {s['win_rate']:.2%}")
        print(f"\n總 PnL: ${s['total_pnl']:,.2f}")
        print(f"總報酬: {s['total_return_pct']:.2%}")
        print(f"最終資金: ${s['final_capital']:,.2f}")
        print(f"\n平均獲利: ${s['avg_win']:,.2f}")
        print(f"平均虧損: ${s['avg_loss']:,.2f}")
        print(f"盈虧比: {s['profit_factor']:.2f}")
        print(f"最大回撤: {s['max_drawdown']:.2%}")
        print(f"Sharpe: {s['sharpe_ratio']:.2f}")
        print(f"平均持有: {s['avg_bars_held']:.1f} bars")
        
        print(f"\n分邊:")
        for side, data in report['by_side'].items():
            print(f"  {side.upper()}: {data['trades']} trades, {data['win_rate']:.1%} win, ${data['pnl']:,.2f}")
        
        print(f"\n{'='*80}")


if __name__ == '__main__':
    models_dir = Path('models_output')
    bearish_models = sorted(models_dir.glob('reversal_bearish_*_v9_*.pkl'))
    bullish_models = sorted(models_dir.glob('reversal_bullish_*_v9_*.pkl'))
    
    if not bearish_models or not bullish_models:
        print("錯誤: 找不到 v9 模型")
        print("請先執行: python train_v9_momentum_reversal.py")
        sys.exit(1)
    
    bearish_path = str(bearish_models[-1])
    bullish_path = str(bullish_models[-1])
    
    print(f"Using models:")
    print(f"  Bearish: {Path(bearish_path).name}")
    print(f"  Bullish: {Path(bullish_path).name}")
    
    backtester = ReversalBacktester(
        bearish_model_path=bearish_path,
        bullish_model_path=bullish_path,
        initial_capital=10000,
        position_size=0.02,
        leverage=10,
        threshold=0.5,
        tp_atr_multiplier=1.5,
        sl_atr_multiplier=1.0
    )
    
    print(f"\nLoading data...")
    from utils.hf_data_loader import load_klines
    
    df = load_klines(
        symbol='BTCUSDT',
        timeframe='15m',
        start_date=(datetime.now() - timedelta(days=9999)).strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    oos_start = int(len(df) * 0.9)
    
    report = backtester.run_backtest(df, start_idx=oos_start)
    backtester.print_report(report)
    
    # 保存
    output_dir = Path('backtest_results')
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with open(output_dir / f'report_v9_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    pd.DataFrame(backtester.trades).to_csv(output_dir / f'trades_v9_{timestamp}.csv', index=False)
    
    print(f"\n結果已保存到 {output_dir}")
