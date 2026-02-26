"""
Backtest Engine
回測引擎
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import requests

class BacktestEngine:
    def __init__(self, config: dict):
        self.initial_capital = config.get('initial_capital', 10)
        self.leverage = config.get('leverage', 3)
        self.maker_fee = config.get('maker_fee', 0.0002)
        self.taker_fee = config.get('taker_fee', 0.0004)
        self.slippage = config.get('slippage', 0.0001)
    
    def fetch_latest_data(self, symbol: str, timeframe: str, days: int = 30) -> pd.DataFrame:
        """從Binance API獲取最新數據"""
        
        interval_map = {
            '15m': '15m',
            '1h': '1h',
            '4h': '4h'
        }
        
        interval = interval_map.get(timeframe, '15m')
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        url = 'https://api.binance.com/api/v3/klines'
        
        all_data = []
        current_start = start_ms
        
        print(f"正在從Binance獲取 {symbol} {timeframe} 最近 {days} 天的數據...")
        
        while current_start < end_ms:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'endTime': end_ms,
                'limit': 1000
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                current_start = data[-1][0] + 1
                
                if len(data) < 1000:
                    break
                    
            except Exception as e:
                print(f"獲取數據失敗: {str(e)}")
                break
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        print(f"成功獲取 {len(df)} 筆數據")
        
        return df
    
    def run_backtest(self, df: pd.DataFrame, min_signal_strength: int = 2, 
                    min_confidence: float = 0.6) -> Dict:
        """執行回測"""
        
        capital = self.initial_capital
        position = None
        trades = []
        equity_curve = [{'time': df.iloc[0]['timestamp'], 'equity': capital}]
        
        for i in range(len(df)):
            row = df.iloc[i]
            current_price = row['close']
            current_time = row['timestamp']
            
            if position is not None:
                if position['type'] == 'LONG':
                    if current_price <= position['stop_loss']:
                        exit_price = position['stop_loss'] * (1 - self.slippage)
                        pnl = self._calculate_pnl(position, exit_price)
                        capital += pnl
                        
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': current_time,
                            'type': 'LONG',
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'size': position['size'],
                            'pnl': pnl,
                            'return_pct': (pnl / position['margin']) * 100,
                            'reason': 'stop_loss'
                        })
                        position = None
                        
                    elif current_price >= position['take_profit']:
                        exit_price = position['take_profit'] * (1 + self.slippage)
                        pnl = self._calculate_pnl(position, exit_price)
                        capital += pnl
                        
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': current_time,
                            'type': 'LONG',
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'size': position['size'],
                            'pnl': pnl,
                            'return_pct': (pnl / position['margin']) * 100,
                            'reason': 'take_profit'
                        })
                        position = None
                
                elif position['type'] == 'SHORT':
                    if current_price >= position['stop_loss']:
                        exit_price = position['stop_loss'] * (1 + self.slippage)
                        pnl = self._calculate_pnl(position, exit_price)
                        capital += pnl
                        
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': current_time,
                            'type': 'SHORT',
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'size': position['size'],
                            'pnl': pnl,
                            'return_pct': (pnl / position['margin']) * 100,
                            'reason': 'stop_loss'
                        })
                        position = None
                        
                    elif current_price <= position['take_profit']:
                        exit_price = position['take_profit'] * (1 - self.slippage)
                        pnl = self._calculate_pnl(position, exit_price)
                        capital += pnl
                        
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': current_time,
                            'type': 'SHORT',
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'size': position['size'],
                            'pnl': pnl,
                            'return_pct': (pnl / position['margin']) * 100,
                            'reason': 'take_profit'
                        })
                        position = None
            
            if position is None:
                if (row.get('signal_long', 0) == 1 and 
                    row.get('pred_long_valid', 0) == 1 and
                    row.get('signal_strength_long', 0) >= min_signal_strength and
                    row.get('pred_long_confidence', 0) >= min_confidence):
                    
                    entry_price = current_price * (1 + self.slippage)
                    position_value = capital * self.leverage
                    position_size = position_value / entry_price
                    margin = capital
                    fee = position_value * self.taker_fee
                    capital -= fee
                    
                    position = {
                        'type': 'LONG',
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'size': position_size,
                        'margin': margin,
                        'stop_loss': row.get('stop_loss', entry_price * 0.99),
                        'take_profit': row.get('take_profit', entry_price * 1.02)
                    }
                
                elif (row.get('signal_short', 0) == 1 and 
                      row.get('pred_short_valid', 0) == 1 and
                      row.get('signal_strength_short', 0) >= min_signal_strength and
                      row.get('pred_short_confidence', 0) >= min_confidence):
                    
                    entry_price = current_price * (1 - self.slippage)
                    position_value = capital * self.leverage
                    position_size = position_value / entry_price
                    margin = capital
                    fee = position_value * self.taker_fee
                    capital -= fee
                    
                    position = {
                        'type': 'SHORT',
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'size': position_size,
                        'margin': margin,
                        'stop_loss': row.get('stop_loss', entry_price * 1.01),
                        'take_profit': row.get('take_profit', entry_price * 0.98)
                    }
            
            equity_curve.append({'time': current_time, 'equity': capital})
        
        if len(trades) == 0:
            return {'error': '無交易產生'}
        
        results = self._calculate_metrics(trades, equity_curve)
        results['trades'] = trades
        results['equity_curve'] = equity_curve
        
        return results
    
    def _calculate_pnl(self, position: Dict, exit_price: float) -> float:
        """計算盈虧"""
        if position['type'] == 'LONG':
            price_change = exit_price - position['entry_price']
        else:
            price_change = position['entry_price'] - exit_price
        
        pnl = price_change * position['size']
        fee = exit_price * position['size'] * self.taker_fee
        
        return pnl - fee
    
    def _calculate_metrics(self, trades: List[Dict], equity_curve: List[Dict]) -> Dict:
        """計算績效指標"""
        trades_df = pd.DataFrame(trades)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        total_win = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        total_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
        profit_factor = (total_win / total_loss) if total_loss > 0 else 0
        
        equity_df = pd.DataFrame(equity_curve)
        final_capital = equity_df['equity'].iloc[-1]
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min() * 100
        
        returns = equity_df['equity'].pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
