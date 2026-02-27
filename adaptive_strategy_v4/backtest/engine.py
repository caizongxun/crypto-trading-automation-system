"""
V4 Backtest Engine - Kelly + LSTM
Kelly準則 + LSTM回測引擎
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import sys
from pathlib import Path

# 添加上級目錄到路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class BacktestEngine:
    """
    V4回測引擎
    
    核心特點:
    1. Kelly動態倉位管理
    2. LSTM預測勝率/賠率
    3. 多層風險控制
    4. 動態槓桿調整
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 資金設定
        self.initial_capital = config.get('initial_capital', 10000)
        self.commission = config.get('commission', 0.001)
        self.slippage = config.get('slippage', 0.0005)
        
        # Kelly配置
        self.kelly_fraction = config.get('kelly_fraction', 0.25)
        self.max_position = config.get('max_position', 0.20)
        self.min_kelly = config.get('min_kelly', 0.10)
        
        # 槓桿配置
        self.max_leverage = config.get('max_leverage', 3)
        
        # ATR止盈止損
        self.atr_tp_multiplier = config.get('atr_tp_multiplier', 2.0)
        self.atr_sl_multiplier = config.get('atr_sl_multiplier', 1.0)
        
        # 狀態追蹤
        self.trades = []
        self.equity_curve = []
        self.current_position = None
    
    def run(self, df: pd.DataFrame, 
            predictions: np.ndarray,
            win_rates: np.ndarray,
            payoffs: np.ndarray,
            confidences: np.ndarray) -> Dict:
        """
        執行回測
        
        Args:
            df: K線數據 (close, high, low, atr_14)
            predictions: 方向預測 (-1, 0, 1)
            win_rates: 勝率預測 (0-1)
            payoffs: 賠率預測 (>0)
            confidences: 信心度 (0-1)
        
        Returns:
            回測結果
        """
        print(f"\n[V4回測引擎] 開始回測")
        print(f"數據筆數: {len(df)}")
        print(f"信號筆數: {(predictions != 0).sum()}")
        print(f"Kelly分數: {self.kelly_fraction}")
        print(f"最大槓桿: {self.max_leverage}x")
        
        self.trades = []
        self.equity_curve = []
        self.current_position = None
        
        capital = self.initial_capital
        position_size = 0
        
        # Kelly管理器
        kelly_stats = {
            'trade_history': [],
            'recent_win_rate': 0.5,
            'recent_payoff': 1.5
        }
        
        # 驗證必要欄位
        required_cols = ['close', 'high', 'low', 'atr_14']
        for col in required_cols:
            if col not in df.columns:
                return {'metrics': {'error': f'缺少必要欄位: {col}'}}
        
        # 逐根K線處理
        for i in range(len(df)):
            current_price = df.iloc[i]['close']
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            atr = df.iloc[i]['atr_14']
            
            # 處理現有倉位
            if self.current_position is not None:
                pos = self.current_position
                
                # 檢查止盈止損
                exit_signal, exit_reason, exit_price = self._check_exit(
                    pos, current_high, current_low, current_price, i
                )
                
                # 平倉
                if exit_signal:
                    pnl = self._calculate_pnl(pos['entry_price'], exit_price, 
                                             position_size, pos['side'], pos['leverage'])
                    capital += pnl
                    
                    is_win = pnl > 0
                    kelly_stats['trade_history'].append({
                        'pnl': pnl,
                        'is_win': is_win,
                        'return_pct': abs(pnl) / (pos['entry_price'] * position_size)
                    })
                    
                    # 保持50筆歷史
                    if len(kelly_stats['trade_history']) > 50:
                        kelly_stats['trade_history'] = kelly_stats['trade_history'][-50:]
                    
                    # 更新Kelly統計
                    self._update_kelly_stats(kelly_stats)
                    
                    trade_record = {
                        'entry_time': pos['entry_time'],
                        'exit_time': df.iloc[i]['timestamp'] if 'timestamp' in df.columns else i,
                        'side': pos['side'],
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'size': position_size,
                        'leverage': pos['leverage'],
                        'pnl': pnl,
                        'pnl_pct': pnl / (pos['entry_price'] * position_size),
                        'exit_reason': exit_reason,
                        'kelly_pct': pos['kelly_pct'],
                        'predicted_win_rate': pos['predicted_win_rate'],
                        'predicted_payoff': pos['predicted_payoff']
                    }
                    self.trades.append(trade_record)
                    
                    self.current_position = None
                    position_size = 0
            
            # 開倉信號
            if self.current_position is None and i < len(predictions):
                signal = predictions[i]
                
                if signal != 0:
                    win_rate = win_rates[i]
                    payoff = payoffs[i]
                    confidence = confidences[i]
                    
                    # 計算Kelly倉位
                    kelly_pct, leverage, position_value = self._calculate_kelly_position(
                        win_rate, payoff, confidence, capital
                    )
                    
                    # Kelly門檻過濾
                    if kelly_pct < self.min_kelly:
                        continue
                    
                    position_size = position_value / current_price
                    
                    # 計算保證金
                    margin_required = position_value / leverage
                    
                    # 資金檢查
                    if margin_required > capital * 0.9:
                        continue
                    
                    # 計算止盈止損
                    tp, sl = self._calculate_tp_sl(signal, current_price, atr)
                    
                    # 開倉
                    entry_cost = position_value * (self.commission + self.slippage)
                    capital -= (margin_required + entry_cost)
                    
                    self.current_position = {
                        'side': 'long' if signal == 1 else 'short',
                        'entry_price': current_price * (1 + self.slippage * signal),
                        'entry_time': df.iloc[i]['timestamp'] if 'timestamp' in df.columns else i,
                        'entry_bar': i,
                        'take_profit': tp,
                        'stop_loss': sl,
                        'leverage': leverage,
                        'margin': margin_required,
                        'kelly_pct': kelly_pct,
                        'predicted_win_rate': win_rate,
                        'predicted_payoff': payoff,
                        'confidence': confidence
                    }
            
            # 記錄權益
            equity = capital
            if self.current_position is not None:
                unrealized_pnl = self._calculate_pnl(
                    self.current_position['entry_price'],
                    current_price,
                    position_size,
                    self.current_position['side'],
                    self.current_position['leverage']
                )
                equity += unrealized_pnl + self.current_position['margin']
            
            self.equity_curve.append({
                'timestamp': df.iloc[i]['timestamp'] if 'timestamp' in df.columns else i,
                'equity': equity
            })
        
        # 計算績效指標
        metrics = self._calculate_metrics()
        
        print(f"\n[V4回測完成] 總交易: {len(self.trades)} 筆")
        
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'metrics': metrics,
            'kelly_stats': kelly_stats
        }
    
    def _calculate_kelly_position(self, win_rate: float, payoff: float, 
                                 confidence: float, capital: float) -> Tuple[float, int, float]:
        """計算Kelly倉位"""
        # Kelly公式
        p = win_rate
        q = 1 - p
        b = payoff
        
        if b <= 0:
            return 0, 1, 0
        
        kelly_pct = (p * b - q) / b
        
        # 分數Kelly
        adjusted_kelly = kelly_pct * self.kelly_fraction
        
        # 信心度調整
        confidence_adjusted = adjusted_kelly * confidence
        
        # 應用限制
        final_position = min(confidence_adjusted, self.max_position)
        
        # 計算槓桿
        if kelly_pct > 0.4 and confidence > 0.7:
            leverage = min(3, self.max_leverage)
        elif kelly_pct > 0.3 and confidence > 0.6:
            leverage = min(2, self.max_leverage)
        else:
            leverage = 1
        
        position_value = capital * final_position * leverage
        
        return kelly_pct, leverage, position_value
    
    def _calculate_tp_sl(self, signal: int, price: float, atr: float) -> Tuple[float, float]:
        """計算止盈止損"""
        if signal == 1:  # 做多
            take_profit = price + atr * self.atr_tp_multiplier
            stop_loss = price - atr * self.atr_sl_multiplier
        else:  # 做空
            take_profit = price - atr * self.atr_tp_multiplier
            stop_loss = price + atr * self.atr_sl_multiplier
        
        return take_profit, stop_loss
    
    def _check_exit(self, pos: Dict, high: float, low: float, 
                   close: float, current_bar: int) -> Tuple[bool, str, float]:
        """檢查出場"""
        if pos['side'] == 'long':
            if high >= pos['take_profit']:
                return True, 'take_profit', pos['take_profit']
            if low <= pos['stop_loss']:
                return True, 'stop_loss', pos['stop_loss']
        else:
            if low <= pos['take_profit']:
                return True, 'take_profit', pos['take_profit']
            if high >= pos['stop_loss']:
                return True, 'stop_loss', pos['stop_loss']
        
        # 時間止損
        if current_bar - pos['entry_bar'] >= 16:  # 4小時
            return True, 'time_stop', close
        
        return False, None, None
    
    def _calculate_pnl(self, entry_price: float, exit_price: float, 
                      size: float, side: str, leverage: int) -> float:
        """計算盈虧"""
        if side == 'long':
            gross_pnl = (exit_price - entry_price) * size * leverage
        else:
            gross_pnl = (entry_price - exit_price) * size * leverage
        
        trade_value = entry_price * size * leverage
        cost = trade_value * (self.commission + self.slippage) * 2
        
        return gross_pnl - cost
    
    def _update_kelly_stats(self, kelly_stats: Dict):
        """更新Kelly統計"""
        history = kelly_stats['trade_history']
        if len(history) >= 10:
            wins = [t for t in history if t['is_win']]
            losses = [t for t in history if not t['is_win']]
            
            kelly_stats['recent_win_rate'] = len(wins) / len(history)
            
            if len(wins) > 0 and len(losses) > 0:
                avg_win = np.mean([t['return_pct'] for t in wins])
                avg_loss = np.mean([t['return_pct'] for t in losses])
                kelly_stats['recent_payoff'] = avg_win / avg_loss if avg_loss > 0 else 1.5
    
    def _calculate_metrics(self) -> Dict:
        """計算績效指標"""
        if len(self.trades) == 0:
            return {'error': '無交易記錄'}
        
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = winning_trades / total_trades
        
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if (total_trades - winning_trades) > 0 else 0
        
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        total_return = (equity_df['equity'].iloc[-1] - self.initial_capital) / self.initial_capital
        
        equity_returns = equity_df['equity'].pct_change().dropna()
        sharpe_ratio = equity_returns.mean() / equity_returns.std() * np.sqrt(252) if len(equity_returns) > 0 else 0
        
        equity_series = equity_df['equity']
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Kelly特定指標
        avg_kelly = trades_df['kelly_pct'].mean()
        avg_leverage = trades_df['leverage'].mean()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_kelly': avg_kelly,
            'avg_leverage': avg_leverage
        }
