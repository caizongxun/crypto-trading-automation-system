"""
V3 Backtest Engine - ATR Dynamic Risk Management
ATR動態風險管理回測引擎

修正:
1. 添加槓桿支持
2. 最大倉位限制
3. 正確處理過濾後的信號
4. 改進交易邏輯
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime

class BacktestEngine:
    """
    回測引擎
    
    核心改進:
    1. ATR動態止盈止損
    2. 移動止損 (Trailing Stop)
    3. 時間止損
    4. 分批平倉
    5. 趨勢強度自適應
    6. 槓桿支持
    7. 最大倉位限制
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 資金設定
        self.initial_capital = config.get('initial_capital', 10000)
        self.commission = config.get('commission', 0.001)  # 0.1%
        self.slippage = config.get('slippage', 0.0005)    # 0.05%
        
        # 槓桿和倉位
        self.leverage = config.get('leverage', 1)  # 槓桿倍數
        self.max_position_pct = config.get('max_position_pct', 0.1)  # 最大倉位10%
        
        # ATR動態參數
        self.atr_tp_strong = config.get('atr_tp_strong', 2.5)  # 強趨勢止盈
        self.atr_sl_strong = config.get('atr_sl_strong', 1.0)  # 強趨勢止損
        self.atr_tp_weak = config.get('atr_tp_weak', 1.5)    # 弱趨勢止盈
        self.atr_sl_weak = config.get('atr_sl_weak', 0.8)    # 弱趨勢止損
        
        # 移動止損參數
        self.trailing_start = config.get('trailing_start', 1.0)  # 盈利>1 ATR啟動
        self.trailing_distance = config.get('trailing_distance', 0.5)  # 0.5 ATR距離
        
        # 時間止損
        self.max_hold_bars = config.get('max_hold_bars', 16)  # 15m: 4小時
        
        # 趨勢強度閾值
        self.strong_trend_threshold = config.get('strong_trend_threshold', 0.7)
        
        # 分批平倉參數
        self.partial_take_profit = config.get('partial_take_profit', True)
        self.partial_tp_atr = config.get('partial_tp_atr', 1.5)  # 1.5 ATR平倉50%
        self.partial_ratio = config.get('partial_ratio', 0.5)  # 平倉比例
        
        # 狀態追蹤
        self.trades = []
        self.equity_curve = []
        self.current_position = None
    
    def run(self, df: pd.DataFrame, predictions: np.ndarray, 
            confidences: np.ndarray = None) -> Dict:
        """
        執行回測
        
        Args:
            df: K線數據 (必須包含: close, high, low, atr_14, trend_strength)
            predictions: 預測信號 (-1, 0, 1) - 已經過濾
            confidences: 預測信心度 (0-1)
        
        Returns:
            回測結果
        """
        print(f"\n[回測引擎] 開始回測")
        print(f"數據筆數: {len(df)}")
        print(f"信號筆數: {(predictions != 0).sum()}")
        print(f"槓桿: {self.leverage}x")
        print(f"最大倉位: {self.max_position_pct*100:.0f}%")
        
        self.trades = []
        self.equity_curve = []
        self.current_position = None
        
        capital = self.initial_capital
        position_size = 0
        partial_closed = False  # 追蹤是否已經分批平倉
        
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
                    pos, current_high, current_low, current_price, atr, i
                )
                
                # 分批平倉檢查
                if not partial_closed and self.partial_take_profit:
                    should_partial, partial_price = self._check_partial_exit(
                        pos, current_high, current_low, atr
                    )
                    
                    if should_partial:
                        # 平倉50%倉位
                        partial_size = position_size * self.partial_ratio
                        pnl = self._calculate_pnl(pos['entry_price'], partial_price, 
                                                 partial_size, pos['side'])
                        capital += pnl
                        position_size -= partial_size
                        partial_closed = True
                        
                        # 調整止損到成本價
                        if pos['side'] == 'long':
                            pos['stop_loss'] = max(pos['stop_loss'], pos['entry_price'])
                        else:
                            pos['stop_loss'] = min(pos['stop_loss'], pos['entry_price'])
                
                # 移動止損更新
                if not exit_signal:
                    self._update_trailing_stop(pos, current_price, atr)
                
                # 平倉
                if exit_signal:
                    pnl = self._calculate_pnl(pos['entry_price'], exit_price, 
                                             position_size, pos['side'])
                    capital += pnl
                    
                    trade_record = {
                        'entry_time': pos['entry_time'],
                        'exit_time': df.iloc[i]['timestamp'] if 'timestamp' in df.columns else i,
                        'side': pos['side'],
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'size': pos['original_size'],
                        'pnl': pnl,
                        'pnl_pct': pnl / (pos['entry_price'] * pos['original_size']),
                        'exit_reason': exit_reason,
                        'bars_held': i - pos['entry_bar'],
                        'partial_closed': partial_closed
                    }
                    self.trades.append(trade_record)
                    
                    self.current_position = None
                    position_size = 0
                    partial_closed = False
            
            # 開倉信號 - 關鍵修正: 只在有信號且無持倉時才開倉
            if self.current_position is None and i < len(predictions):
                signal = predictions[i]
                
                if signal != 0:  # 有交易信號
                    # 計算倉位大小 (使用槓桿)
                    position_value = capital * self.max_position_pct * self.leverage
                    position_size = position_value / current_price
                    
                    # 實際佔用保證金
                    margin_required = position_value / self.leverage
                    
                    # 檢查資金是否足夠
                    if margin_required > capital * 0.9:  # 保留10%緩衝
                        continue  # 跳過此信號
                    
                    # 計算止盈止損
                    tp, sl = self._calculate_tp_sl(df, i, signal, current_price, atr)
                    
                    # 開倉 (扣除保證金和手續費)
                    entry_cost = position_value * (self.commission + self.slippage)
                    capital -= (margin_required + entry_cost)
                    
                    self.current_position = {
                        'side': 'long' if signal == 1 else 'short',
                        'entry_price': current_price * (1 + self.slippage * signal),
                        'entry_time': df.iloc[i]['timestamp'] if 'timestamp' in df.columns else i,
                        'entry_bar': i,
                        'take_profit': tp,
                        'stop_loss': sl,
                        'original_size': position_size,
                        'margin': margin_required,
                        'atr': atr,
                        'trailing_active': False
                    }
            
            # 記錄權益
            equity = capital
            if self.current_position is not None:
                unrealized_pnl = self._calculate_pnl(
                    self.current_position['entry_price'],
                    current_price,
                    position_size,
                    self.current_position['side']
                )
                equity += unrealized_pnl + self.current_position['margin']
            
            self.equity_curve.append({
                'timestamp': df.iloc[i]['timestamp'] if 'timestamp' in df.columns else i,
                'equity': equity
            })
        
        # 計算績效指標
        metrics = self._calculate_metrics()
        
        print(f"\n[回測完成] 總交易: {len(self.trades)} 筆")
        
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'metrics': metrics
        }
    
    def _calculate_tp_sl(self, df: pd.DataFrame, idx: int, signal: int, 
                        entry_price: float, atr: float) -> Tuple[float, float]:
        """
        計算動態止盈止損
        """
        # 獲取趨勢強度
        trend_strength = df.iloc[idx].get('trend_strength', 0.5)
        
        # 根據趨勢強度調整
        if abs(trend_strength) > self.strong_trend_threshold:
            tp_multiplier = self.atr_tp_strong
            sl_multiplier = self.atr_sl_strong
        else:
            tp_multiplier = self.atr_tp_weak
            sl_multiplier = self.atr_sl_weak
        
        if signal == 1:  # 做多
            take_profit = entry_price + atr * tp_multiplier
            stop_loss = entry_price - atr * sl_multiplier
        else:  # 做空
            take_profit = entry_price - atr * tp_multiplier
            stop_loss = entry_price + atr * sl_multiplier
        
        return take_profit, stop_loss
    
    def _check_exit(self, pos: Dict, high: float, low: float, close: float,
                   atr: float, current_bar: int) -> Tuple[bool, str, float]:
        """
        檢查是否觸發出場
        
        Returns:
            (should_exit, exit_reason, exit_price)
        """
        # 1. 止盈檢查
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
        
        # 2. 時間止損
        bars_held = current_bar - pos['entry_bar']
        if bars_held >= self.max_hold_bars:
            return True, 'time_stop', close
        
        return False, None, None
    
    def _check_partial_exit(self, pos: Dict, high: float, low: float, 
                           atr: float) -> Tuple[bool, float]:
        """
        檢查是否觸發分批平倉
        """
        partial_target = atr * self.partial_tp_atr
        
        if pos['side'] == 'long':
            partial_price = pos['entry_price'] + partial_target
            if high >= partial_price:
                return True, partial_price
        else:
            partial_price = pos['entry_price'] - partial_target
            if low <= partial_price:
                return True, partial_price
        
        return False, None
    
    def _update_trailing_stop(self, pos: Dict, current_price: float, atr: float):
        """
        更新移動止損
        """
        if pos['side'] == 'long':
            profit = current_price - pos['entry_price']
            
            # 盈利超過閾值,啟動trailing stop
            if profit > atr * self.trailing_start:
                new_sl = current_price - atr * self.trailing_distance
                pos['stop_loss'] = max(pos['stop_loss'], new_sl)
                pos['trailing_active'] = True
        else:
            profit = pos['entry_price'] - current_price
            
            if profit > atr * self.trailing_start:
                new_sl = current_price + atr * self.trailing_distance
                pos['stop_loss'] = min(pos['stop_loss'], new_sl)
                pos['trailing_active'] = True
    
    def _calculate_pnl(self, entry_price: float, exit_price: float, 
                      size: float, side: str) -> float:
        """
        計算盈虧 (考慮槓桿)
        """
        if side == 'long':
            gross_pnl = (exit_price - entry_price) * size
        else:
            gross_pnl = (entry_price - exit_price) * size
        
        # 扣除成本
        trade_value = entry_price * size
        cost = trade_value * (self.commission + self.slippage) * 2  # 進+出
        
        return gross_pnl - cost
    
    def _calculate_metrics(self) -> Dict:
        """
        計算績效指標
        """
        if len(self.trades) == 0:
            return {'error': '無交易記錄'}
        
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        # 基礎指標
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 盈虧統計
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
        
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # 報酬指標
        total_return = (equity_df['equity'].iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Sharpe比率
        equity_returns = equity_df['equity'].pct_change().dropna()
        sharpe_ratio = equity_returns.mean() / equity_returns.std() * np.sqrt(252) if len(equity_returns) > 0 and equity_returns.std() > 0 else 0
        
        # 最大回撤
        equity_series = equity_df['equity']
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_bars_held': trades_df['bars_held'].mean()
        }
