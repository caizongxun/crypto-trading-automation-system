"""
激進複利回測引擎
目標: 30天翻倉 (2x)

特性:
- 全倉交易 (100% 資金)
- 動態複利 (每次用當前全部資金)
- 高頻交易 (低 TP 1.2%)
- 緊止損 (0.6%)
- 多空雙向
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class AggressiveBacktester:
    """
    激進複利回測引擎
    """
    
    def __init__(
        self,
        initial_capital: float = 1000.0,
        position_size: float = 1.0,  # 100% 全倉
        tp_pct: float = 1.2,
        sl_pct: float = 0.6,
        fee_rate: float = 0.0004,  # 0.04% Binance fee
        max_trades_per_day: int = 20,
        enable_compound: bool = True
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size = position_size
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.fee_rate = fee_rate
        self.max_trades_per_day = max_trades_per_day
        self.enable_compound = enable_compound
        
        self.trades = []
        self.position = None
        self.daily_trades = {}
        
        logger.info(f"Aggressive Backtester initialized")
        logger.info(f"Initial capital: ${initial_capital}")
        logger.info(f"Position size: {position_size*100}%")
        logger.info(f"TP/SL: {tp_pct}%/{sl_pct}%")
        logger.info(f"Compound: {enable_compound}")
    
    def _get_trade_count_today(self, timestamp: pd.Timestamp) -> int:
        """獲取當天交易數"""
        date_key = timestamp.date()
        return self.daily_trades.get(date_key, 0)
    
    def _increment_trade_count(self, timestamp: pd.Timestamp):
        """增加當天交易數"""
        date_key = timestamp.date()
        self.daily_trades[date_key] = self.daily_trades.get(date_key, 0) + 1
    
    def open_position(
        self,
        side: str,
        price: float,
        timestamp: pd.Timestamp,
        confidence: float
    ) -> bool:
        """
        開倉
        """
        # 檢查是否已有持倉
        if self.position is not None:
            return False
        
        # 檢查當日交易次數
        if self._get_trade_count_today(timestamp) >= self.max_trades_per_day:
            return False
        
        # 檢查資金
        if self.capital <= 0:
            logger.warning("No capital left!")
            return False
        
        # 計算交易金額
        trade_amount = self.capital * self.position_size
        entry_fee = trade_amount * self.fee_rate
        
        # 計算 TP/SL 價格
        if side == 'LONG':
            tp_price = price * (1 + self.tp_pct / 100)
            sl_price = price * (1 - self.sl_pct / 100)
        else:  # SHORT
            tp_price = price * (1 - self.tp_pct / 100)
            sl_price = price * (1 + self.sl_pct / 100)
        
        self.position = {
            'side': side,
            'entry_price': price,
            'entry_time': timestamp,
            'entry_capital': self.capital,
            'trade_amount': trade_amount,
            'entry_fee': entry_fee,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'confidence': confidence
        }
        
        logger.debug(f"Opened {side} @ {price:.2f}, Capital: ${self.capital:.2f}")
        
        return True
    
    def close_position(
        self,
        exit_price: float,
        exit_time: pd.Timestamp,
        exit_reason: str
    ) -> Dict:
        """
        平倉
        """
        if self.position is None:
            return None
        
        pos = self.position
        side = pos['side']
        entry_price = pos['entry_price']
        trade_amount = pos['trade_amount']
        
        # 計算 PnL
        if side == 'LONG':
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:  # SHORT
            pnl_pct = (entry_price - exit_price) / entry_price * 100
        
        # 計算實際盈虧 (扣除手續費)
        exit_fee = trade_amount * self.fee_rate
        pnl_amount = trade_amount * (pnl_pct / 100) - pos['entry_fee'] - exit_fee
        
        # 更新資金 (複利)
        if self.enable_compound:
            self.capital += pnl_amount
        else:
            # 固定交易金額
            profit_on_initial = (self.initial_capital * self.position_size) * (pnl_pct / 100)
            self.capital = self.initial_capital + (profit_on_initial - pos['entry_fee'] - exit_fee)
        
        # 記錄交易
        trade = {
            'side': side,
            'entry_time': pos['entry_time'],
            'entry_price': entry_price,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'confidence': pos['confidence'],
            'pnl_pct': pnl_pct,
            'pnl_amount': pnl_amount,
            'entry_capital': pos['entry_capital'],
            'exit_capital': self.capital,
            'return_on_capital': (self.capital - pos['entry_capital']) / pos['entry_capital'] * 100,
            'fees': pos['entry_fee'] + exit_fee
        }
        
        self.trades.append(trade)
        self._increment_trade_count(exit_time)
        
        logger.debug(
            f"Closed {side} @ {exit_price:.2f}, "
            f"PnL: {pnl_pct:+.2f}%, "
            f"Capital: ${pos['entry_capital']:.2f} -> ${self.capital:.2f}"
        )
        
        self.position = None
        
        return trade
    
    def check_exit(
        self,
        current_high: float,
        current_low: float,
        current_time: pd.Timestamp
    ) -> bool:
        """
        檢查是否觸發 TP/SL
        """
        if self.position is None:
            return False
        
        pos = self.position
        
        if pos['side'] == 'LONG':
            # 先檢查 TP
            if current_high >= pos['tp_price']:
                self.close_position(pos['tp_price'], current_time, 'TP')
                return True
            # 再檢查 SL
            elif current_low <= pos['sl_price']:
                self.close_position(pos['sl_price'], current_time, 'SL')
                return True
        
        else:  # SHORT
            # 先檢查 TP
            if current_low <= pos['tp_price']:
                self.close_position(pos['tp_price'], current_time, 'TP')
                return True
            # 再檢查 SL
            elif current_high >= pos['sl_price']:
                self.close_position(pos['sl_price'], current_time, 'SL')
                return True
        
        return False
    
    def get_stats(self) -> Dict:
        """
        獲取統計數據
        """
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_return': 0,
                'final_capital': self.initial_capital,
                'max_drawdown': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'total_fees': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        total_trades = len(trades_df)
        wins = len(trades_df[trades_df['pnl_pct'] > 0])
        losses = len(trades_df[trades_df['pnl_pct'] < 0])
        win_rate = wins / total_trades * 100
        
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        # 計算最大回撤
        capital_curve = trades_df['exit_capital'].values
        running_max = np.maximum.accumulate(capital_curve)
        drawdown = (capital_curve - running_max) / running_max * 100
        max_drawdown = abs(drawdown.min())
        
        # Profit Factor
        total_profit = trades_df[trades_df['pnl_amount'] > 0]['pnl_amount'].sum()
        total_loss = abs(trades_df[trades_df['pnl_amount'] < 0]['pnl_amount'].sum())
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        # 平均盈虧
        avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if wins > 0 else 0
        avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() if losses > 0 else 0
        
        # 最大/最小
        largest_win = trades_df['pnl_pct'].max()
        largest_loss = trades_df['pnl_pct'].min()
        
        # 總手續費
        total_fees = trades_df['fees'].sum()
        
        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_return': total_return,
            'final_capital': self.capital,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'total_fees': total_fees,
            'avg_trades_per_day': total_trades / max(len(self.daily_trades), 1),
            'compound_enabled': self.enable_compound
        }
    
    def print_summary(self):
        """
        列印結果摘要
        """
        stats = self.get_stats()
        
        print("\n" + "="*80)
        print("激進複利回測結果")
        print("="*80)
        
        print(f"\n基本資訊:")
        print(f"  初始資金: ${self.initial_capital:,.2f}")
        print(f"  最終資金: ${stats['final_capital']:,.2f}")
        print(f"  總報酬: {stats['total_return']:+.2f}% ({stats['final_capital']/self.initial_capital:.2f}x)")
        
        print(f"\n交易統計:")
        print(f"  總交易數: {stats['total_trades']}")
        print(f"  勝場: {stats['wins']} | 負場: {stats['losses']}")
        print(f"  勝率: {stats['win_rate']:.2f}%")
        print(f"  平均每日: {stats['avg_trades_per_day']:.1f} 筆")
        
        print(f"\n盈虧分析:")
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")
        print(f"  平均盈利: {stats['avg_win']:+.2f}%")
        print(f"  平均虧損: {stats['avg_loss']:+.2f}%")
        print(f"  最大單筆盈利: {stats['largest_win']:+.2f}%")
        print(f"  最大單筆虧損: {stats['largest_loss']:+.2f}%")
        
        print(f"\n風險指標:")
        print(f"  最大回撤: {stats['max_drawdown']:.2f}%")
        print(f"  總手續費: ${stats['total_fees']:.2f}")
        
        print(f"\n設定:")
        print(f"  TP/SL: {self.tp_pct}%/{self.sl_pct}%")
        print(f"  複利: {'\u2705 開啟' if self.enable_compound else '\u274c 關閉'}")
        print(f"  倉位: {self.position_size*100}%")
        print(f"  日交易上限: {self.max_trades_per_day}")
        
        print("\n" + "="*80)
        
        # 評估是否達標
        if stats['total_return'] >= 100:
            print("\n🎉 恣喜! 達成 2x 目標!")
        elif stats['total_return'] >= 50:
            print("\n👍 不錯! 達成 1.5x!")
        elif stats['total_return'] >= 20:
            print("\n✅ 達成 1.2x!")
        elif stats['total_return'] > 0:
            print("\n📈 有盈利,但未達標")
        else:
            print("\n❌ 虧損,需調整策略")
        
        print("="*80 + "\n")
