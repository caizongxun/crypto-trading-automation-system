import pandas as pd
import numpy as np
from enum import Enum
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger('execution_agent', 'logs/execution_agent.log')

class AgentState(Enum):
    """智能體狀態"""
    IDLE = "idle"              # 空手待機
    HUNTING = "hunting"        # 限價狙擊中
    POSITION = "position"      # 持有部位

class OrderType(Enum):
    """訂單類型"""
    MAKER = "maker"    # 限價單
    TAKER = "taker"    # 市價單

class ExecutionAgent:
    """
    執行智能體 - 機構級別的微觀流動性狙擊
    
    核心功能:
    1. FSM 狀態機: IDLE -> HUNTING -> POSITION -> IDLE
    2. Maker 限價進場: 減少手續費
    3. 動態出場: 硬停損/硬停利/動能衰竭平倉
    4. 非對稱手續費: 停損用 Taker，停利用 Maker
    5. 幽靈部位防護: 新訊號自動取消舊單
    """
    
    def __init__(self, capital: float = 10000,
                 stop_loss_pct: float = 0.01,
                 take_profit_pct: float = 0.02,
                 maker_fee: float = 0.0001,
                 taker_fee: float = 0.0004,
                 slippage: float = 0.0002,
                 probability_threshold: float = 0.65,
                 limit_order_offset_pct: float = 0.0005,
                 order_ttl_minutes: int = 10):
        """
        Args:
            capital: 初始資金
            stop_loss_pct: 硬停損百分比 (0.01 = 1%)
            take_profit_pct: 硬停利百分比 (0.02 = 2%)
            maker_fee: Maker 手續費 (0.0001 = 0.01%)
            taker_fee: Taker 手續費 (0.0004 = 0.04%)
            slippage: 滑價 (0.0002 = 0.02%)
            probability_threshold: 進場機率閉值
            limit_order_offset_pct: 限價單偏移百分比
            order_ttl_minutes: 訂單過期時間 (分鐘)
        """
        logger.info("Initializing ExecutionAgent")
        
        # 資金設定
        self.initial_capital = capital
        self.capital = capital
        
        # 風險參數
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # 成本參數
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage
        
        # 進場參數
        self.probability_threshold = probability_threshold
        self.limit_order_offset_pct = limit_order_offset_pct
        self.order_ttl_minutes = order_ttl_minutes
        
        # 狀態變數
        self.state = AgentState.IDLE
        self.pending_order = None
        self.position = None
        self.trades = []
        
        logger.info(f"Agent initialized:")
        logger.info(f"  Capital: ${capital:,.2f}")
        logger.info(f"  Stop Loss: {stop_loss_pct*100:.2f}%")
        logger.info(f"  Take Profit: {take_profit_pct*100:.2f}%")
        logger.info(f"  Maker Fee: {maker_fee*100:.4f}%")
        logger.info(f"  Taker Fee: {taker_fee*100:.4f}% + Slippage {slippage*100:.3f}%")
        logger.info(f"  Probability Threshold: {probability_threshold:.2f}")
    
    def process_bar(self, timestamp: pd.Timestamp, bar_1m: dict, 
                   probability: float = None) -> dict:
        """
        處理一根 1m K 線
        
        Args:
            timestamp: 當前時間
            bar_1m: 1m K 線資料 {'open', 'high', 'low', 'close', 'volume'}
            probability: 15m 大腦機率 (僅在 xx:00/15/30/45 時提供)
        
        Returns:
            action: 當前動作記錄
        """
        action = {
            'timestamp': timestamp,
            'state': self.state.value,
            'action': 'none',
            'price': bar_1m['close'],
            'capital': self.capital
        }
        
        if self.state == AgentState.IDLE:
            action.update(self._handle_idle(timestamp, bar_1m, probability))
        
        elif self.state == AgentState.HUNTING:
            action.update(self._handle_hunting(timestamp, bar_1m))
        
        elif self.state == AgentState.POSITION:
            action.update(self._handle_position(timestamp, bar_1m))
        
        return action
    
    def _handle_idle(self, timestamp: pd.Timestamp, bar_1m: dict, 
                    probability: float) -> dict:
        """處理 IDLE 狀態"""
        # 只有在 15m 整點時才有機率
        if probability is None:
            return {'action': 'waiting'}
        
        # 檢查機率是否達到閉值
        if probability < self.probability_threshold:
            return {
                'action': 'signal_rejected',
                'probability': probability,
                'threshold': self.probability_threshold
            }
        
        # 進入 HUNTING 狀態，掉最佳接刀價位
        limit_price = self._calculate_limit_price(bar_1m)
        
        self.pending_order = {
            'type': OrderType.MAKER,
            'price': limit_price,
            'timestamp': timestamp,
            'expiry': timestamp + pd.Timedelta(minutes=self.order_ttl_minutes),
            'probability': probability
        }
        
        self.state = AgentState.HUNTING
        
        logger.info(f"[{timestamp}] HUNTING: Limit order @ ${limit_price:.2f} (prob={probability:.3f})")
        
        return {
            'action': 'limit_order_placed',
            'limit_price': limit_price,
            'probability': probability,
            'expiry': self.pending_order['expiry']
        }
    
    def _handle_hunting(self, timestamp: pd.Timestamp, bar_1m: dict) -> dict:
        """處理 HUNTING 狀態"""
        # 檢查訂單是否過期
        if timestamp >= self.pending_order['expiry']:
            logger.info(f"[{timestamp}] Order expired, returning to IDLE")
            self.pending_order = None
            self.state = AgentState.IDLE
            return {'action': 'order_expired'}
        
        # 悲觀成交規則: 最低價必須嚴格小於 (<) 掉單價
        if bar_1m['low'] < self.pending_order['price']:
            # 訂單成交，進入 POSITION 狀態
            entry_price = self.pending_order['price']
            entry_fee = entry_price * self.maker_fee
            
            position_size = self.capital / (entry_price + entry_fee)
            position_cost = position_size * entry_price
            total_cost = position_cost + (position_size * entry_fee)
            
            self.position = {
                'entry_timestamp': timestamp,
                'entry_price': entry_price,
                'entry_fee': entry_fee,
                'size': position_size,
                'cost': total_cost,
                'stop_loss': entry_price * (1 - self.stop_loss_pct),
                'take_profit': entry_price * (1 + self.take_profit_pct),
                'probability': self.pending_order['probability']
            }
            
            self.capital -= total_cost
            self.state = AgentState.POSITION
            self.pending_order = None
            
            logger.info(f"[{timestamp}] POSITION: Entry @ ${entry_price:.2f}, Size={position_size:.4f}")
            logger.info(f"  SL: ${self.position['stop_loss']:.2f}, TP: ${self.position['take_profit']:.2f}")
            
            return {
                'action': 'order_filled',
                'entry_price': entry_price,
                'size': position_size,
                'fee': entry_fee * position_size,
                'stop_loss': self.position['stop_loss'],
                'take_profit': self.position['take_profit']
            }
        
        return {'action': 'hunting'}
    
    def _handle_position(self, timestamp: pd.Timestamp, bar_1m: dict) -> dict:
        """處理 POSITION 狀態"""
        current_price = bar_1m['close']
        high_price = bar_1m['high']
        low_price = bar_1m['low']
        
        # 1. 檢查硬停損 (Taker + Slippage)
        if low_price <= self.position['stop_loss']:
            exit_reason = 'stop_loss'
            exit_price = self.position['stop_loss'] * (1 - self.slippage)
            exit_type = OrderType.TAKER
            
            return self._close_position(timestamp, exit_price, exit_reason, exit_type)
        
        # 2. 檢查硬停利 (Maker)
        if high_price >= self.position['take_profit']:
            exit_reason = 'take_profit'
            exit_price = self.position['take_profit']
            exit_type = OrderType.MAKER
            
            return self._close_position(timestamp, exit_price, exit_reason, exit_type)
        
        # 3. 檢查動能衰竭平倉 (Early Exit)
        early_exit_signal = self._check_early_exit(timestamp, bar_1m)
        if early_exit_signal:
            exit_reason = 'early_exit_momentum'
            exit_price = current_price * (1 - self.slippage)
            exit_type = OrderType.TAKER
            
            logger.info(f"[{timestamp}] Early exit triggered: {early_exit_signal['reason']}")
            
            return self._close_position(timestamp, exit_price, exit_reason, exit_type)
        
        # 持有中
        unrealized_pnl_pct = (current_price - self.position['entry_price']) / self.position['entry_price']
        
        return {
            'action': 'holding',
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'current_price': current_price
        }
    
    def _close_position(self, timestamp: pd.Timestamp, exit_price: float, 
                       exit_reason: str, exit_type: OrderType) -> dict:
        """平倉"""
        # 計算出場手續費
        if exit_type == OrderType.MAKER:
            exit_fee = exit_price * self.maker_fee
        else:
            exit_fee = exit_price * (self.taker_fee + self.slippage)
        
        # 計算收益
        gross_proceeds = self.position['size'] * exit_price
        total_fee = self.position['size'] * exit_fee
        net_proceeds = gross_proceeds - total_fee
        
        pnl = net_proceeds - self.position['cost']
        pnl_pct = pnl / self.position['cost']
        
        # 更新資金
        self.capital += net_proceeds
        
        # 記錄交易
        trade = {
            'entry_timestamp': self.position['entry_timestamp'],
            'exit_timestamp': timestamp,
            'entry_price': self.position['entry_price'],
            'exit_price': exit_price,
            'size': self.position['size'],
            'exit_reason': exit_reason,
            'exit_type': exit_type.value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'entry_fee': self.position['size'] * self.position['entry_fee'],
            'exit_fee': total_fee,
            'capital': self.capital,
            'probability': self.position['probability']
        }
        
        self.trades.append(trade)
        
        logger.info(f"[{timestamp}] EXIT: {exit_reason.upper()} @ ${exit_price:.2f}")
        logger.info(f"  PnL: ${pnl:.2f} ({pnl_pct*100:.2f}%), Capital: ${self.capital:.2f}")
        
        # 重置狀態
        self.position = None
        self.state = AgentState.IDLE
        
        return {
            'action': 'position_closed',
            'exit_reason': exit_reason,
            'exit_price': exit_price,
            'exit_type': exit_type.value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'capital': self.capital
        }
    
    def _calculate_limit_price(self, bar_1m: dict) -> float:
        """
        計算最佳接刀價位
        
        策略: 當前 1m 最低價再往下 0.05%
        """
        limit_price = bar_1m['low'] * (1 - self.limit_order_offset_pct)
        return limit_price
    
    def _check_early_exit(self, timestamp: pd.Timestamp, bar_1m: dict) -> dict:
        """
        檢查動能衰竭平倉訊號
        
        觸發條件:
        1. 價格上漲超過 1.2%
        2. 連續兩根 1m K 線爆量 + 收長上影線
        
        Returns:
            signal: {'triggered': bool, 'reason': str}
        """
        # TODO: 這裡需要存储最近幾根 K 線的資料
        # 簡化實現: 只檢查是否超過 1.5% 利潤
        current_price = bar_1m['close']
        unrealized_pnl_pct = (current_price - self.position['entry_price']) / self.position['entry_price']
        
        # 如果已獲利 > 1.5%，且當前 K 線有長上影線，提早出場
        if unrealized_pnl_pct > 0.015:
            upper_wick_pct = (bar_1m['high'] - bar_1m['close']) / bar_1m['close']
            
            if upper_wick_pct > 0.003:  # 上影線 > 0.3%
                return {
                    'triggered': True,
                    'reason': f'Profit={unrealized_pnl_pct*100:.2f}%, UpperWick={upper_wick_pct*100:.2f}%'
                }
        
        return None
    
    def get_statistics(self) -> dict:
        """獲取統計資訊"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'final_capital': self.capital
            }
        
        df_trades = pd.DataFrame(self.trades)
        
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        win_rate = winning_trades / total_trades
        
        total_pnl = df_trades['pnl'].sum()
        avg_pnl = df_trades['pnl'].mean()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'final_capital': self.capital,
            'return_pct': (self.capital - self.initial_capital) / self.initial_capital,
            'trades_df': df_trades
        }
    
    def cancel_all_pending_orders(self):
        """取消所有未成交訂單 (幽靈部位防護)"""
        if self.state == AgentState.HUNTING:
            logger.info("Cancelling pending order (ghost position prevention)")
            self.pending_order = None
            self.state = AgentState.IDLE