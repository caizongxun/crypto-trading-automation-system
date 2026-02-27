"""
Dynamic Risk Manager
動態風險管理器
"""
import pandas as pd
import numpy as np
from typing import Dict

class RiskManager:
    """根據市場狀態動態調整風險參數"""
    def __init__(self, config: Dict):
        self.initial_capital = config.get('initial_capital', 10)
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.015)  # 1.5%
        self.max_leverage = config.get('max_leverage', 5)
        self.default_leverage = config.get('default_leverage', 3)
        
        # 動態止損止盈倍數
        self.base_sl_pct = config.get('base_sl_pct', 0.003)  # 0.3%
        self.base_tp_pct = config.get('base_tp_pct', 0.005)  # 0.5%
        
        # 跟蹤止盈
        self.trailing_stop = config.get('trailing_stop', True)
        self.trailing_start_pct = config.get('trailing_start_pct', 0.005)  # 0.5%啓動
        self.trailing_distance_pct = config.get('trailing_distance_pct', 0.003)  # 0.3%距離
        
        # 時間止損
        self.max_hold_hours = config.get('max_hold_hours', 8)  # 8小時
    
    def calculate_stop_loss_take_profit(self, entry_price: float,
                                       direction: str,
                                       market_state: str = 'ranging',
                                       volatility: float = None) -> Dict:
        """計算止損止盈"""
        # 根據市場狀態調整
        from .market_classifier import MarketClassifier
        classifier = MarketClassifier({})
        strategy_params = classifier.get_optimal_strategy(market_state)
        
        sl_pct = strategy_params['stop_loss_pct']
        tp_pct = strategy_params['take_profit_pct']
        
        # 根據波動率調整(如果提供)
        if volatility is not None:
            # 高波動時擴大止損空間
            if volatility > 0.02:  # 2%以上
                sl_pct *= 1.5
                tp_pct *= 1.5
        
        if direction == 'LONG':
            stop_loss = entry_price * (1 - sl_pct)
            take_profit = entry_price * (1 + tp_pct)
        else:  # SHORT
            stop_loss = entry_price * (1 + sl_pct)
            take_profit = entry_price * (1 - tp_pct)
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'sl_pct': sl_pct,
            'tp_pct': tp_pct
        }
    
    def calculate_position_size(self, capital: float, 
                               entry_price: float,
                               stop_loss: float,
                               leverage: int = None) -> Dict:
        """計算倉位大小"""
        if leverage is None:
            leverage = self.default_leverage
        
        leverage = min(leverage, self.max_leverage)
        
        # 風險金額
        risk_amount = capital * self.max_risk_per_trade
        
        # 價格風險
        price_risk = abs(entry_price - stop_loss) / entry_price
        
        if price_risk == 0:
            position_value = capital * 0.95 * leverage
        else:
            # 根據風險計算倉位
            position_value = risk_amount / price_risk
            position_value = min(position_value, capital * 0.95 * leverage)
        
        position_size = position_value / entry_price
        
        return {
            'position_size': position_size,
            'position_value': position_value,
            'leverage': leverage,
            'risk_amount': risk_amount
        }
    
    def update_trailing_stop(self, entry_price: float,
                            current_price: float,
                            direction: str,
                            current_stop: float) -> float:
        """更新跟蹤止損"""
        if not self.trailing_stop:
            return current_stop
        
        if direction == 'LONG':
            profit_pct = (current_price - entry_price) / entry_price
            
            # 達到啟動條件
            if profit_pct >= self.trailing_start_pct:
                new_stop = current_price * (1 - self.trailing_distance_pct)
                return max(new_stop, current_stop)
        
        else:  # SHORT
            profit_pct = (entry_price - current_price) / entry_price
            
            if profit_pct >= self.trailing_start_pct:
                new_stop = current_price * (1 + self.trailing_distance_pct)
                return min(new_stop, current_stop)
        
        return current_stop
    
    def check_time_stop(self, entry_time: pd.Timestamp, 
                       current_time: pd.Timestamp) -> bool:
        """檢查是否觸發時間止損"""
        hold_hours = (current_time - entry_time).total_seconds() / 3600
        return hold_hours >= self.max_hold_hours
