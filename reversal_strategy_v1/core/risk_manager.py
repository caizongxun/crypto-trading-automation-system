"""
Risk Manager
風險管理和倉位計算
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional

class RiskManager:
    def __init__(self, config: dict):
        self.initial_capital = config.get('initial_capital', 10)
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)
        self.max_leverage = config.get('max_leverage', 10)
        self.default_leverage = config.get('default_leverage', 3)
        self.atr_multiplier_sl = config.get('atr_multiplier_sl', 1.5)
        self.atr_multiplier_tp = config.get('atr_multiplier_tp', 3.0)
        
    def calculate_position_size(self, capital: float, entry_price: float, 
                               stop_loss: float, leverage: int = None) -> Dict:
        """計算倉位大小"""
        if leverage is None:
            leverage = self.default_leverage
        
        leverage = min(leverage, self.max_leverage)
        
        risk_amount = capital * self.max_risk_per_trade
        
        price_risk = abs(entry_price - stop_loss) / entry_price
        
        if price_risk == 0:
            position_value = capital * leverage
        else:
            position_value = risk_amount / price_risk
            position_value = min(position_value, capital * leverage)
        
        position_size = position_value / entry_price
        
        return {
            'position_size': position_size,
            'position_value': position_value,
            'leverage': leverage,
            'risk_amount': risk_amount
        }
    
    def calculate_stop_loss_take_profit(self, df: pd.DataFrame, 
                                       direction: str) -> Dict:
        """基於ATR計算止損和止盈"""
        if len(df) < 14:
            atr = (df['high'] - df['low']).mean()
        else:
            if 'atr_14' in df.columns:
                atr = df['atr_14'].iloc[-1]
            else:
                high_low = df['high'] - df['low']
                high_close = abs(df['high'] - df['close'].shift())
                low_close = abs(df['low'] - df['close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
        
        if pd.isna(atr) or atr == 0:
            atr = df['close'].iloc[-1] * 0.01
        
        current_price = df['close'].iloc[-1]
        
        if direction == 'LONG':
            stop_loss = current_price - (atr * self.atr_multiplier_sl)
            take_profit = current_price + (atr * self.atr_multiplier_tp)
        else:
            stop_loss = current_price + (atr * self.atr_multiplier_sl)
            take_profit = current_price - (atr * self.atr_multiplier_tp)
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr': atr
        }
    
    def check_risk_limits(self, current_positions: int, total_exposure: float, 
                         capital: float) -> bool:
        """檢查是否超過風險限制"""
        max_exposure = capital * self.max_leverage
        
        if total_exposure >= max_exposure:
            return False
        
        return True
