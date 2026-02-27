"""
Market State Classifier
市場狀態分類器
"""
import pandas as pd
import numpy as np
from typing import Dict

class MarketClassifier:
    """分類市場狀態: 越勢/震盪/反轉"""
    def __init__(self, config: Dict):
        self.config = config
        self.lookback = config.get('lookback', 50)
    
    def classify_market(self, df: pd.DataFrame) -> pd.DataFrame:
        """分類市場狀態"""
        df = df.copy()
        
        # 越勢強度 (ADX)
        df['trend_strength'] = df.get('adx', 0)
        
        # 價格跟移動平均的關係
        ma20 = df['close'].rolling(20).mean()
        ma50 = df['close'].rolling(50).mean()
        
        # 分類邏輯
        df['market_state'] = 'ranging'  # 默認震盪
        
        # 上升越勢: 價格 > MA20 > MA50, ADX > 25
        df.loc[(df['close'] > ma20) & (ma20 > ma50) & 
               (df['trend_strength'] > 25), 'market_state'] = 'uptrend'
        
        # 下降越勢: 價格 < MA20 < MA50, ADX > 25  
        df.loc[(df['close'] < ma20) & (ma20 < ma50) & 
               (df['trend_strength'] > 25), 'market_state'] = 'downtrend'
        
        # 反轉警告: RSI極端 + 價格偏離
        if 'rsi_14' in df.columns and 'bb_position' in df.columns:
            # 看漲反轉
            df.loc[(df['rsi_14'] > 70) & (df['bb_position'] > 0.9), 
                   'market_state'] = 'reversal_down'
            
            # 看跌反轉
            df.loc[(df['rsi_14'] < 30) & (df['bb_position'] < 0.1),
                   'market_state'] = 'reversal_up'
        
        return df
    
    def get_optimal_strategy(self, market_state: str) -> Dict:
        """根據市場狀態推薦策略"""
        strategies = {
            'uptrend': {
                'prefer': 'long',
                'stop_loss_pct': 0.003,  # 0.3%
                'take_profit_pct': 0.005,  # 0.5%
                'confidence_threshold': 0.65
            },
            'downtrend': {
                'prefer': 'short',
                'stop_loss_pct': 0.003,
                'take_profit_pct': 0.005,
                'confidence_threshold': 0.65
            },
            'ranging': {
                'prefer': 'both',
                'stop_loss_pct': 0.002,  # 震盪市更緊止損
                'take_profit_pct': 0.004,  # 更小目標
                'confidence_threshold': 0.75  # 更高閾值
            },
            'reversal_up': {
                'prefer': 'long',
                'stop_loss_pct': 0.004,
                'take_profit_pct': 0.008,  # 反轉空間大
                'confidence_threshold': 0.7
            },
            'reversal_down': {
                'prefer': 'short',
                'stop_loss_pct': 0.004,
                'take_profit_pct': 0.008,
                'confidence_threshold': 0.7
            }
        }
        
        return strategies.get(market_state, strategies['ranging'])
