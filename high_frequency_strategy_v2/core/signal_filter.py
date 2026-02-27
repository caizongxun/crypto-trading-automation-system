"""
Multi-layer Signal Filter
多層信號過濾器
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple

class SignalFilter:
    """三層信號過濾系統"""
    def __init__(self, config: Dict):
        self.config = config
        
        # 第一層: 模型置信度閾值
        self.min_confidence_long = config.get('min_confidence_long', 0.7)
        self.min_confidence_short = config.get('min_confidence_short', 0.7)
        
        # 第二層: 市場狀態篩選
        self.allowed_regimes_long = config.get('allowed_regimes_long', [1, 2])  # 中高波動
        self.allowed_regimes_short = config.get('allowed_regimes_short', [1, 2])
        
        # 第三層: 技術指標確認
        self.use_technical_filter = config.get('use_technical_filter', True)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        
        # 動態閾值調整
        self.dynamic_threshold = config.get('dynamic_threshold', True)
    
    def filter_signals(self, df: pd.DataFrame, 
                      predictions: np.ndarray,
                      confidences: np.ndarray) -> pd.DataFrame:
        """應用三層過濾"""
        df = df.copy()
        df['raw_prediction'] = predictions
        df['raw_confidence'] = confidences
        
        # 初始化信號
        df['signal_long'] = 0
        df['signal_short'] = 0
        df['signal_confidence'] = 0.0
        
        for i in range(len(df)):
            pred = predictions[i]
            conf = confidences[i]
            
            # 第一層: 置信度過濾
            if pred == 1 and conf >= self.min_confidence_long:
                signal_type = 'long'
            elif pred == -1 and conf >= self.min_confidence_short:
                signal_type = 'short'
            else:
                continue
            
            # 第二層: 市場狀態過濾
            regime = df.iloc[i]['volatility_regime']
            if signal_type == 'long' and regime not in self.allowed_regimes_long:
                continue
            if signal_type == 'short' and regime not in self.allowed_regimes_short:
                continue
            
            # 第三層: 技術指標確認
            if self.use_technical_filter:
                if not self._check_technical_conditions(df.iloc[i], signal_type):
                    continue
            
            # 通過所有過濾
            if signal_type == 'long':
                df.loc[df.index[i], 'signal_long'] = 1
                df.loc[df.index[i], 'signal_confidence'] = conf
            else:
                df.loc[df.index[i], 'signal_short'] = 1
                df.loc[df.index[i], 'signal_confidence'] = conf
        
        return df
    
    def _check_technical_conditions(self, row: pd.Series, signal_type: str) -> bool:
        """檢查技術指標條件"""
        # 做多條件
        if signal_type == 'long':
            # RSI不能過高
            if 'rsi_14' in row and row['rsi_14'] > self.rsi_overbought:
                return False
            
            # MACD應該是多頭
            if 'macd_hist' in row and row['macd_hist'] < 0:
                return False
            
            # 價格應該在布林帶中下方
            if 'bb_position' in row and row['bb_position'] > 0.8:
                return False
        
        # 做空條件
        elif signal_type == 'short':
            # RSI不能過低
            if 'rsi_14' in row and row['rsi_14'] < self.rsi_oversold:
                return False
            
            # MACD應該是空頭
            if 'macd_hist' in row and row['macd_hist'] > 0:
                return False
            
            # 價格應該在布林帶中上方
            if 'bb_position' in row and row['bb_position'] < 0.2:
                return False
        
        return True
    
    def adjust_thresholds(self, recent_performance: Dict):
        """根據近期表現調整閾值"""
        if not self.dynamic_threshold:
            return
        
        win_rate = recent_performance.get('win_rate', 0.5)
        
        # 如果勝率過低,提高閾值
        if win_rate < 0.55:
            self.min_confidence_long = min(0.85, self.min_confidence_long + 0.05)
            self.min_confidence_short = min(0.85, self.min_confidence_short + 0.05)
        # 如果勝率過高,降低閾值增加交易
        elif win_rate > 0.65:
            self.min_confidence_long = max(0.6, self.min_confidence_long - 0.02)
            self.min_confidence_short = max(0.6, self.min_confidence_short - 0.02)
