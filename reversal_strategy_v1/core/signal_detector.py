"""
Order Flow Imbalance & Liquidity Zone Signal Detector
檢測訂單流不平衡和流動性區域的反轉信號
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List

class SignalDetector:
    def __init__(self, config: dict):
        self.lookback = config.get('lookback', 20)
        self.imbalance_threshold = config.get('imbalance_threshold', 0.6)
        self.liquidity_strength = config.get('liquidity_strength', 1.5)
        self.microstructure_window = config.get('microstructure_window', 10)
        
    def detect_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """主要信號檢測函數"""
        df = df.copy()
        
        df = self._calculate_order_flow_imbalance(df)
        df = self._detect_liquidity_zones(df)
        df = self._analyze_microstructure(df)
        df = self._detect_momentum_exhaustion(df)
        df = self._generate_reversal_signals(df)
        
        return df
    
    def _calculate_order_flow_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算訂單流不平衡指標"""
        df['buy_volume'] = df['volume'].where(df['close'] > df['open'], 0)
        df['sell_volume'] = df['volume'].where(df['close'] <= df['open'], 0)
        
        window = self.lookback
        df['buy_volume_sum'] = df['buy_volume'].rolling(window).sum()
        df['sell_volume_sum'] = df['sell_volume'].rolling(window).sum()
        
        total_volume = df['buy_volume_sum'] + df['sell_volume_sum']
        df['ofi_ratio'] = (df['buy_volume_sum'] - df['sell_volume_sum']) / total_volume.replace(0, np.nan)
        
        df['ofi_extreme_long'] = (df['ofi_ratio'] < -self.imbalance_threshold).astype(int)
        df['ofi_extreme_short'] = (df['ofi_ratio'] > self.imbalance_threshold).astype(int)
        
        df['ofi_delta'] = df['ofi_ratio'].diff()
        df['ofi_acceleration'] = df['ofi_delta'].diff()
        
        return df
    
    def _detect_liquidity_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """檢測流動性區域和流動性掃蕩"""
        window = self.lookback
        df['local_high'] = df['high'].rolling(window, center=True).max()
        df['local_low'] = df['low'].rolling(window, center=True).min()
        
        df['at_resistance'] = (df['high'] >= df['local_high'] * 0.998).astype(int)
        df['at_support'] = (df['low'] <= df['local_low'] * 1.002).astype(int)
        
        df['wick_ratio_upper'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low']).replace(0, np.nan)
        df['wick_ratio_lower'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
        
        df['liquidity_sweep_long'] = (
            (df['at_support'] == 1) & 
            (df['wick_ratio_lower'] > 0.6) &
            (df['close'] > df['open'])
        ).astype(int)
        
        df['liquidity_sweep_short'] = (
            (df['at_resistance'] == 1) & 
            (df['wick_ratio_upper'] > 0.6) &
            (df['close'] < df['open'])
        ).astype(int)
        
        avg_volume = df['volume'].rolling(window).mean()
        df['liquidity_strength'] = df['volume'] / avg_volume.replace(0, np.nan)
        
        return df
    
    def _analyze_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        """分析市場微觀結構"""
        df['body_size'] = abs(df['close'] - df['open'])
        df['range_size'] = df['high'] - df['low']
        df['body_ratio'] = df['body_size'] / df['range_size'].replace(0, np.nan)
        
        df['consecutive_bull'] = (df['close'] > df['open']).rolling(self.microstructure_window).sum()
        df['consecutive_bear'] = (df['close'] < df['open']).rolling(self.microstructure_window).sum()
        
        df['body_shrinking'] = df['body_size'].rolling(3).apply(
            lambda x: 1 if len(x) == 3 and x.iloc[-1] < x.iloc[-2] < x.iloc[-3] else 0, 
            raw=False
        )
        
        df['bullish_engulfing'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'] > df['open']) &
            (df['open'] <= df['close'].shift(1)) &
            (df['close'] >= df['open'].shift(1))
        ).astype(int)
        
        df['bearish_engulfing'] = (
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['close'] < df['open']) &
            (df['open'] >= df['close'].shift(1)) &
            (df['close'] <= df['open'].shift(1))
        ).astype(int)
        
        return df
    
    def _detect_momentum_exhaustion(self, df: pd.DataFrame) -> pd.DataFrame:
        """檢測價格動能衰竭"""
        df['price_change'] = df['close'].pct_change()
        df['momentum'] = df['price_change'].rolling(5).mean()
        df['momentum_change'] = df['momentum'].diff()
        
        df['price_higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['price_lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        
        df['momentum_exhaustion_long'] = (
            (df['price_lower_low'] == 1) &
            (df['momentum_change'] > 0) &
            (df['momentum'] < 0)
        ).astype(int)
        
        df['momentum_exhaustion_short'] = (
            (df['price_higher_high'] == 1) &
            (df['momentum_change'] < 0) &
            (df['momentum'] > 0)
        ).astype(int)
        
        return df
    
    def _generate_reversal_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """綜合各項指標生成最終反轉信號"""
        df['signal_long'] = (
            (df['ofi_extreme_long'] == 1) |
            (df['liquidity_sweep_long'] == 1) |
            (df['momentum_exhaustion_long'] == 1) |
            (df['bullish_engulfing'] == 1)
        ).astype(int)
        
        df['signal_short'] = (
            (df['ofi_extreme_short'] == 1) |
            (df['liquidity_sweep_short'] == 1) |
            (df['momentum_exhaustion_short'] == 1) |
            (df['bearish_engulfing'] == 1)
        ).astype(int)
        
        df['signal_strength_long'] = (
            df['ofi_extreme_long'] +
            df['liquidity_sweep_long'] +
            df['momentum_exhaustion_long'] +
            df['bullish_engulfing'] +
            (df['liquidity_strength'] > self.liquidity_strength).astype(int)
        )
        
        df['signal_strength_short'] = (
            df['ofi_extreme_short'] +
            df['liquidity_sweep_short'] +
            df['momentum_exhaustion_short'] +
            df['bearish_engulfing'] +
            (df['liquidity_strength'] > self.liquidity_strength).astype(int)
        )
        
        return df
    
    def get_current_signal(self, df: pd.DataFrame) -> Dict:
        """獲取當前最新信號"""
        if len(df) == 0:
            return {'signal': 0, 'strength': 0, 'type': 'NONE'}
        
        latest = df.iloc[-1]
        
        if latest['signal_long'] == 1:
            return {
                'signal': 1,
                'strength': latest['signal_strength_long'],
                'type': 'LONG',
                'ofi_ratio': latest['ofi_ratio'],
                'liquidity_strength': latest['liquidity_strength']
            }
        elif latest['signal_short'] == 1:
            return {
                'signal': -1,
                'strength': latest['signal_strength_short'],
                'type': 'SHORT',
                'ofi_ratio': latest['ofi_ratio'],
                'liquidity_strength': latest['liquidity_strength']
            }
        else:
            return {'signal': 0, 'strength': 0, 'type': 'NONE'}
