"""
Market Regime Detector
市場狀態檢測器 - 識別趨勢/震盪/突破狀態
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.cluster import KMeans

class MarketRegimeDetector:
    """
    使用無監督學習識別市場狀態
    0: 低波動震盪
    1: 高波動震盪
    2: 上漲趨勢
    3: 下跌趨勢
    """
    
    def __init__(self, config: Dict):
        self.window = config.get('window', 50)
        self.n_regimes = config.get('n_regimes', 4)
        self.kmeans = None
        
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """檢測市場狀態"""
        df = df.copy()
        
        # 計算狀態特徵
        df = self._calculate_regime_features(df)
        
        # 使用K-Means分群
        feature_cols = ['trend_strength', 'volatility_level', 'volume_intensity']
        features = df[feature_cols].dropna().values
        
        if self.kmeans is None:
            self.kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
            self.kmeans.fit(features)
        
        df['market_regime'] = np.nan
        df.loc[df[feature_cols].notna().all(axis=1), 'market_regime'] = self.kmeans.predict(features)
        
        # 狀態標籤
        df['regime_label'] = df['market_regime'].map({
            0: 'low_vol_range',
            1: 'high_vol_range', 
            2: 'uptrend',
            3: 'downtrend'
        })
        
        return df
    
    def _calculate_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算狀態特徵"""
        # 趨勢強度
        ema_short = df['close'].ewm(span=20).mean()
        ema_long = df['close'].ewm(span=50).mean()
        df['trend_strength'] = (ema_short - ema_long) / ema_long
        
        # 波動率等級
        df['volatility_level'] = df['close'].pct_change().rolling(self.window).std()
        
        # 成交量強度
        df['volume_intensity'] = df['volume'] / df['volume'].rolling(self.window).mean()
        
        return df
