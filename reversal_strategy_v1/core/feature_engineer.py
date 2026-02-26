"""
Feature Engineering for ML Model
為機器學習模型生成特徵
"""
import pandas as pd
import numpy as np
from typing import List

class FeatureEngineer:
    def __init__(self, config: dict):
        self.lookback_periods = config.get('lookback_periods', [5, 10, 20, 30])
        self.use_price_features = config.get('use_price_features', True)
        self.use_volume_features = config.get('use_volume_features', True)
        self.use_microstructure = config.get('use_microstructure', True)
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成所有特徵"""
        df = df.copy()
        
        if self.use_price_features:
            df = self._create_price_features(df)
        
        if self.use_volume_features:
            df = self._create_volume_features(df)
        
        if self.use_microstructure:
            df = self._create_microstructure_features(df)
        
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """價格相關特徵"""
        for period in self.lookback_periods:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'high_{period}'] = df['high'].rolling(period).max()
            df[f'low_{period}'] = df['low'].rolling(period).min()
            df[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
            
            df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
            
            ema = df['close'].ewm(span=period).mean()
            df[f'ema_{period}'] = ema
            df[f'distance_from_ema_{period}'] = (df['close'] - ema) / ema
        
        df['atr_14'] = self._calculate_atr(df, 14)
        
        return df
    
    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交量相關特徵"""
        for period in self.lookback_periods:
            df[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_std_{period}'] = df['volume'].rolling(period).std()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_ma_{period}']
            
            df[f'obv_{period}'] = self._calculate_obv(df, period)
        
        return df
    
    def _create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """市場微觀結構特徵"""
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['range_size'] = df['high'] - df['low']
        
        df['body_ratio'] = df['body_size'] / df['range_size'].replace(0, np.nan)
        df['upper_wick_ratio'] = df['upper_wick'] / df['range_size'].replace(0, np.nan)
        df['lower_wick_ratio'] = df['lower_wick'] / df['range_size'].replace(0, np.nan)
        
        return df
    
    def _calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """計算RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """計算ATR"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    
    def _calculate_obv(self, df: pd.DataFrame, period: int) -> pd.Series:
        """計算OBV"""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv.rolling(period).mean()
    
    def create_labels(self, df: pd.DataFrame, forward_window: int = 12, 
                     profit_threshold: float = 0.01, stop_loss: float = 0.005) -> pd.DataFrame:
        """生成訓練標籤"""
        df = df.copy()
        df['label'] = 0
        
        for i in range(len(df) - forward_window):
            entry_price = df.iloc[i]['close']
            future_prices = df.iloc[i+1:i+forward_window+1]
            
            if df.iloc[i]['signal_long'] == 1:
                max_price = future_prices['high'].max()
                min_price = future_prices['low'].min()
                
                if (max_price - entry_price) / entry_price >= profit_threshold:
                    df.loc[df.index[i], 'label'] = 1
                elif (entry_price - min_price) / entry_price >= stop_loss:
                    df.loc[df.index[i], 'label'] = 0
                else:
                    df.loc[df.index[i], 'label'] = 0
            
            elif df.iloc[i]['signal_short'] == 1:
                max_price = future_prices['high'].max()
                min_price = future_prices['low'].min()
                
                if (entry_price - min_price) / entry_price >= profit_threshold:
                    df.loc[df.index[i], 'label'] = -1
                elif (max_price - entry_price) / entry_price >= stop_loss:
                    df.loc[df.index[i], 'label'] = 0
                else:
                    df.loc[df.index[i], 'label'] = 0
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """獲取所有特徵名稱"""
        feature_names = []
        
        if self.use_price_features:
            for period in self.lookback_periods:
                feature_names.extend([
                    f'return_{period}',
                    f'high_{period}',
                    f'low_{period}',
                    f'volatility_{period}',
                    f'rsi_{period}',
                    f'ema_{period}',
                    f'distance_from_ema_{period}'
                ])
            feature_names.append('atr_14')
        
        if self.use_volume_features:
            for period in self.lookback_periods:
                feature_names.extend([
                    f'volume_ma_{period}',
                    f'volume_std_{period}',
                    f'volume_ratio_{period}',
                    f'obv_{period}'
                ])
        
        if self.use_microstructure:
            feature_names.extend([
                'body_size', 'upper_wick', 'lower_wick', 'range_size',
                'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio'
            ])
        
        feature_names.extend([
            'ofi_ratio', 'ofi_delta', 'ofi_acceleration',
            'liquidity_strength', 'signal_strength_long', 'signal_strength_short'
        ])
        
        return feature_names
