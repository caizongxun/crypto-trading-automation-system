"""
Temporal Feature Engineer
時序特徵工程 - 多時間框架和時間序列特徵
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime

class TemporalFeatureEngineer:
    """
    提取時間序列特徵:
    1. 多時間框架特徵
    2. 時間周期性特徵
    3. 動量和趨勢
    4. 波動率特徵
    5. 技術指標
    """
    
    def __init__(self, config: Dict):
        self.lookback_periods = config.get('lookback_periods', [5, 10, 20, 30, 50])
        self.use_multi_timeframe = config.get('use_multi_timeframe', True)
        self.use_time_features = config.get('use_time_features', True)
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """創建所有時序特徵"""
        df = df.copy()
        
        # 1. 價格特徵
        df = self._create_price_features(df)
        
        # 2. 動量特徵
        df = self._create_momentum_features(df)
        
        # 3. 波動率特徵
        df = self._create_volatility_features(df)
        
        # 4. 技術指標
        df = self._create_technical_indicators(df)
        
        # 5. 時間特徵
        if self.use_time_features:
            df = self._create_time_features(df)
        
        # 6. 統計特徵
        df = self._create_statistical_features(df)
        
        return df
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """價格相關特徵"""
        # 多期報酬率
        for period in self.lookback_periods:
            df[f'return_{period}'] = df['close'].pct_change(period)
        
        # 對數報酬率
        for period in self.lookback_periods:
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
        
        # 價格相對位置
        for period in [20, 50, 100]:
            df[f'price_position_{period}'] = (
                (df['close'] - df['close'].rolling(period).min()) / 
                (df['close'].rolling(period).max() - df['close'].rolling(period).min() + 1e-6)
            )
        
        # OHLC特徵
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
        df['body_ratio'] = df['body'] / (df['high'] - df['low'] + 1e-6)
        
        return df
    
    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """動量特徵"""
        # RSI
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-6)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 動量指標
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / 
                                   df['close'].shift(period)) * 100
        
        return df
    
    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """波動率特徵"""
        # 歷史波動率
        for period in self.lookback_periods:
            df[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
        
        # ATR (Average True Range)
        for period in [7, 14, 21]:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f'atr_{period}'] = tr.rolling(period).mean()
        
        # Bollinger Bands
        for period in [10, 20, 30]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'bb_upper_{period}'] = sma + (2 * std)
            df[f'bb_lower_{period}'] = sma - (2 * std)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / \
                                          (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-6)
        
        # 波動率狀態
        df['volatility_regime'] = pd.cut(
            df['volatility_20'], 
            bins=3, 
            labels=[0, 1, 2]  # 0=低, 1=中, 2=高
        ).astype(float)
        
        return df
    
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """技術指標"""
        # EMA距離
        for period in [10, 20, 50, 100, 200]:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'ema_distance_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
        
        # EMA交叉
        df['ema_cross_10_20'] = (df['ema_10'] > df['ema_20']).astype(int)
        df['ema_cross_20_50'] = (df['ema_20'] > df['ema_50']).astype(int)
        df['ema_cross_50_200'] = (df['ema_50'] > df['ema_200']).astype(int)
        
        # ADX (Average Directional Index)
        period = 14
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        pos_di = 100 * (pos_dm.rolling(period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di + 1e-6)
        df['adx'] = dx.rolling(period).mean()
        
        # OBV (On-Balance Volume)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=20).mean()
        df['obv_divergence'] = df['obv'] - df['obv_ema']
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """時間周期性特徵"""
        if 'timestamp' not in df.columns:
            return df
        
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['day_of_month'] = pd.to_datetime(df['timestamp']).dt.day
        
        # 時段特徵 (亞洲/歐洲/美洲時段)
        df['asian_session'] = df['hour'].between(0, 8).astype(int)
        df['european_session'] = df['hour'].between(8, 16).astype(int)
        df['us_session'] = df['hour'].between(16, 24).astype(int)
        
        # 周末/平日
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # 周期性編碼
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """統計特徵"""
        # 滾動統計
        for period in [10, 20, 30]:
            # 峰度
            df[f'kurtosis_{period}'] = df['close'].rolling(period).kurt()
            
            # 偏度
            df[f'skewness_{period}'] = df['close'].rolling(period).skew()
            
            # 極差
            df[f'range_{period}'] = df['high'].rolling(period).max() - df['low'].rolling(period).min()
            df[f'range_pct_{period}'] = df[f'range_{period}'] / df['close']
        
        # Z-Score
        for period in [20, 50]:
            mean = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'zscore_{period}'] = (df['close'] - mean) / (std + 1e-6)
        
        return df
    
    def create_sequence_features(self, df: pd.DataFrame, sequence_length: int = 30) -> np.ndarray:
        """
        創建時間序列特徵矩陣 - 用於LSTM/Transformer
        返回: (samples, sequence_length, features)
        """
        feature_cols = self.get_feature_names()
        available_cols = [col for col in feature_cols if col in df.columns]
        
        data = df[available_cols].values
        sequences = []
        
        for i in range(sequence_length, len(data)):
            sequences.append(data[i-sequence_length:i])
        
        return np.array(sequences)
    
    def get_feature_names(self) -> List[str]:
        """返回所有特徵名稱"""
        features = []
        
        # 價格特徵
        for period in self.lookback_periods:
            features.extend([f'return_{period}', f'log_return_{period}'])
        
        for period in [20, 50, 100]:
            features.append(f'price_position_{period}')
        
        features.extend(['body', 'upper_shadow', 'lower_shadow', 'body_ratio'])
        
        # 動量特徵
        for period in [7, 14, 21]:
            features.append(f'rsi_{period}')
        
        features.extend(['macd', 'macd_signal', 'macd_histogram'])
        
        for period in [5, 10, 20]:
            features.extend([f'momentum_{period}', f'roc_{period}'])
        
        # 波動率特徵
        for period in self.lookback_periods:
            features.append(f'volatility_{period}')
        
        for period in [7, 14, 21]:
            features.append(f'atr_{period}')
        
        for period in [10, 20, 30]:
            features.extend([
                f'bb_width_{period}', 
                f'bb_position_{period}'
            ])
        
        features.append('volatility_regime')
        
        # 技術指標
        for period in [10, 20, 50, 100, 200]:
            features.append(f'ema_distance_{period}')
        
        features.extend([
            'ema_cross_10_20', 'ema_cross_20_50', 'ema_cross_50_200',
            'adx', 'obv_divergence'
        ])
        
        # 時間特徵
        if self.use_time_features:
            features.extend([
                'asian_session', 'european_session', 'us_session', 'is_weekend',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
            ])
        
        # 統計特徵
        for period in [10, 20, 30]:
            features.extend([
                f'kurtosis_{period}', f'skewness_{period}', f'range_pct_{period}'
            ])
        
        for period in [20, 50]:
            features.append(f'zscore_{period}')
        
        return features
