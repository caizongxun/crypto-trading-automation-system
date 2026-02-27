"""
Time-Series Feature Engineering for High-Frequency Trading
高頻交易時序特徵工程
"""
import pandas as pd
import numpy as np
from typing import List, Dict
import talib

class FeatureEngineer:
    def __init__(self, config: Dict):
        self.sequence_length = config.get('sequence_length', 100)
        self.use_orderbook_features = config.get('use_orderbook_features', True)
        self.use_microstructure = config.get('use_microstructure', True)
        self.use_momentum = config.get('use_momentum', True)
        self.lookback_periods = config.get('lookback_periods', [5, 10, 20, 50])
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """創建完整特徵集"""
        df = df.copy()
        df = self._add_price_features(df)
        df = self._add_volume_features(df)
        if self.use_momentum:
            df = self._add_momentum_features(df)
        if self.use_microstructure:
            df = self._add_microstructure_features(df)
        if self.use_orderbook_features and 'bid_volume' in df.columns:
            df = self._add_orderbook_features(df)
        df = self._add_time_features(df)
        df = self._add_volatility_regime(df)
        df = df.dropna()
        
        # 轉揟float32節省記憶體 (50%)
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for period in self.lookback_periods:
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
            df[f'high_low_ratio_{period}'] = (df['high'] - df['low']) / df['close']
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        for period in [5, 10, 20, 50]:
            ma = df['close'].rolling(period).mean()
            df[f'dist_from_ma{period}'] = (df['close'] - ma) / ma
        bb_period = 20
        bb_std = 2
        ma20 = df['close'].rolling(bb_period).mean()
        std20 = df['close'].rolling(bb_period).std()
        df['bb_upper'] = ma20 + (bb_std * std20)
        df['bb_lower'] = ma20 - (bb_std * std20)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for period in self.lookback_periods:
            df[f'volume_change_{period}'] = df['volume'].pct_change(period)
            df[f'volume_ma_ratio_{period}'] = df['volume'] / df['volume'].rolling(period).mean()
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['dist_from_vwap'] = (df['close'] - df['vwap']) / df['vwap']
        vol_ma = df['volume'].rolling(20).mean()
        vol_std = df['volume'].rolling(20).std()
        df['large_trade'] = ((df['volume'] - vol_ma) / vol_std).clip(-3, 3)
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)
        macd, signal, hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'],
                                   fastk_period=14, slowk_period=3, 
                                   slowk_matype=0, slowd_period=3, slowd_matype=0)
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            df[f'acceleration_{period}'] = df[f'momentum_{period}'].diff()
        for period in [5, 10, 20, 50]:
            df[f'volatility_{period}'] = df['log_return_5'].rolling(period).std()
        df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['atr_ratio'] = df['atr_14'] / df['close']
        for period in [10, 20, 50]:
            price_change = abs(df['close'] - df['close'].shift(period))
            path_length = df['close'].diff().abs().rolling(period).sum()
            df[f'efficiency_ratio_{period}'] = price_change / path_length
        returns_vol = df['log_return_5'].rolling(20).std()
        volume_weight = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_weighted_vol'] = returns_vol * volume_weight
        return df
    
    def _add_orderbook_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['orderbook_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
        df['buy_sell_pressure'] = df['bid_volume'] / df['ask_volume']
        if 'bid_price' in df.columns and 'ask_price' in df.columns:
            df['spread'] = (df['ask_price'] - df['bid_price']) / df['close']
            df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
            df['dist_from_mid'] = (df['close'] - df['mid_price']) / df['mid_price']
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['europe_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['us_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
        return df
    
    def _add_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        vol = df['log_return_5'].rolling(20).std()
        vol_20 = vol.rolling(100).quantile(0.2)
        vol_80 = vol.rolling(100).quantile(0.8)
        df['volatility_regime'] = 1
        df.loc[vol < vol_20, 'volatility_regime'] = 0
        df.loc[vol > vol_80, 'volatility_regime'] = 2
        return df
    
    def get_feature_names(self) -> List[str]:
        single_features = [
            'price_position', 'bb_position', 'dist_from_vwap',
            'large_trade', 'macd', 'macd_signal', 'macd_hist',
            'adx', 'cci', 'stoch_k', 'stoch_d',
            'atr_14', 'atr_ratio', 'volume_weighted_vol',
            'volatility_regime', 'hour', 'day_of_week',
            'asia_session', 'europe_session', 'us_session'
        ]
        return single_features
    
    def prepare_sequences(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """
        準備時序數據用於Transformer (向量化優化 + 記憶體優化)
        
        Args:
            df: DataFrame包含特徵
            feature_cols: 特徵列名列表
        
        Returns:
            np.ndarray: shape (n_samples, sequence_length, n_features), dtype=float32
        """
        print(f"  準備序列數據: {len(df)} 筆, 序列長度={self.sequence_length}")
        
        # 提取特徵矩陣 (使用float32節省記憶體)
        feature_matrix = df[feature_cols].values.astype(np.float32)
        n_samples = len(feature_matrix) - self.sequence_length + 1
        n_features = len(feature_cols)
        
        if n_samples <= 0:
            raise ValueError(f"數據不足以建立序列: {len(feature_matrix)} < {self.sequence_length}")
        
        # 計算預期記憶體使用
        expected_memory_gb = (n_samples * self.sequence_length * n_features * 4) / (1024**3)
        print(f"  預期記憶體: {expected_memory_gb:.2f} GB")
        
        # 如果超過4GB，使用批次處理
        if expected_memory_gb > 4.0:
            print(f"  警告: 記憶體需求過大，使用批次處理")
            return self._prepare_sequences_batched(feature_matrix, n_samples, n_features)
        
        # 使用向量化操作建立滾動窗口
        from numpy.lib.stride_tricks import as_strided
        
        shape = (n_samples, self.sequence_length, n_features)
        strides = (feature_matrix.strides[0], feature_matrix.strides[0], feature_matrix.strides[1])
        
        sequences = as_strided(feature_matrix, shape=shape, strides=strides)
        
        print(f"  ✓ 序列數據形狀: {sequences.shape}, dtype={sequences.dtype}")
        
        return sequences.copy()  # copy以確保數據獨立
    
    def _prepare_sequences_batched(self, feature_matrix: np.ndarray, 
                                   n_samples: int, n_features: int) -> np.ndarray:
        """批次處理大型數據集"""
        print(f"  使用批次處理模式")
        
        # 分成20個批次處理
        batch_size = max(1000, n_samples // 20)
        sequences_list = []
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            
            batch_sequences = []
            for i in range(start_idx, end_idx):
                seq = feature_matrix[i:i+self.sequence_length]
                batch_sequences.append(seq)
            
            sequences_list.append(np.array(batch_sequences, dtype=np.float32))
            
            if (start_idx // batch_size) % 5 == 0:
                print(f"    處理進度: {end_idx}/{n_samples} ({100*end_idx/n_samples:.1f}%)")
        
        sequences = np.concatenate(sequences_list, axis=0)
        print(f"  ✓ 序列數據形狀: {sequences.shape}, dtype={sequences.dtype}")
        
        return sequences
