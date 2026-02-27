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
        
        # 基礎價格特徵
        df = self._add_price_features(df)
        
        # 成交量特徵
        df = self._add_volume_features(df)
        
        # 動量指標
        if self.use_momentum:
            df = self._add_momentum_features(df)
        
        # 微觀結構特徵
        if self.use_microstructure:
            df = self._add_microstructure_features(df)
        
        # 訂單簿特徵(如果有數據)
        if self.use_orderbook_features and 'bid_volume' in df.columns:
            df = self._add_orderbook_features(df)
        
        # 時間特徵
        df = self._add_time_features(df)
        
        # 波動率狀態
        df = self._add_volatility_regime(df)
        
        # 去除NaN
        df = df.dropna()
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """價格相關特徵"""
        # 對數收益率(多週期)
        for period in self.lookback_periods:
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
            df[f'high_low_ratio_{period}'] = (df['high'] - df['low']) / df['close']
        
        # 價格位置(在高低點之間的位置)
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # 與移動平均的距離
        for period in [5, 10, 20, 50]:
            ma = df['close'].rolling(period).mean()
            df[f'dist_from_ma{period}'] = (df['close'] - ma) / ma
        
        # 布林帶位置
        bb_period = 20
        bb_std = 2
        ma20 = df['close'].rolling(bb_period).mean()
        std20 = df['close'].rolling(bb_period).std()
        df['bb_upper'] = ma20 + (bb_std * std20)
        df['bb_lower'] = ma20 - (bb_std * std20)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交量特徵"""
        # 成交量變化率
        for period in self.lookback_periods:
            df[f'volume_change_{period}'] = df['volume'].pct_change(period)
            df[f'volume_ma_ratio_{period}'] = df['volume'] / df['volume'].rolling(period).mean()
        
        # 成交量加權價格
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['dist_from_vwap'] = (df['close'] - df['vwap']) / df['vwap']
        
        # 大單檢測(異常成交量)
        vol_ma = df['volume'].rolling(20).mean()
        vol_std = df['volume'].rolling(20).std()
        df['large_trade'] = ((df['volume'] - vol_ma) / vol_std).clip(-3, 3)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """動量技術指標"""
        # RSI (多週期)
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)
        
        # MACD
        macd, signal, hist = talib.MACD(df['close'], 
                                        fastperiod=12, 
                                        slowperiod=26, 
                                        signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # ADX (趨勢強度)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # CCI
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Stochastic
        slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'],
                                   fastk_period=14, slowk_period=3, 
                                   slowk_matype=0, slowd_period=3, slowd_matype=0)
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """市場微觀結構特徵"""
        # 價格動量分解
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            df[f'acceleration_{period}'] = df[f'momentum_{period}'].diff()
        
        # 波動率(多週期)
        for period in [5, 10, 20, 50]:
            df[f'volatility_{period}'] = df['log_return_5'].rolling(period).std()
        
        # ATR
        df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['atr_ratio'] = df['atr_14'] / df['close']
        
        # 價格效率比(趨勢強度)
        for period in [10, 20, 50]:
            price_change = abs(df['close'] - df['close'].shift(period))
            path_length = df['close'].diff().abs().rolling(period).sum()
            df[f'efficiency_ratio_{period}'] = price_change / path_length
        
        # 成交量加權波動率
        returns_vol = df['log_return_5'].rolling(20).std()
        volume_weight = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_weighted_vol'] = returns_vol * volume_weight
        
        return df
    
    def _add_orderbook_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """訂單簿特徵(需要訂單簿數據)"""
        # 訂單簿失衡率
        df['orderbook_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
        
        # 買賣壓力比
        df['buy_sell_pressure'] = df['bid_volume'] / df['ask_volume']
        
        # 價差
        if 'bid_price' in df.columns and 'ask_price' in df.columns:
            df['spread'] = (df['ask_price'] - df['bid_price']) / df['close']
            df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
            df['dist_from_mid'] = (df['close'] - df['mid_price']) / df['mid_price']
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """時間相關特徵"""
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # 交易時段(亞洲/歐洲/美洲)
            df['asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['europe_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['us_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
        
        return df
    
    def _add_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """波動率狀態分類"""
        # 計算20期波動率
        vol = df['log_return_5'].rolling(20).std()
        
        # 使用分位數分類
        vol_20 = vol.rolling(100).quantile(0.2)
        vol_80 = vol.rolling(100).quantile(0.8)
        
        # 狀態: 0=低波動, 1=中波動, 2=高波動
        df['volatility_regime'] = 1  # default
        df.loc[vol < vol_20, 'volatility_regime'] = 0
        df.loc[vol > vol_80, 'volatility_regime'] = 2
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """獲取所有特徵名稱"""
        # 這裡列出所有可能的特徵名稱
        # 實際使用時需要根據配置動態生成
        feature_prefixes = [
            'log_return_', 'high_low_ratio_', 'dist_from_ma',
            'volume_change_', 'volume_ma_ratio_',
            'rsi_', 'momentum_', 'acceleration_', 'volatility_',
            'efficiency_ratio_'
        ]
        
        single_features = [
            'price_position', 'bb_position', 'dist_from_vwap',
            'large_trade', 'macd', 'macd_signal', 'macd_hist',
            'adx', 'cci', 'stoch_k', 'stoch_d',
            'atr_14', 'atr_ratio', 'volume_weighted_vol',
            'volatility_regime', 'hour', 'day_of_week',
            'asia_session', 'europe_session', 'us_session'
        ]
        
        return single_features  # 簡化返回
    
    def prepare_sequences(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """準備時序數據用於Transformer"""
        sequences = []
        
        for i in range(self.sequence_length, len(df)):
            seq = df[feature_cols].iloc[i-self.sequence_length:i].values
            sequences.append(seq)
        
        return np.array(sequences)
