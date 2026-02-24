import pandas as pd
import numpy as np
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineerV3:
    """
    V3 特徵工程 - 簡潔但強大
    
    設計理念:
    1. 精簡特徵 (20-25個)
    2. 更好的標籤定義
    3. 基於實際可獲利的價格移動
    4. 更好的機率校準
    """
    
    def __init__(self):
        self.version = 'v3'
        self.feature_groups = {
            'price': [],
            'momentum': [],
            'volatility': [],
            'volume': [],
            'microstructure': []
        }
    
    def create_features_from_1m(self, 
                                df_1m: pd.DataFrame,
                                tp_pct: float = 0.02,
                                sl_pct: float = 0.01,
                                label_type: str = 'both') -> pd.DataFrame:
        """
        從 1m K線生成 V3 特徵
        
        Args:
            df_1m: 1m K線數據
            tp_pct: 停利百分比 (2%)
            sl_pct: 停損百分比 (1%)
            label_type: 'both', 'long', 'short'
        """
        df = df_1m.copy()
        
        print("=" * 80)
        print("V3 FEATURE ENGINEERING")
        print("=" * 80)
        print(f"Input data: {len(df):,} rows")
        print(f"TP/SL: {tp_pct*100:.1f}% / {sl_pct*100:.1f}%")
        
        # 1. 基礎價格特徵
        df = self._add_price_features(df)
        
        # 2. 動能特徵
        df = self._add_momentum_features(df)
        
        # 3. 波動率特徵
        df = self._add_volatility_features(df)
        
        # 4. 成交量特徵
        df = self._add_volume_features(df)
        
        # 5. 微觀結構特徵
        df = self._add_microstructure_features(df)
        
        # 6. 多時間框架特徵
        df = self._add_multi_timeframe_features(df)
        
        # 7. 生成標籤
        if label_type in ['both', 'long']:
            df = self._create_labels(df, 'long', tp_pct, sl_pct)
        if label_type in ['both', 'short']:
            df = self._create_labels(df, 'short', tp_pct, sl_pct)
        
        # 移除 NaN
        df = df.dropna()
        
        print(f"\nOutput data: {len(df):,} rows")
        print(f"Features: {len(self.get_feature_list())}")
        print("=" * 80)
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """基礎價格特徵"""
        print("\n1. Adding Price Features...")
        
        # 價格變化
        df['returns_1m'] = df['close'].pct_change()
        df['returns_5m'] = df['close'].pct_change(5)
        df['returns_15m'] = df['close'].pct_change(15)
        
        # 價格位置 (relative to recent range)
        high_60 = df['high'].rolling(60).max()
        low_60 = df['low'].rolling(60).min()
        df['price_position'] = (df['close'] - low_60) / (high_60 - low_60 + 1e-8)
        
        self.feature_groups['price'] = ['returns_1m', 'returns_5m', 'returns_15m', 'price_position']
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """動能特徵"""
        print("2. Adding Momentum Features...")
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['rsi_28'] = self._calculate_rsi(df['close'], 28)
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = (ema_12 - ema_26) / df['close']
        
        # 動能強度 (price momentum over different periods)
        df['momentum_5'] = (df['close'] / df['close'].shift(5) - 1)
        df['momentum_15'] = (df['close'] / df['close'].shift(15) - 1)
        df['momentum_30'] = (df['close'] / df['close'].shift(30) - 1)
        
        self.feature_groups['momentum'] = ['rsi_14', 'rsi_28', 'macd', 'momentum_5', 'momentum_15', 'momentum_30']
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """波動率特徵"""
        print("3. Adding Volatility Features...")
        
        # ATR
        df['atr_14'] = self._calculate_atr(df, 14)
        df['atr_pct'] = df['atr_14'] / df['close']
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_width'] = (std_20 * 2) / sma_20
        df['bb_position'] = (df['close'] - sma_20) / (std_20 * 2 + 1e-8)
        
        # 波動率越勢 (volatility regime)
        df['volatility_ratio'] = df['atr_pct'] / df['atr_pct'].rolling(60).mean()
        
        self.feature_groups['volatility'] = ['atr_pct', 'bb_width', 'bb_position', 'volatility_ratio']
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交量特徵"""
        print("4. Adding Volume Features...")
        
        # 成交量比率
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # 價量相關
        df['price_volume_corr'] = df['close'].rolling(20).corr(df['volume'])
        
        # VWAP 偏離
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
        
        self.feature_groups['volume'] = ['volume_ratio', 'price_volume_corr', 'vwap_deviation']
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """微觀結構特徵"""
        print("5. Adding Microstructure Features...")
        
        # 價格效率 (price efficiency)
        price_change = abs(df['close'] - df['close'].shift(10))
        path_length = abs(df['close'].diff()).rolling(10).sum()
        df['efficiency_ratio'] = price_change / (path_length + 1e-8)
        
        # 極端價格時間 (time since extreme)
        high_10 = df['high'].rolling(10).max()
        low_10 = df['low'].rolling(10).min()
        df['time_since_high'] = (df['high'] == high_10).rolling(10).apply(lambda x: 10 - np.argmax(x[::-1]) if x.any() else 10, raw=True)
        df['time_since_low'] = (df['low'] == low_10).rolling(10).apply(lambda x: 10 - np.argmax(x[::-1]) if x.any() else 10, raw=True)
        
        self.feature_groups['microstructure'] = ['efficiency_ratio', 'time_since_high', 'time_since_low']
        return df
    
    def _add_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """多時間框架特徵"""
        print("6. Adding Multi-Timeframe Features...")
        
        # 5m 特徵
        df_5m = df.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        df_5m['rsi_5m'] = self._calculate_rsi(df_5m['close'], 14)
        df_5m['atr_5m'] = self._calculate_atr(df_5m, 14) / df_5m['close']
        
        # 15m 特徵
        df_15m = df.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        df_15m['momentum_15m'] = df_15m['close'].pct_change(4)
        
        # 合併回 1m
        df = df.join(df_5m[['rsi_5m', 'atr_5m']], how='left')
        df = df.join(df_15m[['momentum_15m']], how='left')
        
        # Forward fill
        df['rsi_5m'] = df['rsi_5m'].fillna(method='ffill')
        df['atr_5m'] = df['atr_5m'].fillna(method='ffill')
        df['momentum_15m'] = df['momentum_15m'].fillna(method='ffill')
        
        return df
    
    def _create_labels(self, df: pd.DataFrame, direction: str, tp_pct: float, sl_pct: float) -> pd.DataFrame:
        """
        生成標籤 - 基於實際 TP/SL
        
        標籤定義:
        1 = 能在 TP 之前達到目標 (profitable)
        0 = 在 SL 之前被停損 (unprofitable)
        """
        print(f"\n7. Creating {direction.upper()} labels (TP={tp_pct*100:.1f}%, SL={sl_pct*100:.1f}%)...")
        
        labels = []
        lookforward = 120  # 2小時
        
        for i in range(len(df) - lookforward):
            entry_price = df['close'].iloc[i]
            future_prices = df['close'].iloc[i+1:i+lookforward+1]
            future_highs = df['high'].iloc[i+1:i+lookforward+1]
            future_lows = df['low'].iloc[i+1:i+lookforward+1]
            
            if direction == 'long':
                tp_price = entry_price * (1 + tp_pct)
                sl_price = entry_price * (1 - sl_pct)
                
                # 檢查是否先達到 TP
                tp_hit = (future_highs >= tp_price).any()
                sl_hit = (future_lows <= sl_price).any()
                
                if tp_hit:
                    tp_time = (future_highs >= tp_price).idxmax()
                    if sl_hit:
                        sl_time = (future_lows <= sl_price).idxmax()
                        label = 1 if tp_time < sl_time else 0
                    else:
                        label = 1
                else:
                    label = 0
            
            else:  # short
                tp_price = entry_price * (1 - tp_pct)
                sl_price = entry_price * (1 + sl_pct)
                
                # 檢查是否先達到 TP
                tp_hit = (future_lows <= tp_price).any()
                sl_hit = (future_highs >= sl_price).any()
                
                if tp_hit:
                    tp_time = (future_lows <= tp_price).idxmax()
                    if sl_hit:
                        sl_time = (future_highs >= sl_price).idxmax()
                        label = 1 if tp_time < sl_time else 0
                    else:
                        label = 1
                else:
                    label = 0
            
            labels.append(label)
        
        # 加入 NaN 以匹配長度
        labels.extend([np.nan] * lookforward)
        df[f'label_{direction}'] = labels
        
        # 統計
        valid_labels = [l for l in labels if not np.isnan(l)]
        if valid_labels:
            positive_rate = sum(valid_labels) / len(valid_labels) * 100
            print(f"  Positive rate: {positive_rate:.2f}%")
            print(f"  Total labels: {len(valid_labels):,}")
        
        return df
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """計算 RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """計算 ATR"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def get_feature_list(self) -> List[str]:
        """獲取所有特徵名稱"""
        features = []
        for group in self.feature_groups.values():
            features.extend(group)
        
        # 加入多時間框架特徵
        features.extend(['rsi_5m', 'atr_5m', 'momentum_15m'])
        
        return features
    
    def get_feature_groups(self) -> dict:
        """獲取特徵分組"""
        groups = self.feature_groups.copy()
        groups['multi_timeframe'] = ['rsi_5m', 'atr_5m', 'momentum_15m']
        return groups