"""
V4 Feature Engineer - Enhanced for Neural Network
V4特徵工程 - 為神經網路優化
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import ta

class FeatureEngineer:
    """
    為V4 LSTM/GRU生成高質量特徵
    
    核心特徵類別:
    1. 市場微觀結構 (價格效率, VWAP偏離)
    2. 波動率特徵 (ATR標準化)
    3. 動量品質 (趨勢一致性)
    4. 多週期確認 (15m + 1h)
    5. 時序特徵 (為LSTM/GRU優化)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.lookback_periods = config.get('lookback_periods', [5, 10, 20])
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成所有特徵
        """
        df = df.copy()
        
        # 1. 基礎技術指標
        df = self._add_basic_indicators(df)
        
        # 2. 市場微觀結構
        df = self._add_microstructure_features(df)
        
        # 3. 波動率特徵
        df = self._add_volatility_features(df)
        
        # 4. 動量品質
        df = self._add_momentum_features(df)
        
        # 5. 成交量特徵
        df = self._add_volume_features(df)
        
        # 6. 統計特徵
        df = self._add_statistical_features(df)
        
        # 7. 時間特徵
        df = self._add_time_features(df)
        
        # 8. V4特有: 時序增強特徵
        df = self._add_sequence_features(df)
        
        return df
    
    def _add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """基礎技術指標"""
        # ATR
        df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['atr_50'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=50)
        
        # RSI
        df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / (df['bb_mid'] + 1e-10)
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'] + 1e-10)
        
        # EMA
        df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """市場微觀結構特徵"""
        # 價格效率
        df['price_efficiency'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        
        # VWAP
        df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-10)
        df['vwap_deviation'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-10)
        
        # 上影線/下影線
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
        df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        
        # 價格位置
        df['price_position_20'] = (df['close'] - df['close'].rolling(20).min()) / (df['close'].rolling(20).max() - df['close'].rolling(20).min() + 1e-10)
        
        # 真實波動率
        df['true_range'] = df[['high', 'close']].max(axis=1) - df[['low', 'close']].min(axis=1)
        df['true_range_pct'] = df['true_range'] / (df['close'] + 1e-10)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """波動率特徵"""
        df['price_change'] = df['close'].diff()
        df['price_change_atr_norm'] = df['price_change'] / (df['atr_14'] + 1e-10)
        df['atr_ratio'] = df['atr_14'] / (df['atr_50'] + 1e-10)
        df['atr_change'] = df['atr_14'].pct_change(5)
        
        df['realized_vol_5'] = df['price_change'].rolling(5).std() / (df['close'] + 1e-10)
        df['realized_vol_20'] = df['price_change'].rolling(20).std() / (df['close'] + 1e-10)
        df['vol_deviation'] = (df['realized_vol_5'] - df['realized_vol_20']) / (df['realized_vol_20'] + 1e-10)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """動量品質特徵"""
        for period in [5, 10, 20]:
            df[f'trend_consistency_{period}'] = df['price_change'].rolling(period).apply(
                lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
            )
            df[f'trend_consistency_{period}'] = df[f'trend_consistency_{period}'] * 2 - 1
            df[f'momentum_{period}'] = df['close'].pct_change(period)
        
        df['roc_5'] = ta.momentum.roc(df['close'], window=5)
        df['roc_10'] = ta.momentum.roc(df['close'], window=10)
        df['price_acceleration'] = df['price_change'].diff()
        df['price_acceleration_norm'] = df['price_acceleration'] / (df['atr_14'] + 1e-10)
        df['rsi_slope'] = df['rsi_14'].diff(5)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交量特徵"""
        df['volume_ma_5'] = df['volume'].rolling(5).mean()
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ma_50'] = df['volume'].rolling(50).mean()
        
        df['volume_ratio_5_20'] = df['volume_ma_5'] / (df['volume_ma_20'] + 1e-10)
        df['volume_ratio_20_50'] = df['volume_ma_20'] / (df['volume_ma_50'] + 1e-10)
        df['volume_relative'] = df['volume'] / (df['volume_ma_20'] + 1e-10)
        
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['obv_ma'] = df['obv'].rolling(20).mean()
        df['obv_divergence'] = (df['obv'] - df['obv_ma']) / (df['obv_ma'].abs() + 1e-10)
        
        df['price_volume_corr'] = df['close'].rolling(20).corr(df['volume'])
        df['vpt'] = (df['close'].pct_change() * df['volume']).cumsum()
        df['vpt_ma'] = df['vpt'].rolling(20).mean()
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """統計特徵"""
        df['price_skew_20'] = df['close'].rolling(20).skew()
        df['price_kurt_20'] = df['close'].rolling(20).kurt()
        df['price_zscore_20'] = (df['close'] - df['close'].rolling(20).mean()) / (df['close'].rolling(20).std() + 1e-10)
        df['price_range_20'] = (df['close'].rolling(20).max() - df['close'].rolling(20).min()) / (df['close'].rolling(20).mean() + 1e-10)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """時間特徵"""
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['trading_session'] = df['hour'].apply(lambda x: 0 if x < 8 else (1 if x < 16 else 2))
        
        return df
    
    def _add_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        V4特有: 時序增強特徵
        為LSTM/GRU優化的特徵
        """
        # 1. 價格趨勢強度 (連續上漲/下跌)
        df['price_trend_strength'] = df['close'].diff().rolling(10).apply(
            lambda x: abs((x > 0).sum() - (x < 0).sum()) / len(x) if len(x) > 0 else 0
        )
        
        # 2. 波動率趨勢 (擴張/收縮)
        df['volatility_trend'] = df['atr_14'].diff(5) / (df['atr_14'].shift(5) + 1e-10)
        
        # 3. 成交量趨勢
        df['volume_trend'] = df['volume_ma_5'].diff(5) / (df['volume_ma_5'].shift(5) + 1e-10)
        
        # 4. 動能收斂/發散 (MACD交叉)
        df['macd_crossover'] = (df['macd'] > df['macd_signal']).astype(int) * 2 - 1
        df['macd_crossover_change'] = df['macd_crossover'].diff().fillna(0)
        
        # 5. 多空能量對比
        df['bull_power'] = df['high'] - df['ema_21']
        df['bear_power'] = df['low'] - df['ema_21']
        df['bull_bear_ratio'] = df['bull_power'] / (abs(df['bear_power']) + 1e-10)
        
        return df
    
    def get_feature_names(self, exclude_cols: List[str] = None) -> List[str]:
        """獲取特徵欄位名稱"""
        if exclude_cols is None:
            exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label',
                          'target_win_rate', 'target_payoff']
        return exclude_cols
