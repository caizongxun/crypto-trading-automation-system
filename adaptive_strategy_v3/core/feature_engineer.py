"""
V3 Feature Engineer - Market Microstructure & Multi-Timeframe Features
市場微觀結構特徵工程
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import ta

class FeatureEngineer:
    """
    生成高預測力特徵
    
    核心特徵類別:
    1. 市場微觀結構 (價格效率, VWAP偏離)
    2. 波動率特徵 (ATR標準化)
    3. 動量品質 (趨勢一致性)
    4. 多週期確認 (15m + 1h)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.lookback_periods = config.get('lookback_periods', [5, 10, 20])
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成所有特徵
        """
        df = df.copy()
        
        print("\n[特徵工程] 開始...")
        
        # 1. 基礎技術指標
        df = self._add_basic_indicators(df)
        print("[OK] 基礎指標")
        
        # 2. 市場微觀結構
        df = self._add_microstructure_features(df)
        print("[OK] 微觀結構")
        
        # 3. 波動率特徵
        df = self._add_volatility_features(df)
        print("[OK] 波動率特徵")
        
        # 4. 動量品質
        df = self._add_momentum_features(df)
        print("[OK] 動量品質")
        
        # 5. 成交量特徵
        df = self._add_volume_features(df)
        print("[OK] 成交量特徵")
        
        # 6. 統計特徵
        df = self._add_statistical_features(df)
        print("[OK] 統計特徵")
        
        # 7. 時間特徵
        df = self._add_time_features(df)
        print("[OK] 時間特徵")
        
        print(f"[OK] 特徵工程完成 - 總特徵數: {len(df.columns)}")
        
        return df
    
    def _add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基礎技術指標
        """
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
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        
        # EMA
        df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
        df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        市場微觀結構特徵
        """
        # 1. 價格效率 (Price Efficiency)
        # 衡量趨勢vs震盪: 值越高趨勢越強
        df['price_efficiency'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        
        # 2. VWAP (Volume Weighted Average Price)
        df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['vwap_deviation'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-10)
        
        # 3. 上影線/下影線比例
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
        df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        
        # 4. 價格位置 (Price Position)
        df['price_position_20'] = (df['close'] - df['close'].rolling(20).min()) / (df['close'].rolling(20).max() - df['close'].rolling(20).min() + 1e-10)
        
        # 5. 真實波動率
        df['true_range'] = df[['high', 'close']].max(axis=1) - df[['low', 'close']].min(axis=1)
        df['true_range_pct'] = df['true_range'] / df['close']
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        波動率特徵
        """
        # 1. ATR標準化價格變動
        df['price_change'] = df['close'].diff()
        df['price_change_atr_norm'] = df['price_change'] / (df['atr_14'] + 1e-10)
        
        # 2. 波動率比率 (短期vs長期)
        df['atr_ratio'] = df['atr_14'] / (df['atr_50'] + 1e-10)
        
        # 3. 波動率趨勢
        df['atr_change'] = df['atr_14'].pct_change(5)
        
        # 4. Realized Volatility
        df['realized_vol_5'] = df['price_change'].rolling(5).std() / df['close']
        df['realized_vol_20'] = df['price_change'].rolling(20).std() / df['close']
        
        # 5. 波動率偏离
        df['vol_deviation'] = (df['realized_vol_5'] - df['realized_vol_20']) / (df['realized_vol_20'] + 1e-10)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        動量品質特徵
        """
        # 1. 趨勢一致性 (Trend Consistency)
        for period in [5, 10, 20]:
            df[f'trend_consistency_{period}'] = df['price_change'].rolling(period).apply(
                lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0.5
            )
            # 轉換為 -1 到 1
            df[f'trend_consistency_{period}'] = df[f'trend_consistency_{period}'] * 2 - 1
        
        # 2. 價格動能 (Price Momentum)
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
        
        # 3. ROC (Rate of Change)
        df['roc_5'] = ta.momentum.roc(df['close'], window=5)
        df['roc_10'] = ta.momentum.roc(df['close'], window=10)
        
        # 4. 價格加速度
        df['price_acceleration'] = df['price_change'].diff()
        df['price_acceleration_norm'] = df['price_acceleration'] / (df['atr_14'] + 1e-10)
        
        # 5. 相對強弱指數 (RSI Slope)
        df['rsi_slope'] = df['rsi_14'].diff(5)
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        成交量特徵
        """
        # 1. 成交量均線
        df['volume_ma_5'] = df['volume'].rolling(5).mean()
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ma_50'] = df['volume'].rolling(50).mean()
        
        # 2. 成交量比率
        df['volume_ratio_5_20'] = df['volume_ma_5'] / (df['volume_ma_20'] + 1e-10)
        df['volume_ratio_20_50'] = df['volume_ma_20'] / (df['volume_ma_50'] + 1e-10)
        
        # 3. 相對成交量
        df['volume_relative'] = df['volume'] / (df['volume_ma_20'] + 1e-10)
        
        # 4. OBV (On Balance Volume)
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['obv_ma'] = df['obv'].rolling(20).mean()
        df['obv_divergence'] = (df['obv'] - df['obv_ma']) / (df['obv_ma'].abs() + 1e-10)
        
        # 5. 價量相關
        df['price_volume_corr'] = df['close'].rolling(20).corr(df['volume'])
        
        # 6. Volume-Price Trend
        df['vpt'] = (df['close'].pct_change() * df['volume']).cumsum()
        df['vpt_ma'] = df['vpt'].rolling(20).mean()
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        統計特徵
        """
        # 1. 價格偏度 (Skewness)
        df['price_skew_20'] = df['close'].rolling(20).skew()
        
        # 2. 價格峰度 (Kurtosis)
        df['price_kurt_20'] = df['close'].rolling(20).kurt()
        
        # 3. Z-Score
        df['price_zscore_20'] = (df['close'] - df['close'].rolling(20).mean()) / (df['close'].rolling(20).std() + 1e-10)
        
        # 4. 價格範圍
        df['price_range_20'] = (df['close'].rolling(20).max() - df['close'].rolling(20).min()) / (df['close'].rolling(20).mean() + 1e-10)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        時間特徵
        """
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            
            # 交易時段 (0: 亞洲, 1: 歐洲, 2: 美洲)
            df['trading_session'] = df['hour'].apply(lambda x: 0 if x < 8 else (1 if x < 16 else 2))
        
        return df
    
    def get_feature_names(self, exclude_cols: List[str] = None) -> List[str]:
        """
        獲取特徵欄位名稱
        """
        if exclude_cols is None:
            exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label']
        
        return exclude_cols
