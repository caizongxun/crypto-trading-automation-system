"""
Feature Engineering V3 - Optimized for High Win Rate

Key Improvements:
1. More aggressive label definitions (1.2% target vs 2%)
2. Directional features (separate for long/short)
3. Market regime features
4. Probability-focused design

Author: Zong
Version: 3.0.0
Date: 2026-02-25
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineerV3:
    """
    V3 Feature Engineering - Optimized for practical trading
    
    Design Philosophy:
    - Higher signal frequency (more trades)
    - Better probability calibration
    - Directional bias features
    - Market regime awareness
    """
    
    def __init__(self):
        self.version = "3.0.0"
        print(f"[V3 Feature Engineer] Initialized (v{self.version})")
    
    def create_features_from_1m(self, 
                                df_1m: pd.DataFrame,
                                label_type: str = 'both',
                                tp_target: float = 0.012,  # 1.2% target
                                sl_stop: float = 0.008,    # 0.8% stop
                                lookahead_bars: int = 240  # 4 hours
                               ) -> pd.DataFrame:
        """
        Generate V3 features and labels from 1m data
        
        Args:
            df_1m: 1m OHLCV data
            label_type: 'long', 'short', or 'both'
            tp_target: Take profit target (1.2% default)
            sl_stop: Stop loss (0.8% default)
            lookahead_bars: Forward looking window (240 = 4h)
        
        Returns:
            DataFrame with features and labels
        """
        print(f"\n[V3] Creating features from {len(df_1m):,} 1m bars")
        print(f"[V3] Label config: TP={tp_target*100:.1f}%, SL={sl_stop*100:.1f}%, Lookahead={lookahead_bars}")
        
        df = df_1m.copy()
        
        # CRITICAL FIX: Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'open_time' in df.columns:
                print("[V3] Converting open_time to datetime index...")
                df['open_time'] = pd.to_datetime(df['open_time'])
                df = df.set_index('open_time')
            elif 'timestamp' in df.columns:
                print("[V3] Converting timestamp to datetime index...")
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            else:
                print("[V3 WARNING] No time column found, creating synthetic datetime index")
                df.index = pd.date_range(start='2020-01-01', periods=len(df), freq='1min')
        
        print(f"[V3] Index type: {type(df.index).__name__}")
        
        # ========================================
        # 1. Core Price Features
        # ========================================
        print("[V3] Computing core price features...")
        
        # Price momentum (multiple timeframes)
        df['returns_5m'] = df['close'].pct_change(5)
        df['returns_15m'] = df['close'].pct_change(15)
        df['returns_30m'] = df['close'].pct_change(30)
        df['returns_1h'] = df['close'].pct_change(60)
        
        # Price position (where are we in recent range?)
        df['price_position_1h'] = (df['close'] - df['close'].rolling(60).min()) / \
                                  (df['close'].rolling(60).max() - df['close'].rolling(60).min())
        df['price_position_4h'] = (df['close'] - df['close'].rolling(240).min()) / \
                                  (df['close'].rolling(240).max() - df['close'].rolling(240).min())
        
        # ========================================
        # 2. Volatility Features
        # ========================================
        print("[V3] Computing volatility features...")
        
        # ATR (multiple periods)
        df['high_low'] = df['high'] - df['low']
        df['atr_14'] = df['high_low'].rolling(14).mean()
        df['atr_60'] = df['high_low'].rolling(60).mean()
        df['atr_pct_14'] = df['atr_14'] / df['close']
        df['atr_pct_60'] = df['atr_60'] / df['close']
        
        # Volatility regime
        df['vol_ratio'] = df['atr_pct_14'] / df['atr_pct_60'].replace(0, np.nan)
        df['vol_expanding'] = (df['vol_ratio'] > 1.2).astype(int)  # High vol regime
        
        # ========================================
        # 3. Trend Features
        # ========================================
        print("[V3] Computing trend features...")
        
        # EMAs
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # Trend strength
        df['trend_9_21'] = (df['ema_9'] - df['ema_21']) / df['close']
        df['trend_21_50'] = (df['ema_21'] - df['ema_50']) / df['close']
        
        # Price vs EMAs
        df['above_ema9'] = (df['close'] > df['ema_9']).astype(int)
        df['above_ema21'] = (df['close'] > df['ema_21']).astype(int)
        df['above_ema50'] = (df['close'] > df['ema_50']).astype(int)
        
        # ========================================
        # 4. Volume Features
        # ========================================
        print("[V3] Computing volume features...")
        
        # Volume ratios
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma20'].replace(0, np.nan)
        
        # Volume trend
        df['volume_trend'] = df['volume'].rolling(10).mean() / \
                            df['volume'].rolling(30).mean().replace(0, np.nan)
        
        # High volume bars
        df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)
        
        # ========================================
        # 5. Microstructure Features
        # ========================================
        print("[V3] Computing microstructure features...")
        
        # Candle patterns
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['body_pct'] = df['body'] / df['high_low'].replace(0, np.nan)
        
        # Bullish/Bearish candles
        df['bullish_candle'] = (df['close'] > df['open']).astype(int)
        df['bearish_candle'] = (df['close'] < df['open']).astype(int)
        
        # ========================================
        # 6. Directional Pressure Features
        # ========================================
        print("[V3] Computing directional pressure...")
        
        # Buying/Selling pressure (last 30 mins)
        df['bullish_bars_30m'] = df['bullish_candle'].rolling(30).sum()
        df['bearish_bars_30m'] = df['bearish_candle'].rolling(30).sum()
        df['pressure_ratio_30m'] = df['bullish_bars_30m'] / (df['bearish_bars_30m'] + 1)
        
        # Recent momentum direction
        df['green_streak'] = (df['close'] > df['open']).astype(int)
        df['green_streak'] = df['green_streak'].groupby(
            (df['green_streak'] != df['green_streak'].shift()).cumsum()
        ).cumsum()
        
        # ========================================
        # 7. RSI and Oscillators
        # ========================================
        print("[V3] Computing oscillators...")
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # RSI zones
        df['rsi_oversold'] = (df['rsi_14'] < 35).astype(int)
        df['rsi_overbought'] = (df['rsi_14'] > 65).astype(int)
        
        # ========================================
        # 8. Market Regime Features
        # ========================================
        print("[V3] Computing market regime...")
        
        # Hourly aggregates - NOW SAFE because we have DatetimeIndex
        df['hour'] = df.index.hour
        df['is_asian'] = df['hour'].between(0, 8).astype(int)
        df['is_london'] = df['hour'].between(8, 16).astype(int)
        df['is_nyc'] = df['hour'].between(13, 21).astype(int)
        
        # ========================================
        # 9. Labels (V3 - More Aggressive)
        # ========================================
        print("[V3] Generating labels...")
        
        if label_type in ['long', 'both']:
            df['label_long'] = self._create_label_long_v3(
                df, tp_target, sl_stop, lookahead_bars
            )
        
        if label_type in ['short', 'both']:
            df['label_short'] = self._create_label_short_v3(
                df, tp_target, sl_stop, lookahead_bars
            )
        
        # ========================================
        # 10. Clean and Fill
        # ========================================
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(0)
        
        # Statistics
        if label_type in ['long', 'both']:
            long_rate = (df['label_long'] == 1).sum() / len(df) * 100
            print(f"[V3] Long signals: {(df['label_long']==1).sum():,} ({long_rate:.2f}%)")
        
        if label_type in ['short', 'both']:
            short_rate = (df['label_short'] == 1).sum() / len(df) * 100
            print(f"[V3] Short signals: {(df['label_short']==1).sum():,} ({short_rate:.2f}%)")
        
        print(f"[V3] Features created: {len(df.columns)} columns")
        return df
    
    def _create_label_long_v3(self, df: pd.DataFrame, tp: float, sl: float, 
                              lookahead: int) -> pd.Series:
        """
        V3 Long Label: More aggressive, higher hit rate
        
        Criteria:
        - Hit TP (1.2%) before SL (0.8%) within lookahead
        - OR price > entry + 0.8% at bar 120 (2h) and never hit SL
        """
        labels = pd.Series(0, index=df.index)
        
        for i in range(len(df) - lookahead):
            entry_price = df['close'].iloc[i]
            tp_price = entry_price * (1 + tp)
            sl_price = entry_price * (1 - sl)
            
            future_high = df['high'].iloc[i+1:i+1+lookahead]
            future_low = df['low'].iloc[i+1:i+1+lookahead]
            
            # Check if hit TP before SL
            hit_tp_idx = np.where(future_high >= tp_price)[0]
            hit_sl_idx = np.where(future_low <= sl_price)[0]
            
            if len(hit_tp_idx) > 0:
                if len(hit_sl_idx) == 0 or hit_tp_idx[0] < hit_sl_idx[0]:
                    labels.iloc[i] = 1
            # Additional: Partial profit condition
            elif len(hit_sl_idx) == 0:
                # If never hit SL and up 0.8% at 2h mark
                if len(future_high) >= 120:
                    if future_high.iloc[119] >= entry_price * 1.008:
                        labels.iloc[i] = 1
        
        return labels
    
    def _create_label_short_v3(self, df: pd.DataFrame, tp: float, sl: float,
                               lookahead: int) -> pd.Series:
        """
        V3 Short Label: More aggressive, higher hit rate
        
        Criteria:
        - Hit TP (1.2%) before SL (0.8%) within lookahead
        - OR price < entry - 0.8% at bar 120 (2h) and never hit SL
        """
        labels = pd.Series(0, index=df.index)
        
        for i in range(len(df) - lookahead):
            entry_price = df['close'].iloc[i]
            tp_price = entry_price * (1 - tp)
            sl_price = entry_price * (1 + sl)
            
            future_high = df['high'].iloc[i+1:i+1+lookahead]
            future_low = df['low'].iloc[i+1:i+1+lookahead]
            
            # Check if hit TP before SL
            hit_tp_idx = np.where(future_low <= tp_price)[0]
            hit_sl_idx = np.where(future_high >= sl_price)[0]
            
            if len(hit_tp_idx) > 0:
                if len(hit_sl_idx) == 0 or hit_tp_idx[0] < hit_sl_idx[0]:
                    labels.iloc[i] = 1
            # Additional: Partial profit condition
            elif len(hit_sl_idx) == 0:
                # If never hit SL and down 0.8% at 2h mark
                if len(future_low) >= 120:
                    if future_low.iloc[119] <= entry_price * 0.992:
                        labels.iloc[i] = 1
        
        return labels
    
    def get_feature_list(self) -> list:
        """
        Return list of V3 features (excluding labels)
        """
        features = [
            # Price momentum
            'returns_5m', 'returns_15m', 'returns_30m', 'returns_1h',
            'price_position_1h', 'price_position_4h',
            
            # Volatility
            'atr_pct_14', 'atr_pct_60', 'vol_ratio', 'vol_expanding',
            
            # Trend
            'trend_9_21', 'trend_21_50',
            'above_ema9', 'above_ema21', 'above_ema50',
            
            # Volume
            'volume_ratio', 'volume_trend', 'high_volume',
            
            # Microstructure
            'body_pct', 'bullish_candle', 'bearish_candle',
            
            # Directional pressure
            'pressure_ratio_30m', 'green_streak',
            
            # Oscillators
            'rsi_14', 'rsi_oversold', 'rsi_overbought',
            
            # Market regime
            'is_asian', 'is_london', 'is_nyc'
        ]
        
        return features
    
    def get_version(self) -> str:
        return self.version