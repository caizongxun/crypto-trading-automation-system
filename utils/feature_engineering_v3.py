#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Engineering v3

完整版特徵工程，用於混合回測 (29個特徵)
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging
import talib

logger = logging.getLogger(__name__)


def engineer_features_v3(
    df: pd.DataFrame,
    symbol: str = 'BTCUSDT',
    timeframe: str = '1h'
) -> pd.DataFrame:
    """
    工程化特徵 v3 (29個特徵)
    
    Args:
        df: K線數據
        symbol: 交易對
        timeframe: 時間周期
    
    Returns:
        特徵 DataFrame
    """
    logger.info(f"Engineering features v3 for {symbol} {timeframe}...")
    
    df_features = pd.DataFrame(index=df.index)
    
    # 1. Returns系列 (4個)
    df_features['returns_5m'] = df['close'].pct_change(5)
    df_features['returns_15m'] = df['close'].pct_change(15)
    df_features['returns_30m'] = df['close'].pct_change(30)
    df_features['returns_1h'] = df['close'].pct_change(60) if timeframe != '1h' else df['close'].pct_change(1)
    
    # 2. Price Position (2個)
    high_1h = df['high'].rolling(60 if timeframe != '1h' else 1).max()
    low_1h = df['low'].rolling(60 if timeframe != '1h' else 1).min()
    df_features['price_position_1h'] = (df['close'] - low_1h) / (high_1h - low_1h + 1e-8)
    
    high_4h = df['high'].rolling(240 if timeframe != '1h' else 4).max()
    low_4h = df['low'].rolling(240 if timeframe != '1h' else 4).min()
    df_features['price_position_4h'] = (df['close'] - low_4h) / (high_4h - low_4h + 1e-8)
    
    # 3. ATR (2個)
    atr_14 = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    atr_60 = talib.ATR(df['high'], df['low'], df['close'], timeperiod=60)
    df_features['atr_pct_14'] = atr_14 / df['close']
    df_features['atr_pct_60'] = atr_60 / df['close']
    
    # 4. Volatility (2個)
    df_features['vol_ratio'] = atr_14 / (atr_60 + 1e-8)
    df_features['vol_expanding'] = (atr_14 > atr_60).astype(int)
    
    # 5. Trend EMA (5個)
    ema9 = talib.EMA(df['close'], timeperiod=9)
    ema21 = talib.EMA(df['close'], timeperiod=21)
    ema50 = talib.EMA(df['close'], timeperiod=50)
    
    df_features['trend_9_21'] = (ema9 - ema21) / ema21
    df_features['trend_21_50'] = (ema21 - ema50) / ema50
    df_features['above_ema9'] = (df['close'] > ema9).astype(int)
    df_features['above_ema21'] = (df['close'] > ema21).astype(int)
    df_features['above_ema50'] = (df['close'] > ema50).astype(int)
    
    # 6. Volume (3個)
    vol_sma = df['volume'].rolling(20).mean()
    df_features['volume_ratio'] = df['volume'] / (vol_sma + 1e-8)
    df_features['volume_trend'] = df['volume'].rolling(5).mean() / (df['volume'].rolling(20).mean() + 1e-8)
    df_features['high_volume'] = (df['volume'] > vol_sma * 1.5).astype(int)
    
    # 7. Candle (3個)
    body = abs(df['close'] - df['open'])
    full_range = df['high'] - df['low'] + 1e-8
    df_features['body_pct'] = body / full_range
    df_features['bullish_candle'] = (df['close'] > df['open']).astype(int)
    df_features['bearish_candle'] = (df['close'] < df['open']).astype(int)
    
    # 8. Pressure (2個)
    # Buying/Selling pressure
    buy_pressure = (df['close'] - df['low']) / full_range
    sell_pressure = (df['high'] - df['close']) / full_range
    df_features['pressure_ratio_30m'] = buy_pressure.rolling(30).sum() / (sell_pressure.rolling(30).sum() + 1e-8)
    
    # Green streak
    green = (df['close'] > df['open']).astype(int)
    df_features['green_streak'] = green.rolling(5).sum()
    
    # 9. RSI (3個)
    rsi = talib.RSI(df['close'], timeperiod=14)
    df_features['rsi_14'] = rsi / 100.0  # Normalize
    df_features['rsi_oversold'] = (rsi < 30).astype(int)
    df_features['rsi_overbought'] = (rsi > 70).astype(int)
    
    # 10. Time (3個) - 交易時段
    # 假設index是 DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
    elif 'open_time' in df.columns:
        hour = pd.to_datetime(df['open_time']).dt.hour
    else:
        hour = pd.Series(0, index=df.index)  # Fallback
    
    df_features['is_asian'] = ((hour >= 0) & (hour < 8)).astype(int)
    df_features['is_london'] = ((hour >= 8) & (hour < 16)).astype(int)
    df_features['is_nyc'] = ((hour >= 16) & (hour < 24)).astype(int)
    
    # 填充 NaN
    df_features = df_features.fillna(0)
    
    # 確保正確的特徵順序 (29個)
    feature_order = [
        'returns_5m', 'returns_15m', 'returns_30m', 'returns_1h',
        'price_position_1h', 'price_position_4h',
        'atr_pct_14', 'atr_pct_60',
        'vol_ratio', 'vol_expanding',
        'trend_9_21', 'trend_21_50',
        'above_ema9', 'above_ema21', 'above_ema50',
        'volume_ratio', 'volume_trend', 'high_volume',
        'body_pct', 'bullish_candle', 'bearish_candle',
        'pressure_ratio_30m', 'green_streak',
        'rsi_14', 'rsi_oversold', 'rsi_overbought',
        'is_asian', 'is_london', 'is_nyc'
    ]
    
    df_features = df_features[feature_order]
    
    logger.info(f"[OK] Features engineered: {len(df_features.columns)} columns")
    
    return df_features


if __name__ == "__main__":
    # 測試
    import sys
    sys.path.append('..')
    from utils.hf_data_loader import load_data_from_hf
    
    df = load_data_from_hf('BTCUSDT', '1h', days=30)
    features = engineer_features_v3(df)
    
    print(f"\nFeatures shape: {features.shape}")
    print(f"\nFeatures columns ({len(features.columns)}):個):")
    print(features.columns.tolist())
    print(f"\nFirst 5 rows:")
    print(features.head())
    print(f"\nLast 5 rows:")
    print(features.tail())
    print(f"\nNull counts:")
    print(features.isnull().sum())
