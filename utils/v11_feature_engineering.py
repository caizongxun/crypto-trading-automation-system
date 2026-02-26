#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11 特徵工程 - 專注於反轉交易
"""

import pandas as pd
import numpy as np
import ta


def create_v11_features(df: pd.DataFrame, feature_config: dict) -> pd.DataFrame:
    """
    創建 V11 特徵
    
    Parameters:
    -----------
    feature_config : dict
        {'price': bool, 'volume': bool, 'trend': bool, 'reversal': bool, 'pattern': bool}
    """
    
    df = df.copy()
    
    features = []
    
    # 1. 價格特徵
    if feature_config.get('price', True):
        df = add_price_features(df)
        features.extend([
            'returns', 'log_returns', 'high_low_range', 'close_open_range',
            'upper_shadow', 'lower_shadow', 'body_size'
        ])
    
    # 2. 量能特徵
    if feature_config.get('volume', True):
        df = add_volume_features(df)
        features.extend([
            'volume_ratio', 'volume_ma_ratio', 'volume_std',
            'obv', 'obv_ma', 'vwap'
        ])
    
    # 3. 趨勢特徵
    if feature_config.get('trend', True):
        df = add_trend_features(df)
        features.extend([
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'adx', 'adx_pos', 'adx_neg',
            'price_vs_sma20', 'price_vs_sma50'
        ])
    
    # 4. 反轉特徵 (核心!)
    if feature_config.get('reversal', True):
        features.extend([
            'rsi', 'rsi_bullish_div', 'rsi_bearish_div',
            'macd', 'macd_signal', 'macd_hist',
            'macd_bullish_cross', 'macd_bearish_cross',
            'bb_high', 'bb_low', 'bb_width',
            'bb_bounce_up', 'bb_bounce_down',
            'volume_bullish_div', 'volume_bearish_div',
            'near_support', 'near_resistance'
        ])
    
    # 5. 型態特徵
    if feature_config.get('pattern', True):
        df = add_pattern_features(df)
        features.extend([
            'doji', 'hammer', 'shooting_star',
            'engulfing_bullish', 'engulfing_bearish',
            'three_consecutive_up', 'three_consecutive_down'
        ])
    
    # 在 ZigZag 特徵
    df['has_pivot'] = df['pivot_type'].notna().astype(int)
    df['is_high_pivot'] = (df['pivot_type'] == 'high').astype(int)
    df['is_low_pivot'] = (df['pivot_type'] == 'low').astype(int)
    df['pivot_swing_pct'] = df['zigzag_swing'].fillna(0)
    
    features.extend(['has_pivot', 'is_high_pivot', 'is_low_pivot', 'pivot_swing_pct'])
    
    # 儲存特徵列表
    df.attrs['feature_columns'] = features
    
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['close_open_range'] = (df['close'] - df['open']) / df['open']
    
    # K線形態
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
    df['body_size'] = abs(df['close'] - df['open']) / df['close']
    
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_std'] = df['volume'].rolling(20).std()
    
    # OBV
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['obv_ma'] = df['obv'].rolling(20).mean()
    
    # VWAP
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    return df


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    # 移動平均
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    
    # ADX
    adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
    df['adx'] = adx.adx()
    df['adx_pos'] = adx.adx_pos()
    df['adx_neg'] = adx.adx_neg()
    
    # 價格相對位置
    df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
    
    return df


def add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """
K線型態
    """
    
    # Doji
    df['doji'] = (abs(df['close'] - df['open']) / (df['high'] - df['low']) < 0.1).astype(int)
    
    # Hammer
    df['hammer'] = (
        ((df['high'] - df['low']) > 3 * abs(df['close'] - df['open'])) &
        ((df['close'] - df['low']) / (0.001 + df['high'] - df['low']) > 0.6)
    ).astype(int)
    
    # Shooting Star
    df['shooting_star'] = (
        ((df['high'] - df['low']) > 3 * abs(df['close'] - df['open'])) &
        ((df['high'] - df['close']) / (0.001 + df['high'] - df['low']) > 0.6)
    ).astype(int)
    
    # Engulfing
    df['engulfing_bullish'] = (
        (df['close'] > df['open']) &
        (df['close'].shift(1) < df['open'].shift(1)) &
        (df['close'] > df['open'].shift(1)) &
        (df['open'] < df['close'].shift(1))
    ).astype(int)
    
    df['engulfing_bearish'] = (
        (df['close'] < df['open']) &
        (df['close'].shift(1) > df['open'].shift(1)) &
        (df['close'] < df['open'].shift(1)) &
        (df['open'] > df['close'].shift(1))
    ).astype(int)
    
    # 連續上漲/下跌
    df['three_consecutive_up'] = (
        (df['close'] > df['close'].shift(1)) &
        (df['close'].shift(1) > df['close'].shift(2)) &
        (df['close'].shift(2) > df['close'].shift(3))
    ).astype(int)
    
    df['three_consecutive_down'] = (
        (df['close'] < df['close'].shift(1)) &
        (df['close'].shift(1) < df['close'].shift(2)) &
        (df['close'].shift(2) < df['close'].shift(3))
    ).astype(int)
    
    return df
