#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11 反轉指標 - RSI背離, MACD交叉, 支撐阻力等
"""

import pandas as pd
import numpy as np
import ta


def calculate_reversal_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算所有反轉相關指標
    """
    
    df = df.copy()
    
    # 1. RSI 背離
    df = calculate_rsi_divergence(df)
    
    # 2. MACD 交叉與背離
    df = calculate_macd_signals(df)
    
    # 3. 布林帶反彈
    df = calculate_bb_reversal(df)
    
    # 4. 量能發散
    df = calculate_volume_divergence(df)
    
    # 5. 支撐阻力
    df = calculate_support_resistance(df)
    
    return df


def calculate_rsi_divergence(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    計算 RSI 背離
    
    Regular Bullish: 價格低點更低, RSI低點更高
    Regular Bearish: 價格高點更高, RSI高點更低
    """
    
    # 計算 RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
    
    df['rsi_bullish_div'] = 0
    df['rsi_bearish_div'] = 0
    
    lookback = 20
    
    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i+1]
        
        # 找近期的高低點
        recent_low_idx = window['low'].idxmin()
        recent_high_idx = window['high'].idxmax()
        
        # Bullish Divergence
        if i > recent_low_idx:
            prev_low_window = df.iloc[max(0, recent_low_idx-lookback):recent_low_idx]
            if len(prev_low_window) > 0:
                prev_low_idx = prev_low_window['low'].idxmin()
                
                if (df.loc[i, 'low'] < df.loc[prev_low_idx, 'low'] and 
                    df.loc[i, 'rsi'] > df.loc[prev_low_idx, 'rsi']):
                    df.loc[i, 'rsi_bullish_div'] = 1
        
        # Bearish Divergence
        if i > recent_high_idx:
            prev_high_window = df.iloc[max(0, recent_high_idx-lookback):recent_high_idx]
            if len(prev_high_window) > 0:
                prev_high_idx = prev_high_window['high'].idxmax()
                
                if (df.loc[i, 'high'] > df.loc[prev_high_idx, 'high'] and 
                    df.loc[i, 'rsi'] < df.loc[prev_high_idx, 'rsi']):
                    df.loc[i, 'rsi_bearish_div'] = 1
    
    return df


def calculate_macd_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    MACD 交叉與背離
    """
    
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    
    # 交叉信號
    df['macd_bullish_cross'] = ((df['macd'] > df['macd_signal']) & 
                                 (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
    
    df['macd_bearish_cross'] = ((df['macd'] < df['macd_signal']) & 
                                 (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
    
    return df


def calculate_bb_reversal(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    """
    布林帶反彈信號
    """
    
    bb = ta.volatility.BollingerBands(df['close'], window=period, window_dev=std)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_low'] = bb.bollinger_lband()
    
    # 觸碰下軌反彈
    df['bb_bounce_up'] = ((df['low'] <= df['bb_low']) & 
                          (df['close'] > df['bb_low'])).astype(int)
    
    # 觸碰上軌反彈
    df['bb_bounce_down'] = ((df['high'] >= df['bb_high']) & 
                            (df['close'] < df['bb_high'])).astype(int)
    
    return df


def calculate_volume_divergence(df: pd.DataFrame) -> pd.DataFrame:
    """
    量能發散 - 價格上漲但量能衰減
    """
    
    df['volume_ma'] = df['volume'].rolling(20).mean()
    
    # 量能發散信號
    df['volume_bullish_div'] = 0
    df['volume_bearish_div'] = 0
    
    for i in range(20, len(df)):
        # Bearish: 價格新高但量能減少
        if (df['high'].iloc[i] > df['high'].iloc[i-10:i].max() and 
            df['volume'].iloc[i] < df['volume'].iloc[i-10:i].mean()):
            df.loc[df.index[i], 'volume_bearish_div'] = 1
        
        # Bullish: 價格新低但量能增加
        if (df['low'].iloc[i] < df['low'].iloc[i-10:i].min() and 
            df['volume'].iloc[i] > df['volume'].iloc[i-10:i].mean()):
            df.loc[df.index[i], 'volume_bullish_div'] = 1
    
    return df


def calculate_support_resistance(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """
    支撐阻力位識別
    """
    
    df['near_support'] = 0
    df['near_resistance'] = 0
    
    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i]
        
        # 找支撐位 (近期低點)
        support_levels = window.nsmallest(3, 'low')['low'].values
        support = np.mean(support_levels)
        
        # 找阻力位 (近期高點)
        resistance_levels = window.nlargest(3, 'high')['high'].values
        resistance = np.mean(resistance_levels)
        
        current_price = df['close'].iloc[i]
        
        # 判斷是否接近支撐/阻力 (2% 範圍內)
        if abs(current_price - support) / support < 0.02:
            df.loc[df.index[i], 'near_support'] = 1
        
        if abs(current_price - resistance) / resistance < 0.02:
            df.loc[df.index[i], 'near_resistance'] = 1
    
    return df
