#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11 ZigZag 計算 - 識別重要的高低點反轉
"""

import pandas as pd
import numpy as np


def calculate_zigzag_pivots(df: pd.DataFrame, threshold_pct: float = 3.0) -> pd.DataFrame:
    """
    計算 ZigZag 反轉點
    
    Parameters:
    -----------
    df : DataFrame
        必須包含 'high', 'low', 'close' 欄位
    threshold_pct : float
        反轉門檼百分比 (e.g., 3.0 = 3%)
    
    Returns:
    --------
    DataFrame 增加以下欄位:
        - pivot_type: 'high' | 'low' | None
        - pivot_price: 反轉點價格
        - pivot_idx: 反轉點索引
        - zigzag_swing: 反轉振幅 (%)
    """
    
    df = df.copy()
    
    # 初始化
    df['pivot_type'] = None
    df['pivot_price'] = np.nan
    df['pivot_idx'] = -1
    df['zigzag_swing'] = 0.0
    
    if len(df) < 10:
        return df
    
    # ZigZag 核心算法
    pivots = []
    
    # 找第一個反轉點 (使用前 10 根K線)
    first_high = df['high'].iloc[:10].max()
    first_low = df['low'].iloc[:10].min()
    first_high_idx = df['high'].iloc[:10].idxmax()
    first_low_idx = df['low'].iloc[:10].idxmin()
    
    # 決定起始方向
    if first_high_idx < first_low_idx:
        # 先出現高點
        current_trend = 'down'
        last_pivot_price = first_high
        last_pivot_idx = first_high_idx
        last_pivot_type = 'high'
    else:
        # 先出現低點
        current_trend = 'up'
        last_pivot_price = first_low
        last_pivot_idx = first_low_idx
        last_pivot_type = 'low'
    
    pivots.append({
        'idx': last_pivot_idx,
        'type': last_pivot_type,
        'price': last_pivot_price,
        'swing': 0.0
    })
    
    # 遍歷數據尋找反轉點
    threshold = threshold_pct / 100
    
    for i in range(10, len(df)):
        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        
        if current_trend == 'up':
            # 上升趨勢,尋找高點
            if current_high > last_pivot_price:
                # 更新高點
                last_pivot_price = current_high
                last_pivot_idx = i
            
            # 檢查是否回撤超過門檼
            drop_pct = (last_pivot_price - current_low) / last_pivot_price
            
            if drop_pct >= threshold:
                # 確認高點,轉向下降
                swing_pct = drop_pct * 100
                pivots.append({
                    'idx': last_pivot_idx,
                    'type': 'high',
                    'price': last_pivot_price,
                    'swing': swing_pct
                })
                
                current_trend = 'down'
                last_pivot_price = current_low
                last_pivot_idx = i
                last_pivot_type = 'low'
        
        else:  # current_trend == 'down'
            # 下降趨勢,尋找低點
            if current_low < last_pivot_price:
                # 更新低點
                last_pivot_price = current_low
                last_pivot_idx = i
            
            # 檢查是否反彈超過門檼
            rise_pct = (current_high - last_pivot_price) / last_pivot_price
            
            if rise_pct >= threshold:
                # 確認低點,轉向上升
                swing_pct = rise_pct * 100
                pivots.append({
                    'idx': last_pivot_idx,
                    'type': 'low',
                    'price': last_pivot_price,
                    'swing': swing_pct
                })
                
                current_trend = 'up'
                last_pivot_price = current_high
                last_pivot_idx = i
                last_pivot_type = 'high'
    
    # 將反轉點標記到 DataFrame
    for pivot in pivots:
        idx = pivot['idx']
        if idx in df.index:
            df.loc[idx, 'pivot_type'] = pivot['type']
            df.loc[idx, 'pivot_price'] = pivot['price']
            df.loc[idx, 'pivot_idx'] = idx
            df.loc[idx, 'zigzag_swing'] = pivot['swing']
    
    return df


def get_zigzag_trend(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    根據 ZigZag 判斷當前趨勢
    
    Returns:
    --------
    Series: 1 (up), -1 (down), 0 (neutral)
    """
    
    trend = pd.Series(0, index=df.index)
    
    if 'pivot_type' not in df.columns:
        return trend
    
    last_high_idx = None
    last_low_idx = None
    
    for i in range(len(df)):
        if df['pivot_type'].iloc[i] == 'high':
            last_high_idx = i
        elif df['pivot_type'].iloc[i] == 'low':
            last_low_idx = i
        
        # 判斷趨勢
        if last_high_idx is not None and last_low_idx is not None:
            if last_low_idx > last_high_idx:
                trend.iloc[i] = 1  # 上升趨勢
            else:
                trend.iloc[i] = -1  # 下降趨勢
    
    return trend
