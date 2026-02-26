#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11 標籤系統 - 基於 ZigZag 反轉點
"""

import pandas as pd
import numpy as np


def create_v11_labels(
    df: pd.DataFrame,
    lookahead_bars: int = 2,
    tp_multiplier: float = 1.5,
    sl_multiplier: float = 0.5,
    require_rsi_div: bool = True,
    require_volume: bool = False,
    require_sr: bool = False
) -> pd.DataFrame:
    """
    使用 ZigZag 反轉點生成訓練標籤
    
    Parameters:
    -----------
    lookahead_bars : int
        提前 N根K線標記信號
    tp_multiplier : float
        止盈倍數 (相對於 ZigZag 振幅)
    sl_multiplier : float
        止損倍數
    require_rsi_div : bool
        是否需要 RSI 背離確認
    require_volume : bool
        是否需要量能確認
    require_sr : bool
        是否需要在支撐/阻力位
    
    Returns:
    --------
    DataFrame 增加:
        - label: 1 (long), -1 (short), 0 (no signal)
        - target_tp: 目標止盈價
        - target_sl: 目標止損價
        - signal_strength: 信號強度 (0-1)
    """
    
    df = df.copy()
    
    df['label'] = 0
    df['target_tp'] = np.nan
    df['target_sl'] = np.nan
    df['signal_strength'] = 0.0
    
    if 'pivot_type' not in df.columns:
        return df
    
    # 找所有反轉點
    pivot_indices = df[df['pivot_type'].notna()].index
    
    for pivot_idx in pivot_indices:
        pivot_type = df.loc[pivot_idx, 'pivot_type']
        pivot_price = df.loc[pivot_idx, 'pivot_price']
        swing_pct = df.loc[pivot_idx, 'zigzag_swing']
        
        # 跳過太小的反轉
        if swing_pct < 1.0:  # < 1%
            continue
        
        # 確認條件
        if not check_confirmation(
            df, pivot_idx, pivot_type,
            require_rsi_div, require_volume, require_sr
        ):
            continue
        
        # 計算信號強度
        strength = calculate_signal_strength(
            df, pivot_idx, pivot_type, swing_pct
        )
        
        # 計算 TP/SL
        if pivot_type == 'low':
            # Long 信號
            tp_distance = (swing_pct / 100) * tp_multiplier
            sl_distance = (swing_pct / 100) * sl_multiplier
            
            tp_price = pivot_price * (1 + tp_distance)
            sl_price = pivot_price * (1 - sl_distance)
            
            signal_label = 1
        
        else:  # pivot_type == 'high'
            # Short 信號
            tp_distance = (swing_pct / 100) * tp_multiplier
            sl_distance = (swing_pct / 100) * sl_multiplier
            
            tp_price = pivot_price * (1 - tp_distance)
            sl_price = pivot_price * (1 + sl_distance)
            
            signal_label = -1
        
        # 標記提前 N 根K線
        signal_idx = max(0, pivot_idx - lookahead_bars)
        
        if signal_idx in df.index:
            df.loc[signal_idx, 'label'] = signal_label
            df.loc[signal_idx, 'target_tp'] = tp_price
            df.loc[signal_idx, 'target_sl'] = sl_price
            df.loc[signal_idx, 'signal_strength'] = strength
    
    return df


def check_confirmation(
    df: pd.DataFrame,
    idx: int,
    pivot_type: str,
    require_rsi_div: bool,
    require_volume: bool,
    require_sr: bool
) -> bool:
    """
    檢查反轉信號確認條件
    """
    
    if idx not in df.index:
        return False
    
    # RSI 背離確認
    if require_rsi_div:
        if pivot_type == 'low':
            if df.loc[idx, 'rsi_bullish_div'] != 1:
                return False
        else:
            if df.loc[idx, 'rsi_bearish_div'] != 1:
                return False
    
    # 量能確認
    if require_volume:
        if pivot_type == 'low':
            if df.loc[idx, 'volume_bullish_div'] != 1:
                return False
        else:
            if df.loc[idx, 'volume_bearish_div'] != 1:
                return False
    
    # 支撐/阻力確認
    if require_sr:
        if pivot_type == 'low':
            if df.loc[idx, 'near_support'] != 1:
                return False
        else:
            if df.loc[idx, 'near_resistance'] != 1:
                return False
    
    return True


def calculate_signal_strength(
    df: pd.DataFrame,
    idx: int,
    pivot_type: str,
    swing_pct: float
) -> float:
    """
    計算信號強度 (0-1)
    """
    
    strength = 0.0
    
    if idx not in df.index:
        return strength
    
    # 1. 振幅越大,強度越高 (max 0.3)
    swing_score = min(swing_pct / 10.0, 1.0) * 0.3
    strength += swing_score
    
    # 2. RSI 背離 (+0.2)
    if pivot_type == 'low' and df.loc[idx, 'rsi_bullish_div'] == 1:
        strength += 0.2
    elif pivot_type == 'high' and df.loc[idx, 'rsi_bearish_div'] == 1:
        strength += 0.2
    
    # 3. MACD 交叉 (+0.15)
    if pivot_type == 'low' and df.loc[idx, 'macd_bullish_cross'] == 1:
        strength += 0.15
    elif pivot_type == 'high' and df.loc[idx, 'macd_bearish_cross'] == 1:
        strength += 0.15
    
    # 4. 布林帶反彈 (+0.15)
    if pivot_type == 'low' and df.loc[idx, 'bb_bounce_up'] == 1:
        strength += 0.15
    elif pivot_type == 'high' and df.loc[idx, 'bb_bounce_down'] == 1:
        strength += 0.15
    
    # 5. 支撐/阻力 (+0.2)
    if pivot_type == 'low' and df.loc[idx, 'near_support'] == 1:
        strength += 0.2
    elif pivot_type == 'high' and df.loc[idx, 'near_resistance'] == 1:
        strength += 0.2
    
    return min(strength, 1.0)
