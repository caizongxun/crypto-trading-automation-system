"""
V4 Label Generator
標籤生成器 - 包含勝率和賠率目標
"""
import pandas as pd
import numpy as np
from typing import Dict

class LabelGenerator:
    """
    V4標籤生成器
    
    除了生成方向標籤(-1/0/1)外,
    還計算每筆交易的期望勝率和賠率供LSTM學習
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.forward_window = config.get('forward_window', 8)
        self.atr_profit_multiplier = config.get('atr_profit_multiplier', 0.7)
        self.atr_loss_multiplier = config.get('atr_loss_multiplier', 1.5)
        self.min_volume_ratio = config.get('min_volume_ratio', 0.7)
        self.min_trend_strength = config.get('min_trend_strength', 0.15)
        self.max_atr_ratio = config.get('max_atr_ratio', 0.08)
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成標籤、勝率目標、賠率目標
        
        Returns:
            df with 'label', 'target_win_rate', 'target_payoff'
        """
        df = df.copy()
        
        print(f"\n[V4標籤生成] 開始...")
        
        # 1. 計算輔助特徵
        df = self._calculate_helper_features(df)
        
        # 2. 生成方向標籤
        labels = []
        win_rates = []
        payoffs = []
        
        for i in range(len(df) - self.forward_window):
            label, win_rate, payoff = self._label_single_bar(df, i)
            labels.append(label)
            win_rates.append(win_rate)
            payoffs.append(payoff)
        
        # 補齊最後幾根
        labels.extend([0] * self.forward_window)
        win_rates.extend([0.5] * self.forward_window)
        payoffs.extend([1.5] * self.forward_window)
        
        df['label'] = labels
        df['target_win_rate'] = win_rates
        df['target_payoff'] = payoffs
        
        long_count = (df['label'] == 1).sum()
        short_count = (df['label'] == -1).sum()
        valid_rate = (long_count + short_count) / len(df)
        
        print(f"[OK] 做多標籤: {long_count} ({long_count/len(df)*100:.1f}%)")
        print(f"[OK] 做空標籤: {short_count} ({short_count/len(df)*100:.1f}%)")
        print(f"[OK] 有效標籤率: {valid_rate*100:.1f}%")
        print(f"[OK] 平均目標勝率: {df['target_win_rate'].mean():.3f}")
        print(f"[OK] 平均目標賠率: {df['target_payoff'].mean():.3f}")
        
        return df
    
    def _calculate_helper_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算輔助特徵"""
        # 成交量比率
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-10)
        
        # 趨勢強度
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['trend_strength'] = (df['ema_9'] - df['ema_21']) / (df['ema_21'] + 1e-10)
        
        # ATR比率
        df['atr_ratio'] = df['atr_14'] / (df['close'] + 1e-10)
        
        return df
    
    def _label_single_bar(self, df: pd.DataFrame, idx: int) -> tuple:
        """
        為單根K線生成標籤
        
        Returns:
            (label, target_win_rate, target_payoff)
        """
        current_price = df.iloc[idx]['close']
        atr = df.iloc[idx]['atr_14']
        volume_ratio = df.iloc[idx]['volume_ratio']
        trend_strength = df.iloc[idx]['trend_strength']
        atr_ratio = df.iloc[idx]['atr_ratio']
        
        # 過濾條件
        if volume_ratio < self.min_volume_ratio:
            return 0, 0.5, 1.5
        
        if abs(trend_strength) < self.min_trend_strength:
            return 0, 0.5, 1.5
        
        if atr_ratio > self.max_atr_ratio:
            return 0, 0.5, 1.5
        
        # 未來價格
        future_highs = df.iloc[idx+1:idx+1+self.forward_window]['high'].values
        future_lows = df.iloc[idx+1:idx+1+self.forward_window]['low'].values
        
        if len(future_highs) < self.forward_window:
            return 0, 0.5, 1.5
        
        # 做多檢查
        profit_target_long = current_price + atr * self.atr_profit_multiplier
        loss_target_long = current_price - atr * self.atr_loss_multiplier
        
        hit_profit_long = np.any(future_highs >= profit_target_long)
        hit_loss_long = np.any(future_lows <= loss_target_long)
        
        # 做空檢查
        profit_target_short = current_price - atr * self.atr_profit_multiplier
        loss_target_short = current_price + atr * self.atr_loss_multiplier
        
        hit_profit_short = np.any(future_lows <= profit_target_short)
        hit_loss_short = np.any(future_highs >= loss_target_short)
        
        # 計算勝率和賠率
        # 勝率: 基於歷史趨勢強度
        base_win_rate = 0.55
        if abs(trend_strength) > 0.5:
            base_win_rate = 0.65
        elif abs(trend_strength) > 0.3:
            base_win_rate = 0.60
        
        # 賠率: 基於ATR倍數
        payoff = self.atr_profit_multiplier / self.atr_loss_multiplier
        
        # 做多標籤
        if hit_profit_long and not hit_loss_long and trend_strength > 0:
            return 1, base_win_rate, payoff
        
        # 做空標籤
        if hit_profit_short and not hit_loss_short and trend_strength < 0:
            return -1, base_win_rate, payoff
        
        return 0, 0.5, 1.5
