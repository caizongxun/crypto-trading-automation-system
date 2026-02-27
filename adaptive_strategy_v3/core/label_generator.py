"""
V3 Label Generator - ATR Dynamic Labels with Quality Filters
ATR動態標籤生成器
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple

class LabelGenerator:
    """
    生成高質量交易標籤
    
    核心改進:
    1. ATR動態調整利潤要求
    2. 成交量確認
    3. 趨勢強度過濾
    4. 交易成本考慮
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 基礎參數
        self.forward_window = config.get('forward_window', 8)
        self.cost_buffer = config.get('cost_buffer', 0.002)  # 0.2% 交易成本緩衝
        
        # ATR動態調整
        self.atr_profit_multiplier = config.get('atr_profit_multiplier', 1.5)
        self.atr_loss_multiplier = config.get('atr_loss_multiplier', 0.8)
        
        # 質量過濾閾值
        self.min_volume_ratio = config.get('min_volume_ratio', 1.2)  # 成交量至少是平均的1.2倍
        self.min_trend_strength = config.get('min_trend_strength', 0.5)  # 趨勢強度閾值
        self.max_atr_ratio = config.get('max_atr_ratio', 0.05)  # ATR不超過價格5%
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易標籤
        
        Returns:
            df with 'label' column: -1 (做空), 0 (中立), 1 (做多)
        """
        df = df.copy()
        df['label'] = 0
        
        # 確保必要欄位存在
        required_cols = ['close', 'high', 'low', 'volume', 'atr_14']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # 計算輔助指標
        df = self._calculate_helper_features(df)
        
        # 逐行生成標籤
        for i in range(len(df) - self.forward_window):
            current_price = df.iloc[i]['close']
            atr = df.iloc[i]['atr_14']
            
            # 跳過異常值
            if pd.isna(atr) or atr == 0:
                continue
            
            # 質量過濾 - 先檢查是否值得分析
            if not self._pass_quality_filter(df, i):
                continue
            
            # 計算動態閾值
            min_profit = atr * self.atr_profit_multiplier + self.cost_buffer
            max_loss = atr * self.atr_loss_multiplier
            
            # 前瞻窗口分析
            future_prices = df.iloc[i+1:i+self.forward_window+1]['close'].values
            future_high = df.iloc[i+1:i+self.forward_window+1]['high'].max()
            future_low = df.iloc[i+1:i+self.forward_window+1]['low'].min()
            
            # 做多標籤判斷
            max_profit = future_high - current_price
            max_drawdown = current_price - future_low
            
            if max_profit >= min_profit and max_drawdown < max_loss:
                # 額外確認: 實際能獲利平倉
                if self._can_take_profit(future_prices, current_price, min_profit):
                    df.loc[df.index[i], 'label'] = 1
                    continue
            
            # 做空標籤判斷
            short_profit = current_price - future_low
            short_loss = future_high - current_price
            
            if short_profit >= min_profit and short_loss < max_loss:
                if self._can_take_profit(future_prices, current_price, -min_profit):
                    df.loc[df.index[i], 'label'] = -1
        
        # 統計標籤分布
        self._print_label_stats(df)
        
        return df
    
    def _calculate_helper_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算輔助特徵用於質量過濾
        """
        # 成交量均線
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        
        # 趨勢強度 (10根K線方向一致性)
        df['price_change'] = df['close'].diff()
        df['trend_strength'] = df['price_change'].rolling(10).apply(
            lambda x: (x > 0).sum() / 10 if len(x) == 10 else 0.5
        )
        # 轉換為 -1 到 1 的範圍
        df['trend_strength'] = df['trend_strength'] * 2 - 1
        
        # ATR相對價格比例
        df['atr_ratio'] = df['atr_14'] / df['close']
        
        return df
    
    def _pass_quality_filter(self, df: pd.DataFrame, idx: int) -> bool:
        """
        質量過濾器
        
        確保只在高質量條件下生成標籤:
        1. 成交量充足
        2. 趨勢明確
        3. 波動率合理
        """
        row = df.iloc[idx]
        
        # 檢查1: 成交量
        if pd.isna(row['volume_ma_20']) or row['volume_ma_20'] == 0:
            return False
        
        volume_ratio = row['volume'] / row['volume_ma_20']
        if volume_ratio < self.min_volume_ratio:
            return False
        
        # 檢查2: 趨勢強度
        if pd.isna(row['trend_strength']):
            return False
        
        if abs(row['trend_strength']) < self.min_trend_strength:
            return False
        
        # 檢查3: ATR比例 (避免極端波動)
        if pd.isna(row['atr_ratio']):
            return False
        
        if row['atr_ratio'] > self.max_atr_ratio:
            return False
        
        return True
    
    def _can_take_profit(self, future_prices: np.ndarray, entry_price: float, 
                        target_profit: float) -> bool:
        """
        確認是否實際能夠在目標利潤平倉
        
        避免標籤虛假: 價格雖然達到目標,但立即回撤
        """
        if target_profit > 0:  # 做多
            target_price = entry_price + target_profit
            # 檢查是否有價格達到目標
            reached = future_prices >= target_price
        else:  # 做空
            target_price = entry_price + target_profit  # target_profit是負數
            reached = future_prices <= target_price
        
        if not reached.any():
            return False
        
        # 確認達到目標後,價格維持至少1-2根K線
        first_reach_idx = np.where(reached)[0][0]
        if first_reach_idx >= len(future_prices) - 1:
            return True  # 窗口末尾達到,認為有效
        
        # 檢查達到後是否立即大幅回撤
        next_prices = future_prices[first_reach_idx:first_reach_idx+2]
        if target_profit > 0:
            # 做多: 檢查後續是否跌破95%利潤
            min_next = next_prices.min()
            if min_next < entry_price + target_profit * 0.95:
                return False
        else:
            # 做空: 檢查後續是否漲破95%利潤
            max_next = next_prices.max()
            if max_next > entry_price + target_profit * 0.95:
                return False
        
        return True
    
    def _print_label_stats(self, df: pd.DataFrame):
        """
        打印標籤分布統計
        """
        total = len(df)
        long_count = (df['label'] == 1).sum()
        short_count = (df['label'] == -1).sum()
        neutral_count = (df['label'] == 0).sum()
        
        print("\n[標籤生成統計]")
        print(f"總樣本: {total}")
        print(f"做多標籤: {long_count} ({long_count/total*100:.1f}%)")
        print(f"做空標籤: {short_count} ({short_count/total*100:.1f}%)")
        print(f"中立標籤: {neutral_count} ({neutral_count/total*100:.1f}%)")
        print(f"有效交易標籤: {long_count + short_count} ({(long_count + short_count)/total*100:.1f}%)")
        
        # 警告: 標籤過少
        if (long_count + short_count) / total < 0.05:
            print("[警告] 有效標籤少於5%,可能需要調整參數")
        
        # 警告: 標籤過多
        if (long_count + short_count) / total > 0.30:
            print("[警告] 有效標籤超過30%,質量可能不足")
    
    def get_label_quality_metrics(self, df: pd.DataFrame) -> Dict:
        """
        計算標籤質量指標
        """
        labeled = df[df['label'] != 0]
        
        if len(labeled) == 0:
            return {'error': '沒有有效標籤'}
        
        metrics = {
            'total_labels': len(labeled),
            'long_ratio': (df['label'] == 1).sum() / len(df),
            'short_ratio': (df['label'] == -1).sum() / len(df),
            'avg_volume_ratio': labeled['volume'].mean() / df['volume'].mean(),
            'avg_trend_strength': labeled['trend_strength'].abs().mean(),
            'avg_atr_ratio': labeled['atr_ratio'].mean()
        }
        
        return metrics
