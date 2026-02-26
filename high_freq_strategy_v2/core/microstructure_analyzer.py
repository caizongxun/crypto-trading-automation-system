"""
Market Microstructure Analyzer
市場微觀結構分析器 - 檢測訂單流和大單異常
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class MicrostructureAnalyzer:
    """
    分析市場微觀結構特徵:
    1. 訂單簿失衡 (Order Book Imbalance)
    2. 大單檢測 (Large Order Detection)
    3. 價格衝擊 (Price Impact)
    4. 成交量分布 (Volume Distribution)
    5. Bid-Ask Spread動態
    """
    
    def __init__(self, config: Dict):
        self.volume_threshold = config.get('volume_threshold', 2.0)  # 大單閾值(倍數)
        self.imbalance_window = config.get('imbalance_window', 20)
        self.spread_window = config.get('spread_window', 10)
        
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """完整微觀結構分析"""
        df = df.copy()
        
        # 1. 訂單簿失衡指標
        df = self._calculate_order_imbalance(df)
        
        # 2. 大單檢測
        df = self._detect_large_orders(df)
        
        # 3. 價格衝擊
        df = self._calculate_price_impact(df)
        
        # 4. 成交量特徵
        df = self._analyze_volume_profile(df)
        
        # 5. Spread動態
        df = self._calculate_spread_dynamics(df)
        
        # 6. 流動性指標
        df = self._calculate_liquidity_metrics(df)
        
        return df
    
    def _calculate_order_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算訂單流不平衡 (Order Flow Imbalance)
        模擬: 使用價格和成交量推測買賣壓力
        """
        # 使用價格變化方向作為訂單流指標
        df['price_change'] = df['close'].diff()
        df['volume_weighted_flow'] = df['volume'] * np.sign(df['price_change'])
        
        # 訂單流不平衡率 (OFI)
        buy_volume = df['volume_weighted_flow'].rolling(self.imbalance_window).apply(
            lambda x: x[x > 0].sum()
        )
        sell_volume = df['volume_weighted_flow'].rolling(self.imbalance_window).apply(
            lambda x: abs(x[x < 0].sum())
        )
        
        total_volume = buy_volume + sell_volume
        df['ofi'] = np.where(total_volume > 0, 
                            (buy_volume - sell_volume) / total_volume, 
                            0)
        
        # OFI動量
        df['ofi_momentum'] = df['ofi'].diff()
        
        # OFI波動率
        df['ofi_volatility'] = df['ofi'].rolling(self.imbalance_window).std()
        
        return df
    
    def _detect_large_orders(self, df: pd.DataFrame) -> pd.DataFrame:
        """大單檢測 - 識別異常成交量"""
        # 成交量移動平均
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_std'] = df['volume'].rolling(20).std()
        
        # 大單閾值
        threshold = df['volume_ma'] + (self.volume_threshold * df['volume_std'])
        df['large_order'] = (df['volume'] > threshold).astype(int)
        
        # 大單方向
        df['large_order_direction'] = np.where(
            df['large_order'] == 1,
            np.sign(df['price_change']),
            0
        )
        
        # 連續大單
        df['consecutive_large_orders'] = df['large_order'].rolling(5).sum()
        
        return df
    
    def _calculate_price_impact(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算價格衝擊 - 成交量對價格的影響"""
        # 價格變化百分比
        df['price_change_pct'] = df['close'].pct_change()
        
        # 成交量標準化
        df['volume_normalized'] = (df['volume'] - df['volume'].rolling(50).mean()) / df['volume'].rolling(50).std()
        
        # 價格衝擊 = 價格變化 / 成交量
        df['price_impact'] = np.where(
            df['volume'] > 0,
            abs(df['price_change_pct']) / (df['volume_normalized'] + 1e-6),
            0
        )
        
        # 價格衝擊移動平均
        df['price_impact_ma'] = df['price_impact'].rolling(10).mean()
        
        return df
    
    def _analyze_volume_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交量分布分析"""
        # 成交量動量
        df['volume_momentum'] = df['volume'].pct_change(5)
        
        # 成交量趨勢
        df['volume_trend'] = df['volume'].rolling(20).mean() / df['volume'].rolling(50).mean()
        
        # 成交量分位數
        df['volume_percentile'] = df['volume'].rolling(100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        
        # 成交量突破
        df['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
        
        return df
    
    def _calculate_spread_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bid-Ask Spread動態
        模擬: 使用high-low作為spread代理
        """
        # Spread代理
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Spread移動平均
        df['spread_ma'] = df['spread_proxy'].rolling(self.spread_window).mean()
        
        # Spread波動率
        df['spread_volatility'] = df['spread_proxy'].rolling(self.spread_window).std()
        
        # Spread擴張/收縮
        df['spread_expansion'] = df['spread_proxy'] > df['spread_ma'] * 1.5
        
        return df
    
    def _calculate_liquidity_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """流動性指標"""
        # Amihud流動性比率
        df['amihud_illiquidity'] = abs(df['price_change_pct']) / (df['volume'] + 1e-6)
        
        # 流動性指數 (成交量 * (收盤-開盤))
        df['liquidity_index'] = df['volume'] * abs(df['close'] - df['open'])
        
        # 流動性趨勢
        df['liquidity_trend'] = df['liquidity_index'].rolling(20).mean() / df['liquidity_index'].rolling(50).mean()
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """返回所有微觀結構特徵名稱"""
        return [
            # 訂單流特徵
            'ofi', 'ofi_momentum', 'ofi_volatility',
            
            # 大單特徵
            'large_order', 'large_order_direction', 'consecutive_large_orders',
            
            # 價格衝擊
            'price_impact', 'price_impact_ma',
            
            # 成交量特徵
            'volume_momentum', 'volume_trend', 'volume_percentile', 'volume_spike',
            'volume_normalized',
            
            # Spread特徵
            'spread_proxy', 'spread_ma', 'spread_volatility', 'spread_expansion',
            
            # 流動性特徵
            'amihud_illiquidity', 'liquidity_index', 'liquidity_trend'
        ]
    
    def detect_market_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """檢測市場異常狀況"""
        df = df.copy()
        
        # 異常高OFI
        df['anomaly_high_ofi'] = (abs(df['ofi']) > 0.7).astype(int)
        
        # 異常高成交量
        df['anomaly_high_volume'] = (df['volume_percentile'] > 0.95).astype(int)
        
        # 異常高Spread
        df['anomaly_wide_spread'] = (df['spread_proxy'] > df['spread_ma'] * 2).astype(int)
        
        # 綜合異常評分
        df['anomaly_score'] = (
            df['anomaly_high_ofi'] + 
            df['anomaly_high_volume'] + 
            df['anomaly_wide_spread']
        )
        
        return df
