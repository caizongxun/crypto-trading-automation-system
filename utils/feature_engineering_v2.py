import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger('feature_engineering_v2', 'logs/feature_engineering_v2.log')

class FeatureEngineerV2:
    """
    終極版特徵工程 - 整合所有優化方向
    
    **核心特性**:
    - 訂單流特徵 (Order Flow)
    - 市場微觀結構 (Microstructure)
    - 多時間框架 (MTF)
    - 機器學習衍生特徵
    - 動態標籤生成
    - 自動特徵選擇
    """
    
    def __init__(self, enable_advanced_features: bool = True,
                 enable_ml_features: bool = True):
        logger.info("="*80)
        logger.info("INITIALIZING FEATURE ENGINEER V2")
        logger.info("="*80)
        
        self.enable_advanced = enable_advanced_features
        self.enable_ml = enable_ml_features
        
        logger.info(f"Advanced Features: {enable_advanced_features}")
        logger.info(f"ML Features: {enable_ml_features}")
    
    # ==================== 基礎特徵 ====================
    
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        創建基礎技術指標特徵
        """
        logger.info("Creating basic features...")
        
        # 1. Efficiency Ratio (趨勢效率)
        df['price_change'] = df['close'] - df['close'].shift(10)
        df['volatility'] = (df['high'] - df['low']).rolling(10).sum()
        df['efficiency_ratio'] = df['price_change'].abs() / (df['volatility'] + 1e-8)
        
        # 2. Extreme Time Diff (極值時間差)
        df['extreme_idx'] = df['high'].rolling(20).apply(
            lambda x: x.argmax() if len(x) > 0 else 0, raw=True
        )
        df['extreme_time_diff'] = 20 - df['extreme_idx']
        
        # 3. Volume Imbalance (量能失衡)
        up_volume = df[df['close'] > df['open']]['volume'].rolling(10).sum()
        down_volume = df[df['close'] <= df['open']]['volume'].rolling(10).sum()
        df['vol_imbalance_ratio'] = up_volume / (down_volume + 1e-8)
        
        # 4. Z-Score (價格標準化)
        df['z_score'] = (
            (df['close'] - df['close'].rolling(20).mean()) /
            (df['close'].rolling(20).std() + 1e-8)
        )
        
        # 5. Bollinger Bands Width
        bb_upper = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
        bb_lower = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
        df['bb_width_pct'] = (bb_upper - bb_lower) / df['close']
        
        # 6. RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 7. ATR Percentage
        df['atr'] = (df['high'] - df['low']).rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']
        
        # 8. Multi-timeframe Z-Score
        df['z_score_1h'] = (
            (df['close'] - df['close'].rolling(60).mean()) /
            (df['close'].rolling(60).std() + 1e-8)
        )
        
        # 9. Multi-timeframe ATR
        df['atr_1d'] = (df['high'] - df['low']).rolling(1440).mean()
        df['atr_pct_1d'] = df['atr_1d'] / df['close']
        
        logger.info("Basic features created: 9 features")
        return df
    
    # ==================== 訂單流特徵 ====================
    
    def create_order_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        訂單流特徵 - 捕捉買賣壓力
        """
        logger.info("Creating order flow features...")
        
        # 1. Delta (買賣力道差)
        df['delta'] = df['close'] - df['open']
        df['delta_volume'] = df['volume'] * np.sign(df['delta'])
        
        # 2. 累積 Delta
        df['cumulative_delta_5'] = df['delta_volume'].rolling(5).sum()
        df['cumulative_delta_15'] = df['delta_volume'].rolling(15).sum()
        df['cumulative_delta_60'] = df['delta_volume'].rolling(60).sum()
        
        # 3. Delta 強度
        total_volume_5 = df['volume'].rolling(5).sum()
        total_volume_15 = df['volume'].rolling(15).sum()
        
        df['delta_strength_5'] = df['cumulative_delta_5'] / (total_volume_5 + 1e-8)
        df['delta_strength_15'] = df['cumulative_delta_15'] / (total_volume_15 + 1e-8)
        
        # 4. 買賣壓力比
        up_mask = df['delta'] > 0
        down_mask = df['delta'] <= 0
        
        up_vol = df.loc[up_mask, 'volume'].rolling(10, min_periods=1).sum()
        down_vol = df.loc[down_mask, 'volume'].rolling(10, min_periods=1).sum()
        
        df.loc[up_mask, 'buy_volume'] = up_vol
        df.loc[down_mask, 'sell_volume'] = down_vol
        df['buy_volume'] = df['buy_volume'].fillna(method='ffill')
        df['sell_volume'] = df['sell_volume'].fillna(method='ffill')
        
        df['buy_sell_ratio'] = df['buy_volume'] / (df['sell_volume'] + 1e-8)
        
        # 5. Volume Profile (成交量分布)
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
        
        # 6. Aggressive Buy/Sell (激進買賣)
        df['aggressive_buy'] = np.where(
            (df['close'] >= df['high'] * 0.995) & (df['delta'] > 0),
            df['volume'],
            0
        )
        df['aggressive_sell'] = np.where(
            (df['close'] <= df['low'] * 1.005) & (df['delta'] < 0),
            df['volume'],
            0
        )
        
        df['aggressive_ratio'] = (
            df['aggressive_buy'].rolling(10).sum() /
            (df['aggressive_sell'].rolling(10).sum() + 1e-8)
        )
        
        logger.info("Order flow features created: 10 features")
        return df
    
    # ==================== 市場微觀結構特徵 ====================
    
    def create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        市場微觀結構特徵 - 捕捉市場摩擦和流動性
        """
        logger.info("Creating microstructure features...")
        
        # 1. Tick Imbalance (tick 失衡)
        df['tick_direction'] = np.sign(df['close'] - df['close'].shift(1))
        df['tick_imbalance_10'] = df['tick_direction'].rolling(10).sum()
        df['tick_imbalance_20'] = df['tick_direction'].rolling(20).sum()
        df['tick_imbalance_50'] = df['tick_direction'].rolling(50).sum()
        
        # 2. Price Impact (價格衝擊)
        df['price_impact'] = (df['high'] - df['low']) / (df['volume'] + 1e-8)
        df['price_impact_ma'] = df['price_impact'].rolling(20).mean()
        
        # 3. Spread Proxy (價差代理)
        df['spread_proxy'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
        df['spread_ma'] = df['spread_proxy'].rolling(20).mean()
        
        # 4. Liquidity Score (流動性評分)
        df['liquidity_score'] = df['volume'] / (df['spread_proxy'] + 1e-8)
        df['liquidity_score_norm'] = (
            df['liquidity_score'] / (df['liquidity_score'].rolling(50).mean() + 1e-8)
        )
        
        # 5. Reversal Strength (反轉強度)
        df['reversal_strength'] = (
            (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        )
        df['reversal_ma'] = df['reversal_strength'].rolling(5).mean()
        
        # 6. Trade Intensity (交易強度)
        df['trade_intensity'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)
        
        # 7. Market Efficiency (市場效率)
        price_range = df['high'] - df['low']
        close_range = (df['close'] - df['open']).abs()
        df['market_efficiency'] = close_range / (price_range + 1e-8)
        
        logger.info("Microstructure features created: 10 features")
        return df
    
    # ==================== 多時間框架特徵 ====================
    
    def create_mtf_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        多時間框架特徵 - 捕捉不同週期的信號
        """
        logger.info("Creating multi-timeframe features...")
        
        # 1. 趨勢一致性
        df['ema_5'] = df['close'].ewm(span=5).mean()
        df['ema_15'] = df['close'].ewm(span=15).mean()
        df['ema_60'] = df['close'].ewm(span=60).mean()
        df['ema_240'] = df['close'].ewm(span=240).mean()
        
        df['trend_alignment'] = (
            np.sign(df['close'] - df['ema_5']) +
            np.sign(df['close'] - df['ema_15']) +
            np.sign(df['close'] - df['ema_60']) +
            np.sign(df['close'] - df['ema_240'])
        ) / 4
        
        # 2. 波動率比率
        df['vol_5m'] = df['close'].rolling(5).std()
        df['vol_15m'] = df['close'].rolling(15).std()
        df['vol_1h'] = df['close'].rolling(60).std()
        df['vol_4h'] = df['close'].rolling(240).std()
        
        df['vol_ratio_5m_1h'] = df['vol_5m'] / (df['vol_1h'] + 1e-8)
        df['vol_ratio_15m_4h'] = df['vol_15m'] / (df['vol_4h'] + 1e-8)
        
        # 3. 動量散度
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_15'] = df['close'].pct_change(15)
        df['momentum_60'] = df['close'].pct_change(60)
        
        df['momentum_divergence_5_15'] = df['momentum_5'] - df['momentum_15']
        df['momentum_divergence_15_60'] = df['momentum_15'] - df['momentum_60']
        
        # 4. 多週期 RSI
        def calc_rsi(series, period):
            delta = series.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = -delta.where(delta < 0, 0).rolling(period).mean()
            rs = gain / (loss + 1e-8)
            return 100 - (100 / (1 + rs))
        
        df['rsi_5'] = calc_rsi(df['close'], 5)
        df['rsi_14'] = calc_rsi(df['close'], 14)
        df['rsi_30'] = calc_rsi(df['close'], 30)
        
        df['rsi_divergence'] = df['rsi_5'] - df['rsi_30']
        
        # 5. 價格位置 (多週期)
        def price_position(close, high, low, period):
            highest = high.rolling(period).max()
            lowest = low.rolling(period).min()
            return (close - lowest) / (highest - lowest + 1e-8)
        
        df['price_pos_20'] = price_position(df['close'], df['high'], df['low'], 20)
        df['price_pos_60'] = price_position(df['close'], df['high'], df['low'], 60)
        df['price_pos_240'] = price_position(df['close'], df['high'], df['low'], 240)
        
        logger.info("Multi-timeframe features created: 15 features")
        return df
    
    # ==================== 機器學習衍生特徵 ====================
    
    def create_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        機器學習衍生特徵 - 自動發現特徵交互
        """
        if not self.enable_ml:
            logger.info("ML features disabled")
            return df
        
        logger.info("Creating ML-derived features...")
        
        # 1. 特徵交互 (多項式特徵)
        df['rsi_x_vol'] = df['rsi'] * df['atr_pct']
        df['rsi_x_bb'] = df['rsi'] * df['bb_width_pct']
        df['vol_x_bb'] = df['atr_pct'] * df['bb_width_pct']
        
        # 2. 比率特徵
        df['rsi_volatility_ratio'] = df['rsi'] / (df['atr_pct'] * 1000 + 1e-8)
        df['momentum_volatility_ratio'] = (
            df['momentum_15'].abs() / (df['vol_15m'] + 1e-8)
        )
        
        # 3. 市場狀態聚類
        try:
            from sklearn.cluster import KMeans
            
            cluster_features = [
                'rsi', 'atr_pct', 'vol_imbalance_ratio',
                'delta_strength_15', 'tick_imbalance_20'
            ]
            
            X_cluster = df[cluster_features].fillna(0).values
            
            if len(X_cluster) > 100:
                kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                df['market_regime'] = kmeans.fit_predict(X_cluster)
            else:
                df['market_regime'] = 0
                
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            df['market_regime'] = 0
        
        # 4. 統計特徵
        df['returns'] = df['close'].pct_change()
        df['returns_skew'] = df['returns'].rolling(50).skew()
        df['returns_kurt'] = df['returns'].rolling(50).kurt()
        
        # 5. Hurst Exponent (趨勢持續性)
        def hurst_exponent(ts, max_lag=20):
            lags = range(2, max_lag)
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            return np.polyfit(np.log(lags), np.log(tau), 1)[0]
        
        df['hurst'] = df['close'].rolling(100).apply(
            lambda x: hurst_exponent(x.values) if len(x) >= 20 else 0.5,
            raw=False
        )
        
        logger.info("ML-derived features created: 10 features")
        return df
    
    # ==================== 動態標籤生成 ====================
    
    def compute_adaptive_labels(self, df: pd.DataFrame,
                               direction: str = 'long') -> pd.DataFrame:
        """
        動態標籤生成 - 根據波動率調整 TP/SL
        """
        logger.info(f"Computing adaptive labels for {direction}...")
        
        # 1. 計算波動率狀態
        atr_pct = df['atr_pct'].fillna(df['atr_pct'].median())
        
        # 2. 動態 TP/SL
        conditions = [
            atr_pct < 0.02,  # 低波動
            atr_pct > 0.04,  # 高波動
        ]
        
        # TP
        tp_choices = [0.015, 0.025]
        tp_default = 0.020
        df['dynamic_tp'] = np.select(conditions, tp_choices, default=tp_default)
        
        # SL
        sl_choices = [0.0075, 0.0125]
        sl_default = 0.010
        df['dynamic_sl'] = np.select(conditions, sl_choices, default=sl_default)
        
        # 3. 計算標籤
        labels = []
        
        for idx in range(len(df)):
            if idx >= len(df) - 100:  # 最後100根無法計算
                labels.append(0)
                continue
            
            entry_price = df.iloc[idx]['close']
            tp_pct = df.iloc[idx]['dynamic_tp']
            sl_pct = df.iloc[idx]['dynamic_sl']
            
            if direction == 'long':
                tp_price = entry_price * (1 + tp_pct)
                sl_price = entry_price * (1 - sl_pct)
                
                future = df.iloc[idx+1:idx+101]
                tp_hit = (future['high'] >= tp_price).any()
                sl_hit = (future['low'] <= sl_price).any()
                
                if tp_hit and sl_hit:
                    tp_idx = future[future['high'] >= tp_price].index[0]
                    sl_idx = future[future['low'] <= sl_price].index[0]
                    label = 1 if tp_idx < sl_idx else 0
                elif tp_hit:
                    label = 1
                else:
                    label = 0
            
            else:  # short
                tp_price = entry_price * (1 - tp_pct)
                sl_price = entry_price * (1 + sl_pct)
                
                future = df.iloc[idx+1:idx+101]
                tp_hit = (future['low'] <= tp_price).any()
                sl_hit = (future['high'] >= sl_price).any()
                
                if tp_hit and sl_hit:
                    tp_idx = future[future['low'] <= tp_price].index[0]
                    sl_idx = future[future['high'] >= sl_price].index[0]
                    label = 1 if tp_idx < sl_idx else 0
                elif tp_hit:
                    label = 1
                else:
                    label = 0
            
            labels.append(label)
        
        df[f'label_{direction}_adaptive'] = labels
        
        positive_rate = sum(labels) / len(labels) * 100
        logger.info(f"Adaptive labels created: {sum(labels)}/{len(labels)} ({positive_rate:.2f}%)")
        
        return df
    
    # ==================== 主函數 ====================
    
    def create_features_from_1m(self, df_1m: pd.DataFrame,
                               use_adaptive_labels: bool = True,
                               label_type: str = 'both') -> pd.DataFrame:
        """
        從 1m K線創建完整特徵集
        
        Args:
            df_1m: 1分鐘K線數據
            use_adaptive_labels: 是否使用動態標籤
            label_type: 'long', 'short', 'both'
        """
        logger.info("="*80)
        logger.info("CREATING COMPLETE FEATURE SET")
        logger.info("="*80)
        logger.info(f"Input data: {len(df_1m):,} rows")
        
        df = df_1m.copy()
        
        # 1. 基礎特徵
        df = self.create_basic_features(df)
        
        # 2. 訂單流特徵
        df = self.create_order_flow_features(df)
        
        # 3. 市場微觀結構
        if self.enable_advanced:
            df = self.create_microstructure_features(df)
        
        # 4. 多時間框架
        if self.enable_advanced:
            df = self.create_mtf_features(df)
        
        # 5. 機器學習特徵
        if self.enable_ml:
            df = self.create_ml_features(df)
        
        # 6. 標籤生成
        if use_adaptive_labels:
            if label_type in ['long', 'both']:
                df = self.compute_adaptive_labels(df, direction='long')
            if label_type in ['short', 'both']:
                df = self.compute_adaptive_labels(df, direction='short')
        
        # 清理
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 統計
        total_features = len([col for col in df.columns if col not in 
                             ['open', 'high', 'low', 'close', 'volume'] and
                             not col.startswith('label_') and
                             not col.startswith('dynamic_')])
        
        logger.info("="*80)
        logger.info("FEATURE CREATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Total features created: {total_features}")
        logger.info(f"Output data: {len(df):,} rows")
        logger.info(f"Missing values: {df.isnull().sum().sum():,}")
        
        return df
    
    def get_feature_list(self) -> List[str]:
        """
        返回所有特徵名稱列表
        """
        features = [
            # 基礎特徵 (9)
            'efficiency_ratio', 'extreme_time_diff', 'vol_imbalance_ratio',
            'z_score', 'bb_width_pct', 'rsi', 'atr_pct', 'z_score_1h', 'atr_pct_1d',
            
            # 訂單流特徵 (10)
            'cumulative_delta_5', 'cumulative_delta_15', 'cumulative_delta_60',
            'delta_strength_5', 'delta_strength_15', 'buy_sell_ratio',
            'volume_ratio', 'aggressive_ratio',
        ]
        
        if self.enable_advanced:
            # 微觀結構 (10)
            features += [
                'tick_imbalance_10', 'tick_imbalance_20', 'tick_imbalance_50',
                'price_impact', 'spread_proxy', 'liquidity_score_norm',
                'reversal_strength', 'trade_intensity', 'market_efficiency'
            ]
            
            # 多時間框架 (15)
            features += [
                'trend_alignment', 'vol_ratio_5m_1h', 'vol_ratio_15m_4h',
                'momentum_divergence_5_15', 'momentum_divergence_15_60',
                'rsi_divergence', 'price_pos_20', 'price_pos_60', 'price_pos_240'
            ]
        
        if self.enable_ml:
            # ML特徵 (10)
            features += [
                'rsi_x_vol', 'rsi_x_bb', 'vol_x_bb',
                'rsi_volatility_ratio', 'momentum_volatility_ratio',
                'market_regime', 'returns_skew', 'returns_kurt', 'hurst'
            ]
        
        return features