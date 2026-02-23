import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Tuple, List, Optional
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger('enhanced_feature_engineering', 'logs/enhanced_feature_engineering.log')

class EnhancedFeatureEngineer:
    """
    增強版特徵工程模組
    
    **整合所有優化方向**:
    1. 訂單流特徵 (Order Flow Features)
    2. 市場微觀結構 (Market Microstructure)
    3. 多時間框架 (Multi-Timeframe)
    4. ML 衍生特徵 (ML-derived Features)
    5. 動態標籤生成 (Adaptive Labels)
    6. 原有特徵 (Original Features)
    """
    
    def __init__(self):
        logger.info("Initializing EnhancedFeatureEngineer")
        logger.info("Feature modules: OrderFlow + Microstructure + MTF + ML + Adaptive Labels")
        
        self.kmeans_model = None
        self.poly_features = None
    
    # ========== 1. 訂單流特徵 ==========
    
    def create_order_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        訂單流特徵 - 捕捉買賣壓力
        
        **核心特徵**:
        - Delta: 單根 K 棒買賣壓力
        - Cumulative Delta: 累積買賣壓力
        - Delta Strength: 買賣力道強度
        - Buy/Sell Ratio: 買賣量比例
        """
        logger.info("Creating Order Flow features...")
        
        # 1. Delta (買賣壓力差)
        df['delta'] = df['close'] - df['open']
        df['delta_volume'] = df['volume'] * np.sign(df['delta'])
        
        # 2. 累積 Delta (5 分鐘 & 15 分鐘)
        df['cumulative_delta_5'] = df['delta_volume'].rolling(5).sum()
        df['cumulative_delta_15'] = df['delta_volume'].rolling(15).sum()
        df['cumulative_delta_60'] = df['delta_volume'].rolling(60).sum()
        
        # 3. Delta 強度 (標準化)
        df['delta_strength_5'] = (
            df['cumulative_delta_5'] / (df['volume'].rolling(5).sum() + 1e-8)
        )
        df['delta_strength_15'] = (
            df['cumulative_delta_15'] / (df['volume'].rolling(15).sum() + 1e-8)
        )
        
        # 4. 買賣壓力比 (Buy/Sell Ratio)
        up_mask = df['delta'] > 0
        down_mask = df['delta'] <= 0
        
        up_volume = df['volume'].where(up_mask, 0).rolling(20).sum()
        down_volume = df['volume'].where(down_mask, 0).rolling(20).sum()
        df['buy_sell_ratio'] = up_volume / (down_volume + 1e-8)
        
        # 5. Delta 加速度
        df['delta_acceleration'] = (
            df['cumulative_delta_5'] - df['cumulative_delta_5'].shift(5)
        )
        
        # 6. Volume Profile (量價分布)
        df['volume_delta_ratio'] = df['delta_volume'] / (df['volume'] + 1e-8)
        
        logger.info("  ✅ Order Flow features: 10 features created")
        
        return df
    
    # ========== 2. 市場微觀結構 ==========
    
    def create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        市場微觀結構特徵 - 捕捉市場摩擦
        
        **核心特徵**:
        - Tick Imbalance: Tick 方向不平衡
        - Price Impact: 價格衝擊
        - Liquidity Score: 流動性指標
        - Reversal Strength: 反轉強度
        """
        logger.info("Creating Microstructure features...")
        
        # 1. Tick Imbalance (Tick 不平衡)
        df['tick_direction'] = np.sign(df['close'] - df['close'].shift(1))
        df['tick_imbalance_10'] = df['tick_direction'].rolling(10).sum()
        df['tick_imbalance_20'] = df['tick_direction'].rolling(20).sum()
        df['tick_imbalance_60'] = df['tick_direction'].rolling(60).sum()
        
        # 2. Price Impact (價格衝擊)
        df['price_impact'] = (df['high'] - df['low']) / (df['volume'] + 1e-8)
        df['price_impact_ma'] = df['price_impact'].rolling(20).mean()
        
        # 3. Spread Proxy (價差代理)
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        df['spread_ma'] = df['spread_proxy'].rolling(20).mean()
        df['spread_volatility'] = df['spread_proxy'].rolling(20).std()
        
        # 4. Liquidity Score (流動性指標)
        df['liquidity_score'] = df['volume'] / (df['spread_proxy'] + 1e-8)
        df['liquidity_ma'] = df['liquidity_score'].rolling(20).mean()
        
        # 5. Reversal Strength (反轉強度)
        df['reversal_strength'] = (
            (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        )
        df['reversal_ma_5'] = df['reversal_strength'].rolling(5).mean()
        df['reversal_ma_15'] = df['reversal_strength'].rolling(15).mean()
        
        # 6. Volatility Clustering (波動率聚集)
        df['volatility_cluster'] = (
            df['spread_proxy'].rolling(5).std() / 
            (df['spread_proxy'].rolling(20).std() + 1e-8)
        )
        
        # 7. Trade Intensity (交易強度)
        df['trade_intensity'] = df['volume'] / df['volume'].rolling(60).mean()
        
        logger.info("  ✅ Microstructure features: 15 features created")
        
        return df
    
    # ========== 3. 多時間框架特徵 ==========
    
    def create_mtf_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        多時間框架特徵 - 捕捉不同週期信號
        
        **核心特徵**:
        - Trend Alignment: 趨勢一致性
        - Volatility Ratio: 波動率比率
        - Momentum Divergence: 動量散度
        - Multi-Period Correlation: 多週期相關
        """
        logger.info("Creating Multi-Timeframe features...")
        
        # 1. 多期均線
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()
        df['ema_60'] = df['close'].ewm(span=60, adjust=False).mean()
        df['ema_240'] = df['close'].ewm(span=240, adjust=False).mean()
        
        # 2. 趨勢一致性 (Trend Alignment)
        df['trend_alignment'] = (
            np.sign(df['close'] - df['ema_5']) +
            np.sign(df['close'] - df['ema_15']) +
            np.sign(df['close'] - df['ema_60']) +
            np.sign(df['close'] - df['ema_240'])
        ) / 4
        
        # 3. 波動率比率
        df['vol_5'] = df['close'].rolling(5).std()
        df['vol_15'] = df['close'].rolling(15).std()
        df['vol_60'] = df['close'].rolling(60).std()
        
        df['vol_ratio_5_15'] = df['vol_5'] / (df['vol_15'] + 1e-8)
        df['vol_ratio_5_60'] = df['vol_5'] / (df['vol_60'] + 1e-8)
        df['vol_ratio_15_60'] = df['vol_15'] / (df['vol_60'] + 1e-8)
        
        # 4. 動量散度 (Momentum Divergence)
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_15'] = df['close'].pct_change(15)
        df['momentum_60'] = df['close'].pct_change(60)
        
        df['momentum_div_5_15'] = df['momentum_5'] - df['momentum_15']
        df['momentum_div_5_60'] = df['momentum_5'] - df['momentum_60']
        
        # 5. 距離均線 (標準化)
        df['distance_ema_5'] = (df['close'] - df['ema_5']) / df['ema_5']
        df['distance_ema_15'] = (df['close'] - df['ema_15']) / df['ema_15']
        df['distance_ema_60'] = (df['close'] - df['ema_60']) / df['ema_60']
        
        # 6. 多期相關 (Rolling Correlation)
        df['corr_5_15'] = (
            df['close'].rolling(5).corr(df['close'].rolling(15).mean())
        )
        
        logger.info("  ✅ Multi-Timeframe features: 20 features created")
        
        return df
    
    # ========== 4. ML 衍生特徵 ==========
    
    def create_ml_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ML 衍生特徵 - 自動發現特徵交互
        
        **核心特徵**:
        - Polynomial Features: 特徵交互
        - Market Regime Clustering: 市場狀態聚類
        - Feature Ratios: 特徵比率
        """
        logger.info("Creating ML-derived features...")
        
        # 1. 關鍵特徵交互 (Polynomial)
        base_features = ['rsi', 'bb_width_pct', 'atr_pct']
        
        if all(feat in df.columns for feat in base_features):
            X_base = df[base_features].fillna(0).values
            
            if self.poly_features is None:
                self.poly_features = PolynomialFeatures(
                    degree=2, include_bias=False, interaction_only=True
                )
                X_poly = self.poly_features.fit_transform(X_base)
            else:
                X_poly = self.poly_features.transform(X_base)
            
            # 只保留交互項
            feature_names = self.poly_features.get_feature_names_out(base_features)
            for i, name in enumerate(feature_names):
                if '*' in name:
                    df[f'interaction_{name.replace(" ", "")}'] = X_poly[:, i]
        
        # 2. 市場狀態聚類 (Market Regime)
        cluster_features = ['atr_pct', 'volume', 'delta_strength_5']
        
        if all(feat in df.columns for feat in cluster_features):
            X_cluster = df[cluster_features].fillna(0).values
            
            # 標準化
            X_cluster_norm = (X_cluster - X_cluster.mean(axis=0)) / (X_cluster.std(axis=0) + 1e-8)
            
            if self.kmeans_model is None:
                self.kmeans_model = KMeans(n_clusters=5, random_state=42, n_init=10)
                df['market_regime'] = self.kmeans_model.fit_predict(X_cluster_norm)
            else:
                df['market_regime'] = self.kmeans_model.predict(X_cluster_norm)
            
            # One-hot encoding
            for i in range(5):
                df[f'regime_{i}'] = (df['market_regime'] == i).astype(int)
        
        # 3. 特徵比率 (Feature Ratios)
        if 'volume' in df.columns and 'atr_pct' in df.columns:
            df['volume_volatility_ratio'] = (
                df['volume'] / (df['atr_pct'] + 1e-8)
            )
        
        if 'rsi' in df.columns and 'bb_width_pct' in df.columns:
            df['rsi_bbwidth_product'] = df['rsi'] * df['bb_width_pct']
        
        logger.info("  ✅ ML-derived features: 12 features created")
        
        return df
    
    # ========== 5. 原有特徵 ==========
    
    def create_original_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        原有特徵 - 保留原系統的核心特徵
        """
        logger.info("Creating Original features...")
        
        # 1. Efficiency Ratio
        price_change = abs(df['close'] - df['close'].shift(20))
        volatility_sum = abs(df['close'] - df['close'].shift(1)).rolling(20).sum()
        df['efficiency_ratio'] = price_change / (volatility_sum + 1e-8)
        
        # 2. Extreme Time Diff
        df['high_time'] = df['high'].rolling(20).apply(lambda x: np.argmax(x), raw=True)
        df['low_time'] = df['low'].rolling(20).apply(lambda x: np.argmin(x), raw=True)
        df['extreme_time_diff'] = abs(df['high_time'] - df['low_time'])
        
        # 3. Volume Imbalance
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['vol_imbalance_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
        
        # 4. Z-Score (1 分鐘)
        df['z_score'] = (
            (df['close'] - df['close'].rolling(20).mean()) /
            (df['close'].rolling(20).std() + 1e-8)
        )
        
        # 5. Bollinger Bands
        bb_middle = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = bb_middle + 2 * bb_std
        df['bb_lower'] = bb_middle - 2 * bb_std
        df['bb_width_pct'] = (df['bb_upper'] - df['bb_lower']) / bb_middle
        
        # 6. RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 7. ATR (1 分鐘)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']
        
        # 8. 多時框 Z-Score
        df['z_score_1h'] = (
            (df['close'] - df['close'].rolling(60).mean()) /
            (df['close'].rolling(60).std() + 1e-8)
        )
        
        # 9. 多時框 ATR
        df['atr_1d'] = true_range.rolling(1440).mean()
        df['atr_pct_1d'] = df['atr_1d'] / df['close']
        
        logger.info("  ✅ Original features: 14 features created")
        
        return df
    
    # ========== 6. 動態標籤生成 ==========
    
    def create_adaptive_labels(
        self,
        df: pd.DataFrame,
        direction: str = 'long',
        base_tp_pct: float = 0.020,
        base_sl_pct: float = 0.010,
        adaptive: bool = True
    ) -> pd.Series:
        """
        動態標籤生成 - 根據市場波動率調整 TP/SL
        
        **模式**:
        - adaptive=True: 根據 ATR 動態調整
        - adaptive=False: 使用固定 TP/SL
        """
        logger.info(f"Creating adaptive labels for {direction.upper()}...")
        
        if adaptive:
            # 動態 TP/SL
            atr_pct = df['atr_pct_1d'] if 'atr_pct_1d' in df.columns else df['atr_pct']
            
            tp_pct = np.where(
                atr_pct < 0.02, base_tp_pct * 0.75,  # 低波動: 1.5%
                np.where(
                    atr_pct > 0.04, base_tp_pct * 1.25,  # 高波動: 2.5%
                    base_tp_pct  # 中等波動: 2.0%
                )
            )
            
            sl_pct = tp_pct / 2
            
            logger.info(f"  Adaptive TP range: {tp_pct.min()*100:.2f}% - {tp_pct.max()*100:.2f}%")
        else:
            tp_pct = base_tp_pct
            sl_pct = base_sl_pct
            logger.info(f"  Fixed TP/SL: {tp_pct*100:.2f}% / {sl_pct*100:.2f}%")
        
        # 計算標籤
        labels = np.zeros(len(df), dtype=int)
        
        for i in range(len(df) - 100):
            entry_price = df['close'].iloc[i]
            
            if direction == 'long':
                tp_price = entry_price * (1 + (tp_pct[i] if adaptive else tp_pct))
                sl_price = entry_price * (1 - (sl_pct[i] if adaptive else sl_pct))
            else:
                tp_price = entry_price * (1 - (tp_pct[i] if adaptive else tp_pct))
                sl_price = entry_price * (1 + (sl_pct[i] if adaptive else sl_pct))
            
            # 查找未來 100 根 K 棒
            future_high = df['high'].iloc[i+1:i+101].max()
            future_low = df['low'].iloc[i+1:i+101].min()
            
            if direction == 'long':
                if future_high >= tp_price:
                    labels[i] = 1
                elif future_low <= sl_price:
                    labels[i] = 0
            else:
                if future_low <= tp_price:
                    labels[i] = 1
                elif future_high >= sl_price:
                    labels[i] = 0
        
        positive_rate = labels.sum() / len(labels) * 100
        logger.info(f"  Positive sample rate: {positive_rate:.2f}%")
        
        return pd.Series(labels, index=df.index)
    
    # ========== 7. 完整流程 ==========
    
    def create_enhanced_features(
        self,
        df_1m: pd.DataFrame,
        use_adaptive_labels: bool = True,
        label_type: str = 'both'
    ) -> pd.DataFrame:
        """
        創建完整增強特徵集
        
        **參數**:
        - df_1m: 1 分鐘 K 線數據
        - use_adaptive_labels: 是否使用動態標籤
        - label_type: 'long', 'short', 'both'
        """
        logger.info("="*80)
        logger.info("CREATING ENHANCED FEATURES")
        logger.info("="*80)
        
        df = df_1m.copy()
        
        # 1. 原有特徵
        df = self.create_original_features(df)
        
        # 2. 訂單流特徵
        df = self.create_order_flow_features(df)
        
        # 3. 市場微觀結構
        df = self.create_microstructure_features(df)
        
        # 4. 多時間框架
        df = self.create_mtf_features(df)
        
        # 5. ML 衍生特徵
        df = self.create_ml_derived_features(df)
        
        # 6. 生成標籤
        if label_type in ['long', 'both']:
            df['label_long'] = self.create_adaptive_labels(
                df, direction='long', adaptive=use_adaptive_labels
            )
        
        if label_type in ['short', 'both']:
            df['label_short'] = self.create_adaptive_labels(
                df, direction='short', adaptive=use_adaptive_labels
            )
        
        # 7. 清理
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # 統計
        total_features = len([col for col in df.columns if col not in 
                             ['open', 'high', 'low', 'close', 'volume', 'label_long', 'label_short']])
        
        logger.info("="*80)
        logger.info("FEATURE CREATION COMPLETED")
        logger.info("="*80)
        logger.info(f"Total features: {total_features}")
        logger.info(f"  - Original: 14")
        logger.info(f"  - Order Flow: 10")
        logger.info(f"  - Microstructure: 15")
        logger.info(f"  - Multi-Timeframe: 20")
        logger.info(f"  - ML-derived: 12")
        logger.info(f"Final dataset shape: {df.shape}")
        
        if 'label_long' in df.columns:
            logger.info(f"Long positive rate: {df['label_long'].mean()*100:.2f}%")
        if 'label_short' in df.columns:
            logger.info(f"Short positive rate: {df['label_short'].mean()*100:.2f}%")
        
        logger.info("="*80)
        
        return df
    
    def get_feature_list(self) -> List[str]:
        """
        獲取所有特徵列表
        """
        features = [
            # Original (14)
            'efficiency_ratio', 'extreme_time_diff', 'vol_imbalance_ratio',
            'z_score', 'bb_width_pct', 'rsi', 'atr_pct', 'z_score_1h', 'atr_pct_1d',
            'bb_upper', 'bb_lower', 'volume_ma', 'atr', 'atr_1d',
            
            # Order Flow (10)
            'delta', 'delta_volume', 'cumulative_delta_5', 'cumulative_delta_15',
            'cumulative_delta_60', 'delta_strength_5', 'delta_strength_15',
            'buy_sell_ratio', 'delta_acceleration', 'volume_delta_ratio',
            
            # Microstructure (15)
            'tick_direction', 'tick_imbalance_10', 'tick_imbalance_20', 'tick_imbalance_60',
            'price_impact', 'price_impact_ma', 'spread_proxy', 'spread_ma',
            'spread_volatility', 'liquidity_score', 'liquidity_ma',
            'reversal_strength', 'reversal_ma_5', 'reversal_ma_15',
            'volatility_cluster', 'trade_intensity',
            
            # Multi-Timeframe (20)
            'ema_5', 'ema_15', 'ema_60', 'ema_240', 'trend_alignment',
            'vol_5', 'vol_15', 'vol_60', 'vol_ratio_5_15', 'vol_ratio_5_60',
            'vol_ratio_15_60', 'momentum_5', 'momentum_15', 'momentum_60',
            'momentum_div_5_15', 'momentum_div_5_60', 'distance_ema_5',
            'distance_ema_15', 'distance_ema_60', 'corr_5_15',
        ]
        
        # ML-derived features (動態生成,可能不同)
        # interaction_* (3-5 features)
        # regime_* (5 features)
        # volume_volatility_ratio, rsi_bbwidth_product
        
        return features