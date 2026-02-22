import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger('micro_structure', 'logs/micro_structure.log')

class MicroStructureEngineer:
    """
    微觀結構特徵工程引擎 - 廣播架構
    
    關鍵設計: 保持 1m 級別解析度，將 15m 軌跡特徵廣播回每根 1m K 線
    """
    
    @staticmethod
    def calculate_15m_micro_features(df_1m: pd.DataFrame) -> pd.DataFrame:
        """
        計算 15m 微觀軌跡特徵 (不壓縮 1m 數據)
        
        Args:
            df_1m: 1m K 線資料 (open, high, low, close, volume)
                   必須有 DatetimeIndex 且經過時間排序
        
        Returns:
            df_15m_features: 15m 微觀特徵 (7 萬筆)
                    - efficiency_ratio: 效率比 (ER)
                    - extreme_time_diff: 極值時序差 (分鐘)
                    - vol_imbalance_ratio: 量能失衡率
        """
        logger.info("Calculating 15m micro-structure features...")
        logger.info(f"Input 1m data shape: {df_1m.shape}")
        
        if not isinstance(df_1m.index, pd.DatetimeIndex):
            raise ValueError("Input DataFrame must have a DatetimeIndex.")
        
        df = df_1m.copy()
        
        # 1. 計算 1m 級別的基礎變數
        df['price_diff'] = df['close'].diff().abs()
        df['is_up'] = (df['close'] > df['open']).astype(int)
        df['up_vol'] = df['volume'] * df['is_up']
        df['down_vol'] = df['volume'] * (1 - df['is_up'])
        
        # 2. 15 分鐘重採樣聚合器
        grouper = pd.Grouper(freq='15min', closed='left', label='left')
        
        # 3. 聚合標準 OHLCV 與累積特徵
        df_15m = df.groupby(grouper).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'price_diff': 'sum',
            'up_vol': 'sum',
            'down_vol': 'sum'
        })
        
        # 4. 提取極值發生的精確時間
        high_times = df.groupby(grouper)['high'].idxmax()
        low_times = df.groupby(grouper)['low'].idxmin()
        
        # 5. 構建微結構特徵
        logger.info("Building micro-structure features...")
        
        # Feature A: 效率比 (Efficiency Ratio)
        net_change = (df_15m['close'] - df_15m['open']).abs()
        df_15m['efficiency_ratio'] = np.where(
            df_15m['price_diff'] == 0,
            0,
            net_change / df_15m['price_diff']
        )
        
        # Feature B: 極值時序差 (Sweep Sequence)
        df_15m['extreme_time_diff'] = (high_times - low_times).dt.total_seconds() / 60.0
        
        # Feature C: 微觀量能失衡率
        vol_imbalance = df_15m['up_vol'] - df_15m['down_vol']
        df_15m['vol_imbalance_ratio'] = np.where(
            df_15m['volume'] == 0,
            0,
            vol_imbalance / df_15m['volume']
        )
        
        # 6. 只保留特徵欄位
        feature_cols = ['efficiency_ratio', 'extreme_time_diff', 'vol_imbalance_ratio']
        df_15m_features = df_15m[feature_cols].copy()
        
        # 7. 移除 NaN
        df_15m_features = df_15m_features.dropna()
        
        logger.info(f"Output 15m features shape: {df_15m_features.shape}")
        logger.info(f"  - efficiency_ratio: mean={df_15m_features['efficiency_ratio'].mean():.4f}")
        logger.info(f"  - extreme_time_diff: mean={df_15m_features['extreme_time_diff'].mean():.2f} min")
        logger.info(f"  - vol_imbalance_ratio: mean={df_15m_features['vol_imbalance_ratio'].mean():.4f}")
        
        return df_15m_features
    
    @staticmethod
    def broadcast_15m_to_1m(df_1m: pd.DataFrame, df_15m_features: pd.DataFrame) -> pd.DataFrame:
        """
        廣播架構: 將 15m 微觀特徵廣播回 1m K 線
        
        關鍵: 使用 merge_asof 向後填充 + shift(1) 防止未來洩漏
        
        Args:
            df_1m: 1m K 線 (105 萬筆)
            df_15m_features: 15m 微觀特徵 (7 萬筆)
        
        Returns:
            df_final: 1m K 線 + 15m 微觀特徵 (105 萬筆)
        """
        logger.info("="*80)
        logger.info("BROADCAST ARCHITECTURE - 15m Features -> 1m Resolution")
        logger.info("="*80)
        logger.info(f"1m data: {len(df_1m):,} samples")
        logger.info(f"15m features: {len(df_15m_features):,} samples")
        
        # 1. 向後位移一格，避免偷看未來
        logger.info("Shifting 15m features by 1 bar to prevent lookahead bias...")
        df_15m_shifted = df_15m_features.shift(1).dropna()
        
        # 2. 廣播合併 (merge_asof)
        logger.info("Broadcasting 15m features to 1m timeframe...")
        df_final = pd.merge_asof(
            df_1m.sort_index(),
            df_15m_shifted.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )
        
        # 3. 移除 NaN
        initial_count = len(df_final)
        df_final = df_final.dropna()
        logger.info(f"Dropped {initial_count - len(df_final):,} rows with NaN")
        
        logger.info(f"Final data shape: {df_final.shape}")
        logger.info("="*80)
        
        return df_final
    
    @staticmethod
    def add_bidirectional_labels_1m(df_1m: pd.DataFrame,
                                    lookahead_bars: int = 240,
                                    tp_pct_long: float = 0.02,
                                    sl_pct_long: float = 0.01,
                                    tp_pct_short: float = 0.02,
                                    sl_pct_short: float = 0.01) -> pd.DataFrame:
        """
        在 1m 級別生成雙向標籤 (Long + Short Oracles)
        
        Args:
            df_1m: 1m K 線 + 15m 微觀特徵 (105 萬筆)
            lookahead_bars: 向前看幾根 1m K 線 (240 = 4 小時)
            tp_pct_long: 做多停利百分比 (0.02 = 2%)
            sl_pct_long: 做多停損百分比 (0.01 = 1%)
            tp_pct_short: 做空停利百分比 (0.02 = 2%)
            sl_pct_short: 做空停損百分比 (0.01 = 1%)
        
        Returns:
            df_1m: 加入 label_long 和 label_short 欄位
        """
        logger.info("="*80)
        logger.info("BIDIRECTIONAL LABEL GENERATION (1m Resolution)")
        logger.info("="*80)
        logger.info(f"Lookahead: {lookahead_bars} bars (4 hours)")
        logger.info(f"Long  -> TP: +{tp_pct_long*100:.1f}%, SL: -{sl_pct_long*100:.1f}%")
        logger.info(f"Short -> TP: -{tp_pct_short*100:.1f}%, SL: +{sl_pct_short*100:.1f}%")
        
        df = df_1m.copy()
        n = len(df)
        
        # 初始化標籤
        df['label_long'] = 0
        df['label_short'] = 0
        
        logger.info("Processing labels (this may take a few minutes)...")
        
        # 向量化計算未來價格
        for i in range(n - lookahead_bars):
            entry_price = df.iloc[i]['close']
            future_prices = df.iloc[i+1:i+1+lookahead_bars][['high', 'low']]
            
            if len(future_prices) < lookahead_bars:
                break
            
            # --- 做多標籤 (Long) ---
            tp_price_long = entry_price * (1 + tp_pct_long)
            sl_price_long = entry_price * (1 - sl_pct_long)
            
            tp_hit_long = (future_prices['high'] >= tp_price_long)
            sl_hit_long = (future_prices['low'] <= sl_price_long)
            
            if tp_hit_long.any():
                tp_bar_long = tp_hit_long.idxmax()
                if sl_hit_long.any():
                    sl_bar_long = sl_hit_long.idxmax()
                    if tp_bar_long <= sl_bar_long:
                        df.iloc[i, df.columns.get_loc('label_long')] = 1
                else:
                    df.iloc[i, df.columns.get_loc('label_long')] = 1
            
            # --- 做空標籤 (Short) ---
            tp_price_short = entry_price * (1 - tp_pct_short)
            sl_price_short = entry_price * (1 + sl_pct_short)
            
            tp_hit_short = (future_prices['low'] <= tp_price_short)
            sl_hit_short = (future_prices['high'] >= sl_price_short)
            
            if tp_hit_short.any():
                tp_bar_short = tp_hit_short.idxmax()
                if sl_hit_short.any():
                    sl_bar_short = sl_hit_short.idxmax()
                    if tp_bar_short <= sl_bar_short:
                        df.iloc[i, df.columns.get_loc('label_short')] = 1
                else:
                    df.iloc[i, df.columns.get_loc('label_short')] = 1
            
            # 進度顯示
            if (i + 1) % 100000 == 0:
                logger.info(f"  Processed {i+1:,}/{n:,} samples ({(i+1)/n*100:.1f}%)")
        
        # 移除無法計算的最後 N 根
        df = df.iloc[:-lookahead_bars]
        
        # 統計
        long_rate = df['label_long'].mean()
        short_rate = df['label_short'].mean()
        both_rate = ((df['label_long'] == 1) & (df['label_short'] == 1)).mean()
        neither_rate = ((df['label_long'] == 0) & (df['label_short'] == 0)).mean()
        
        logger.info("="*80)
        logger.info("LABEL STATISTICS (1m Resolution)")
        logger.info("="*80)
        logger.info(f"Total samples: {len(df):,}")
        logger.info(f"Long  positive: {df['label_long'].sum():,} ({long_rate*100:.2f}%)")
        logger.info(f"Short positive: {df['label_short'].sum():,} ({short_rate*100:.2f}%)")
        logger.info(f"Both  positive: {((df['label_long'] == 1) & (df['label_short'] == 1)).sum():,} ({both_rate*100:.2f}%)")
        logger.info(f"Neither positive: {((df['label_long'] == 0) & (df['label_short'] == 0)).sum():,} ({neither_rate*100:.2f}%)")
        logger.info("="*80)
        
        return df
    
    @staticmethod
    def compress_1m_to_15m(df_1m: pd.DataFrame) -> pd.DataFrame:
        """舊版相容 - 已不使用"""
        logger.warning("compress_1m_to_15m is deprecated. Use broadcast architecture instead.")
        return df_1m
    
    @staticmethod
    def add_micro_labels(df_15m: pd.DataFrame, forward_bars: int = 16) -> pd.DataFrame:
        """舊版相容 - 已不使用"""
        logger.warning("add_micro_labels is deprecated. Use add_bidirectional_labels_1m instead.")
        return df_15m
    
    @staticmethod
    def add_bidirectional_labels(df_15m: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """舊版相容 - 已不使用"""
        logger.warning("add_bidirectional_labels is deprecated. Use add_bidirectional_labels_1m instead.")
        return df_15m
    
    @staticmethod
    def validate_features(df: pd.DataFrame):
        """驗證特徵矩陣質量"""
        logger.info("="*80)
        logger.info("FEATURE VALIDATION")
        logger.info("="*80)
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"\nFeature Statistics:")
        logger.info(df.describe())
        logger.info("="*80)