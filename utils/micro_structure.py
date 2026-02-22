import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger('micro_structure', 'logs/micro_structure.log')

class MicroStructureEngineer:
    """
    微觀結構特徵工程引擎
    
    解決傳統 OHLCV 的資訊遺失問題，捕捉市場微結構 (Market Microstructure)
    的動態博弈過程。
    """
    
    @staticmethod
    def compress_1m_to_15m(df_1m: pd.DataFrame) -> pd.DataFrame:
        """
        將 1m K 線壓縮為 15m，提取微觀軌跡特徵
        
        Args:
            df_1m: 1m K 線資料 (open, high, low, close, volume)
                   必須有 DatetimeIndex 且經過時間排序
        
        Returns:
            df_15m: 15m K 線 + 微觀特徵矩陣
                    - efficiency_ratio: 效率比 (ER)
                    - extreme_time_diff: 極值時序差 (分鐘)
                    - vol_imbalance_ratio: 量能失衡率
        """
        logger.info("Starting micro-structure compression: 1m -> 15m")
        logger.info(f"Input 1m data shape: {df_1m.shape}")
        
        if not isinstance(df_1m.index, pd.DatetimeIndex):
            raise ValueError("Input DataFrame must have a DatetimeIndex.")
        
        df = df_1m.copy()
        
        # 1. 計算 1m 級別的基礎變數 (向量化處理)
        logger.info("Calculating 1m micro features...")
        
        df['price_diff'] = df['close'].diff().abs()
        df['is_up'] = (df['close'] > df['open']).astype(int)
        df['up_vol'] = df['volume'] * df['is_up']
        df['down_vol'] = df['volume'] * (1 - df['is_up'])
        
        # 2. 定義 15 分鐘重採樣聚合器
        grouper = pd.Grouper(freq='15min', closed='left', label='left')
        
        logger.info("Aggregating to 15m timeframe...")
        
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
        logger.info("Extracting extreme points timing...")
        
        high_times = df.groupby(grouper)['high'].idxmax()
        low_times = df.groupby(grouper)['low'].idxmin()
        
        # --- 5. 構建進階微結構特徵 ---
        logger.info("Building advanced micro-structure features...")
        
        # Feature A: 效率比 (Efficiency Ratio)
        # ER = |Net Change| / Sum(|Price Diff|)
        # 接近 1 = 無阻力單邊趨勢
        # 接近 0 = 激烈震盪 (洗盤)
        net_change = (df_15m['close'] - df_15m['open']).abs()
        df_15m['efficiency_ratio'] = np.where(
            df_15m['price_diff'] == 0,
            0,
            net_change / df_15m['price_diff']
        )
        
        # Feature B: 極值時序差 (Sweep Sequence)
        # 正值 = Low 先於 High (先跌後漲的獵取結構, W 底)
        # 負值 = High 先於 Low (誘多結構, M 頭)
        df_15m['extreme_time_diff'] = (high_times - low_times).dt.total_seconds() / 60.0
        
        # Feature C: 微觀量能失衡率 (Volume Imbalance Ratio)
        # CVD Proxy: 範圍 [-1.0, 1.0]
        # 正值 = 買盤吸收 (Bullish Absorption)
        # 負值 = 賣盤壓力 (Bearish Pressure)
        vol_imbalance = df_15m['up_vol'] - df_15m['down_vol']
        df_15m['vol_imbalance_ratio'] = np.where(
            df_15m['volume'] == 0,
            0,
            vol_imbalance / df_15m['volume']
        )
        
        # 6. 清理過渡欄位
        drop_cols = ['price_diff', 'up_vol', 'down_vol']
        df_15m.drop(columns=drop_cols, inplace=True)
        
        # 7. 移除 NaN
        df_15m = df_15m.dropna()
        
        logger.info(f"Output 15m data shape: {df_15m.shape}")
        logger.info("Micro-structure features:")
        logger.info(f"  - efficiency_ratio: mean={df_15m['efficiency_ratio'].mean():.4f}, std={df_15m['efficiency_ratio'].std():.4f}")
        logger.info(f"  - extreme_time_diff: mean={df_15m['extreme_time_diff'].mean():.2f} min")
        logger.info(f"  - vol_imbalance_ratio: mean={df_15m['vol_imbalance_ratio'].mean():.4f}")
        
        return df_15m
    
    @staticmethod
    def add_micro_labels(df_15m: pd.DataFrame, forward_bars: int = 16) -> pd.DataFrame:
        """
        為 15m 的微觀特徵生成預測標籤 (僅做多)
        
        Args:
            df_15m: 15m K 線 + 微觀特徵
            forward_bars: 向前看幾根 15m K 線 (16 = 4 小時)
        
        Returns:
            df_15m: 加入 target 欄位
        """
        logger.info(f"Generating micro-structure labels (forward_bars={forward_bars})")
        
        df = df_15m.copy()
        
        # 計算未來 N 根 15m K 線的最高價
        df['future_high'] = df['high'].shift(-forward_bars).rolling(forward_bars).max()
        
        # 標籤定義: 未來 4 小時內上漲 > 1%
        df['return'] = (df['future_high'] - df['close']) / df['close']
        df['target'] = (df['return'] > 0.01).astype(int)
        
        # 移除無法計算的最後 N 根
        df = df.dropna(subset=['target'])
        
        positive_rate = df['target'].mean()
        logger.info(f"Label distribution: Positive={positive_rate*100:.2f}%, Negative={100-positive_rate*100:.2f}%")
        logger.info(f"Total samples after labeling: {len(df)}")
        
        # 清理過渡欄位
        df.drop(columns=['future_high', 'return'], inplace=True)
        
        return df
    
    @staticmethod
    def add_bidirectional_labels(df_15m: pd.DataFrame, 
                                 lookahead_bars: int = 16,
                                 tp_pct_long: float = 0.02,
                                 sl_pct_long: float = 0.01,
                                 tp_pct_short: float = 0.02,
                                 sl_pct_short: float = 0.01) -> pd.DataFrame:
        """
        生成雙向標籤 (Long + Short Oracles)
        
        Args:
            df_15m: 15m K 線 + 微觀特徵
            lookahead_bars: 向前看幾根 15m K 線 (16 = 4 小時)
            tp_pct_long: 做多停利百分比 (0.02 = 2%)
            sl_pct_long: 做多停損百分比 (0.01 = 1%)
            tp_pct_short: 做空停利百分比 (0.02 = 2%)
            sl_pct_short: 做空停損百分比 (0.01 = 1%)
        
        Returns:
            df_15m: 加入 label_long 和 label_short 欄位
        """
        logger.info("="*80)
        logger.info("BIDIRECTIONAL LABEL GENERATION")
        logger.info("="*80)
        logger.info(f"Lookahead: {lookahead_bars} bars (4 hours)")
        logger.info(f"Long  -> TP: +{tp_pct_long*100:.1f}%, SL: -{sl_pct_long*100:.1f}%")
        logger.info(f"Short -> TP: -{tp_pct_short*100:.1f}%, SL: +{sl_pct_short*100:.1f}%")
        
        df = df_15m.copy()
        n = len(df)
        
        # 初始化標籤
        df['label_long'] = 0
        df['label_short'] = 0
        
        logger.info("Processing labels (vectorized)...")
        
        # 向量化計算未來價格
        for i in range(n - lookahead_bars):
            entry_price = df.iloc[i]['close']
            future_prices = df.iloc[i+1:i+1+lookahead_bars][['high', 'low']]
            
            if len(future_prices) < lookahead_bars:
                break
            
            # --- 做多標籤 (Long) ---
            tp_price_long = entry_price * (1 + tp_pct_long)
            sl_price_long = entry_price * (1 - sl_pct_long)
            
            # 檢查是否先碰到停利
            tp_hit_long = (future_prices['high'] >= tp_price_long)
            sl_hit_long = (future_prices['low'] <= sl_price_long)
            
            if tp_hit_long.any():
                tp_bar_long = tp_hit_long.idxmax()
                if sl_hit_long.any():
                    sl_bar_long = sl_hit_long.idxmax()
                    # 先碰到停利
                    if tp_bar_long <= sl_bar_long:
                        df.iloc[i, df.columns.get_loc('label_long')] = 1
                else:
                    # 只碰到停利
                    df.iloc[i, df.columns.get_loc('label_long')] = 1
            
            # --- 做空標籤 (Short) ---
            tp_price_short = entry_price * (1 - tp_pct_short)
            sl_price_short = entry_price * (1 + sl_pct_short)
            
            # 檢查是否先碰到停利
            tp_hit_short = (future_prices['low'] <= tp_price_short)
            sl_hit_short = (future_prices['high'] >= sl_price_short)
            
            if tp_hit_short.any():
                tp_bar_short = tp_hit_short.idxmax()
                if sl_hit_short.any():
                    sl_bar_short = sl_hit_short.idxmax()
                    # 先碰到停利
                    if tp_bar_short <= sl_bar_short:
                        df.iloc[i, df.columns.get_loc('label_short')] = 1
                else:
                    # 只碰到停利
                    df.iloc[i, df.columns.get_loc('label_short')] = 1
            
            # 進度顯示
            if (i + 1) % 10000 == 0:
                logger.info(f"  Processed {i+1:,}/{n:,} samples ({(i+1)/n*100:.1f}%)")
        
        # 移除無法計算的最後 N 根
        df = df.iloc[:-lookahead_bars]
        
        # 統計
        long_rate = df['label_long'].mean()
        short_rate = df['label_short'].mean()
        both_rate = ((df['label_long'] == 1) & (df['label_short'] == 1)).mean()
        neither_rate = ((df['label_long'] == 0) & (df['label_short'] == 0)).mean()
        
        logger.info("="*80)
        logger.info("LABEL STATISTICS")
        logger.info("="*80)
        logger.info(f"Total samples: {len(df):,}")
        logger.info(f"Long  positive: {df['label_long'].sum():,} ({long_rate*100:.2f}%)")
        logger.info(f"Short positive: {df['label_short'].sum():,} ({short_rate*100:.2f}%)")
        logger.info(f"Both  positive: {((df['label_long'] == 1) & (df['label_short'] == 1)).sum():,} ({both_rate*100:.2f}%)")
        logger.info(f"Neither positive: {((df['label_long'] == 0) & (df['label_short'] == 0)).sum():,} ({neither_rate*100:.2f}%)")
        logger.info("="*80)
        
        return df
    
    @staticmethod
    def validate_features(df_15m: pd.DataFrame):
        """驗證特徵矩陣質量"""
        logger.info("="*80)
        logger.info("MICRO-STRUCTURE FEATURE VALIDATION")
        logger.info("="*80)
        
        logger.info(f"\nDataFrame Info:")
        logger.info(f"Shape: {df_15m.shape}")
        logger.info(f"Columns: {df_15m.columns.tolist()}")
        logger.info(f"Index type: {type(df_15m.index)}")
        logger.info(f"Date range: {df_15m.index.min()} to {df_15m.index.max()}")
        
        logger.info(f"\nFeature Statistics:")
        logger.info(df_15m.describe())
        
        logger.info(f"\nMissing Values:")
        missing = df_15m.isnull().sum()
        logger.info(missing[missing > 0] if missing.sum() > 0 else "No missing values")
        
        logger.info(f"\nSample Data (First 5 rows):")
        logger.info(df_15m.head())
        
        logger.info("="*80)