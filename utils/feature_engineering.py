import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger
from utils.micro_structure import MicroStructureEngineer

logger = setup_logger('feature_engineering', 'logs/feature_engineering.log')

class FeatureEngineer:
    """特徵工程 - 全面採用 Rolling Window (滾動視窗) 零延遲架構
    
    **核心概念**:
    - 完全拋棄 resample + shift + merge_asof (會造成 14 分鐘盲區)
    - 改用 rolling(期數) 直接在 1m 級別計算
    - 當時間走到 10:14,模型看到的就是 10:00~10:14 的軌跡
    - 零延遲、零盲區、且絕對沒有未來函數
    """
    
    def __init__(self):
        logger.info("Initialized FeatureEngineer with Rolling Window Architecture")
        self.micro_engineer = MicroStructureEngineer()
    
    def create_features_from_1m(self, df_1m: pd.DataFrame, 
                               use_micro_structure: bool = True,
                               label_type: str = 'both') -> pd.DataFrame:
        """主流程: 從 1m K 線生成所有特徵 (滾動視窗架構)
        
        Args:
            df_1m: 原始 1m K 線 (OHLCV)
            use_micro_structure: 是否計算 15m 微觀特徵
            label_type: 'both', 'long', 'short'
        
        Returns:
            df: 完整特徵矩陣 + 標籤
        """
        logger.info("="*80)
        logger.info("FEATURE ENGINEERING PIPELINE (ROLLING WINDOW ARCHITECTURE)")
        logger.info("="*80)
        
        df = df_1m.copy()
        logger.info(f"Input: {len(df):,} 1m bars")
        
        # 步驟 1: 滾動計算 15m 微觀軌跡 (零延遲)
        if use_micro_structure:
            logger.info("Step 1: Calculating Rolling 15m Micro Features...")
            df = self.add_rolling_micro_features(df, period=15)
            logger.info(f"  -> Added: efficiency_ratio, vol_imbalance_ratio, extreme_time_diff")
        
        # 步驟 2: 滾動計算 15m 技術指標
        logger.info("Step 2: Calculating Rolling 15m Technicals...")
        df = self.add_rolling_technicals(df, period=15)
        logger.info(f"  -> Added: z_score, bb_width_pct, rsi, atr_pct")
        
        # 步驟 3: 滾動計算巨觀指標 (1h, 1d)
        logger.info("Step 3: Calculating Rolling Macro Features (1h, 1d)...")
        df = self.add_rolling_macro_features(df)
        logger.info(f"  -> Added: z_score_1h, atr_pct_1d")
        
        # 步驟 4: 在 1m 級別生成精準的雙向標籤
        logger.info(f"Step 4: Generating Labels (type={label_type})...")
        if label_type == 'both':
            df = self.micro_engineer.add_bidirectional_labels_1m(
                df, 
                lookahead_bars=240,  # 4 小時
                tp_pct_long=0.02, sl_pct_long=0.01,
                tp_pct_short=0.02, sl_pct_short=0.01
            )
            logger.info(f"  -> Added: label_long, label_short")
        elif label_type == 'long':
            df = self.micro_engineer.add_bidirectional_labels_1m(
                df, lookahead_bars=240,
                tp_pct_long=0.02, sl_pct_long=0.01,
                tp_pct_short=0, sl_pct_short=0
            )
            df = df.drop(columns=['label_short'], errors='ignore')
        elif label_type == 'short':
            df = self.micro_engineer.add_bidirectional_labels_1m(
                df, lookahead_bars=240,
                tp_pct_long=0, sl_pct_long=0,
                tp_pct_short=0.02, sl_pct_short=0.01
            )
            df = df.drop(columns=['label_long'], errors='ignore')
        
        # 移除因 rolling 產生的 NaN 列 (前 1440 根 K 線會是 NaN)
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        logger.info(f"Step 5: Dropped {dropped:,} rows due to rolling windows.")
        logger.info(f"Final output: {len(df):,} clean samples")
        logger.info("="*80)
        
        return df

    def add_rolling_micro_features(self, df: pd.DataFrame, period: int = 15) -> pd.DataFrame:
        """核心: 在 1m 上直接計算過去 15 根 K 線的微結構
        
        解決 14 分鐘盲區問題:
        - 舊版: 10:14 看到的是 09:45~09:59 (延遲 15 分鐘)
        - 新版: 10:14 看到的是 10:00~10:14 (零延遲)
        """
        # 基礎計算
        df['price_diff'] = df['close'].diff().abs()
        df['is_up'] = (df['close'] > df['open']).astype(int)
        df['up_vol'] = df['volume'] * df['is_up']
        df['down_vol'] = df['volume'] * (1 - df['is_up'])
        
        # Feature A: 效率比 (Efficiency Ratio)
        # 滾動 15 分鐘的總價格波動 vs 淨變化
        rolling_price_diff = df['price_diff'].rolling(period).sum()
        rolling_net_change = (df['close'] - df['close'].shift(period)).abs()
        df['efficiency_ratio'] = np.where(
            rolling_price_diff == 0, 
            0, 
            rolling_net_change / rolling_price_diff
        )
        
        # Feature B: 微觀量能失衡率
        # 滾動 15 分鐘的買賣壓力
        rolling_up_vol = df['up_vol'].rolling(period).sum()
        rolling_down_vol = df['down_vol'].rolling(period).sum()
        rolling_tot_vol = df['volume'].rolling(period).sum()
        df['vol_imbalance_ratio'] = np.where(
            rolling_tot_vol == 0, 
            0, 
            (rolling_up_vol - rolling_down_vol) / rolling_tot_vol
        )
        
        # Feature C: 極值時序差 (使用 apply 取得 index)
        logger.info("  -> Calculating extreme sequence (takes ~15 seconds)...")
        high_pos = df['high'].rolling(period).apply(np.argmax, raw=True)
        low_pos = df['low'].rolling(period).apply(np.argmin, raw=True)
        df['extreme_time_diff'] = high_pos - low_pos
        
        # 清理暫存欄位
        df.drop(columns=['price_diff', 'is_up', 'up_vol', 'down_vol'], inplace=True)
        return df

    def add_rolling_technicals(self, df: pd.DataFrame, period: int = 15) -> pd.DataFrame:
        """滾動 15 分鐘技術指標: 布林帶、RSI、ATR"""
        # Z-score (15m)
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        df['z_score'] = np.where(std == 0, 0, (df['close'] - sma) / std)
        
        # 布林帶寬度 (15m)
        df['bb_width_pct'] = np.where(sma == 0, 0, 4 * std / sma)
        
        # RSI (15m)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR (15m)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr_pct'] = tr.rolling(period).mean() / df['close']
        
        return df

    def add_rolling_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """完全捨棄 resample，使用 rolling 計算大週期，徹底消滅未來洩漏"""
        # 1h (60分鐘) Z-score
        period_1h = 60
        sma_1h = df['close'].rolling(period_1h).mean()
        std_1h = df['close'].rolling(period_1h).std()
        df['z_score_1h'] = np.where(std_1h == 0, 0, (df['close'] - sma_1h) / std_1h)
        
        # 1d (1440分鐘) 真實波動率
        period_1d = 1440
        rolling_range_1d = df['high'].rolling(period_1d).max() - df['low'].rolling(period_1d).min()
        df['atr_pct_1d'] = rolling_range_1d / df['close']
        
        return df
    
    def validate_features(self, df: pd.DataFrame) -> dict:
        """驗證特徵品質"""
        report = {
            'total_samples': len(df),
            'feature_count': len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]),
            'null_count': df.isnull().sum().sum(),
            'inf_count': np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        }
        
        if 'label_long' in df.columns:
            report['long_positive_rate'] = df['label_long'].mean()
        if 'label_short' in df.columns:
            report['short_positive_rate'] = df['label_short'].mean()
        
        logger.info(f"Feature validation: {report}")
        return report