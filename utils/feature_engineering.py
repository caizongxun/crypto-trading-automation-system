import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger
from utils.micro_structure import MicroStructureEngineer

logger = setup_logger('feature_engineering', 'logs/feature_engineering.log')

class FeatureEngineer:
    """特徵工程 - 整合微觀結構與宏觀指標"""
    
    def __init__(self):
        logger.info("Initialized FeatureEngineer with micro-structure support")
        self.micro_engineer = MicroStructureEngineer()
    
    def create_features_from_1m(self, df_1m: pd.DataFrame, 
                               use_micro_structure: bool = True,
                               label_type: str = 'long') -> pd.DataFrame:
        """
        從 1m K 線生成完整特徵矩陣
        
        Args:
            df_1m: 1m K 線資料
            use_micro_structure: 是否使用微觀結構特徵
            label_type: 標籤類型
                - 'long': 做多標籤 (target)
                - 'short': 做空標籤 (target)
                - 'both': 雙向標籤 (label_long, label_short)
        
        Returns:
            features_df: 完整特徵矩陣 (15m 基礎 + 微觀特徵 + 宏觀指標 + 標籤)
        """
        logger.info("="*80)
        logger.info("FEATURE ENGINEERING PIPELINE")
        logger.info(f"Use micro-structure: {use_micro_structure}")
        logger.info(f"Label type: {label_type}")
        logger.info("="*80)
        
        # 步驟 1: 微觀軌跡壓縮 (1m -> 15m)
        if use_micro_structure:
            logger.info("Step 1: Micro-structure compression (1m -> 15m)")
            df_15m = self.micro_engineer.compress_1m_to_15m(df_1m)
        else:
            logger.info("Step 1: Standard OHLCV resampling (1m -> 15m)")
            df_15m = df_1m.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        
        # 步驟 2: 添加宏觀技術指標
        logger.info("Step 2: Adding macro technical indicators")
        df_15m = self.add_technical_indicators(df_15m)
        
        # 步驟 3: 添加跨週期特徵 (1h, 1d)
        logger.info("Step 3: Adding multi-timeframe features")
        df_15m = self.add_higher_timeframe_features(df_15m, df_1m)
        
        # 步驟 4: 生成標籤
        logger.info(f"Step 4: Generating labels (type={label_type})")
        if use_micro_structure:
            if label_type == 'both':
                # 雙向標籤
                df_15m = self.micro_engineer.add_bidirectional_labels(
                    df_15m, 
                    lookahead_bars=16,
                    tp_pct_long=0.02,
                    sl_pct_long=0.01,
                    tp_pct_short=0.02,
                    sl_pct_short=0.01
                )
            elif label_type == 'long':
                # 做多標籤 (向下相容)
                df_15m = self.micro_engineer.add_micro_labels(df_15m, forward_bars=16)
                df_15m.rename(columns={'target': 'label_long'}, inplace=True)
            elif label_type == 'short':
                # 做空標籤
                df_15m = self.micro_engineer.add_bidirectional_labels(
                    df_15m, 
                    lookahead_bars=16,
                    tp_pct_long=0.02,
                    sl_pct_long=0.01,
                    tp_pct_short=0.02,
                    sl_pct_short=0.01
                )
                df_15m.drop(columns=['label_long'], inplace=True)
                df_15m.rename(columns={'label_short': 'target'}, inplace=True)
            else:
                raise ValueError(f"Invalid label_type: {label_type}. Must be 'long', 'short', or 'both'")
        else:
            # 傳統模式: 預測下一根 15m
            df_15m = self.add_traditional_labels(df_15m)
        
        # 步驟 5: 驗證特徵矩陣
        logger.info("Step 5: Feature validation")
        self.micro_engineer.validate_features(df_15m)
        
        return df_15m
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加 15m 技術指標"""
        df = df.copy()
        
        # Z-Score
        df['z_score'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma_20 + 2 * std_20
        df['bb_lower'] = sma_20 - 2 * std_20
        df['bb_width_pct'] = (df['bb_upper'] - df['bb_lower']) / sma_20
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['atr'] = ranges.max(axis=1).rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close']
        
        return df
    
    def add_higher_timeframe_features(self, df_15m: pd.DataFrame, df_1m: pd.DataFrame) -> pd.DataFrame:
        """添加跨週期特徵 (1h, 1d)"""
        df = df_15m.copy()
        
        # 1h Z-Score
        df_1h = df_1m.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        df_1h['z_score_1h'] = (df_1h['close'] - df_1h['close'].rolling(20).mean()) / df_1h['close'].rolling(20).std()
        
        # 1d ATR
        df_1d = df_1m.resample('1d').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        high_low = df_1d['high'] - df_1d['low']
        high_close = (df_1d['high'] - df_1d['close'].shift()).abs()
        low_close = (df_1d['low'] - df_1d['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df_1d['atr_1d'] = ranges.max(axis=1).rolling(14).mean()
        df_1d['atr_pct_1d'] = df_1d['atr_1d'] / df_1d['close']
        
        # 合併
        df = df.join(df_1h[['z_score_1h']], how='left')
        df = df.join(df_1d[['atr_pct_1d']], how='left')
        
        # 前向填充
        df['z_score_1h'] = df['z_score_1h'].fillna(method='ffill')
        df['atr_pct_1d'] = df['atr_pct_1d'].fillna(method='ffill')
        
        return df
    
    def add_traditional_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """傳統標籤: 下一根 15m 上漲 > 0.3%"""
        df = df.copy()
        df['return'] = df['close'].pct_change().shift(-1)
        df['target'] = (df['return'] > 0.003).astype(int)
        df = df.dropna(subset=['target'])
        df.drop(columns=['return'], inplace=True)
        return df