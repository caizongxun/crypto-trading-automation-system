import pandas as pd
import numpy as np
from typing import Dict, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from huggingface_hub import hf_hub_download
from config import Config
from utils.logger import setup_logger

logger = setup_logger('feature_engineer', 'logs/feature_engineer.log')

class MultiTimeframeFeatureEngineer:
    """多時間框架特徵工程師 - 純價格行為與市場微結構"""
    
    def __init__(self):
        logger.info("Initialized MultiTimeframeFeatureEngineer")
    
    def load_klines(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """載入 HuggingFace 資料"""
        try:
            repo_id = Config.HF_REPO_ID
            base = symbol.replace("USDT", "")
            filename = f"{base}_{timeframe}.parquet"
            path_in_repo = f"klines/{symbol}/{filename}"
            
            logger.info(f"Loading {symbol} {timeframe} from HuggingFace")
            
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=path_in_repo,
                repo_type="dataset"
            )
            df = pd.read_parquet(local_path)
            logger.info(f"Loaded {len(df)} records for {symbol} {timeframe}")
            return df
        
        except Exception as e:
            logger.error(f"Failed to load {symbol} {timeframe}: {str(e)}")
            return pd.DataFrame()
    
    def load_all_timeframes(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """載入所有時間框架的資料"""
        logger.info(f"Loading all timeframes for {symbol}")
        
        timeframes = ['1m', '15m', '1h', '1d']
        data = {}
        
        for tf in timeframes:
            df = self.load_klines(symbol, tf)
            if not df.empty:
                data[tf] = df
        
        return data
    
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """策略 1: 提取時間特徵 (交易時段/流動性)"""
        logger.info("Extracting time-based features")
        
        df = df.copy()
        df['open_time'] = pd.to_datetime(df['open_time'])
        
        # 1. 小時 (0-23)
        df['hour'] = df['open_time'].dt.hour
        
        # 2. 星期 (0=星期一, 6=星期日)
        df['day_of_week'] = df['open_time'].dt.dayofweek
        
        # 3. 交易時段 (Session)
        # 亞洲盤: 0-7 (UTC)
        # 歐洲盤: 7-15 (UTC)
        # 美國盤: 13-21 (UTC) - 最高流動性
        df['session_asia'] = ((df['hour'] >= 0) & (df['hour'] < 7)).astype(int)
        df['session_europe'] = ((df['hour'] >= 7) & (df['hour'] < 13)).astype(int)
        df['session_us'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
        df['session_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 15)).astype(int)  # 歐美重疊
        
        # 4. 週末標記 (流動性低)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # 5. 循環編碼 (Cyclical Encoding) - 保持連續性
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        logger.info("Time features extracted: hour, day_of_week, sessions, cyclical")
        return df
    
    def extract_microstructure_features(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        """第一層: 1分鐘微觀結構特徵"""
        logger.info("Extracting 1m microstructure features")
        
        df = df_1m.copy()
        
        # 先提取時間特徵
        df = self.extract_time_features(df)
        
        # 1. 流動性獵取 (Liquidity Sweeps)
        df['pivot_high'] = df['high'].rolling(5, center=True).max() == df['high']
        df['pivot_low'] = df['low'].rolling(5, center=True).min() == df['low']
        
        df['prev_pivot_high'] = df[df['pivot_high']]['high'].shift(1)
        df['prev_pivot_low'] = df[df['pivot_low']]['low'].shift(1)
        
        df['prev_pivot_high'] = df['prev_pivot_high'].ffill()
        df['prev_pivot_low'] = df['prev_pivot_low'].ffill()
        
        # Bull Sweep
        df['1m_bull_sweep'] = (
            (df['high'] > df['prev_pivot_high']) & 
            (df['close'] < df['prev_pivot_high'])
        ).astype(int)
        
        # Bear Sweep
        df['1m_bear_sweep'] = (
            (df['low'] < df['prev_pivot_low']) & 
            (df['close'] > df['prev_pivot_low'])
        ).astype(int)
        
        # 2. 結構突破 (Break of Structure)
        df['body_high'] = np.maximum(df['open'], df['close'])
        df['body_low'] = np.minimum(df['open'], df['close'])
        
        df['recent_high'] = df['high'].rolling(20).max().shift(1)
        df['recent_low'] = df['low'].rolling(20).min().shift(1)
        
        df['1m_bull_bos'] = (df['body_high'] > df['recent_high']).astype(int)
        df['1m_bear_bos'] = (df['body_low'] < df['recent_low']).astype(int)
        
        # 3. 動態 POC (Volume Weighted Price)
        df['vwap'] = (df['close'] * df['volume']).rolling(240).sum() / df['volume'].rolling(240).sum()
        df['1m_dist_to_poc'] = (df['close'] - df['vwap']) / df['vwap']
        
        logger.info("1m microstructure features extracted")
        return df
    
    def extract_tactical_features(self, df_15m: pd.DataFrame) -> pd.DataFrame:
        """第二層: 15分鐘戰術特徵"""
        logger.info("Extracting 15m tactical features")
        
        df = df_15m.copy()
        
        # 布林帶 Z-Score
        window = 20
        df['ma_15m'] = df['close'].rolling(window).mean()
        df['std_15m'] = df['close'].rolling(window).std()
        df['15m_z_score'] = (df['close'] - df['ma_15m']) / df['std_15m']
        
        # 布林帶寬度百分位
        df['upper_bb'] = df['ma_15m'] + (df['std_15m'] * 2)
        df['lower_bb'] = df['ma_15m'] - (df['std_15m'] * 2)
        df['bb_width'] = (df['upper_bb'] - df['lower_bb']) / df['ma_15m']
        df['15m_bb_width_pct'] = df['bb_width'].rolling(100).rank(pct=True)
        
        logger.info("15m tactical features extracted")
        return df
    
    def extract_macro_features(self, df_1h: pd.DataFrame, df_1d: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """第三層: 1h 和 1d 宏觀趨勢特徵"""
        logger.info("Extracting macro trend features")
        
        # 1h 趨勢動能
        df_1h = df_1h.copy()
        window = 20
        df_1h['ma_1h'] = df_1h['close'].rolling(window).mean()
        df_1h['std_1h'] = df_1h['close'].rolling(window).std()
        df_1h['1h_z_score'] = (df_1h['close'] - df_1h['ma_1h']) / df_1h['std_1h']
        
        # 1d ATR 百分比
        df_1d = df_1d.copy()
        df_1d['tr'] = np.maximum(
            df_1d['high'] - df_1d['low'],
            np.maximum(
                abs(df_1d['high'] - df_1d['close'].shift(1)),
                abs(df_1d['low'] - df_1d['close'].shift(1))
            )
        )
        df_1d['atr_1d'] = df_1d['tr'].rolling(14).mean()
        df_1d['1d_atr_pct'] = df_1d['atr_1d'] / df_1d['close']
        
        logger.info("Macro features extracted")
        return df_1h, df_1d
    
    def align_features_to_1m(self, df_1m: pd.DataFrame, df_15m_feat: pd.DataFrame, 
                           df_1h_feat: pd.DataFrame, df_1d_feat: pd.DataFrame) -> pd.DataFrame:
        """關鍵步驟: 防洩漏對齊所有特徵到 1m"""
        logger.info("Aligning features to 1m with anti-leakage")
        
        # 轉換 open_time 為 datetime
        df_1m['open_time'] = pd.to_datetime(df_1m['open_time'])
        df_15m_feat['open_time'] = pd.to_datetime(df_15m_feat['open_time'])
        df_1h_feat['open_time'] = pd.to_datetime(df_1h_feat['open_time'])
        df_1d_feat['open_time'] = pd.to_datetime(df_1d_feat['open_time'])
        
        # 15m 特徵: 位移一根 K 線後合併
        df_15m_shift = df_15m_feat[['open_time', '15m_z_score', '15m_bb_width_pct']].copy()
        df_15m_shift = df_15m_shift.shift(1)
        
        # 1h 特徵: 位移一根 K 線後合併
        df_1h_shift = df_1h_feat[['open_time', '1h_z_score']].copy()
        df_1h_shift = df_1h_shift.shift(1)
        
        # 1d 特徵: 位移一根 K 線後合併
        df_1d_shift = df_1d_feat[['open_time', '1d_atr_pct']].copy()
        df_1d_shift = df_1d_shift.shift(1)
        
        # 合併到 1m
        features_df = df_1m.merge(df_15m_shift, on='open_time', how='left')
        features_df = features_df.merge(df_1h_shift, on='open_time', how='left')
        features_df = features_df.merge(df_1d_shift, on='open_time', how='left')
        
        # 前向填充空值
        feature_cols = ['15m_z_score', '15m_bb_width_pct', '1h_z_score', '1d_atr_pct']
        features_df[feature_cols] = features_df[feature_cols].ffill()
        
        logger.info(f"Feature alignment completed: {len(features_df)} records")
        return features_df
    
    def create_target(self, df: pd.DataFrame, look_forward: int = 5, threshold: float = 0.001) -> pd.Series:
        """創建目標變數: 未來 N 分鐘漲跌標籤"""
        logger.info(f"Creating target with look_forward={look_forward}, threshold={threshold}")
        
        df['future_return'] = df['close'].shift(-look_forward) / df['close'] - 1
        df['target'] = (df['future_return'] > threshold).astype(int)
        
        return df['target']
    
    def process_symbol(self, symbol: str) -> pd.DataFrame:
        """處理單一幣種的完整特徵工程流程"""
        logger.info(f"Processing {symbol}")
        
        try:
            # 載入資料
            data = self.load_all_timeframes(symbol)
            if '1m' not in data or data['1m'].empty:
                logger.error(f"No 1m data for {symbol}")
                return None
            
            # 特徵工程
            df_1m_feat = self.extract_microstructure_features(data['1m'])
            
            if '15m' in data and not data['15m'].empty:
                df_15m_feat = self.extract_tactical_features(data['15m'])
            else:
                df_15m_feat = pd.DataFrame()
            
            if '1h' in data and not data['1h'].empty and '1d' in data and not data['1d'].empty:
                df_1h_feat, df_1d_feat = self.extract_macro_features(data['1h'], data['1d'])
            else:
                df_1h_feat = pd.DataFrame()
                df_1d_feat = pd.DataFrame()
            
            # 對齊特徵
            if not df_15m_feat.empty and not df_1h_feat.empty and not df_1d_feat.empty:
                features_df = self.align_features_to_1m(
                    df_1m_feat, df_15m_feat, df_1h_feat, df_1d_feat
                )
            else:
                features_df = df_1m_feat
                logger.warning(f"Some timeframes missing for {symbol}, using 1m features only")
            
            # 創建目標
            features_df['target'] = self.create_target(features_df)
            
            # 移除 NaN
            initial_count = len(features_df)
            features_df = features_df.dropna()
            logger.info(f"Dropped {initial_count - len(features_df)} rows with NaN")
            
            logger.info(f"{symbol} processed: {len(features_df)} records")
            return features_df
        
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)
            return None