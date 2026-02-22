# Multi-Timeframe Feature Engineering
# 第一步：多時間框架特徵工程

from typing import Dict, Tuple
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from tabs.data_fetcher_tab import load_klines  # 使用您現有的 HF 讀取函數
from utils.logger import setup_logger

logger = setup_logger('feature_engineering', 'logs/feature_engineering.log')

class MultiTimeframeFeatureEngineer:
    """多時間框架特徵工程師 - 純價格行為與市場微結構"""
    
    def __init__(self):
        self.symbols = [
            "BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "MATICUSDT",
            "AVAXUSDT", "DOTUSDT", "LINKUSDT", "UNIUSDT", "LTCUSDT"
        ]  # 先用 10 個主要幣種測試
        logger.info("Initialized MultiTimeframeFeatureEngineer")
    
    def load_all_timeframes(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """載入所有時間框架的資料"""
        logger.info(f"Loading all timeframes for {symbol}")
        
        timeframes = ['1m', '15m', '1h', '1d']
        data = {}
        
        for tf in timeframes:
            try:
                df = load_klines(symbol, tf)
                data[tf] = df
                logger.info(f"Loaded {len(df)} records for {symbol} {tf}")
            except Exception as e:
                logger.warning(f"Failed to load {symbol} {tf}: {e}")
                continue
        
        return data
    
    def extract_microstructure_features(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        """第一層：1分鐘微觀結構特徵"""
        logger.info("Extracting 1m microstructure features")
        
        df = df_1m.copy()
        
        # 1. 流動性獵取 (Liquidity Sweeps)
        df['pivot_high'] = df['high'].rolling(5, center=True).max() == df['high']
        df['pivot_low'] = df['low'].rolling(5, center=True).min() == df['low']
        
        df['prev_pivot_high'] = df['pivot_high'].shift(1)
        df['prev_pivot_low'] = df['pivot_low'].shift(1)
        
        # Bull Sweep: 價格突破前高後迅速回落
        df['1m_bull_sweep'] = (
            (df['high'] > df['prev_pivot_high']) & 
            (df['close'] < df['prev_pivot_high']) & 
            df['prev_pivot_high']
        ).astype(int)
        
        # Bear Sweep: 價格突破前低後迅速反彈
        df['1m_bear_sweep'] = (
            (df['low'] < df['prev_pivot_low']) & 
            (df['close'] > df['prev_pivot_low']) & 
            df['prev_pivot_low']
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
        
        return df
    
    def extract_tactical_features(self, df_15m: pd.DataFrame) -> pd.DataFrame:
        """第二層：15分鐘戰術特徵"""
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
        
        return df
    
    def extract_macro_features(self, df_1h: pd.DataFrame, df_1d: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """第三層：1h 和 1d 宏觀趨勢特徵"""
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
        
        return df_1h, df_1d
    
    def align_features_to_1m(self, df_1m: pd.DataFrame, df_15m_feat: pd.DataFrame, 
                           df_1h_feat: pd.DataFrame, df_1d_feat: pd.DataFrame) -> pd.DataFrame:
        """關鍵步驟：防洩漏對齊所有特徵到 1m"""
        logger.info("Aligning features to 1m with anti-leakage")
        
        # 15m 特徵：位移一根 15m K 線後前向填充
        df_15m_aligned = df_15m_feat.shift(1).ffill()
        
        # 1h 特徵：位移一根 1h K 線後前向填充
        df_1h_aligned = df_1h_feat.shift(1).ffill()
        
        # 1d 特徵：位移一根 1d K 線後前向填充
        df_1d_aligned = df_1d_feat.shift(1).ffill()
        
        # 合併到 1m（使用 left join 確保時間對齊）
        features_df = df_1m.merge(
            df_15m_aligned[['open_time', '15m_z_score', '15m_bb_width_pct']], 
            on='open_time', how='left'
        ).merge(
            df_1h_aligned[['open_time', '1h_z_score']], 
            on='open_time', how='left'
        ).merge(
            df_1d_aligned[['open_time', '1d_atr_pct']], 
            on='open_time', how='left'
        )
        
        # 前向填充空值
        feature_cols = ['15m_z_score', '15m_bb_width_pct', '1h_z_score', '1d_atr_pct']
        features_df[feature_cols] = features_df[feature_cols].ffill()
        
        logger.info(f"Feature alignment completed: {len(features_df)} records")
        return features_df
    
    def create_target(self, df: pd.DataFrame, look_forward: int = 5) -> pd.Series:
        """創建目標變數：未來 N 分鐘漲跌標籤"""
        df['future_return'] = df['close'].shift(-look_forward) / df['close'] - 1
        df['target'] = (df['future_return'] > 0.001).astype(int)  # 0.1% 漲幅
        return df['target']
    
    def process_symbol(self, symbol: str) -> pd.DataFrame:
        """處理單一幣種的完整特徵工程流程"""
        logger.info(f"Processing {symbol}")
        
        # 載入資料
        data = self.load_all_timeframes(symbol)
        if '1m' not in data:
            logger.error(f"No 1m data for {symbol}")
            return None
        
        # 特徵工程
        df_1m_feat = self.extract_microstructure_features(data['1m'])
        df_15m_feat = self.extract_tactical_features(data.get('15m', pd.DataFrame()))
        df_1h_feat, df_1d_feat = self.extract_macro_features(
            data.get('1h', pd.DataFrame()), 
            data.get('1d', pd.DataFrame())
        )
        
        # 對齊特徵
        features_df = self.align_features_to_1m(
            df_1m_feat, df_15m_feat, df_1h_feat, df_1d_feat
        )
        
        # 創建目標
        features_df['target'] = self.create_target(features_df)
        
        # 移除 NaN
        features_df = features_df.dropna()
        
        logger.info(f"{symbol} processed: {len(features_df)} records")
        return features_df


def main():
    st.title("多時間框架特徵工程 - 第一步")
    
    engineer = MultiTimeframeFeatureEngineer()
    
    symbol = st.selectbox("選擇幣種", engineer.symbols, index=0)
    
    if st.button("執行特徵工程"):
        with st.spinner(f"處理 {symbol} ..."):
            features_df = engineer.process_symbol(symbol)
            
            if features_df is not None:
                st.success(f"特徵工程完成！生成 {len(features_df):,} 筆資料")
                
                st.subheader("特徵統計")
                feature_cols = [
                    '1m_bull_sweep', '1m_bear_sweep', '1m_bull_bos', '1m_bear_bos',
                    '1m_dist_to_poc', '15m_z_score', '15m_bb_width_pct',
                    '1h_z_score', '1d_atr_pct'
                ]
                st.dataframe(features_df[feature_cols + ['target']].describe())
                
                st.subheader("目標分佈")
                st.bar_chart(features_df['target'].value_counts())
                
                # 保存檔案
                output_path = f"features_{symbol}_multi_tf.parquet"
                features_df.to_parquet(output_path, index=False)
                st.success(f"已保存至: {output_path}")
            else:
                st.error("特徵工程失敗")


if __name__ == "__main__":
    main()

# 使用方式：
# 1. python feature_engineering.py
# 2. 或在 Streamlit 中執行
# 3. 產出 features_BTCUSDT_multi_tf.parquet
# 4. 此檔案可直接用於模型訓練