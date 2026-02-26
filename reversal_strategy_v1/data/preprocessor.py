"""
Data Preprocessor
數據預處理工具
"""
import pandas as pd
import numpy as np

class DataPreprocessor:
    @staticmethod
    def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """清理OHLCV數據"""
        df = df.copy()
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"缺少必要列: {col}")
        
        df = df.dropna(subset=required_columns)
        
        df = df[(df['high'] >= df['low']) & 
                (df['high'] >= df['open']) & 
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) & 
                (df['low'] <= df['close']) &
                (df['volume'] >= 0)]
        
        return df
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, column: str, n_std: float = 3.0) -> pd.DataFrame:
        """移除異常值"""
        df = df.copy()
        mean = df[column].mean()
        std = df[column].std()
        df = df[abs(df[column] - mean) <= n_std * std]
        return df
