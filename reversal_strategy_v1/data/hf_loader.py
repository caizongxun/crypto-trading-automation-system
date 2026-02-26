"""
HuggingFace Dataset Loader
從HuggingFace加載加密貨幣歷史數據
"""
import pandas as pd
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from pathlib import Path
import os
from datetime import datetime, timedelta

class HFDataLoader:
    def __init__(self, repo_id: str = "zongowo111/v2-crypto-ohlcv-data"):
        self.repo_id = repo_id
        
        self.available_symbols = [
            'AAVEUSDT', 'ADAUSDT', 'ALGOUSDT', 'ARBUSDT', 'ATOMUSDT',
            'AVAXUSDT', 'BALUSDT', 'BATUSDT', 'BCHUSDT', 'BNBUSDT',
            'BTCUSDT', 'COMPUSDT', 'CRVUSDT', 'DOGEUSDT', 'DOTUSDT',
            'ENJUSDT', 'ENSUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT',
            'GALAUSDT', 'GRTUSDT', 'IMXUSDT', 'KAVAUSDT', 'LINKUSDT',
            'LTCUSDT', 'MANAUSDT', 'MATICUSDT', 'MKRUSDT', 'NEARUSDT',
            'OPUSDT', 'SANDUSDT', 'SNXUSDT', 'SOLUSDT', 'SPELLUSDT',
            'UNIUSDT', 'XRPUSDT', 'ZRXUSDT'
        ]
        
        self.available_timeframes = ['1m', '15m', '1h', '1d']
    
    def load_klines(self, symbol: str, timeframe: str, recent_days: int = None) -> pd.DataFrame:
        """
        加載K線數據
        
        Args:
            symbol: 交易對
            timeframe: 時間框架
            recent_days: 只加載最近N天的數據(可選,None則加載所有數據)
        """
        
        if symbol not in self.available_symbols:
            print(f"警告: {symbol} 不在可用符號列表中")
            return pd.DataFrame()
        
        if timeframe not in self.available_timeframes:
            print(f"警告: {timeframe} 不在可用時間框架列表中")
            return pd.DataFrame()
        
        try:
            base = symbol.replace("USDT", "")
            filename = f"{base}_{timeframe}.parquet"
            path_in_repo = f"klines/{symbol}/{filename}"
            
            print(f"正在從HuggingFace下載: {path_in_repo}")
            
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=path_in_repo,
                repo_type="dataset"
            )
            
            df = pd.read_parquet(local_path)
            
            print(f"原始數據欄位: {df.columns.tolist()}")
            print(f"原始數據行數: {len(df)}")
            
            column_mapping = {
                'open_time': 'timestamp',
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            for col in required_cols:
                if col not in df.columns:
                    original_cols = df.columns.tolist()
                    raise ValueError(f"缺少必要欄位: {col}. 當前欄位: {original_cols}")
            
            df = df[required_cols]
            
            if 'timestamp' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            
            # 如果指定了recent_days,只保留最近的數據
            if recent_days is not None:
                cutoff_date = datetime.now() - timedelta(days=recent_days)
                df = df[df['timestamp'] >= cutoff_date]
                print(f"築選最近{recent_days}天數據")
            
            print(f"成功加載 {len(df)} 筆數據")
            if len(df) > 0:
                print(f"時間範圍: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            print(f"加載失敗: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return pd.DataFrame()
    
    def get_available_symbols(self):
        """獲取可用符號列表"""
        return self.available_symbols
    
    def get_available_timeframes(self):
        """獲取可用時間框架列表"""
        return self.available_timeframes
