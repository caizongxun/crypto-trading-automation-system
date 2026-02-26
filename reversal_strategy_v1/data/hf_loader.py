"""
HuggingFace Dataset Loader
從HuggingFace加載加密貨幣歷史數據
"""
import pandas as pd
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from pathlib import Path
import os

class HFDataLoader:
    def __init__(self, repo_id: str = "zongowo111/v2-crypto-ohlcv-data"):
        self.repo_id = repo_id
        
        self.available_symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOGEUSDT', 'MATICUSDT', 'DOTUSDT', 'AVAXUSDT',
            'UNIUSDT', 'LINKUSDT', 'ATOMUSDT', 'LTCUSDT', 'NEARUSDT',
            'ALGOUSDT', 'VETUSDT', 'ICPUSDT', 'FILUSDT', 'HBARUSDT',
            'APTUSDT', 'ARBUSDT', 'OPUSDT', 'INJUSDT', 'STXUSDT',
            'TIAUSDT', 'SEIUSDT', 'PENDLEUSDT', 'WLDUSDT', 'TAOUSDT',
            'AAVEUSDT', 'COMPUSDT', 'MKRUSDT', 'CRVUSDT', 'SNXUSDT',
            'LDOUSDT', 'RNDRUSDT', 'SUSHIUSDT'
        ]
        
        self.available_timeframes = ['15m', '1h', '4h']
    
    def load_klines(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """加載K線數據"""
        
        if symbol not in self.available_symbols:
            print(f"警告: {symbol} 不在可用符號列表中")
            return pd.DataFrame()
        
        if timeframe not in self.available_timeframes:
            print(f"警告: {timeframe} 不在可用時間框架列表中")
            return pd.DataFrame()
        
        try:
            filename = f"{symbol}_{timeframe}.parquet"
            
            print(f"正在從HuggingFace下載: {filename}")
            
            file_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                repo_type="dataset"
            )
            
            df = pd.read_parquet(file_path)
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"成功加載 {len(df)} 筆數據")
            
            return df
            
        except Exception as e:
            print(f"加載失敗: {str(e)}")
            return pd.DataFrame()
    
    def get_available_symbols(self):
        """獲取可用符號列表"""
        return self.available_symbols
    
    def get_available_timeframes(self):
        """獲取可用時間框架列表"""
        return self.available_timeframes
