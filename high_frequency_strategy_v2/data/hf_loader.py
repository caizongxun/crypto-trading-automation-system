"""
HuggingFace Data Loader for V2
"""
import pandas as pd
from datasets import load_dataset
from typing import Optional

class HFDataLoader:
    def __init__(self, repo_id: str = "caizongxun/crypto_market_data"):
        self.repo_id = repo_id
    
    def load_klines(self, symbol: str, timeframe: str, 
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> pd.DataFrame:
        """加載K線數據"""
        dataset_name = f"{symbol}_{timeframe}"
        
        try:
            dataset = load_dataset(
                self.repo_id,
                dataset_name,
                split='train'
            )
            
            df = dataset.to_pandas()
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            
            if start_date:
                df = df[df['timestamp'] >= start_date]
            if end_date:
                df = df[df['timestamp'] <= end_date]
            
            return df
            
        except Exception as e:
            print(f"加載失敗: {str(e)}")
            return pd.DataFrame()
