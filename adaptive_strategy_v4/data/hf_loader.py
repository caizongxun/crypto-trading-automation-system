"""
HuggingFace Data Loader
"""
import pandas as pd
from datasets import load_dataset

class HFDataLoader:
    def __init__(self):
        self.dataset_name = "CaizxMech/crypto-klines-ohlcv"
    
    def load_klines(self, symbol: str, timeframe: str) -> pd.DataFrame:
        print(f"\n[HuggingFace] 加載 {symbol} {timeframe}")
        dataset = load_dataset(self.dataset_name, f"{symbol.lower()}_{timeframe}", split="train")
        df = dataset.to_pandas()
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        print(f"[OK] 加載: {len(df)} 筆")
        return df
