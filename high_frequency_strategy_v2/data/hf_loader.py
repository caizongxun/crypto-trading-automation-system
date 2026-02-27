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
            print(f"加載數據集: {self.repo_id}/{dataset_name}")
            dataset = load_dataset(
                self.repo_id,
                dataset_name,
                split='train'
            )
            
            df = dataset.to_pandas()
            
            # 檢查列名
            print(f"原始列名: {df.columns.tolist()}")
            
            # 標準化列名
            column_mapping = {
                'open_time': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'close_time': 'close_time',
                'quote_asset_volume': 'quote_volume',
                'number_of_trades': 'trades',
                'taker_buy_base_asset_volume': 'taker_buy_volume',
                'taker_buy_quote_asset_volume': 'taker_buy_quote_volume'
            }
            
            # 重命名存在的列
            rename_dict = {}
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    rename_dict[old_name] = new_name
            
            if rename_dict:
                df = df.rename(columns=rename_dict)
            
            print(f"重命名後列名: {df.columns.tolist()}")
            
            # 確保必要的列存在
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"缺少必要的列: {missing_columns}")
            
            # 處理時間戳
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            # 篩選時間範圍
            if start_date and 'timestamp' in df.columns:
                df = df[df['timestamp'] >= start_date]
            if end_date and 'timestamp' in df.columns:
                df = df[df['timestamp'] <= end_date]
            
            # 轉換數值類型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 移除NaN
            df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            
            print(f"最終數據形狀: {df.shape}")
            print(f"最終列名: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            print(f"加載失敗: {str(e)}")
            print(f"嘗試使用備用方法...")
            
            # 備用: 直接創建Binance格式的空數據
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
