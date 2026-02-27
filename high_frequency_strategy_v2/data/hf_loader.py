"""
HuggingFace Data Loader for V2 with Binance API fallback
"""
import pandas as pd
from typing import Optional
import requests
from datetime import datetime, timedelta

class HFDataLoader:
    def __init__(self, repo_id: str = "caizongxun/crypto_market_data"):
        self.repo_id = repo_id
        self.use_binance_fallback = True
    
    def load_klines(self, symbol: str, timeframe: str, 
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> pd.DataFrame:
        """加載K線數據"""
        
        # 嘗試從HuggingFace加載
        try:
            from datasets import load_dataset
            dataset_name = f"{symbol}_{timeframe}"
            print(f"嘗試加載HuggingFace數據集: {self.repo_id}/{dataset_name}")
            
            dataset = load_dataset(self.repo_id, dataset_name, split='train')
            df = dataset.to_pandas()
            
            if len(df) > 0:
                df = self._process_dataframe(df)
                print(f"✓ HuggingFace加載成功: {len(df)} 筆")
                return df
            else:
                print(f"HuggingFace數據集為空，使用Binance API")
        except Exception as e:
            print(f"HuggingFace加載失敗: {str(e)}")
            print(f"切換到Binance API")
        
        # 備用: Binance API
        if self.use_binance_fallback:
            return self._load_from_binance(symbol, timeframe, start_date, end_date)
        else:
            raise ValueError("無法加載數據")
    
    def _load_from_binance(self, symbol: str, timeframe: str,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """從Binance API加載數據"""
        print(f"從Binance API加載 {symbol} {timeframe}")
        
        # 時間框架轉換
        interval_map = {
            '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m',
            '30m': '30m', '1h': '1h', '2h': '2h', '4h': '4h',
            '6h': '6h', '8h': '8h', '12h': '12h', '1d': '1d',
            '3d': '3d', '1w': '1w', '1M': '1M'
        }
        interval = interval_map.get(timeframe, '1h')
        
        # 預設時間範圍: 最近180天
        if not end_date:
            end_time = datetime.now()
        else:
            end_time = pd.to_datetime(end_date)
        
        if not start_date:
            start_time = end_time - timedelta(days=180)
        else:
            start_time = pd.to_datetime(start_date)
        
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        # Binance API
        url = 'https://api.binance.com/api/v3/klines'
        all_data = []
        
        # 分批拉取(每次最套1000筆)
        current_start = start_ms
        max_requests = 10
        request_count = 0
        
        while current_start < end_ms and request_count < max_requests:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'endTime': end_ms,
                'limit': 1000
            }
            
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                current_start = data[-1][0] + 1
                request_count += 1
                
                print(f"  拉取第 {request_count} 批: {len(data)} 筆")
                
            except Exception as e:
                print(f"  Binance API錯誤: {str(e)}")
                break
        
        if not all_data:
            raise ValueError(f"無法從Binance拉取 {symbol} {timeframe} 數據")
        
        # 轉揝ataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        
        df = self._process_dataframe(df)
        print(f"✓ Binance API加載成功: {len(df)} 筆")
        
        return df
    
    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """處理DataFrame"""
        # 標準化列名
        column_mapping = {
            'open_time': 'timestamp',
            'close_time': 'close_time',
            'quote_asset_volume': 'quote_volume',
            'number_of_trades': 'trades',
            'taker_buy_base_asset_volume': 'taker_buy_volume',
            'taker_buy_quote_asset_volume': 'taker_buy_quote_volume'
        }
        
        rename_dict = {old: new for old, new in column_mapping.items() if old in df.columns}
        if rename_dict:
            df = df.rename(columns=rename_dict)
        
        # 確保必要的列存在
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"缺少必要的列: {missing_columns}")
        
        # 處理時間戳
        if 'timestamp' in df.columns:
            # 如果是毫秒，轉換為datetime
            if df['timestamp'].dtype in ['int64', 'float64']:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 轉換數值類型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 移除NaN
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        # 移除異常值
        df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & 
                (df['close'] > 0) & (df['volume'] >= 0)]
        
        return df.reset_index(drop=True)
