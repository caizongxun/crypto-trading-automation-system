"""
Binance API Data Loader (copy from V3)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time

class BinanceDataLoader:
    BASE_URL = "https://api.binance.com/api/v3/klines"
    
    TIMEFRAME_MAP = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360, '12h': 720, '1d': 1440
    }
    
    def __init__(self, max_retries: int = 3, retry_delay: int = 1):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def load_klines(self, symbol: str, timeframe: str, days: int = 30) -> pd.DataFrame:
        print(f"\n[Binance API] 加載 {symbol} {timeframe} 最近 {days} 天")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        minutes_per_bar = self.TIMEFRAME_MAP.get(timeframe, 15)
        total_bars = (days * 24 * 60) // minutes_per_bar
        num_requests = (total_bars // 1000) + 1
        
        all_data = []
        current_start = start_ts
        
        for i in range(num_requests):
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'startTime': current_start,
                'endTime': end_ts,
                'limit': 1000
            }
            
            data = self._fetch_with_retry(params)
            if not data:
                break
            
            all_data.extend(data)
            
            if len(data) > 0:
                current_start = data[-1][0] + 1
            
            if len(data) < 1000:
                break
            
            time.sleep(0.1)
        
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        df = df[df['volume'] > 0]
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"[OK] 加載完成: {len(df)} 筆")
        return df
    
    def _fetch_with_retry(self, params: dict) -> list:
        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.BASE_URL, params=params, timeout=10)
                if response.status_code == 200:
                    return response.json()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        return []
