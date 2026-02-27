"""
Binance API Data Loader for V3
Binance API數據加載器
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import requests
import time

class BinanceDataLoader:
    """
    Binance API數據加載器
    
    功能:
    1. 使用Binance公開API加載數據
    2. 支持自定義日期範圍
    3. 自動重試機制
    """
    
    BASE_URL = "https://api.binance.com/api/v3/klines"
    
    # 時間框架對應表
    TIMEFRAME_MAP = {
        '1m': 1,
        '3m': 3,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '2h': 120,
        '4h': 240,
        '6h': 360,
        '12h': 720,
        '1d': 1440
    }
    
    def __init__(self, max_retries: int = 3, retry_delay: int = 1):
        """
        Args:
            max_retries: 最大重試次數
            retry_delay: 重試延遲(秒)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def load_klines(self, symbol: str, timeframe: str, days: int = 30) -> pd.DataFrame:
        """
        加載K線數據
        
        Args:
            symbol: 交易對 (e.g., 'BTCUSDT')
            timeframe: 時間框架 (e.g., '15m', '1h')
            days: 回測天數
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        print(f"\n[Binance API] 加載 {symbol} {timeframe} 最近 {days} 天")
        
        # 計算時間範圍
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # 轉換為毫秒時間戳
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        # 計算需要的查詢次數 (Binance單次最多1000根K線)
        minutes_per_bar = self.TIMEFRAME_MAP.get(timeframe, 15)
        total_bars = (days * 24 * 60) // minutes_per_bar
        num_requests = (total_bars // 1000) + 1
        
        print(f"預計K線數: {total_bars}")
        print(f"需要查詢: {num_requests} 次")
        
        all_data = []
        current_start = start_ts
        
        for i in range(num_requests):
            # 构造請求參數
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'startTime': current_start,
                'endTime': end_ts,
                'limit': 1000
            }
            
            # 發送請求並重試
            data = self._fetch_with_retry(params)
            
            if not data:
                print(f"[警告] 第 {i+1} 次查詢失敗")
                break
            
            all_data.extend(data)
            
            # 更新下次查詢的起始時間
            if len(data) > 0:
                current_start = data[-1][0] + 1
            
            # 如果返回數據少於1000,說明已經到達最新
            if len(data) < 1000:
                break
            
            # 避免 API限制
            time.sleep(0.1)
            
            if (i + 1) % 5 == 0:
                print(f"進度: {i+1}/{num_requests}")
        
        if not all_data:
            raise ValueError("無法加載數據,請檢查網絡連接和交易對名稱")
        
        # 轉換DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # 只保留必要欄位
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # 轉換數據類型
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 清理異常值
        df = df.dropna()
        df = df[df['volume'] > 0]  # 移除成交量為0的K線
        
        # 按時間排序
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"[OK] 加載完成: {len(df)} 筆")
        print(f"時間範圍: {df['timestamp'].iloc[0]} 至 {df['timestamp'].iloc[-1]}")
        
        return df
    
    def _fetch_with_retry(self, params: dict) -> list:
        """
        帶重試機制的數據獲取
        
        Args:
            params: API請求參數
        
        Returns:
            K線數據列表
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.BASE_URL, params=params, timeout=10)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # 限頻
                    print(f"[警告] API限頻,等待60秒...")
                    time.sleep(60)
                else:
                    print(f"[錯誤] HTTP {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                print(f"[錯誤] 請求失敗: {e}")
            
            if attempt < self.max_retries - 1:
                wait_time = self.retry_delay * (attempt + 1)
                print(f"重試 {attempt + 1}/{self.max_retries}, 等待 {wait_time} 秒...")
                time.sleep(wait_time)
        
        return []
    
    def get_available_symbols(self) -> list:
        """
        獲取所有可用交易對
        
        Returns:
            交易對列表
        """
        try:
            response = requests.get('https://api.binance.com/api/v3/exchangeInfo', timeout=10)
            if response.status_code == 200:
                data = response.json()
                symbols = [s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING']
                return symbols
        except Exception as e:
            print(f"[錯誤] 無法獲取交易對列表: {e}")
        
        return []
