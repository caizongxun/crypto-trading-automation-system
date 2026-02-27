"""
HuggingFace Data Loader for V2 - zongowo111/v2-crypto-ohlcv-data
"""
import pandas as pd
from typing import Optional
from huggingface_hub import hf_hub_download
import requests
from datetime import datetime, timedelta

class HFDataLoader:
    def __init__(self, repo_id: str = "zongowo111/v2-crypto-ohlcv-data"):
        self.repo_id = repo_id
        self.use_binance_fallback = True
    
    def load_klines(self, symbol: str, timeframe: str, 
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> pd.DataFrame:
        """
        加載K線數據
        
        Args:
            symbol: 交易對，例如 'BTCUSDT', 'ETHUSDT'
            timeframe: '1m', '15m', '1h', '1d'
            start_date: 開始時間 (optional)
            end_date: 結束時間 (optional)
        
        Returns:
            pd.DataFrame: K線數據
        """
        
        # 嘗試從HuggingFace加載
        try:
            print(f"從HuggingFace加載 {symbol} {timeframe}")
            df = self._load_from_huggingface(symbol, timeframe)
            
            if len(df) > 0:
                # 篩選時間範圍
                if start_date:
                    df = df[df['timestamp'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['timestamp'] <= pd.to_datetime(end_date)]
                
                print(f"✓ HuggingFace加載成功: {len(df)} 筆")
                return df
            else:
                print(f"HuggingFace數據集為空")
        except Exception as e:
            print(f"HuggingFace加載失敗: {str(e)}")
        
        # 備用: Binance API
        if self.use_binance_fallback:
            print(f"切換到Binance API")
            return self._load_from_binance(symbol, timeframe, start_date, end_date)
        else:
            raise ValueError(f"無法加載 {symbol} {timeframe} 數據")
    
    def _load_from_huggingface(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        從HuggingFace加載K線數據
        
        路徑規則: klines/{SYMBOL}/{BASE}_{TIMEFRAME}.parquet
        例如: klines/BTCUSDT/BTC_1m.parquet
        """
        # 提取BASE (移除USDT)
        base = symbol.replace("USDT", "")
        filename = f"{base}_{timeframe}.parquet"
        path_in_repo = f"klines/{symbol}/{filename}"
        
        print(f"  下載: {path_in_repo}")
        
        # 下載parquet檔案
        local_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=path_in_repo,
            repo_type="dataset"
        )
        
        # 讀取parquet
        df = pd.read_parquet(local_path)
        
        # 處理DataFrame
        df = self._process_huggingface_df(df)
        
        return df
    
    def _process_huggingface_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """處理HuggingFace數據集格式"""
        # 重命名列名以符合V2格式
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
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"缺少必要的列: {missing_columns}")
        
        # 處理時間戳 (已經是datetime格式)
        if df['timestamp'].dtype != 'datetime64[ns]':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 確保按時間排序
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 轉換數值類型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 移除NaN和異常值
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & 
                (df['close'] > 0) & (df['volume'] >= 0)]
        
        return df.reset_index(drop=True)
    
    def _load_from_binance(self, symbol: str, timeframe: str,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """從Binance API加載數據 (備用)"""
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
        
        # 轉揟DataFrame
        df = pd.DataFrame(all_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        
        # 處理時間戳
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # 轉換數值類型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 移除NaN和異常值
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & 
                (df['close'] > 0) & (df['volume'] >= 0)]
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✓ Binance API加載成功: {len(df)} 筆")
        
        return df

# 支援的交易對列表
SUPPORTED_SYMBOLS = [
    'AAVEUSDT', 'ADAUSDT', 'ALGOUSDT', 'ARBUSDT', 'ATOMUSDT',
    'AVAXUSDT', 'BALUSDT', 'BATUSDT', 'BCHUSDT', 'BNBUSDT',
    'BTCUSDT', 'COMPUSDT', 'CRVUSDT', 'DOGEUSDT', 'DOTUSDT',
    'ENJUSDT', 'ENSUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT',
    'GALAUSDT', 'GRTUSDT', 'IMXUSDT', 'KAVAUSDT', 'LINKUSDT',
    'LTCUSDT', 'MANAUSDT', 'MATICUSDT', 'MKRUSDT', 'NEARUSDT',
    'OPUSDT', 'SANDUSDT', 'SNXUSDT', 'SOLUSDT', 'SPELLUSDT',
    'UNIUSDT', 'XRPUSDT', 'ZRXUSDT'
]

# 支援的時間框架
SUPPORTED_TIMEFRAMES = ['1m', '15m', '1h', '1d']
