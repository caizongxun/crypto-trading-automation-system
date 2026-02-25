from huggingface_hub import hf_hub_download
import pandas as pd
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def load_klines(
    symbol: str, 
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fill_missing: bool = True
) -> pd.DataFrame:
    """
    讀取並修正指定幣種和時間週期的 K 線資料
    
    Args:
        symbol: 交易對 'BTCUSDT', 'ETHUSDT'
        timeframe: '1m', '15m', '1h', '1d'
        start_date: 開始日期 'YYYY-MM-DD'
        end_date: 結束日期 'YYYY-MM-DD'
        fill_missing: 是否填補缺失資料
    
    Returns:
        pd.DataFrame: 修正後的 K 線資料
    """
    repo_id = "zongowo111/v2-crypto-ohlcv-data"
    base = symbol.replace("USDT", "")
    filename = f"{base}_{timeframe}.parquet"
    path_in_repo = f"klines/{symbol}/{filename}"

    try:
        # 下載並讀取
        logger.info(f"Loading {symbol} {timeframe} from HuggingFace...")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type="dataset"
        )
        df = pd.read_parquet(local_path)
        
    except Exception as e:
        logger.error(f"Failed to load {symbol} {timeframe}: {e}")
        return pd.DataFrame()
    
    # 確保時間欄位為 datetime
    df['open_time'] = pd.to_datetime(df['open_time'], utc=True)
    df['close_time'] = pd.to_datetime(df['close_time'], utc=True)
    
    # 排序並去重
    df = df.sort_values('open_time').drop_duplicates(subset=['open_time'])
    
    # 填補缺失的時間間隔
    if fill_missing:
        df = fill_missing_candles(df, timeframe)
    
    # 篩選日期範圍
    if start_date:
        df = df[df['open_time'] >= pd.to_datetime(start_date, utc=True)]
    if end_date:
        df = df[df['open_time'] <= pd.to_datetime(end_date, utc=True)]
    
    df = df.reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} candles for {symbol} {timeframe}")
    return df


def fill_missing_candles(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    填補缺失的 K 線資料
    
    策略:
    - 缺失 < 3 根: 用前後值線性插值
    - 缺失 >= 3 根: 保留缺失 (可能是市場休市)
    
    Args:
        df: K線資料
        timeframe: 時間週期
    
    Returns:
        填補後的 DataFrame
    """
    # 時間間隔映射
    interval_map = {
        '1m': '1T',
        '15m': '15T',
        '1h': '1H',
        '1d': '1D'
    }
    
    if timeframe not in interval_map:
        logger.warning(f"Unknown timeframe: {timeframe}")
        return df
    
    freq = interval_map[timeframe]
    
    # 生成完整時間序列
    full_range = pd.date_range(
        start=df['open_time'].min(),
        end=df['open_time'].max(),
        freq=freq
    )
    
    original_len = len(df)
    
    # 重新索引
    df = df.set_index('open_time').reindex(full_range)
    
    # 計算連續缺失數
    missing_mask = df['close'].isna()
    missing_groups = (missing_mask != missing_mask.shift()).cumsum()
    missing_counts = missing_mask.groupby(missing_groups).transform('sum')
    
    # 只填補連續缺失 < 3 的資料
    fill_mask = missing_mask & (missing_counts < 3)
    
    # 用線性插值填補
    cols_to_fill = ['open', 'high', 'low', 'close', 'volume']
    for col in cols_to_fill:
        if col in df.columns:
            df.loc[fill_mask, col] = df[col].interpolate(method='linear', limit=2)
    
    # 移除仍然缺失的行
    df = df.dropna(subset=['close'])
    
    # 恢復 close_time
    df['close_time'] = df.index + pd.Timedelta(freq) - pd.Timedelta('1ms')
    
    df = df.reset_index().rename(columns={'index': 'open_time'})
    
    filled_count = len(df) - original_len
    if filled_count > 0:
        logger.info(f"Filled {filled_count} missing candles")
    
    return df


def load_multi_timeframe(
    symbol: str,
    timeframes: List[str] = ['15m', '1h', '1d'],
    start_date: str = "2024-01-01",
    end_date: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    載入多個時間週期的資料
    
    Args:
        symbol: 交易對
        timeframes: 時間週期列表
        start_date: 開始日期
        end_date: 結束日期
    
    Returns:
        dict: {timeframe: DataFrame}
    """
    data = {}
    for tf in timeframes:
        data[tf] = load_klines(symbol, tf, start_date, end_date)
    return data


def get_available_symbols() -> List[str]:
    """
    返回所有可用的交易對
    
    Returns:
        List[str]: 交易對列表
    """
    return [
        'AAVEUSDT', 'ADAUSDT', 'ALGOUSDT', 'ARBUSDT', 'ATOMUSDT',
        'AVAXUSDT', 'BALUSDT', 'BATUSDT', 'BCHUSDT', 'BNBUSDT',
        'BTCUSDT', 'COMPUSDT', 'CRVUSDT', 'DOGEUSDT', 'DOTUSDT',
        'ENJUSDT', 'ENSUSDT', 'ETCUSDT', 'ETHUSDT', 'FILUSDT',
        'GALAUSDT', 'GRTUSDT', 'IMXUSDT', 'KAVAUSDT', 'LINKUSDT',
        'LTCUSDT', 'MANAUSDT', 'MATICUSDT', 'MKRUSDT', 'NEARUSDT',
        'OPUSDT', 'SANDUSDT', 'SNXUSDT', 'SOLUSDT', 'SPELLUSDT',
        'UNIUSDT', 'XRPUSDT', 'ZRXUSDT'
    ]
