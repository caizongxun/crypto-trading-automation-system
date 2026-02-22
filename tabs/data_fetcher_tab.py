import streamlit as st
import pandas as pd
import ccxt
from datetime import datetime, timedelta
from huggingface_hub import HfApi
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import Config
from utils.logger import setup_logger

logger = setup_logger('data_fetcher', 'logs/data_fetcher.log')

class DataFetcherTab:
    def __init__(self):
        logger.info("Initializing DataFetcherTab")
        self.symbols = [
            "AAVEUSDT", "ADAUSDT", "ALGOUSDT", "ARBUSDT", "ATOMUSDT",
            "AVAXUSDT", "BALUSDT", "BATUSDT", "BCHUSDT", "BNBUSDT",
            "BTCUSDT", "COMPUSDT", "CRVUSDT", "DOGEUSDT", "DOTUSDT",
            "ENJUSDT", "ENSUSDT", "ETCUSDT", "ETHUSDT", "FILUSDT",
            "GALAUSDT", "GRTUSDT", "IMXUSDT", "KAVAUSDT", "LINKUSDT",
            "LTCUSDT", "MANAUSDT", "MATICUSDT", "MKRUSDT", "NEARUSDT",
            "OPUSDT", "SANDUSDT", "SNXUSDT", "SOLUSDT", "SPELLUSDT",
            "UNIUSDT", "XRPUSDT", "ZRXUSDT"
        ]
        logger.info(f"Loaded {len(self.symbols)} trading pairs")
        
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        logger.info("Initialized Binance exchange connection")
        
    def render(self):
        logger.info("Rendering Data Fetcher Tab")
        st.header("K棒資料抓取")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_symbol = st.selectbox(
                "選擇交易對",
                self.symbols,
                index=10  # Default to BTCUSDT
            )
            logger.info(f"Selected symbol: {selected_symbol}")
        
        with col2:
            timeframe = st.selectbox(
                "時間週期",
                ["1m", "15m", "1h", "1d"],
                index=0
            )
            logger.info(f"Selected timeframe: {timeframe}")
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            if st.button("抓取單一幣種資料", use_container_width=True):
                logger.info(f"Button clicked: Fetch single symbol - {selected_symbol} {timeframe}")
                self.fetch_single_symbol(selected_symbol, timeframe)
        
        with col4:
            if st.button("一鍵抓取所有幣種1m資料", use_container_width=True):
                logger.info("Button clicked: Fetch all symbols 1m data")
                self.fetch_all_symbols_1m()
    
    def fetch_single_symbol(self, symbol, timeframe):
        logger.info(f"Starting fetch for {symbol} {timeframe}")
        with st.spinner(f"正在抓取 {symbol} {timeframe} 資料..."):
            try:
                # Fetch data from Binance
                logger.info(f"Fetching klines from Binance for {symbol} {timeframe}")
                data = self.fetch_klines_from_binance(symbol, timeframe)
                
                if data is not None and len(data) > 0:
                    logger.info(f"Successfully fetched {len(data)} records for {symbol} {timeframe}")
                    
                    # Upload to HuggingFace
                    logger.info(f"Uploading data to HuggingFace for {symbol} {timeframe}")
                    success = self.upload_to_huggingface(symbol, timeframe, data)
                    
                    if success:
                        logger.info(f"Upload successful for {symbol} {timeframe}")
                        st.success(f"成功抓取並上傳 {symbol} {timeframe} 資料，共 {len(data)} 筆")
                        
                        # Display sample data
                        st.subheader("資料預覽")
                        st.dataframe(data.head(10))
                        
                        st.info(f"資料時間範圍: {data['open_time'].min()} 至 {data['open_time'].max()}")
                    else:
                        logger.error(f"Upload failed for {symbol} {timeframe}")
                        st.error("上傳資料至 HuggingFace 失敗")
                else:
                    logger.warning(f"No data fetched for {symbol} {timeframe}")
                    st.warning(f"無法獲取 {symbol} 的資料")
            
            except Exception as e:
                logger.error(f"Error in fetch_single_symbol: {str(e)}", exc_info=True)
                st.error(f"錯誤: {str(e)}")
    
    def fetch_all_symbols_1m(self):
        logger.info("Starting batch fetch for all symbols 1m data")
        st.info("開始批次抓取所有幣種的1分鐘K棒資料...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_symbols = len(self.symbols)
        success_count = 0
        failed_symbols = []
        
        logger.info(f"Total symbols to process: {total_symbols}")
        
        for idx, symbol in enumerate(self.symbols):
            logger.info(f"Processing {symbol} ({idx + 1}/{total_symbols})")
            status_text.text(f"正在處理 {symbol} ({idx + 1}/{total_symbols})...")
            
            try:
                logger.info(f"Fetching klines for {symbol}")
                data = self.fetch_klines_from_binance(symbol, "1m")
                
                if data is not None and len(data) > 0:
                    logger.info(f"Fetched {len(data)} records for {symbol}")
                    
                    logger.info(f"Uploading {symbol} to HuggingFace")
                    success = self.upload_to_huggingface(symbol, "1m", data)
                    
                    if success:
                        success_count += 1
                        logger.info(f"Successfully processed {symbol}")
                        st.success(f"完成 {symbol}: {len(data)} 筆資料")
                    else:
                        failed_symbols.append(symbol)
                        logger.warning(f"Upload failed for {symbol}")
                        st.warning(f"上傳失敗: {symbol}")
                else:
                    failed_symbols.append(symbol)
                    logger.warning(f"No data available for {symbol}")
                    st.warning(f"無資料: {symbol}")
                
                # Update progress
                progress_bar.progress((idx + 1) / total_symbols)
                
                # Rate limiting
                logger.info(f"Sleeping 1 second for rate limiting")
                time.sleep(1)
            
            except Exception as e:
                failed_symbols.append(symbol)
                logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)
                st.error(f"錯誤 {symbol}: {str(e)}")
        
        # Final summary
        status_text.text("完成")
        logger.info(f"Batch fetch completed: {success_count}/{total_symbols} successful")
        st.success(f"批次抓取完成: 成功 {success_count}/{total_symbols}")
        
        if failed_symbols:
            logger.warning(f"Failed symbols: {', '.join(failed_symbols)}")
            st.warning(f"失敗的幣種: {', '.join(failed_symbols)}")
    
    def fetch_klines_from_binance(self, symbol, timeframe):
        logger.info(f"fetch_klines_from_binance called for {symbol} {timeframe}")
        try:
            # Calculate start time
            if timeframe == "1m":
                since = int((datetime.now() - timedelta(days=730)).timestamp() * 1000)
                logger.info(f"Fetching 1m data from {datetime.fromtimestamp(since/1000)}")
            else:
                since = int((datetime.now() - timedelta(days=1825)).timestamp() * 1000)
                logger.info(f"Fetching {timeframe} data from {datetime.fromtimestamp(since/1000)}")
            
            all_candles = []
            fetch_count = 0
            
            while True:
                fetch_count += 1
                logger.info(f"Fetch iteration {fetch_count} for {symbol}")
                
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=1000
                )
                
                logger.info(f"Received {len(candles)} candles in iteration {fetch_count}")
                
                if not candles:
                    logger.info(f"No more candles to fetch for {symbol}")
                    break
                
                all_candles.extend(candles)
                logger.info(f"Total candles accumulated: {len(all_candles)}")
                
                # Update since to the last candle's timestamp
                since = candles[-1][0] + 1
                
                # If we got less than 1000 candles, we've reached the end
                if len(candles) < 1000:
                    logger.info(f"Reached end of available data for {symbol} (got {len(candles)} < 1000)")
                    break
                
                # Rate limiting
                time.sleep(0.5)
            
            if not all_candles:
                logger.warning(f"No candles fetched for {symbol}")
                return None
            
            logger.info(f"Converting {len(all_candles)} candles to DataFrame")
            
            # Convert to DataFrame
            df = pd.DataFrame(all_candles, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Convert timestamp to datetime
            df['open_time'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate close_time based on timeframe
            timedelta_str = self.get_timedelta(timeframe)
            logger.info(f"Using timedelta: {timedelta_str}")
            df['close_time'] = df['open_time'] + pd.Timedelta(timedelta_str)
            
            # Create proper column structure
            result_df = pd.DataFrame({
                'open_time': df['open_time'],
                'open': df['open'],
                'high': df['high'],
                'low': df['low'],
                'close': df['close'],
                'volume': df['volume'],
                'close_time': df['close_time'],
                'quote_asset_volume': 0.0,
                'number_of_trades': 0,
                'taker_buy_base_asset_volume': 0.0,
                'taker_buy_quote_asset_volume': 0.0,
                'ignore': 0
            })
            
            # Remove duplicates
            before_dedup = len(result_df)
            result_df = result_df.drop_duplicates(subset=['open_time'], keep='first')
            after_dedup = len(result_df)
            logger.info(f"Removed {before_dedup - after_dedup} duplicate records")
            
            # Sort by open_time
            result_df = result_df.sort_values('open_time').reset_index(drop=True)
            
            logger.info(f"Final DataFrame: {len(result_df)} records")
            logger.info(f"Date range: {result_df['open_time'].min()} to {result_df['open_time'].max()}")
            
            return result_df
        
        except Exception as e:
            logger.error(f"Error in fetch_klines_from_binance: {str(e)}", exc_info=True)
            st.error(f"從 Binance 抓取資料時發生錯誤: {str(e)}")
            return None
    
    def get_timedelta(self, timeframe):
        """Convert timeframe string to pandas Timedelta compatible string"""
        mapping = {
            '1m': '1min',
            '15m': '15min',
            '1h': '1H',
            '1d': '1D'
        }
        result = mapping.get(timeframe, '1min')
        logger.info(f"get_timedelta: {timeframe} -> {result}")
        return result
    
    def upload_to_huggingface(self, symbol, timeframe, data):
        logger.info(f"upload_to_huggingface called for {symbol} {timeframe}")
        try:
            # Check if HF token is configured
            if not Config.HF_TOKEN:
                logger.error("HF_TOKEN not configured")
                st.error("請在 config.py 中設定 HuggingFace Token")
                return False
            
            logger.info("Initializing HuggingFace API")
            api = HfApi(token=Config.HF_TOKEN)
            
            # Prepare file path
            base = symbol.replace("USDT", "")
            filename = f"{base}_{timeframe}.parquet"
            folder_path = f"klines/{symbol}"
            
            logger.info(f"Preparing to upload: {folder_path}/{filename}")
            
            # Save to local temp file
            temp_dir = Path("temp_data")
            temp_dir.mkdir(exist_ok=True)
            logger.info(f"Created temp directory: {temp_dir}")
            
            local_file = temp_dir / filename
            logger.info(f"Saving to local file: {local_file}")
            data.to_parquet(local_file, index=False)
            logger.info(f"Saved {len(data)} records to local file")
            
            # Upload to HuggingFace
            logger.info(f"Uploading to HuggingFace: {Config.HF_REPO_ID}")
            api.upload_file(
                path_or_fileobj=str(local_file),
                path_in_repo=f"{folder_path}/{filename}",
                repo_id=Config.HF_REPO_ID,
                repo_type="dataset",
                commit_message=f"Update {symbol} {timeframe} data"
            )
            logger.info("Upload completed successfully")
            
            # Clean up temp file
            local_file.unlink()
            logger.info(f"Cleaned up temp file: {local_file}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error in upload_to_huggingface: {str(e)}", exc_info=True)
            st.error(f"上傳至 HuggingFace 時發生錯誤: {str(e)}")
            return False