import streamlit as st
import pandas as pd
import ccxt
from datetime import datetime, timedelta
from huggingface_hub import HfApi, hf_hub_download
import os
from pathlib import Path
import time
from config import Config

class DataFetcherTab:
    def __init__(self):
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
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
    def render(self):
        st.header("K棒資料抓取")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_symbol = st.selectbox(
                "選擇交易對",
                self.symbols,
                index=10  # Default to BTCUSDT
            )
        
        with col2:
            timeframe = st.selectbox(
                "時間週期",
                ["1m", "15m", "1h", "1d"],
                index=0
            )
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            if st.button("抓取單一幣種資料", use_container_width=True):
                self.fetch_single_symbol(selected_symbol, timeframe)
        
        with col4:
            if st.button("一鍵抓取所有幣種1m資料", use_container_width=True):
                self.fetch_all_symbols_1m()
    
    def fetch_single_symbol(self, symbol, timeframe):
        """Fetch data for a single symbol and timeframe"""
        with st.spinner(f"正在抓取 {symbol} {timeframe} 資料..."):
            try:
                # Fetch data from Binance
                data = self.fetch_klines_from_binance(symbol, timeframe)
                
                if data is not None and len(data) > 0:
                    # Upload to HuggingFace
                    success = self.upload_to_huggingface(symbol, timeframe, data)
                    
                    if success:
                        st.success(f"成功抓取並上傳 {symbol} {timeframe} 資料，共 {len(data)} 筆")
                        
                        # Display sample data
                        st.subheader("資料預覽")
                        st.dataframe(data.head(10))
                        
                        st.info(f"資料時間範圍: {data['open_time'].min()} 至 {data['open_time'].max()}")
                    else:
                        st.error("上傳資料至 HuggingFace 失敗")
                else:
                    st.warning(f"無法獲取 {symbol} 的資料")
            
            except Exception as e:
                st.error(f"錯誤: {str(e)}")
    
    def fetch_all_symbols_1m(self):
        """Fetch 1m data for all symbols"""
        st.info("開始批次抓取所有幣種的1分鐘K棒資料...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_symbols = len(self.symbols)
        success_count = 0
        failed_symbols = []
        
        for idx, symbol in enumerate(self.symbols):
            status_text.text(f"正在處理 {symbol} ({idx + 1}/{total_symbols})...")
            
            try:
                data = self.fetch_klines_from_binance(symbol, "1m")
                
                if data is not None and len(data) > 0:
                    success = self.upload_to_huggingface(symbol, "1m", data)
                    
                    if success:
                        success_count += 1
                        st.success(f"完成 {symbol}: {len(data)} 筆資料")
                    else:
                        failed_symbols.append(symbol)
                        st.warning(f"上傳失敗: {symbol}")
                else:
                    failed_symbols.append(symbol)
                    st.warning(f"無資料: {symbol}")
                
                # Update progress
                progress_bar.progress((idx + 1) / total_symbols)
                
                # Rate limiting
                time.sleep(1)
            
            except Exception as e:
                failed_symbols.append(symbol)
                st.error(f"錯誤 {symbol}: {str(e)}")
        
        # Final summary
        status_text.text("完成")
        st.success(f"批次抓取完成: 成功 {success_count}/{total_symbols}")
        
        if failed_symbols:
            st.warning(f"失敗的幣種: {', '.join(failed_symbols)}")
    
    def fetch_klines_from_binance(self, symbol, timeframe):
        """Fetch K-line data from Binance"""
        try:
            # Calculate start time (get maximum available history)
            # For 1m data, fetch from as early as possible (Binance allows ~2 years for futures)
            if timeframe == "1m":
                # Start from 2 years ago
                since = int((datetime.now() - timedelta(days=730)).timestamp() * 1000)
            else:
                # For other timeframes, fetch more history
                since = int((datetime.now() - timedelta(days=1825)).timestamp() * 1000)
            
            all_candles = []
            
            while True:
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=1000
                )
                
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # Update since to the last candle's timestamp
                since = candles[-1][0] + 1
                
                # If we got less than 1000 candles, we've reached the end
                if len(candles) < 1000:
                    break
                
                # Rate limiting
                time.sleep(0.5)
            
            if not all_candles:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_candles, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Convert timestamp to datetime
            df['open_time'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['close_time'] = df['open_time'] + pd.Timedelta(self.get_timedelta(timeframe))
            
            # Create proper column structure matching existing format
            result_df = pd.DataFrame({
                'open_time': df['open_time'],
                'open': df['open'],
                'high': df['high'],
                'low': df['low'],
                'close': df['close'],
                'volume': df['volume'],
                'close_time': df['close_time'],
                'quote_asset_volume': 0.0,  # Not available from ccxt
                'number_of_trades': 0,  # Not available from ccxt
                'taker_buy_base_asset_volume': 0.0,  # Not available from ccxt
                'taker_buy_quote_asset_volume': 0.0,  # Not available from ccxt
                'ignore': 0
            })
            
            # Remove duplicates based on open_time
            result_df = result_df.drop_duplicates(subset=['open_time'], keep='first')
            
            # Sort by open_time
            result_df = result_df.sort_values('open_time').reset_index(drop=True)
            
            return result_df
        
        except Exception as e:
            st.error(f"從 Binance 抓取資料時發生錯誤: {str(e)}")
            return None
    
    def get_timedelta(self, timeframe):
        """Convert timeframe string to timedelta"""
        mapping = {
            '1m': 'min',
            '15m': '15min',
            '1h': 'H',
            '1d': 'D'
        }
        return mapping.get(timeframe, 'min')
    
    def upload_to_huggingface(self, symbol, timeframe, data):
        """Upload data to HuggingFace"""
        try:
            # Check if HF token is configured
            if not Config.HF_TOKEN:
                st.error("請在 config.py 中設定 HuggingFace Token")
                return False
            
            api = HfApi(token=Config.HF_TOKEN)
            
            # Prepare file path
            base = symbol.replace("USDT", "")
            filename = f"{base}_{timeframe}.parquet"
            folder_path = f"klines/{symbol}"
            
            # Save to local temp file
            temp_dir = Path("temp_data")
            temp_dir.mkdir(exist_ok=True)
            
            local_file = temp_dir / filename
            data.to_parquet(local_file, index=False)
            
            # Upload to HuggingFace
            api.upload_file(
                path_or_fileobj=str(local_file),
                path_in_repo=f"{folder_path}/{filename}",
                repo_id=Config.HF_REPO_ID,
                repo_type="dataset",
                commit_message=f"Update {symbol} {timeframe} data"
            )
            
            # Clean up temp file
            local_file.unlink()
            
            return True
        
        except Exception as e:
            st.error(f"上傳至 HuggingFace 時發生錯誤: {str(e)}")
            return False