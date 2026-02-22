import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger
from utils.feature_engineer import MultiTimeframeFeatureEngineer

logger = setup_logger('feature_engineering_tab', 'logs/feature_engineering_tab.log')

class FeatureEngineeringTab:
    def __init__(self):
        logger.info("Initializing FeatureEngineeringTab")
        self.engineer = MultiTimeframeFeatureEngineer()
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
    
    def render(self):
        logger.info("Rendering Feature Engineering Tab")
        st.header("特徵工程")
        
        st.markdown("""
        ### 多時間框架特徵工程
        
        基於純價格行為與市場微結構分析，結合 1m、1h、15m 與 1d 四個時間框架。
        
        **特徵層級**:
        - **1m 微觀結構層**: 流動性獲取、結構突破、POC 偏離度
        - **15m 戰術層**: 布林帶 Z-Score、波動率狀態
        - **1h/1d 宏觀層**: 趨勢動能、真實波動率
        """)
        
        st.markdown("---")
        
        # 選擇幣種模式
        col1, col2 = st.columns(2)
        
        with col1:
            mode = st.radio(
                "處理模式",
                ["單一幣種", "批次處理"],
                horizontal=True
            )
        
        with col2:
            if mode == "單一幣種":
                selected_symbol = st.selectbox(
                    "選擇幣種",
                    self.symbols,
                    index=10  # BTCUSDT
                )
            else:
                num_symbols = st.slider(
                    "處理幣種數量",
                    min_value=1,
                    max_value=len(self.symbols),
                    value=10
                )
        
        st.markdown("---")
        
        # 執行按鈕
        if mode == "單一幣種":
            if st.button("執行特徵工程", use_container_width=True):
                self.process_single_symbol(selected_symbol)
        else:
            if st.button("批次執行特徵工程", use_container_width=True):
                self.process_batch_symbols(num_symbols)
    
    def process_single_symbol(self, symbol: str):
        logger.info(f"Processing single symbol: {symbol}")
        
        with st.spinner(f"處理 {symbol} 特徵工程..."):
            try:
                features_df = self.engineer.process_symbol(symbol)
                
                if features_df is not None and len(features_df) > 0:
                    logger.info(f"Feature engineering completed for {symbol}: {len(features_df)} records")
                    
                    st.success(f"特徵工程完成！生成 {len(features_df):,} 筆資料")
                    
                    # 特徵統計
                    st.subheader("特徵統計")
                    feature_cols = [
                        '1m_bull_sweep', '1m_bear_sweep', '1m_bull_bos', '1m_bear_bos',
                        '1m_dist_to_poc', '15m_z_score', '15m_bb_width_pct',
                        '1h_z_score', '1d_atr_pct'
                    ]
                    
                    # 顯示存在的特徵
                    available_features = [col for col in feature_cols if col in features_df.columns]
                    if available_features:
                        st.dataframe(features_df[available_features + ['target']].describe())
                    
                    # 目標分佈
                    st.subheader("目標分佈")
                    target_counts = features_df['target'].value_counts()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("總樣本數", f"{len(features_df):,}")
                    with col2:
                        st.metric("正樣本數", f"{target_counts.get(1, 0):,}")
                    with col3:
                        positive_rate = target_counts.get(1, 0) / len(features_df) * 100
                        st.metric("正樣本比例", f"{positive_rate:.2f}%")
                    
                    st.bar_chart(target_counts)
                    
                    # 保存檔案
                    output_dir = Path("features_output")
                    output_dir.mkdir(exist_ok=True)
                    
                    output_path = output_dir / f"features_{symbol}_multi_tf.parquet"
                    features_df.to_parquet(output_path, index=False)
                    
                    st.success(f"已保存至: {output_path}")
                    logger.info(f"Features saved to {output_path}")
                    
                    # 預覽資料
                    st.subheader("資料預覽")
                    st.dataframe(features_df.head(20))
                    
                else:
                    st.error(f"無法處理 {symbol}")
                    logger.warning(f"No features generated for {symbol}")
            
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)
                st.error(f"處理 {symbol} 時發生錯誤: {str(e)}")
    
    def process_batch_symbols(self, num_symbols: int):
        logger.info(f"Processing batch: {num_symbols} symbols")
        
        st.info(f"開始批次處理 {num_symbols} 個幣種...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        symbols_to_process = self.symbols[:num_symbols]
        success_count = 0
        failed_symbols = []
        
        output_dir = Path("features_output")
        output_dir.mkdir(exist_ok=True)
        
        for idx, symbol in enumerate(symbols_to_process):
            status_text.text(f"正在處理 {symbol} ({idx + 1}/{num_symbols})...")
            logger.info(f"Processing {symbol} ({idx + 1}/{num_symbols})")
            
            try:
                features_df = self.engineer.process_symbol(symbol)
                
                if features_df is not None and len(features_df) > 0:
                    output_path = output_dir / f"features_{symbol}_multi_tf.parquet"
                    features_df.to_parquet(output_path, index=False)
                    
                    success_count += 1
                    logger.info(f"Successfully processed {symbol}: {len(features_df)} records")
                    st.success(f"完成 {symbol}: {len(features_df):,} 筆")
                else:
                    failed_symbols.append(symbol)
                    logger.warning(f"No features for {symbol}")
                    st.warning(f"跳過 {symbol}: 無資料")
            
            except Exception as e:
                failed_symbols.append(symbol)
                logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)
                st.error(f"錯誤 {symbol}: {str(e)}")
            
            progress_bar.progress((idx + 1) / num_symbols)
        
        # 最終摘要
        status_text.text("完成")
        
        st.success(f"批次處理完成: 成功 {success_count}/{num_symbols}")
        logger.info(f"Batch processing completed: {success_count}/{num_symbols} successful")
        
        if failed_symbols:
            st.warning(f"失敗的幣種: {', '.join(failed_symbols)}")
            logger.warning(f"Failed symbols: {', '.join(failed_symbols)}")
        
        st.info(f"所有特徵檔案已保存在: {output_dir}/")