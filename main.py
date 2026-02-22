import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from tabs.data_fetcher_tab import DataFetcherTab
from tabs.model_training_tab import ModelTrainingTab
from tabs.backtesting_tab import BacktestingTab
from tabs.auto_trading_tab import AutoTradingTab
from utils.logger import setup_logger

# Setup logger
logger = setup_logger('main', 'logs/main.log')

def main():
    logger.info("Starting Crypto Trading Automation System")
    
    st.set_page_config(
        page_title="加密貨幣自動交易系統",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("加密貨幣自動交易系統")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "K棒資料抓取",
        "模型訓練",
        "策略回測",
        "自動交易"
    ])
    
    with tab1:
        logger.info("Loading Data Fetcher Tab")
        DataFetcherTab().render()
    
    with tab2:
        logger.info("Loading Model Training Tab")
        ModelTrainingTab().render()
    
    with tab3:
        logger.info("Loading Backtesting Tab")
        BacktestingTab().render()
    
    with tab4:
        logger.info("Loading Auto Trading Tab")
        AutoTradingTab().render()

if __name__ == "__main__":
    main()