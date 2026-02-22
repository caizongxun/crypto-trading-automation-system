import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data_fetcher import DataFetcherTab

def main():
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
        DataFetcherTab().render()
    
    with tab2:
        st.header("模型訓練")
        st.info("模型訓練功能開發中")
    
    with tab3:
        st.header("策略回測")
        st.info("策略回測功能開發中")
    
    with tab4:
        st.header("自動交易")
        st.info("自動交易功能開發中")

if __name__ == "__main__":
    main()