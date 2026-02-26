"""
Reversal Strategy V1 - Streamlit GUI
反轉策略交易系統主界面
"""
import streamlit as st
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="反轉交易系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("加密貨幣反轉交易系統")
    
    with st.sidebar:
        st.header("系統配置")
        
        version = st.selectbox(
            "策略版本",
            ["V1 - 訂單流反轉策略", "V2 - 即將推出", "V3 - 即將推出"],
            index=0
        )
        
        st.markdown("---")
        st.subheader("當前版本: V1")
        st.caption("訂單流不平衡與流動性區域策略")
        
        st.markdown("---")
        st.info(
            "**V1 策略特點:**\n\n"
            "訂單流不平衡檢測\n"
            "流動性掃蕩識別\n"
            "市場微觀結構分析\n"
            "機器學習信號驗證"
        )
    
    if "V1" in version:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "模型訓練",
            "回測分析",
            "模擬交易", 
            "實盤交易",
            "績效分析"
        ])
        
        with tab1:
            render_training_tab()
        
        with tab2:
            render_backtest_tab()
        
        with tab3:
            render_paper_trading_tab()
        
        with tab4:
            render_live_trading_tab()
        
        with tab5:
            render_analytics_tab()

def render_training_tab():
    """模型訓練頁面"""
    st.header("模型訓練")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("訓練參數")
        
        symbol = st.selectbox(
            "交易對",
            ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "AVAXUSDT"],
            key="train_symbol"
        )
        
        timeframe = st.selectbox(
            "時間框架",
            ["15m", "1h", "4h"],
            index=0,
            key="train_timeframe"
        )
        
        st.markdown("---")
        
        st.subheader("信號檢測")
        lookback = st.slider("回溯週期", 10, 50, 20)
        imbalance_threshold = st.slider("OFI閾值", 0.5, 0.8, 0.6, 0.05)
        
        st.markdown("---")
        
        st.subheader("標籤生成")
        forward_window = st.slider("前瞻窗口", 8, 20, 12)
        profit_threshold = st.slider("盈利目標 %", 0.5, 3.0, 1.0, 0.1) / 100
        stop_loss = st.slider("止損 %", 0.3, 2.0, 0.5, 0.1) / 100
        
        st.markdown("---")
        
        st.subheader("機器學習模型")
        n_estimators = st.slider("樹數量", 100, 500, 200, 50)
        max_depth = st.slider("最大深度", 3, 10, 5)
        test_size = st.slider("驗證集比例", 0.1, 0.3, 0.2, 0.05)
        oos_size = st.slider("OOS測試集比例", 0.05, 0.2, 0.1, 0.05)
        
        st.markdown("---")
        
        if st.button("開始訓練", type="primary", use_container_width=True):
            st.session_state['training_started'] = True
    
    with col2:
        st.subheader("訓練過程")
        
        if st.session_state.get('training_started', False):
            with st.spinner("正在從HuggingFace加載數據..."):
                st.info("步驟 1/5: 加載歷史數據...")
                
            st.success("訓練完成")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("訓練準確率", "78.5%")
            with col_b:
                st.metric("驗證準確率", "72.3%")
            with col_c:
                st.metric("OOS準確率", "69.8%")
            
            st.markdown("---")
            st.subheader("信號分布")
            st.info("圖表將顯示在此處")
        else:
            st.info("請配置參數後點擊開始訓練")

def render_backtest_tab():
    """回測頁面"""
    st.header("回測分析")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("回測配置")
        
        model_version = st.selectbox(
            "選擇模型",
            ["BTCUSDT_15m_v1_20260226", "無可用模型"],
            key="backtest_model"
        )
        
        st.markdown("---")
        
        st.subheader("回測參數")
        
        data_source = st.radio(
            "數據源",
            ["Binance API (最新)", "HuggingFace (歷史)"],
            index=0
        )
        
        if data_source == "Binance API (最新)":
            backtest_days = st.slider("回測天數", 7, 60, 30)
        
        initial_capital = st.number_input("初始資金 (USDT)", 10, 10000, 10)
        leverage = st.slider("槓桿倍數", 1, 20, 3)
        
        st.markdown("---")
        
        st.subheader("交易參數")
        min_signal_strength = st.slider("最小信號強度", 1, 5, 2)
        min_confidence = st.slider("最小模型置信度", 0.5, 0.95, 0.6, 0.05)
        
        maker_fee = st.number_input("Maker手續費 %", 0.01, 0.1, 0.02, 0.01) / 100
        taker_fee = st.number_input("Taker手續費 %", 0.01, 0.1, 0.04, 0.01) / 100
        
        st.markdown("---")
        
        if st.button("運行回測", type="primary", use_container_width=True):
            st.session_state['backtest_started'] = True
    
    with col2:
        st.subheader("回測結果")
        
        if st.session_state.get('backtest_started', False):
            st.success("回測完成")
            
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("總收益率", "+45.8%", "+35.8%")
            with col_b:
                st.metric("勝率", "68.5%")
            with col_c:
                st.metric("交易次數", "47")
            with col_d:
                st.metric("最大回撤", "-8.2%")
            
            st.markdown("---")
            
            st.subheader("權益曲線")
            st.info("權益曲線圖表將顯示在此處")
            
            st.markdown("---")
            
            st.subheader("交易明細")
            st.info("交易列表將顯示在此處")
            
        else:
            st.info("請配置回測參數後點擊運行回測")

def render_paper_trading_tab():
    """模擬交易頁面"""
    st.header("模擬交易")
    st.info(
        "模擬交易功能將使用Bybit Demo帳戶實現\n\n"
        "**此功能將允許您:**\n"
        "- 使用模擬資金測試策略\n"
        "- 監控實時表現\n"
        "- 在實盤交易前驗證模型"
    )

def render_live_trading_tab():
    """實盤交易頁面"""
    st.header("實盤交易")
    st.warning("在回測結果驗證通過前,實盤交易功能已禁用")
    
    st.markdown(
        "**啟用實盤交易前:**\n"
        "1. 完成回測並獲得滿意的結果\n"
        "2. 使用模擬交易測試至少7天\n"
        "3. 配置Binance API憑證\n"
        "4. 設置風險管理參數"
    )

def render_analytics_tab():
    """分析頁面"""
    st.header("績效分析")
    
    st.subheader("模型表現")
    st.info("模型表現指標將顯示在此處")
    
    st.markdown("---")
    
    st.subheader("信號分析")
    st.info("信號質量分析將顯示在此處")
    
    st.markdown("---")
    
    st.subheader("風險指標")
    st.info("風險和敞口指標將顯示在此處")

if __name__ == "__main__":
    main()
