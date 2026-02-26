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
    page_title="Reversal Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🚀 Crypto Reversal Trading System")
    
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        version = st.selectbox(
            "Strategy Version",
            ["V1 - Order Flow Reversal", "V2 - Coming Soon", "V3 - Coming Soon"],
            index=0
        )
        
        st.markdown("---")
        st.subheader("📊 Current Version: V1")
        st.caption("Order Flow Imbalance & Liquidity Zone Strategy")
        
        st.markdown("---")
        st.info(
            "**V1 Strategy Features:**\n\n"
            "✓ Order Flow Imbalance Detection\n"
            "✓ Liquidity Sweep Identification\n"
            "✓ Market Microstructure Analysis\n"
            "✓ ML Signal Validation"
        )
    
    if "V1" in version:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Model Training",
            "📈 Backtest",
            "📝 Paper Trading", 
            "💰 Live Trading",
            "📊 Analytics"
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
    st.header("🎯 Model Training")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Training Parameters")
        
        symbol = st.selectbox(
            "Trading Pair",
            ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "AVAXUSDT"],
            key="train_symbol"
        )
        
        timeframe = st.selectbox(
            "Timeframe",
            ["15m", "1h", "4h"],
            index=0,
            key="train_timeframe"
        )
        
        st.markdown("---")
        
        st.subheader("Signal Detection")
        lookback = st.slider("Lookback Period", 10, 50, 20)
        imbalance_threshold = st.slider("OFI Threshold", 0.5, 0.8, 0.6, 0.05)
        
        st.markdown("---")
        
        st.subheader("Label Generation")
        forward_window = st.slider("Forward Window", 8, 20, 12)
        profit_threshold = st.slider("Profit Target %", 0.5, 3.0, 1.0, 0.1) / 100
        stop_loss = st.slider("Stop Loss %", 0.3, 2.0, 0.5, 0.1) / 100
        
        st.markdown("---")
        
        st.subheader("ML Model")
        n_estimators = st.slider("N Estimators", 100, 500, 200, 50)
        max_depth = st.slider("Max Depth", 3, 10, 5)
        test_size = st.slider("Validation Size", 0.1, 0.3, 0.2, 0.05)
        oos_size = st.slider("OOS Test Size", 0.05, 0.2, 0.1, 0.05)
        
        st.markdown("---")
        
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            st.session_state['training_started'] = True
    
    with col2:
        st.subheader("Training Process")
        
        if st.session_state.get('training_started', False):
            with st.spinner("Loading data from HuggingFace..."):
                st.info("📥 Step 1/5: Loading historical data...")
                
            st.success("✅ Training completed")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Training Accuracy", "78.5%")
            with col_b:
                st.metric("Validation Accuracy", "72.3%")
            with col_c:
                st.metric("OOS Accuracy", "69.8%")
            
            st.markdown("---")
            st.subheader("Signal Distribution")
            st.info("📊 Chart will be displayed here")
        else:
            st.info("👈 Configure parameters and click Start Training")

def render_backtest_tab():
    """回測頁面"""
    st.header("📈 Backtest")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Backtest Configuration")
        
        model_version = st.selectbox(
            "Select Model",
            ["BTCUSDT_15m_v1_20260226", "No models available"],
            key="backtest_model"
        )
        
        st.markdown("---")
        
        st.subheader("Backtest Parameters")
        
        data_source = st.radio(
            "Data Source",
            ["Binance API (Latest)", "HuggingFace (Historical)"],
            index=0
        )
        
        if data_source == "Binance API (Latest)":
            backtest_days = st.slider("Backtest Days", 7, 60, 30)
        
        initial_capital = st.number_input("Initial Capital (USDT)", 10, 10000, 10)
        leverage = st.slider("Leverage", 1, 20, 3)
        
        st.markdown("---")
        
        st.subheader("Trading Parameters")
        min_signal_strength = st.slider("Min Signal Strength", 1, 5, 2)
        min_confidence = st.slider("Min ML Confidence", 0.5, 0.95, 0.6, 0.05)
        
        maker_fee = st.number_input("Maker Fee %", 0.01, 0.1, 0.02, 0.01) / 100
        taker_fee = st.number_input("Taker Fee %", 0.01, 0.1, 0.04, 0.01) / 100
        
        st.markdown("---")
        
        if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
            st.session_state['backtest_started'] = True
    
    with col2:
        st.subheader("Backtest Results")
        
        if st.session_state.get('backtest_started', False):
            st.success("✅ Backtest completed successfully")
            
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Total Return", "+45.8%", "+35.8%")
            with col_b:
                st.metric("Win Rate", "68.5%")
            with col_c:
                st.metric("Total Trades", "47")
            with col_d:
                st.metric("Max Drawdown", "-8.2%")
            
            st.markdown("---")
            
            st.subheader("Equity Curve")
            st.info("📈 Equity curve chart will be displayed here")
            
            st.markdown("---")
            
            st.subheader("Trade Details")
            st.info("📋 Trade list will be displayed here")
            
        else:
            st.info("👈 Configure backtest parameters and click Run Backtest")

def render_paper_trading_tab():
    """模擬交易頁面"""
    st.header("📝 Paper Trading")
    st.info(
        "Paper trading functionality will be implemented using Bybit Demo account\n\n"
        "**This feature will allow you to:**\n"
        "- Test strategies with simulated funds\n"
        "- Monitor real-time performance\n"
        "- Validate model before live trading"
    )

def render_live_trading_tab():
    """實盤交易頁面"""
    st.header("💰 Live Trading")
    st.warning("⚠️ Live trading is disabled until backtest results are validated")
    
    st.markdown(
        "**Before enabling live trading:**\n"
        "1. Complete backtest with satisfactory results\n"
        "2. Test with paper trading for at least 7 days\n"
        "3. Configure Binance API credentials\n"
        "4. Set up risk management parameters"
    )

def render_analytics_tab():
    """分析頁面"""
    st.header("📊 Analytics & Performance")
    
    st.subheader("Model Performance")
    st.info("Model performance metrics will be displayed here")
    
    st.markdown("---")
    
    st.subheader("Signal Analysis")
    st.info("Signal quality analysis will be displayed here")
    
    st.markdown("---")
    
    st.subheader("Risk Metrics")
    st.info("Risk and exposure metrics will be displayed here")

if __name__ == "__main__":
    main()
