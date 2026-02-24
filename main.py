import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from tabs.data_fetcher_tab import DataFetcherTab
from tabs.feature_engineering_tab import FeatureEngineeringTab
from tabs.model_training_tab import ModelTrainingTab
from tabs.backtesting_tab import BacktestingTab
from tabs.auto_trading_tab import AutoTradingTab
from tabs.model_management_tab import ModelManagementTab
from utils.logger import setup_logger

# 試圖載入 V2 標籤
try:
    from tabs.model_training_v2_tab import ModelTrainingV2Tab
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False

# Setup logger
logger = setup_logger('main', 'logs/main.log')

def main():
    logger.info("Starting Crypto Trading Automation System")
    
    st.set_page_config(
        page_title="加密貨幣自動交易系統",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 侧邊欄 - 版本選擇
    with st.sidebar:
        st.title("🤖 AI 交易系統")
        
        st.markdown("---")
        
        # 版本選擇器
        if V2_AVAILABLE:
            st.subheader("版本選擇")
            
            system_version = st.radio(
                "選擇系統版本",
                options=[
                    "V1 - 基礎版 (9特徵)",
                    "V2 - 進階版 (44-54特徵)"
                ],
                index=1 if 'system_version' not in st.session_state else 
                      (1 if st.session_state.system_version == 'v2' else 0),
                help="V2 版本整合了所有模型優化,預期 Profit Factor +19-35%"
            )
            
            if "V2" in system_version:
                st.session_state.system_version = 'v2'
                st.success("✅ V2 系統已啟用")
                
                # V2 特性說明
                with st.expander("📊 V2 新特性"):
                    st.markdown("""
                    **特徵強化**
                    - 訂單流特徵 (10個)
                    - 微觀結構 (10個)
                    - 多時間框架 (15個)
                    - ML衍生 (10個)
                    
                    **模型強化**
                    - 動態標籤生成
                    - 集成學習
                    - Optuna 優化
                    - Walk-Forward 驗證
                    - **Metadata 追蹤** ✨
                    
                    **預期效果**
                    - 交易數: +116-224%
                    - 勝率: +11-27%
                    - Profit Factor: +19-35%
                    """)
            else:
                st.session_state.system_version = 'v1'
                st.info("ℹ️ V1 系統已啟用")
            
            st.markdown("---")
        else:
            st.session_state.system_version = 'v1'
            st.info("V2 系統未安裝")
            if st.button("安裝 V2 系統"):
                st.code("python upgrade_to_v2.py", language="bash")
        
        # 系統狀態
        st.subheader("📊 系統狀態")
        
        # 模型狀態
        models_dir = Path("models_output")
        if models_dir.exists():
            v1_long = list(models_dir.glob("catboost_long_[0-9]*.pkl"))
            v1_short = list(models_dir.glob("catboost_short_[0-9]*.pkl"))
            v2_long = list(models_dir.glob("catboost_long_v2_*.pkl"))
            v2_short = list(models_dir.glob("catboost_short_v2_*.pkl"))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("V1 Long", len(v1_long))
                st.metric("V2 Long", len(v2_long))
            with col2:
                st.metric("V1 Short", len(v1_short))
                st.metric("V2 Short", len(v2_short))
        
        st.markdown("---")
        
        # 快速連結
        st.subheader("📚 文檔")
        st.markdown("""
        - [📖 Model Metadata Fix](MODEL_METADATA_FIX.md)
        - [📝 Changelog](CHANGELOG.md)
        - [🚀 V2 系統文檔](docs/V2_SYSTEM_SUMMARY.md)
        """)
        
        st.markdown("---")
        st.caption("Powered by AI | V2.1.0")
    
    # 主頁面標題
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("💹 加密貨幣自動交易系統")
    with col2:
        if st.session_state.get('system_version') == 'v2':
            st.success("✨ V2 進階版")
        else:
            st.info("📌 V1 基礎版")
    
    st.markdown("---")
    
    # 根據版本顯示不同標籤
    if st.session_state.get('system_version') == 'v2' and V2_AVAILABLE:
        # V2 版本 - 7 個標籤 (新增模型管理)
        tab1, tab2, tab3, tab3_v2, tab4, tab5, tab6 = st.tabs([
            "📊 K棒資料抽取",
            "🔧 特徵工程",
            "🎯 V1 模型訓練",
            "🚀 V2 模型訓練",
            "📈 策略回測",
            "🤖 自動交易",
            "📦 模型管理"
        ])
        
        with tab1:
            logger.info("Loading Data Fetcher Tab")
            DataFetcherTab().render()
        
        with tab2:
            logger.info("Loading Feature Engineering Tab")
            FeatureEngineeringTab().render()
        
        with tab3:
            logger.info("Loading V1 Model Training Tab")
            st.info("ℹ️ V1 模型訓練 (9 個特徵)")
            ModelTrainingTab().render()
        
        with tab3_v2:
            logger.info("Loading V2 Model Training Tab")
            ModelTrainingV2Tab().render()
        
        with tab4:
            logger.info("Loading Backtesting Tab")
            BacktestingTab().render()
        
        with tab5:
            logger.info("Loading Auto Trading Tab")
            AutoTradingTab().render()
        
        with tab6:
            logger.info("Loading Model Management Tab")
            ModelManagementTab().render()
    
    else:
        # V1 版本 - 6 個標籤 (原有 + 模型管理)
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 K棒資料抽取",
            "🔧 特徵工程",
            "🎯 模型訓練",
            "📈 策略回測",
            "🤖 自動交易",
            "📦 模型管理"
        ])
        
        with tab1:
            logger.info("Loading Data Fetcher Tab")
            DataFetcherTab().render()
        
        with tab2:
            logger.info("Loading Feature Engineering Tab")
            FeatureEngineeringTab().render()
        
        with tab3:
            logger.info("Loading Model Training Tab")
            ModelTrainingTab().render()
        
        with tab4:
            logger.info("Loading Backtesting Tab")
            BacktestingTab().render()
        
        with tab5:
            logger.info("Loading Auto Trading Tab")
            AutoTradingTab().render()
        
        with tab6:
            logger.info("Loading Model Management Tab")
            ModelManagementTab().render()

if __name__ == "__main__":
    main()