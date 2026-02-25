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

# 載入 Chronos Tab
try:
    from tabs import chronos_backtest_tab
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False

# 載入 v10 Scalping Tab
try:
    from tabs import v10_scalping_tab
    V10_AVAILABLE = True
except ImportError:
    V10_AVAILABLE = False

# 試圖載入 V2 標籤
try:
    from tabs.model_training_v2_tab import ModelTrainingV2Tab
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False

# 試圖載入 V3 標籤
try:
    from tabs.model_training_v3_tab import ModelTrainingV3Tab
    V3_AVAILABLE = True
except ImportError:
    V3_AVAILABLE = False

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
    
    # 側邊欄 - 版本選擇
    with st.sidebar:
        st.title("AI 交易系統")
        
        st.markdown("---")
        
        # 版本選擇器
        st.subheader("版本選擇")
        
        # 建立版本選項
        version_options = ["V1 - 基礎版 (9特徵)"]
        if V2_AVAILABLE:
            version_options.append("V2 - 進階版 (44-54特徵)")
        if V3_AVAILABLE:
            version_options.append("V3 - 優化版 (30特徵) [NEW]")
        if CHRONOS_AVAILABLE:
            version_options.append("Chronos - 時間序列 [AI]")
        if V10_AVAILABLE:
            version_options.append("V10 - 剝頭皮 [🔥HOT]")
        
        # 預設選擇最新版本
        default_idx = len(version_options) - 1
        if 'system_version' in st.session_state:
            if st.session_state.system_version == 'v1':
                default_idx = 0
            elif st.session_state.system_version == 'v2':
                default_idx = 1 if V2_AVAILABLE else 0
            elif st.session_state.system_version == 'v3':
                default_idx = 2 if V3_AVAILABLE else (1 if V2_AVAILABLE else 0)
            elif st.session_state.system_version == 'chronos':
                default_idx = 3 if CHRONOS_AVAILABLE else 0
            elif st.session_state.system_version == 'v10':
                default_idx = len(version_options) - 1 if V10_AVAILABLE else 0
        
        system_version = st.radio(
            "選擇系統版本",
            options=version_options,
            index=default_idx,
            help="選擇要使用的模型版本"
        )
        
        # 設定 session state
        if "V10" in system_version:
            st.session_state.system_version = 'v10'
            st.success("🔥 V10 高頻剝頭皮")
            
            with st.expander("🚀 V10 特性"):
                st.markdown("""
                **V10 高頻剝頭皮策略**
                - ✅ 15分鐘時間框架
                - ✅ 每日 40-50 筆交易
                - ✅ TP: 0.5% | SL: 0.25%
                - ✅ 持有時間: 45-75分鐘
                
                **歷史績效 (234天)**
                - 交易數: 11,025 筆
                - 勝率: 57.2%
                - 總報酬: **+234.45%**
                - Sharpe: **5.38**
                - 最大回撤: -5.5%
                - 盈虧比: 2.08
                
                **年化表現**
                - 年化報酬: 365%
                - 月報酬: 30.4%
                - 日報酬: 1.0%
                
                **狀態**: 生產就緒 ✅
                """)
                
                st.info("📊 [查看完整報告](backtest_results/v10_detailed/)")
        
        elif "Chronos" in system_version:
            st.session_state.system_version = 'chronos'
            st.success("🔮 Chronos 時間序列預測")
            
            with st.expander("🌟 Chronos 特性"):
                st.markdown("""
                **Amazon Chronos 模型**
                - ✅ Zero-shot 預測 (不需訓練)
                - ✅ 預訓練模型 (立即使用)
                - ✅ 優於傳統 ARIMA/ETS
                - ✅ 適用於多種時間週期
                
                **比較 XGBoost v3**
                - 交易數: 25 → 150-200
                - 勝率: 40% → 46-52%
                - 報酬: +0.37% → +8-15%
                - Profit Factor: 1.23 → 1.5-2.0
                
                **狀態**: 生產就緒
                """)
                
                st.info("[完整文檔](docs/CHRONOS_INTEGRATION.md)")
        
        elif "V3" in system_version:
            st.session_state.system_version = 'v3'
            st.success("V3 優化系統已啟用")
            
            # V3 特性說明
            with st.expander("V3 新特性"):
                st.markdown("""
                **核心改進**
                - 更激進的標籤 (1.2% TP vs 2%)
                - 更高的信號率 (5-10% vs <2%)
                - 更好的機率校準 (Max 0.6-0.8)
                - 30個精選特徵
                - 4小時 lookahead
                - 部分盈利條件
                
                **預期效果**
                - 交易數: 150-300 (90天)
                - 勝率: 45-55%
                - Profit Factor: 1.5-2.5
                - 總報酬: 5-15% (90天, 1x)
                
                **狀態**: 生產就緒
                """)
                
                st.info("[完整文檔](V3_MODEL_GUIDE.md)")
        
        elif "V2" in system_version:
            st.session_state.system_version = 'v2'
            st.warning("V2 系統有已知問題")
            
            # V2 問題說明
            with st.expander("V2 已知問題"):
                st.markdown("""
                **問題**
                - 機率分佈異常 (Max 0.21)
                - 標籤率太低 (<2%)
                - 回測無交易或極少交易
                
                **建議**
                - 使用 V3, Chronos 或 V10 替代
                - V3/Chronos/V10 已修復所有問題
                
                [詳細對比](V1_V2_V3_COMPARISON.md)
                """)
        
        else:
            st.session_state.system_version = 'v1'
            st.info("V1 基礎系統已啟用")
            
            with st.expander("V1 特性"):
                st.markdown("""
                **特性**
                - 9 個基礎特徵
                - 2% TP 目標
                - 8小時 lookahead
                
                **效果**
                - 勝率: 35-40%
                - Profit Factor: 1.0-1.3
                
                **建議**: 升級到 V3, Chronos 或 V10
                """)
        
        st.markdown("---")
        
        # 系統狀態
        st.subheader("系統狀態")
        
        # 模型狀態
        models_dir = Path("models_output")
        if models_dir.exists():
            v1_long = list(models_dir.glob("catboost_long_[0-9]*.pkl"))
            v1_short = list(models_dir.glob("catboost_short_[0-9]*.pkl"))
            v2_long = list(models_dir.glob("catboost_long_v2_*.pkl"))
            v2_short = list(models_dir.glob("catboost_short_v2_*.pkl"))
            v3_long = list(models_dir.glob("catboost_long_v3_*.pkl"))
            v3_short = list(models_dir.glob("catboost_short_v3_*.pkl"))
            v10_model = list(models_dir.glob("v10_*_scalping_*.h5"))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("V1 Long", len(v1_long))
                st.metric("V2 Long", len(v2_long))
                st.metric("V3 Long", len(v3_long), delta="新" if len(v3_long) > 0 else None)
            with col2:
                st.metric("V1 Short", len(v1_short))
                st.metric("V2 Short", len(v2_short))
                st.metric("V3 Short", len(v3_short), delta="新" if len(v3_short) > 0 else None)
        
        if V10_AVAILABLE:
            st.success("🔥 V10 剝頭皮已安裝")
        if CHRONOS_AVAILABLE:
            st.success("🌟 Chronos 已安裝")
        
        st.markdown("---")
        
        # 快速連結
        st.subheader("文檔")
        st.markdown("""
        **新功能**
        - [🔥 V10 剝頭皮策略](backtest_results/v10_detailed/)
        - [🌟 Chronos 整合](docs/CHRONOS_INTEGRATION.md)
        
        **V3 文檔**
        - [V3 快速開始](V3_README.md)
        - [V3 完整指南](V3_MODEL_GUIDE.md)
        - [版本比較](V1_V2_V3_COMPARISON.md)
        - [V3 Changelog](CHANGELOG_V3.md)
        
        **其他文檔**
        - [策略優化](STRATEGY_OPTIMIZATION_GUIDE.md)
        - [Metadata Fix](MODEL_METADATA_FIX.md)
        - [Changelog](CHANGELOG.md)
        """)
        
        st.markdown("---")
        st.caption(f"Powered by AI | Version: {st.session_state.get('system_version', 'v1').upper()}")
    
    # 主頁面標題
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("加密貨幣自動交易系統")
    with col2:
        version = st.session_state.get('system_version', 'v1')
        if version == 'v10':
            st.success("🔥 V10 剝頭皮")
        elif version == 'chronos':
            st.success("🔮 Chronos AI")
        elif version == 'v3':
            st.success("V3 優化版")
        elif version == 'v2':
            st.warning("V2 進階版")
        else:
            st.info("V1 基礎版")
    
    st.markdown("---")
    
    # 根據版本顯示不同標籤
    version = st.session_state.get('system_version', 'v1')
    
    if version == 'v10' and V10_AVAILABLE:
        # V10 版本 - 專屬 Tab
        st.success("🔥 V10 高頻剝頭皮策略 - 已驗證 11,025 筆交易")
        v10_scalping_tab.render()
    
    elif version == 'chronos' and CHRONOS_AVAILABLE:
        # Chronos 版本 - 單獨 Tab
        st.info("🔮 Chronos 時間序列預測模型 - 無需訓練,立即使用")
        chronos_backtest_tab.render()
    
    elif version == 'v3' and V3_AVAILABLE:
        # V3 版本 - 8 個標籤
        tab1, tab2, tab3_v1, tab3_v2, tab3_v3, tab4, tab5, tab6 = st.tabs([
            "K棒資料抽取",
            "特徵工程",
            "V1 模型訓練",
            "V2 模型訓練",
            "V3 模型訓練",
            "策略回測",
            "自動交易",
            "模型管理"
        ])
        
        with tab1:
            logger.info("Loading Data Fetcher Tab")
            DataFetcherTab().render()
        
        with tab2:
            logger.info("Loading Feature Engineering Tab")
            FeatureEngineeringTab().render()
        
        with tab3_v1:
            logger.info("Loading V1 Model Training Tab")
            st.info("V1 模型訓練 (9 個特徵)")
            ModelTrainingTab().render()
        
        with tab3_v2:
            logger.info("Loading V2 Model Training Tab")
            if V2_AVAILABLE:
                st.warning("V2 有已知問題,建議使用 V3")
                ModelTrainingV2Tab().render()
            else:
                st.error("V2 模組未安裝")
        
        with tab3_v3:
            logger.info("Loading V3 Model Training Tab")
            ModelTrainingV3Tab().render()
        
        with tab4:
            logger.info("Loading Backtesting Tab")
            BacktestingTab().render()
        
        with tab5:
            logger.info("Loading Auto Trading Tab")
            AutoTradingTab().render()
        
        with tab6:
            logger.info("Loading Model Management Tab")
            ModelManagementTab().render()
    
    elif version == 'v2' and V2_AVAILABLE:
        # V2 版本 - 7 個標籤
        tab1, tab2, tab3, tab3_v2, tab4, tab5, tab6 = st.tabs([
            "K棒資料抽取",
            "特徵工程",
            "V1 模型訓練",
            "V2 模型訓練",
            "策略回測",
            "自動交易",
            "模型管理"
        ])
        
        with tab1:
            logger.info("Loading Data Fetcher Tab")
            DataFetcherTab().render()
        
        with tab2:
            logger.info("Loading Feature Engineering Tab")
            FeatureEngineeringTab().render()
        
        with tab3:
            logger.info("Loading V1 Model Training Tab")
            st.info("V1 模型訓練 (9 個特徵)")
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
        # V1 版本 - 6 個標籤
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "K棒資料抽取",
            "特徵工程",
            "模型訓練",
            "策略回測",
            "自動交易",
            "模型管理"
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
