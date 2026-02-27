"""
V2 High-Frequency Strategy - Streamlit GUI
V2高頻交易系統主界面
"""
import streamlit as st
import sys
from pathlib import Path
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import subprocess

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="V2 高頻交易系統",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("⚡ V2 高頻交易系統")
    st.caption("基於Transformer的高頻交易策略 | 目標: 140-150筆/月, 50%+收益")
    
    with st.sidebar:
        st.header("系統配置")
        
        st.markdown("---")
        st.subheader("當前版本: V2")
        st.caption("Transformer + LightGBM 集成模型")
        
        st.markdown("---")
        st.info(
            "**V2 核心特點:**\n\n"
            "Transformer時序學習\n"
            "多時間框架特徵\n"
            "三層信號過濾\n"
            "動態風險管理\n"
            "市場狀態自適應"
        )
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "模型訓練",
        "回測分析",
        "模型比較",
        "系統狀態"
    ])
    
    with tab1:
        render_training_tab()
    
    with tab2:
        render_backtest_tab()
    
    with tab3:
        render_comparison_tab()
    
    with tab4:
        render_status_tab()

def render_training_tab():
    """V2模型訓練頁面"""
    st.header("🧠 模型訓練")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("訓練參數")
        
        symbol = st.selectbox(
            "交易對",
            ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"],
            key="v2_train_symbol"
        )
        
        timeframe = st.selectbox(
            "時間框架",
            ["15m", "1h"],
            index=0,
            key="v2_train_timeframe"
        )
        
        st.markdown("---")
        
        st.subheader("模型配置")
        sequence_length = st.slider("序列長度", 50, 200, 100, 10,
                                   help="Transformer輸入的K線數量")
        
        use_transformer = st.checkbox("Transformer模型", value=True)
        use_lgb = st.checkbox("LightGBM模型", value=True)
        
        if use_transformer and use_lgb:
            transformer_weight = st.slider("Transformer權重", 0.0, 1.0, 0.5, 0.1)
            lgb_weight = 1.0 - transformer_weight
            st.caption(f"LightGBM權重: {lgb_weight:.1f}")
        
        st.markdown("---")
        
        if st.button("開始訓練 V2 模型", type="primary", use_container_width=True):
            with st.spinner("訓練中..."):
                try:
                    result = subprocess.run([
                        "python", 
                        str(project_root / "train_model.py"),
                        "--symbol", symbol,
                        "--timeframe", timeframe,
                        "--sequence_length", str(sequence_length)
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        st.success("訓練完成!")
                        st.code(result.stdout)
                    else:
                        st.error("訓練失敗")
                        st.code(result.stderr)
                except Exception as e:
                    st.error(f"執行失敗: {str(e)}")
    
    with col2:
        st.subheader("訓練進度")
        st.info("請配置參數後點擊開始訓練")
        
        st.markdown("---")
        
        st.subheader("📊 V2模型架構")
        st.image("https://via.placeholder.com/600x400?text=Transformer+Architecture", 
                caption="Transformer + LightGBM 集成架構")

def render_backtest_tab():
    """V2回測頁面"""
    st.header("📈 回測分析")
    
    models_dir = Path('models')
    if models_dir.exists():
        v2_models = [d.name for d in models_dir.iterdir() 
                    if d.is_dir() and '_v2_' in d.name]
    else:
        v2_models = []
    
    if not v2_models:
        st.warning("無V2模型,請先訓練模型")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("回測配置")
        
        model = st.selectbox("選擇V2模型", v2_models)
        
        st.markdown("---")
        
        backtest_days = st.slider("回測天數", 7, 90, 30)
        initial_capital = st.number_input("初始資金 (USDT)", 10, 10000, 10)
        leverage = st.slider("槓桿倍數", 1, 10, 3)
        
        st.markdown("---")
        
        st.subheader("信號過濾")
        min_confidence = st.slider("最小置信度", 0.5, 0.95, 0.7, 0.05)
        
        if st.button("運行V2回測", type="primary", use_container_width=True):
            st.info("回測功能開發中...")
    
    with col2:
        st.subheader("回測結果")
        st.info("請選擇模型並配置參數")

def render_comparison_tab():
    """V1 vs V2 比較"""
    st.header("🏆 V1 vs V2 比較")
    
    comparison_data = {
        '項目': ['Transformer', 'LightGBM', '時序學習', '三層過濾', '動態風險', '目標交易量/月', '目標報酬率'],
        'V1': ['✘', '✓', '✘', '✘', '✘', '50-80', '30-50%'],
        'V2': ['✓', '✓', '✓', '✓', '✓', '140-150', '50%+']
    }
    
    df_comp = pd.DataFrame(comparison_data)
    st.table(df_comp)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔵 V1 特點")
        st.write("""
        - 訂單流不平衡檢測
        - 流動性掃蕩識別
        - XGBoost單一模型
        - 固定百分比止損止盈
        - 適合中頻交易
        """)
    
    with col2:
        st.subheader("🟢 V2 優勢")
        st.write("""
        - Transformer時序模型
        - 集成學習架構
        - 多層信號過濾
        - 市場狀態自適應
        - 適合高頻交易
        """)

def render_status_tab():
    """V2系統狀態"""
    st.header("📊 V2 系統狀態")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("已訓練模型", "0")
    with col2:
        st.metric("回測完成", "0")
    with col3:
        st.metric("實盤運行", "未啟動")
    
    st.markdown("---")
    
    st.subheader("🔧 系統要求")
    
    requirements = [
        ("Python", "3.8+", "✓"),
        ("PyTorch", "2.0+", "❓"),
        ("LightGBM", "4.0+", "❓"),
        ("CUDA (GPU)", "12.1+", "可選")
    ]
    
    req_df = pd.DataFrame(requirements, columns=['Component', 'Version', 'Status'])
    st.table(req_df)
    
    st.markdown("---")
    
    st.subheader("📝 下一步")
    st.write("""
    1. 安裝依賴: `pip install -r requirements.txt`
    2. 訓練V2模型
    3. 執行回測驗證
    4. 與V1結果比較
    5. 優化超參數
    """)

if __name__ == "__main__":
    main()
