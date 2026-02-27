"""
Unified Trading System GUI
統一交易系統界面 - 整合V1和V2策略
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

st.set_page_config(
    page_title="加密貨幣交易系統",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🚀 加密貨幣智能交易系統")
    
    with st.sidebar:
        st.header("策略選擇")
        
        strategy_version = st.radio(
            "選擇策略版本",
            [
                "V1 - 訂單流反轉策略",
                "V2 - 高頻Transformer策略"
            ],
            index=0
        )
        
        st.markdown("---")
        
        if "V1" in strategy_version:
            st.subheader("🟢 V1 特點")
            st.caption("訂單流不平衡 + XGBoost")
            st.info(
                "訂單流不平衡檢測\n"
                "流動性掃蕩識別\n"
                "市場微觀結構分析\n"
                "XGBoost機器學習"
            )
            st.metric("月交易目標", "50-80筆")
            st.metric("月報酬目標", "30-50%")
        else:
            st.subheader("⚡ V2 特點")
            st.caption("Transformer + 集成學習")
            st.info(
                "Transformer時序學習\n"
                "多時間框架特徵\n"
                "三層信號過濾\n"
                "市場狀態自適應"
            )
            st.metric("月交易目標", "140-150筆")
            st.metric("月報酬目標", "50%+")
        
        st.markdown("---")
        
        st.subheader("📊 策略比較")
        if st.button("查看V1 vs V2對比", use_container_width=True):
            st.session_state['show_comparison'] = True
    
    if "V1" in strategy_version:
        render_v1_interface()
    else:
        render_v2_interface()

def render_v1_interface():
    """V1策略界面"""
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 模型訓練",
        "📈 回測分析",
        "🎮 模擬交易",
        "💰 實盤交易",
        "📊 績效分析"
    ])
    
    with tab1:
        render_v1_training()
    with tab2:
        render_v1_backtest()
    with tab3:
        st.info("V1模擬交易功能開發中...")
    with tab4:
        st.warning("V1實盤交易需先完成回測驗證")
    with tab5:
        render_v1_analytics()

def render_v2_interface():
    """V2策略界面"""
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 V2模型訓練",
        "📈 V2回測分析",
        "🏆 策略對比",
        "⚙️ 系統狀態"
    ])
    
    with tab1:
        render_v2_training()
    with tab2:
        render_v2_backtest()
    with tab3:
        render_comparison()
    with tab4:
        render_v2_status()

def render_v1_training():
    """V1訓練界面"""
    st.header("🎯 V1 模型訓練")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("訓練參數")
        
        symbol = st.selectbox("交易對", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"], key="v1_symbol")
        timeframe = st.selectbox("時間框架", ["15m", "1h", "4h"], key="v1_tf")
        
        st.markdown("---")
        lookback = st.slider("回溯週期", 10, 50, 20)
        n_estimators = st.slider("樹數量", 100, 500, 200, 50)
        
        if st.button("開始V1訓練", type="primary", use_container_width=True):
            with st.spinner("訓練V1模型..."):
                result = subprocess.run([
                    "python", "reversal_strategy_v1/train_model.py",
                    "--symbol", symbol,
                    "--timeframe", timeframe
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    st.success("✅ V1訓練完成!")
                    with st.expander("訓練詳細"):
                        st.code(result.stdout)
                else:
                    st.error("❌ V1訓練失敗")
                    st.code(result.stderr)
    
    with col2:
        st.subheader("訓練說明")
        st.markdown("""
        ### V1訓練流程
        
        1. **加載數據**: 從HuggingFace加載歷史K線
        2. **信號檢測**: 訂單流不平衡 + 流動性掃蕩
        3. **特徵工程**: 提取50+個技術指標
        4. **標籤生成**: 前瞻窗口盈虧標籤
        5. **模型訓練**: XGBoost機器學習
        6. **模型驗證**: 訓練集/驗證集/OOS測試
        
        **預計時間**: 5-10分鐘
        """)

def render_v2_training():
    """V2訓練界面"""
    st.header("🧠 V2 模型訓練")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("訓練參數")
        
        symbol = st.selectbox("交易對", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"], key="v2_symbol")
        timeframe = st.selectbox("時間框架", ["15m", "1h"], key="v2_tf")
        
        st.markdown("---")
        
        st.subheader("模型配置")
        sequence_length = st.slider("序列長度", 50, 200, 100, 10)
        use_transformer = st.checkbox("Transformer模型", value=True)
        use_lgb = st.checkbox("LightGBM模型", value=True)
        
        if use_transformer and use_lgb:
            st.caption("集成模式: Transformer + LightGBM")
        
        st.markdown("---")
        
        if st.button("開始V2訓練", type="primary", use_container_width=True):
            with st.spinner("訓練V2模型 (需要10-20分鐘)..."):
                try:
                    result = subprocess.run([
                        "python", "high_frequency_strategy_v2/train_model.py",
                        "--symbol", symbol,
                        "--timeframe", timeframe,
                        "--sequence_length", str(sequence_length)
                    ], capture_output=True, text=True, timeout=1800)
                    
                    if result.returncode == 0:
                        st.success("✅ V2訓練完成!")
                        with st.expander("訓練詳細"):
                            st.code(result.stdout)
                    else:
                        st.error("❌ V2訓練失敗")
                        st.code(result.stderr)
                except subprocess.TimeoutExpired:
                    st.error("訓練超時 (>30分鐘)")
                except Exception as e:
                    st.error(f"執行失敗: {str(e)}")
    
    with col2:
        st.subheader("💡 V2訓練流程")
        st.markdown("""
        ### V2 Transformer訓練
        
        1. **加載數據**: HuggingFace歷史K線
        2. **特徵提取**: 
           - 50+技術指標
           - 市場微觀結構
           - 時間特徵 (時段/星期)
           - 波動率狀態
        3. **時序準備**: 創建100根K線序列
        4. **模型訓練**:
           - Transformer (4層, 8頭注意力)
           - LightGBM (快速決策)
           - 集成學習
        5. **模型驗證**: 訓練/驗證/測試集
        
        **預計時間**: 10-20分鐘 (GPU加速)
        
        **系統要求**:
        - PyTorch 2.0+
        - 8GB+ RAM
        - GPU可選(快10倍)
        """)

def render_v1_backtest():
    """V1回測界面 - 簡化版"""
    st.header("📈 V1 回測分析")
    st.info("請切換至V1頁面使用完整回測功能")
    st.caption("執行: `streamlit run reversal_strategy_v1/gui/app.py`")

def render_v2_backtest():
    """V2回測界面"""
    st.header("📈 V2 回測分析")
    st.info("V2回測功能開發中,先完成模型訓練")

def render_v1_analytics():
    """V1分析界面"""
    st.header("📊 V1 績效分析")
    st.info("請切換至V1頁面使用完整分析功能")

def render_comparison():
    """V1 vs V2對比"""
    st.header("🏆 V1 vs V2 策略對比")
    
    # 功能對比表
    st.subheader("📊 功能對比")
    comparison_df = pd.DataFrame({
        '項目': [
            '模型架構',
            '時序學習',
            '集成學習',
            '信號過濾',
            '動態風險',
            '市場自適應',
            '月交易量',
            '月報酬目標',
            '訓練時間',
            'GPU需求'
        ],
        'V1 反轉策略': [
            'XGBoost',
            '❌',
            '❌',
            '單層',
            '固定',
            '❌',
            '50-80筆',
            '30-50%',
            '5-10分鐘',
            '不需要'
        ],
        'V2 高頻策略': [
            'Transformer + LightGBM',
            '✅ (100根K線)',
            '✅ (加權集成)',
            '三層過濾',
            '動態調整',
            '✅',
            '140-150筆',
            '50%+',
            '10-20分鐘',
            '建議使用'
        ]
    })
    
    st.table(comparison_df)
    
    st.markdown("---")
    
    # 優勣勢對比
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🟢 V1 優勢")
        st.markdown("""
        - 訓練快速 (5-10分鐘)
        - 不需要GPU
        - 模型簡單穩定
        - 資源需求低
        - 適合新手
        """)
        
        st.subheader("🔴 V1 劃勣")
        st.markdown("""
        - 交易頻率低
        - 無時序學習
        - 無動態調整
        - 報酬目標低
        """)
    
    with col2:
        st.subheader("🟢 V2 優勢")
        st.markdown("""
        - 高頻交易 (140-150筆/月)
        - Transformer時序學習
        - 集成學習架構
        - 動態風險管理
        - 市場自適應
        - 高報酬潛力
        """)
        
        st.subheader("🔴 V2 劃勣")
        st.markdown("""
        - 訓練時間長
        - 需要GPU加速
        - 資源需求高
        - 模型複雜
        """)
    
    st.markdown("---")
    
    # 使用建議
    st.subheader("💡 使用建議")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **選擇V1如果:**
        - 初學者
        - 無GPU設備
        - 偶爾手動交易
        - 中低頻交易偏好
        """)
    
    with col2:
        st.success("""
        **選擇V2如果:**
        - 有深度學習經驗
        - 有GPU設備
        - 追求高報酬
        - 高頻自動化交易
        """)

def render_v2_status():
    """V2系統狀態"""
    st.header("⚙️ V2 系統狀態")
    
    # 模型狀態
    col1, col2, col3 = st.columns(3)
    
    models_dir = Path('models')
    v1_count = 0
    v2_count = 0
    
    if models_dir.exists():
        for d in models_dir.iterdir():
            if d.is_dir():
                if '_v1_' in d.name:
                    v1_count += 1
                elif '_v2_' in d.name:
                    v2_count += 1
    
    with col1:
        st.metric("V1模型數量", v1_count)
    with col2:
        st.metric("V2模型數量", v2_count)
    with col3:
        st.metric("總模型數", v1_count + v2_count)
    
    st.markdown("---")
    
    # 系統要求
    st.subheader("💻 系統要求")
    
    requirements = {
        'Python': '3.8+',
        'NumPy': '1.24+',
        'Pandas': '2.0+',
        'LightGBM': '4.0+',
        'PyTorch': '2.0+ (V2必需)',
        'CUDA': '12.1+ (可選)',
        'RAM': '8GB+',
        'GPU': '4GB+ VRAM (可選)'
    }
    
    req_df = pd.DataFrame(list(requirements.items()), columns=['組件', '版本/規格'])
    st.table(req_df)
    
    st.markdown("---")
    
    # 安裝指引
    st.subheader("🛠️ 安裝指引")
    
    with st.expander("📚 查看安裝步驟"):
        st.code("""
# 1. 克隆倉庫
git clone https://github.com/caizongxun/crypto-trading-automation-system.git
cd crypto-trading-automation-system

# 2. 安裝V1依賴
cd reversal_strategy_v1
pip install -r requirements.txt

# 3. 安裝V2依賴 (如果使用V2)
cd ../high_frequency_strategy_v2
pip install -r requirements.txt

# 4. 安裝TA-Lib (技術指標庫)
# Windows: 下載.whl從 https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# Linux: sudo apt-get install ta-lib
# Mac: brew install ta-lib

# 5. 啟動主界面
streamlit run main_app.py
        """, language="bash")

if __name__ == "__main__":
    main()
