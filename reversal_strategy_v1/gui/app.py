"""
Unified Trading System - V1, V2, V3 Integrated GUI
統一交易系統 - 所有功能直接整合進GUI
"""
import streamlit as st
import sys
from pathlib import Path
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np
import importlib.util
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

project_root = Path(__file__).parent.parent.parent
v1_root = project_root / 'reversal_strategy_v1'
v2_root = project_root / 'high_frequency_strategy_v2'
v3_root = project_root / 'adaptive_strategy_v3'

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(v1_root))
sys.path.insert(0, str(v2_root))
sys.path.insert(0, str(v3_root))

from reversal_strategy_v1.core.signal_detector import SignalDetector
from reversal_strategy_v1.core.feature_engineer import FeatureEngineer
from reversal_strategy_v1.core.ml_predictor import MLPredictor
from reversal_strategy_v1.core.risk_manager import RiskManager
from reversal_strategy_v1.backtest.engine import BacktestEngine
from reversal_strategy_v1.data.hf_loader import HFDataLoader

st.set_page_config(
    page_title="加密貨幣交易系統",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_v2_module(module_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    st.title("加密貨幣智能交易系統")
    
    with st.sidebar:
        st.header("策略選擇")
        strategy_version = st.radio(
            "選擇策略版本",
            [
                "V1 - 訂單流反轉策略",
                "V2 - 高頻Transformer策略 (策略無效)",
                "V3 - 自適應多週期策略 [NEW]"
            ],
            index=2
        )
        st.markdown("---")
        
        if "V1" in strategy_version:
            st.subheader("V1 特點")
            st.caption("訂單流不平衡 + XGBoost")
            st.info("訂單流不平衡檢測\n流動性掃蕩識別\n市場微觀結構分析\nXGBoost機器學習")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("月交易目標", "50-80筆")
            with col2:
                st.metric("月報酬目標", "30-50%")
        elif "V2" in strategy_version:
            st.subheader("V2 特點")
            st.caption("LightGBM / Transformer")
            st.error("[策略無效]\n盈虧因子: 0.90\n勝率51.9%但平均虧損>盈利\n不建議使用")
            st.info("問題分析:\n- 信號過濾不足\n- 止盈止損不當\n- 特徵工程問題")
        else:
            st.subheader("V3 特點")
            st.caption("多週期自適應 + 風險動態調整")
            st.success("[開發中]\n目標: 30天50%報酬\n特色: 多時間框架融合\n市場狀態自適應\n動態倉位管理")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("月交易目標", "100-150筆")
            with col2:
                st.metric("月報酬目標", "50%")
        
        st.markdown("---")
        if st.button("查看策略對比", use_container_width=True):
            st.session_state['show_comparison'] = True
    
    if st.session_state.get('show_comparison', False):
        show_strategy_comparison()
    
    if "V1" in strategy_version:
        render_v1_interface()
    elif "V2" in strategy_version:
        render_v2_interface()
    else:
        render_v3_interface()

def show_strategy_comparison():
    with st.expander("V1 vs V2 vs V3 策略對比", expanded=True):
        comparison_df = pd.DataFrame({
            '項目': ['模型架構', '時序學習', '集成學習', '信號過濾', '風險管理', '市場自適應', '月交易量', '月報酬目標', '訓練時間', '狀態'],
            'V1 反轉策略': ['XGBoost', 'X', 'X', '單層', '固定', 'X', '50-80筆', '30-50%', '5-10分鐘', '可用'],
            'V2 高頻策略': ['LightGBM/Transformer', 'O (可選)', 'O (加權集成)', '三層過濾', '動態調整', 'O', '140-150筆', '50%+', '5-15分鐘', '[無效] 盈虧因子0.90'],
            'V3 自適應策略': ['CatBoost + LSTM', 'O (多週期)', 'O (Stacking)', '五層過濾', '市場狀態動態', 'O (三模式)', '100-150筆', '50%', '10-20分鐘', '[開發中]']
        })
        st.table(comparison_df)
        if st.button("關閉對比", use_container_width=True):
            st.session_state['show_comparison'] = False
            st.rerun()

def render_v3_interface():
    st.header("V3 自適應多週期策略")
    
    st.info("""
    **V3 策略設計目標:**
    
    1. **多時間框架融合** - 15m/1h/4h信號綜合判斷
    2. **市場狀態識別** - 趨勢/震盪/波動三種模式自動切換
    3. **動態倉位管理** - 根據市場狀態調整倉位大小
    4. **五層信號過濾**:
       - Layer 1: 技術指標過濾
       - Layer 2: 成交量確認
       - Layer 3: 多週期一致性
       - Layer 4: 市場狀態匹配
       - Layer 5: 風險評分閾值
    5. **CatBoost + LSTM** - 處理類別特徵 + 時序依賴
    
    **預期績效:**
    - 勝率: 55-60%
    - 盈虧因子: >1.5
    - 月報酬: 50%
    - 最大回撤: <15%
    """)
    
    tab1, tab2, tab3 = st.tabs(["V3 模型訓練", "V3 回測分析", "V3 系統狀態"])
    
    with tab1:
        st.warning("[開發中] V3訓練功能即將上線")
        st.markdown("""
        **開發計劃:**
        1. 多週期特徵工程 (15m + 1h + 4h)
        2. 市場狀態分類器
        3. CatBoost + LSTM集成訓練
        4. 五層過濾器實現
        5. 動態風險管理
        
        **預計完成時間:** 1-2小時
        """)
    
    with tab2:
        st.warning("[開發中] 請先完成V3訓練")
    
    with tab3:
        st.info("V3開發進度: 0%\n預計完成時間: 1-2小時")

def render_v1_interface():
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["V1 模型訓練", "V1 回測分析", "V1 模擬交易", "V1 實盤交易", "V1 績效分析"])
    with tab1:
        render_v1_training()
    with tab2:
        st.info("請先完成V1訓練")
    with tab3:
        st.info("功能開發中")
    with tab4:
        st.warning("需先驗證")
    with tab5:
        st.info("請先回測")

def render_v2_interface():
    st.warning("V2策略已被標記為無效 (盈虧因子0.90)，建議使用V1或V3策略")
    tab1, tab2 = st.tabs(["V2 回測分析 (僅供參考)", "V2 問題診斷"])
    with tab1:
        st.info("V2策略不建議繼續使用")
    with tab2:
        st.markdown("""
        **V2策略失敗原因分析:**
        
        1. **盈虧因子0.90** - 平均虧損(6.28) > 平均盈利(5.20)
        2. **勝率51.9%不足以彌補虧損** - 需要至少55%勝率或更好的盈虧比
        3. **可能問題:**
           - 止盈設定太低(1.5%)
           - 止損設定太高(0.8%)
           - 信號過濾不足,交易過於頻繁(31652筆)
           - LightGBM特徵選擇不當
        
        **改進方向 (已整合進V3):**
        - 更嚴格的信號過濾(五層)
        - 更好的盈虧比(目標2:1)
        - 市場狀態自適應
        - 動態止盈止損
        """)

def render_v1_training():
    st.header("V1 模型訓練")
    st.info("V1訓練功能保持不變")

if __name__ == "__main__":
    main()
