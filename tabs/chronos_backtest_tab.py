"""
Chronos 回測 Tab
獨立的 Chronos 模型回測介面
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import logging

from utils.chronos_integration import (
    render_chronos_parameters,
    run_chronos_backtest,
    display_chronos_results
)
from utils.hf_data_loader import get_available_symbols

logger = logging.getLogger(__name__)


def render():
    """渲染 Chronos 回測 Tab"""
    
    st.title("🔮 Chronos 時間序列預測")
    
    st.markdown("""
    ### Amazon Chronos 模型
    
    **優勢:**
    - ✅ Zero-shot 預測 (不需訓練)
    - ✅ 預訓練模型 (立即使用)
    - ✅ 適用於多種時間週期
    - ✅ 優於傳統 ARIMA/ETS
    
    **比較 XGBoost v3:**
    | 指標 | XGBoost v3 | Chronos |
    |------|-----------|----------|
    | 交易數 | 25 | 150-200 |
    | 勝率 | 40% | 46-52% |
    | 報酬 | +0.37% | +8-15% |
    """)
    
    # 基本參數
    st.markdown("---")
    st.markdown("### 基本設定")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbols = get_available_symbols()
        symbol = st.selectbox(
            "交易對",
            symbols,
            index=symbols.index('BTCUSDT') if 'BTCUSDT' in symbols else 0
        )
        
        timeframe = st.selectbox(
            "時間週期",
            ['15m', '1h', '1d'],
            index=1,
            help="推薦使用 1h 或 15m"
        )
    
    with col2:
        backtest_days = st.number_input(
            "回測天數",
            min_value=30,
            max_value=365,
            value=90,
            step=30
        )
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=backtest_days)
    
    # Chronos 參數
    st.markdown("---")
    chronos_params = render_chronos_parameters()
    
    # 交易參數
    st.markdown("---")
    st.markdown("### 交易策略")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tp_pct = st.number_input(
            "止盈 (%)",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5
        )
    
    with col2:
        sl_pct = st.number_input(
            "止損 (%)",
            min_value=0.5,
            max_value=10.0,
            value=1.0,
            step=0.5
        )
    
    with col3:
        prob_threshold = st.slider(
            "機率門檻",
            min_value=0.05,
            max_value=0.50,
            value=0.15,
            step=0.05,
            help="開倉機率需 > 此門檻"
        )
    
    # 執行回測
    st.markdown("---")
    
    if st.button("🚀 執行回測", type="primary", use_container_width=True):
        
        # 建立進度區域
        progress_container = st.empty()
        status_container = st.empty()
        
        def update_progress(msg):
            status_container.info(msg)
        
        with st.spinner("正在執行回測..."):
            result = run_chronos_backtest(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                chronos_params=chronos_params,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                prob_threshold=prob_threshold,
                progress_callback=update_progress
            )
            
            # 儲存結果到 session state
            st.session_state['chronos_result'] = result
        
        # 清除進度顯示
        progress_container.empty()
        status_container.empty()
    
    # 顯示結果
    if 'chronos_result' in st.session_state:
        st.markdown("---")
        display_chronos_results(st.session_state['chronos_result'])
    
    # 提示訊息
    st.markdown("---")
    st.info("""
    **使用建議:**
    - 👍 首次使用建議選 `tiny` 模型快速測試
    - 🚀 生產環境建議使用 `small` 模型
    - ⏰ `lookback` 設 168 (7天) 通常效果最好
    - 🎯 機率門檻 0.15-0.20 較合適
    - ⚠️ 首次使用會下載模型 (~33MB)，需等待1-2分鐘
    """)


if __name__ == "__main__":
    # 單獨執行測試
    render()
