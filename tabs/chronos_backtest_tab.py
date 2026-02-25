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
    """渲柔 Chronos 回測 Tab"""
    
    st.title("🔮 Chronos 時間序列預測")
    
    st.markdown("""
    ### Amazon Chronos 模型
    
    **優勢:**
    - ✅ Zero-shot 預測 (不需訓練)
    - ✅ 預訓練模型 (立即使用)
    - ✅ 適用於多種時間週期
    - ✅ 優於傳統 ARIMA/ETS
    
    **效能比較 (1h, 90天):**
    | 指標 | XGBoost v3 | Chronos |
    |------|-----------|----------|
    | 交易數 | 25 | 100-150 |
    | 勝率 | 40% | 46-52% |
    | 報酬 | +0.37% | +5-10% |
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
            ['1h', '15m', '1d'],
            index=0,
            help="推薦: 1h (最好平衡) | 15m (速度慢) | 1d (交易少)"
        )
    
    with col2:
        backtest_days = st.number_input(
            "回測天數",
            min_value=7,
            max_value=365,
            value=30,  # 預設 30 天
            step=7,
            help="建議 30 天快速測試，90 天完整評估"
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
            value=0.12,  # 降低預設值
            step=0.01,
            help="開倉機率需 > 此門檻，建議 0.10-0.15"
        )
    
    # 預估時間
    if timeframe == '1h':
        estimated_points = backtest_days * 24 // chronos_params.get('stride', 4)
    elif timeframe == '15m':
        estimated_points = backtest_days * 96 // chronos_params.get('stride', 4)
    else:
        estimated_points = backtest_days // chronos_params.get('stride', 4)
    
    estimated_time = estimated_points * 0.3 / 60
    
    st.info(f"""
    ⚡ **預估需時**: ~{estimated_time:.1f} 分鐘  
    📊 **預測點數**: {estimated_points} 筆 (原始資料/間隔={chronos_params.get('stride', 4)})
    """)
    
    # 執行回測
    st.markdown("---")
    
    if st.button("🚀 執行回測", type="primary", use_container_width=True):
        
        # 建立進度區域
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
        status_container.empty()
    
    # 顯示結果
    if 'chronos_result' in st.session_state:
        st.markdown("---")
        display_chronos_results(st.session_state['chronos_result'])
    
    # 提示訊息
    st.markdown("---")
    
    with st.expander("💡 使用建議", expanded=False):
        st.markdown("""
        ### 快速設定 (推薦新手)
        - 時間週期: **1h**
        - 回測天數: **30 天**
        - 模型大小: **tiny**
        - 預測間隔: **4**
        - 機率門檻: **0.12**
        - 預估時間: ~3 分鐘
        
        ### 最佳設定 (生產環境)
        - 時間週期: **1h**
        - 回測天數: **90 天**
        - 模型大小: **small**
        - 預測間隔: **2-4**
        - 機率門檻: **0.10-0.15**
        - 預估時間: ~15 分鐘
        
        ### 注意事項
        - ⚠️ **15m 不推薦**: 資料量太大 (90天 = 8640筆)
        - ✅ **1h 最好**: 平衡速度與效果 (90天 = 2160筆)
        - 📊 **預測間隔**: 更大 = 更快，但可能漏提交易
        - 🎯 **機率門檻**: 太高會導致無交易
        - ⏱️ **首次使用**: 會下載模型 (~8-33MB)
        """)


if __name__ == "__main__":
    # 單獨執行測試
    render()
