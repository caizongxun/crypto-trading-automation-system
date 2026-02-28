"""
V4 Neural Kelly Strategy - Standalone GUI
V4 LSTM + Kelly策略獨立界面
"""
import streamlit as st
import sys
from pathlib import Path
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# 添加路徑
project_root = Path(__file__).parent.parent
v4_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(v4_root))

st.set_page_config(
    page_title="V4 Neural Kelly Strategy",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🧠 V4 Neural Kelly Strategy")
    st.caption("LSTM神經網絡 + Kelly準則倉位管理")
    
    with st.sidebar:
        st.header("V4 特點")
        st.success("""
        **核心優勢:**
        - LSTM時序學習
        - Kelly數學最優倉位
        - 動態槓桿(1-3x)
        - 六層風險控制
        
        **預期績效:**
        - 月報酬: 80-100%
        - 勝率: 60-65%
        - 回撤: <20%
        - Sharpe: >2.0
        """)
        
        st.markdown("---")
        st.warning("⚠️ 實驗階段\n建議小資金測試")
    
    tab1, tab2, tab3 = st.tabs(["📚 模型訓練", "📊 回測分析", "ℹ️ 關於V4"])
    
    with tab1:
        render_training_tab()
    
    with tab2:
        render_backtest_tab()
    
    with tab3:
        render_info_tab()

def render_training_tab():
    st.header("V4 模型訓練")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("訓練參數")
        
        symbol = st.selectbox("交易對", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"])
        timeframe = st.selectbox("時間框架", ["15m", "1h"], index=0)
        
        st.markdown("---")
        st.markdown("**LSTM參數**")
        epochs = st.slider("訓練輪數", 20, 100, 50, 10)
        hidden_size = st.select_slider("隱藏層大小", [64, 128, 256], value=128)
        num_layers = st.slider("LSTM層數", 1, 3, 2)
        
        st.markdown("---")
        st.markdown("**標籤配置**")
        label_preset = st.radio(
            "選擇預設",
            ["激進版 (推薦)", "平衡版", "自定義"],
            index=0
        )
        
        if label_preset == "自定義":
            atr_profit = st.slider("ATR盈利倍數", 0.5, 2.0, 0.7, 0.1)
            atr_loss = st.slider("ATR止損倍數", 0.8, 2.5, 1.5, 0.1)
        elif "激進" in label_preset:
            st.info("ATR 0.7x盈利, 1.5x止損\n目標: 20-25%有效標籤")
            atr_profit = 0.7
            atr_loss = 1.5
        else:
            st.info("ATR 0.9x盈利, 1.2x止損\n目標: 15-20%有效標籤")
            atr_profit = 0.9
            atr_loss = 1.2
        
        st.markdown("---")
        train_button = st.button("🚀 開始訓練", type="primary", use_container_width=True)
        
        if train_button:
            st.session_state['v4_training_params'] = {
                'symbol': symbol,
                'timeframe': timeframe,
                'epochs': epochs,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'atr_profit': atr_profit,
                'atr_loss': atr_loss
            }
            st.session_state['v4_training_started'] = True
    
    with col2:
        st.subheader("訓練過程")
        
        if st.session_state.get('v4_training_started', False):
            params = st.session_state['v4_training_params']
            
            st.info(f"""**訓練配置:**
            - 交易對: {params['symbol']}
            - 時間框架: {params['timeframe']}
            - Epochs: {params['epochs']}
            - Hidden Size: {params['hidden_size']}
            - LSTM Layers: {params['num_layers']}
            """)
            
            st.warning("""
            **命令行訓練:**
            
            由於GUI整合仍在開發中,請使用命令行訓練:
            
            ```bash
            python v4_neural_kelly_strategy/train.py \\
                --symbol {} \\
                --timeframe {} \\
                --epochs {} \\
                --hidden-size {} \\
                --num-layers {}
            ```
            
            訓練完成後,在「回測分析」頁面選擇模型進行測試。
            """.format(
                params['symbol'],
                params['timeframe'],
                params['epochs'],
                params['hidden_size'],
                params['num_layers']
            ))
            
            if st.button("複製命令"):
                st.code(f"""python v4_neural_kelly_strategy/train.py --symbol {params['symbol']} --timeframe {params['timeframe']} --epochs {params['epochs']} --hidden-size {params['hidden_size']} --num-layers {params['num_layers']}""")
            
            st.session_state['v4_training_started'] = False
        else:
            st.info("""**使用步驟:**
            
            1. 左側設定訓練參數
            2. 點擊「開始訓練」
            3. 複製命令到終端機執行
            4. 等待訓練完成(5-15分鐘)
            5. 切換到「回測分析」頁面
            
            **建議配置:**
            - 初次訓練: BTCUSDT, 15m, 50 epochs
            - 隱藏層: 128 (平衡速度和性能)
            - 標籤: 激進版 (20%+有效標籤)
            """)

def render_backtest_tab():
    st.header("V4 回測分析")
    
    models_dir = project_root / 'models'
    v4_models = []
    if models_dir.exists():
        v4_models = [d.name for d in models_dir.iterdir() if d.is_dir() and '_v4_' in d.name]
    
    if not v4_models:
        st.warning("""⚠️ 沒有V4模型
        
        請先訓練模型:
        ```bash
        python v4_neural_kelly_strategy/train.py --symbol BTCUSDT --timeframe 15m
        ```
        """)
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("回測參數")
        
        selected_model = st.selectbox("選擇模型", v4_models)
        
        # 顯示模型信息
        model_dir = models_dir / selected_model
        if (model_dir / 'model_config.json').exists():
            with open(model_dir / 'model_config.json', 'r') as f:
                config = json.load(f)
                st.markdown("**模型信息**")
                st.text(f"訓練日期: {config.get('training_date', 'N/A')}")
                st.text(f"特徵數: {config.get('feature_count', 'N/A')}")
                if 'train_results' in config:
                    r = config['train_results']
                    st.text(f"準確率: {r.get('final_accuracy', 0):.3f}")
        
        st.markdown("---")
        st.markdown("**資金設定**")
        capital = st.number_input("初始資金", 1000, 100000, 10000, 1000)
        
        st.markdown("**Kelly參數**")
        kelly_fraction = st.slider("Kelly分數", 0.10, 0.50, 0.25, 0.05)
        max_leverage = st.slider("最大槓桿", 1, 5, 3, 1)
        
        st.markdown("---")
        backtest_button = st.button("📊 開始回測", type="primary", use_container_width=True)
        
        if backtest_button:
            st.session_state['v4_backtest_params'] = {
                'model': selected_model,
                'capital': capital,
                'kelly_fraction': kelly_fraction,
                'max_leverage': max_leverage
            }
            st.session_state['v4_backtest_started'] = True
    
    with col2:
        st.subheader("回測結果")
        
        if st.session_state.get('v4_backtest_started', False):
            params = st.session_state['v4_backtest_params']
            
            st.info(f"""**回測配置:**
            - 模型: {params['model']}
            - 資金: ${params['capital']}
            - Kelly分數: {params['kelly_fraction']}
            - 最大槓桿: {params['max_leverage']}x
            """)
            
            st.warning("""
            **命令行回測:**
            
            ```bash
            python v4_neural_kelly_strategy/backtest.py \\
                --model {} \\
                --capital {} \\
                --kelly-fraction {} \\
                --max-leverage {}
            ```
            
            回測結果將保存到:
            `models/{}/backtest_results/`
            """.format(
                params['model'],
                params['capital'],
                params['kelly_fraction'],
                params['max_leverage'],
                params['model']
            ))
            
            # 檢查是否有回測結果
            results_dir = models_dir / params['model'] / 'backtest_results'
            if results_dir.exists() and (results_dir / 'metrics.json').exists():
                st.success("✅ 發現回測結果!")
                
                with open(results_dir / 'metrics.json', 'r') as f:
                    metrics = json.load(f)
                
                st.markdown("---")
                st.subheader("績效指標")
                
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("總交易", metrics.get('total_trades', 0))
                    st.metric("勝率", f"{metrics.get('win_rate', 0)*100:.1f}%")
                with col_b:
                    st.metric("總報酬", f"{metrics.get('total_return', 0)*100:.1f}%")
                    st.metric("盈虧因子", f"{metrics.get('profit_factor', 0):.2f}")
                with col_c:
                    st.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")
                    st.metric("最大回撤", f"{metrics.get('max_drawdown', 0)*100:.1f}%")
                with col_d:
                    st.metric("平均Kelly", f"{metrics.get('avg_kelly', 0)*100:.1f}%")
                    st.metric("平均槓桿", f"{metrics.get('avg_leverage', 0):.2f}x")
                
                # 權益曲線
                if (results_dir / 'equity_curve.csv').exists():
                    equity_df = pd.read_csv(results_dir / 'equity_curve.csv')
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=equity_df['timestamp'],
                        y=equity_df['equity'],
                        mode='lines',
                        name='權益',
                        line=dict(color='#00d4ff', width=2)
                    ))
                    fig.update_layout(
                        title='權益曲線',
                        height=400,
                        xaxis_title='時間',
                        yaxis_title='權益 (USDT)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            st.session_state['v4_backtest_started'] = False
        else:
            st.info("""**使用步驟:**
            
            1. 左側選擇已訓練的模型
            2. 設定Kelly參數和槓桿
            3. 點擊「開始回測」
            4. 複製命令到終端機執行
            5. 等待回測完成
            6. 刷新頁面查看結果
            
            **Kelly分數建議:**
            - 保守: 0.20 (1/5 Kelly)
            - 平衡: 0.25 (1/4 Kelly) 推薦
            - 激進: 0.30 (高風險)
            """)

def render_info_tab():
    st.header("關於V4 Neural Kelly Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("核心技術")
        st.markdown("""
        **1. LSTM神經網絡**
        - 原生時序學習能力
        - 多任務輸出(方向/勝率/賠率/信心度)
        - 記憶長期依賴關係
        
        **2. Kelly準則**
        ```
        Kelly% = (p × b - q) / b
        實際倉位 = Kelly% × 0.25
        ```
        - p: 預測勝率
        - b: 預測賠率
        - 分數Kelly降低風險
        
        **3. 動態槓桿**
        - Kelly > 40% + 信心 > 70% → 3x
        - Kelly > 30% + 信心 > 60% → 2x
        - 其他 → 1x
        
        **4. 六層風控**
        - Kelly門檻過濾
        - 倉位上限控制
        - 連敗保護機制
        - 回撤限制管理
        - 波動率自適應
        - 信心度過濾
        """)
    
    with col2:
        st.subheader("策略對比")
        
        comparison_df = pd.DataFrame({
            '項目': ['模型', '倉位管理', '槓桿', '風控層數', '月報酬', '回撤', '狀態'],
            'V3': ['LightGBM', 'ATR動態', '1x', '5層', '50%', '<30%', '推薦'],
            'V4': ['LSTM', 'Kelly最優', '1-3x動態', '6層+Kelly', '80-100%', '<20%', '實驗']
        })
        st.dataframe(comparison_df, use_container_width=True)
        
        st.markdown("---")
        st.subheader("性能目標")
        
        metrics_df = pd.DataFrame({
            '指標': ['勝率', '盈虧因子', 'Sharpe', '月報酬', '月交易'],
            '目標值': ['60-65%', '>2.0', '>2.0', '80-100%', '100-120']
        })
        st.dataframe(metrics_df, use_container_width=True)
        
        st.markdown("---")
        st.subheader("風險警告")
        st.error("""
        ⚠️ **重要提醒:**
        
        1. V4處於實驗階段
        2. 高報酬伴隨高風險
        3. Kelly依賴準確的勝率/賠率預測
        4. 動態槓桿需嚴格風控
        5. 實盤前充分測試(30天+)
        
        **建議起步:**
        - 資金: 1000-5000 USDT
        - Kelly分數: 0.20-0.25
        - 槓桿: 從1x開始
        - 30天穩定盈利後再增加
        """)
    
    st.markdown("---")
    st.subheader("文檔資源")
    
    doc_col1, doc_col2, doc_col3 = st.columns(3)
    
    with doc_col1:
        st.markdown("""**快速開始**
        - [V4 Quick Start](../V4_QUICKSTART.md)
        - [完整指南](USAGE.md)
        """)
    
    with doc_col2:
        st.markdown("""**技術文檔**
        - [V4總結](../V4_SUMMARY.md)
        - [Kelly準則](https://en.wikipedia.org/wiki/Kelly_criterion)
        """)
    
    with doc_col3:
        st.markdown("""**代碼倉庫**
        - [GitHub](https://github.com/caizongxun/crypto-trading-automation-system)
        - [HuggingFace數據](https://huggingface.co/datasets/caizongxun/crypto_market_data)
        """)

if __name__ == "__main__":
    main()
