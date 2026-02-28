"""
V4 Neural Kelly Strategy - Complete GUI
V4 LSTM + Kelly策略完整界面 - 支持GUI內直接訓練
"""
import streamlit as st
import sys
from pathlib import Path
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import threading
import queue

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
    st.caption("LSTM/GRU神經網路 + Kelly準則倉位管理")
    
    with st.sidebar:
        st.header("V4 特點")
        st.success("""
        **核心優勢:**
        - 多模型選擇(LSTM/GRU/混合)
        - Kelly數學最優倉位
        - 動態槓桿(1-3x)
        - 六層風險控制
        - GUI內直接訓練
        
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
        
        # 基礎設定
        symbol = st.selectbox("交易對", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"])
        timeframe = st.selectbox("時間框架", ["15m", "1h"], index=0)
        
        st.markdown("---")
        st.markdown("**🎯 模型選擇**")
        
        model_type = st.radio(
            "神經網路架構",
            [
                "LSTM (標準) - 10分鐘",
                "GRU (輕量) - 7分鐘 ⭐推薦",
                "CNN+GRU (混合) - 5分鐘"
            ],
            index=1
        )
        
        # 提取模型類型
        if "LSTM" in model_type:
            selected_model = "lstm"
            st.info("標準LSTM,最穩定但較慢")
        elif "GRU" in model_type:
            selected_model = "gru"
            st.success("GRU比LSTM快30%,性能相近")
        else:
            selected_model = "cnn_gru"
            st.success("CNN+GRU混合,最快最輕量")
        
        st.markdown("---")
        st.markdown("**🔧 模型參數**")
        
        epochs = st.slider("訓練輪數", 20, 100, 50, 10)
        batch_size = st.select_slider("批次大小", [32, 64, 128], value=64)
        
        if selected_model != "cnn_gru":
            hidden_size = st.select_slider("隱藏層大小", [64, 128, 256], value=128)
            num_layers = st.slider("網路層數", 1, 3, 2)
        else:
            hidden_size = 128
            num_layers = 2
            st.info("CNN+GRU使用固定架構")
        
        learning_rate = st.select_slider(
            "學習率",
            [0.0001, 0.001, 0.01],
            value=0.001,
            format_func=lambda x: f"{x:.4f}"
        )
        
        st.markdown("---")
        st.markdown("**📊 標籤配置**")
        
        label_preset = st.radio(
            "選擇預設",
            ["激進版 (推薦)", "平衡版", "自定義"],
            index=0
        )
        
        if label_preset == "自定義":
            atr_profit = st.slider("ATR盈利倍數", 0.5, 2.0, 0.7, 0.1)
            atr_loss = st.slider("ATR止損倍數", 0.8, 2.5, 1.5, 0.1)
            min_volume = st.slider("最小成交量比", 0.5, 2.0, 0.7, 0.1)
            min_trend = st.slider("最小趨勢強度", 0.1, 0.5, 0.15, 0.05)
        elif "激進" in label_preset:
            st.info("ATR 0.7x盈利, 1.5x止損\n目標: 20-25%有效標籤")
            atr_profit = 0.7
            atr_loss = 1.5
            min_volume = 0.7
            min_trend = 0.15
        else:
            st.info("ATR 0.9x盈利, 1.2x止損\n目標: 15-20%有效標籤")
            atr_profit = 0.9
            atr_loss = 1.2
            min_volume = 0.8
            min_trend = 0.2
        
        st.markdown("---")
        train_button = st.button(
            "🚀 開始訓練",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.get('training_in_progress', False)
        )
        
        if train_button:
            st.session_state['v4_training_params'] = {
                'symbol': symbol,
                'timeframe': timeframe,
                'model_type': selected_model,
                'epochs': epochs,
                'batch_size': batch_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'learning_rate': learning_rate,
                'atr_profit': atr_profit,
                'atr_loss': atr_loss,
                'min_volume': min_volume,
                'min_trend': min_trend
            }
            st.session_state['v4_training_started'] = True
            st.session_state['training_in_progress'] = True
    
    with col2:
        st.subheader("訓練過程")
        
        if st.session_state.get('v4_training_started', False):
            params = st.session_state['v4_training_params']
            
            # 顯示配置
            config_col1, config_col2 = st.columns(2)
            with config_col1:
                st.info(f"""
                **基礎設定:**
                - 交易對: {params['symbol']}
                - 時間框架: {params['timeframe']}
                - 模型: {params['model_type'].upper()}
                """)
            with config_col2:
                st.info(f"""
                **訓練設定:**
                - Epochs: {params['epochs']}
                - Batch: {params['batch_size']}
                - 學習率: {params['learning_rate']}
                """)
            
            # 訓練進度容器
            progress_placeholder = st.empty()
            log_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            # 執行訓練
            try:
                train_model_in_gui(params, progress_placeholder, log_placeholder, metrics_placeholder)
                st.session_state['training_in_progress'] = False
                st.session_state['v4_training_started'] = False
                st.success("✅ 訓練完成!請切換到「回測分析」頁面測試模型。")
                st.balloons()
            except Exception as e:
                st.error(f"❌ 訓練失敗: {str(e)}")
                st.session_state['training_in_progress'] = False
                st.session_state['v4_training_started'] = False
                with st.expander("錯誤詳情"):
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.info("""
            **使用步驟:**
            
            1. ⬅️ 左側選擇模型類型:
               - **GRU**: 推薦,快速且準確
               - LSTM: 標準選擇
               - CNN+GRU: 最輕量
            
            2. 🔧 調整訓練參數
            
            3. 🚀 點擊「開始訓練」
            
            4. ⏳ 等待訓練完成(5-15分鐘)
            
            5. 📊 切換到「回測分析」測試模型
            
            **GPU加速:**
            系統會自動檢測並使用GPU
            (CPU訓練時間×3-5)
            
            **建議配置:**
            - 首次訓練: GRU, 50 epochs
            - 快速測試: GRU, 30 epochs
            - 最佳性能: LSTM, 100 epochs
            """)
            
            # 顯示模型對比
            st.markdown("---")
            st.subheader("模型對比")
            
            model_comparison = pd.DataFrame({
                '模型': ['LSTM', 'GRU', 'CNN+GRU'],
                '訓練時間': ['10分鐘', '7分鐘', '5分鐘'],
                '參數量': ['100%', '75%', '50%'],
                'GPU需求': ['4GB', '3GB', '2GB'],
                '性能': ['⭐⭐⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐'],
                '推薦': ['標準', '✅ 推薦', '輕量']
            })
            st.dataframe(model_comparison, use_container_width=True)

def train_model_in_gui(params, progress_placeholder, log_placeholder, metrics_placeholder):
    """
    在GUI中執行訓練
    """
    import importlib.util
    
    # 動態加載模組
    def load_module(module_path, module_name):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    try:
        # 載入V4模組 (先嘗試v4_neural_kelly_strategy,失敗則用adaptive_strategy_v4)
        v4_paths = [
            (v4_root, 'v4'),
            (project_root / 'adaptive_strategy_v4', 'v4_fallback')
        ]
        
        modules_loaded = False
        for path, prefix in v4_paths:
            try:
                label_gen = load_module(path / 'core' / 'label_generator.py', f'{prefix}_label_gen')
                feature_eng = load_module(path / 'core' / 'feature_engineer.py', f'{prefix}_feature_eng')
                predictor = load_module(path / 'core' / 'neural_predictor.py', f'{prefix}_predictor')
                loader = load_module(path / 'data' / 'hf_loader.py', f'{prefix}_loader')
                modules_loaded = True
                log_placeholder.info(f"✅ 模組加載成功 (來源: {path.name})")
                break
            except:
                continue
        
        if not modules_loaded:
            raise Exception("無法加載V4模組,請執行: python copy_v4_files.py")
        
        LabelGenerator = label_gen.LabelGenerator
        FeatureEngineer = feature_eng.FeatureEngineer
        NeuralPredictor = predictor.NeuralPredictor
        HFDataLoader = loader.HFDataLoader
        
        # 步驟1: 載入數據
        progress_placeholder.progress(0.1)
        log_placeholder.text("[1/6] 🔄 載入數據...")
        
        loader_instance = HFDataLoader()
        df = loader_instance.load_klines(params['symbol'], params['timeframe'])
        
        log_placeholder.success(f"[1/6] ✅ 載入 {len(df)} 筆數據")
        
        # 步驟2: 特徵工程
        progress_placeholder.progress(0.2)
        log_placeholder.text("[2/6] 🔄 特徵工程...")
        
        feature_config = {'lookback_periods': [5, 10, 20]}
        feature_engineer = FeatureEngineer(feature_config)
        df = feature_engineer.create_features(df)
        
        log_placeholder.success(f"[2/6] ✅ 生成 {len(df.columns)} 個特徵")
        
        # 步驟3: 生成標籤
        progress_placeholder.progress(0.3)
        log_placeholder.text("[3/6] 🔄 生成標籤...")
        
        label_config = {
            'forward_window': 8,
            'atr_profit_multiplier': params['atr_profit'],
            'atr_loss_multiplier': params['atr_loss'],
            'min_volume_ratio': params['min_volume'],
            'min_trend_strength': params['min_trend'],
            'max_atr_ratio': 0.08
        }
        label_generator = LabelGenerator(label_config)
        df = label_generator.generate_labels(df)
        
        long_count = (df['label'] == 1).sum()
        short_count = (df['label'] == -1).sum()
        valid_rate = (long_count + short_count) / len(df)
        
        log_placeholder.success(
            f"[3/6] ✅ 標籤完成 | 做多:{long_count} | 做空:{short_count} | 有效率:{valid_rate*100:.1f}%"
        )
        
        if valid_rate < 0.10:
            log_placeholder.warning("⚠️ 有效標籤率<10%,可能影響性能")
        
        # 步驟4: 準備數據
        progress_placeholder.progress(0.4)
        log_placeholder.text("[4/6] 🔄 準備訓練數據...")
        
        exclude_cols = ['timestamp', 'label', 'target_win_rate', 'target_payoff',
                       'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols 
                       and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
        
        for col in feature_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=feature_cols + ['label'])
        
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.15)
        
        df_train = df.iloc[:train_size]
        df_val = df.iloc[train_size:train_size+val_size]
        
        X_train = df_train[feature_cols].values
        y_train = df_train['label'].values
        X_val = df_val[feature_cols].values
        y_val = df_val['label'].values
        
        log_placeholder.success(
            f"[4/6] ✅ 訓練:{len(X_train)} | 驗證:{len(X_val)} | 特徵:{len(feature_cols)}"
        )
        
        # 步驟5: 訓練模型
        progress_placeholder.progress(0.5)
        log_placeholder.text(f"[5/6] 🔄 訓練{params['model_type'].upper()}模型...")
        
        predictor_config = {
            'input_size': len(feature_cols),
            'hidden_size': params['hidden_size'],
            'num_layers': params['num_layers'],
            'dropout': 0.2,
            'sequence_length': 20,
            'model_type': params['model_type']  # lstm/gru/cnn_gru
        }
        
        predictor_instance = NeuralPredictor(predictor_config)
        
        # 訓練並實時更新進度
        results = predictor_instance.train(
            X_train, y_train,
            X_val, y_val,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate']
        )
        
        progress_placeholder.progress(0.9)
        log_placeholder.success(f"[5/6] ✅ 訓練完成 | 準確率:{results['final_accuracy']:.3f}")
        
        # 顯示訓練指標
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("最終準確率", f"{results['final_accuracy']:.3f}")
        with metrics_col2:
            st.metric("最佳驗證Loss", f"{results['best_val_loss']:.4f}")
        with metrics_col3:
            device = "GPU" if results.get('device') == 'cuda' else "CPU"
            st.metric("訓練設備", device)
        
        # 步驟6: 保存模型
        progress_placeholder.progress(0.95)
        log_placeholder.text("[6/6] 🔄 保存模型...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{params['symbol']}_{params['timeframe']}_v4_{params['model_type']}_{timestamp}"
        model_dir = project_root / 'models' / model_name
        
        predictor_instance.save(model_dir)
        
        # 保存完整配置
        with open(model_dir / 'model_config.json', 'w') as f:
            json.dump({
                'symbol': params['symbol'],
                'timeframe': params['timeframe'],
                'model_version': 'v4',
                'model_type': params['model_type'],
                'training_date': timestamp,
                'data_samples': len(df),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'feature_count': len(feature_cols),
                'feature_names': feature_cols,
                'label_config': label_config,
                'predictor_config': predictor_config,
                'train_results': {
                    'final_accuracy': float(results['final_accuracy']),
                    'best_val_loss': float(results['best_val_loss'])
                }
            }, f, indent=2)
        
        progress_placeholder.progress(1.0)
        log_placeholder.success(f"[6/6] ✅ 模型已保存: {model_name}")
        
        st.session_state['latest_v4_model'] = model_name
        
    except Exception as e:
        raise e

def render_backtest_tab():
    st.header("V4 回測分析")
    
    models_dir = project_root / 'models'
    v4_models = []
    if models_dir.exists():
        v4_models = [d.name for d in models_dir.iterdir() if d.is_dir() and '_v4_' in d.name]
    
    if not v4_models:
        st.warning("""⚠️ 沒有V4模型
        
        請先在「模型訓練」頁面訓練模型
        """)
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("回測參數")
        
        # 優先顯示最新模型
        if 'latest_v4_model' in st.session_state and st.session_state['latest_v4_model'] in v4_models:
            default_idx = v4_models.index(st.session_state['latest_v4_model'])
        else:
            default_idx = 0
        
        selected_model = st.selectbox("選擇模型", v4_models, index=default_idx)
        
        # 顯示模型信息
        model_dir = models_dir / selected_model
        if (model_dir / 'model_config.json').exists():
            with open(model_dir / 'model_config.json', 'r') as f:
                config = json.load(f)
                st.markdown("**模型信息**")
                model_type = config.get('model_type', 'lstm').upper()
                st.text(f"架構: {model_type}")
                st.text(f"訓練: {config.get('training_date', 'N/A')}")
                st.text(f"特徵: {config.get('feature_count', 'N/A')}個")
                if 'train_results' in config:
                    r = config['train_results']
                    st.text(f"準確率: {r.get('final_accuracy', 0):.3f}")
        
        st.markdown("---")
        st.markdown("**💰 資金設定**")
        capital = st.number_input("初始資金", 1000, 100000, 10000, 1000)
        
        st.markdown("**📊 Kelly參數**")
        kelly_fraction = st.slider("Kelly分數", 0.10, 0.50, 0.25, 0.05)
        max_leverage = st.slider("最大槓桿", 1, 5, 3, 1)
        
        st.info(f"""
        **當前配置:**
        Kelly: {kelly_fraction} (1/{int(1/kelly_fraction)} Kelly)
        單筆最大倉位: {kelly_fraction*100:.0f}%
        最大槓桿: {max_leverage}x
        """)
        
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
            
            st.info("⏳ 回測功能開發中...")
            st.warning("""
            **臨時方案 - 使用命令行:**
            
            ```bash
            python v4_neural_kelly_strategy/backtest.py \\
                --model {} \\
                --capital {} \\
                --kelly-fraction {} \\
                --max-leverage {}
            ```
            
            回測結果將自動保存並在下次刷新時顯示
            """.format(
                params['model'],
                params['capital'],
                params['kelly_fraction'],
                params['max_leverage']
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
            st.info("""
            **使用步驟:**
            
            1. 選擇已訓練的模型
            2. 設定Kelly參數和槓桿
            3. 點擊「開始回測」
            4. 查看回測結果
            
            **Kelly分數建議:**
            - 保守: 0.20 (1/5 Kelly)
            - 平衡: 0.25 (1/4 Kelly) ✅
            - 激進: 0.30 (⚠️ 高風險)
            """)

def render_info_tab():
    st.header("關於V4 Neural Kelly Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 核心技術")
        st.markdown("""
        **1. 多模型支持**
        - **LSTM**: 標準長短期記憶網路
        - **GRU**: 門控循環單元(更快)
        - **CNN+GRU**: 混合架構(最輕量)
        
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
        st.subheader("📊 策略對比")
        
        comparison_df = pd.DataFrame({
            '項目': ['模型', '倉位管理', '槓桿', '風控層數', '月報酬', '回撤', '狀態'],
            'V3': ['LightGBM', 'ATR動態', '1x', '5層', '50%', '<30%', '推薦'],
            'V4': ['LSTM/GRU', 'Kelly最優', '1-3x動態', '6層+Kelly', '80-100%', '<20%', '實驗']
        })
        st.dataframe(comparison_df, use_container_width=True)
        
        st.markdown("---")
        st.subheader("🎯 性能目標")
        
        metrics_df = pd.DataFrame({
            '指標': ['勝率', '盈虧因子', 'Sharpe', '月報酬', '月交易'],
            '目標值': ['60-65%', '>2.0', '>2.0', '80-100%', '100-120']
        })
        st.dataframe(metrics_df, use_container_width=True)
        
        st.markdown("---")
        st.subheader("⚠️ 風險警告")
        st.error("""
        **重要提醒:**
        
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
    st.subheader("📚 文檔資源")
    
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
