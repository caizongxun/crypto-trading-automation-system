"""
Unified Trading System - V1 & V2 Integrated GUI
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

# 設置路徑
project_root = Path(__file__).parent.parent.parent  # crypto-trading-automation-system
v1_root = project_root / 'reversal_strategy_v1'
v2_root = project_root / 'high_frequency_strategy_v2'

# 添加到sys.path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(v1_root))
sys.path.insert(0, str(v2_root))

# V1引用
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
    """動態加載V2模組"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    st.title("🚀 加密貨幣智能交易系統")
    
    with st.sidebar:
        st.header("策略選擇")
        strategy_version = st.radio("選擇策略版本", ["V1 - 訂單流反轉策略", "V2 - 高頻Transformer策略"], index=0, help="V1適合中頻交易(50-80筆/月)，V2適合高頻交易(140-150筆/月)")
        st.markdown("---")
        if "V1" in strategy_version:
            st.subheader("🔵 V1 特點")
            st.caption("訂單流不平衡 + XGBoost")
            st.info("訂單流不平衡檢測\n流動性掃蕩識別\n市場微觀結構分析\nXGBoost機器學習")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("月交易目標", "50-80筆")
            with col2:
                st.metric("月報酬目標", "30-50%")
        else:
            st.subheader("⚡ V2 特點")
            st.caption("Transformer + 集成學習")
            st.info("Transformer時序學習\n多時間框架特徵\n三層信號過濾\n市場狀態自適應")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("月交易目標", "140-150筆")
            with col2:
                st.metric("月報酬目標", "50%+")
        st.markdown("---")
        if st.button("📊 查看V1 vs V2對比", use_container_width=True):
            st.session_state['show_comparison'] = True
    
    if st.session_state.get('show_comparison', False):
        show_strategy_comparison()
    
    if "V1" in strategy_version:
        render_v1_interface()
    else:
        render_v2_interface()

def show_strategy_comparison():
    with st.expander("🏆 V1 vs V2 策略對比", expanded=True):
        comparison_df = pd.DataFrame({
            '項目': ['模型架構', '時序學習', '集成學習', '信號過濾', '風險管理', '市場自適應', '月交易量', '月報酬目標', '訓練時間', 'GPU需求'],
            'V1 反轉策略': ['XGBoost', '✘', '✘', '單層', '固定', '✘', '50-80筆', '30-50%', '5-10分鐘', '不需要'],
            'V2 高頻策略': ['Transformer + LightGBM', '✓ (100根K線)', '✓ (加權集成)', '三層過濾', '動態調整', '✓', '140-150筆', '50%+', '10-20分鐘', '建議使用']
        })
        st.table(comparison_df)
        if st.button("關閉對比", use_container_width=True):
            st.session_state['show_comparison'] = False
            st.rerun()

def render_v1_interface():
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 模型訓練", "📈 回測分析", "🎮 模擬交易", "💰 實盤交易", "📊 績效分析"])
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
    tab1, tab2, tab3, tab4 = st.tabs(["🧠 V2模型訓練", "📈 V2回測分析", "🎮 V2模擬交易", "⚙️ 系統狀態"])
    with tab1:
        render_v2_training()
    with tab2:
        st.info("V2回測開發中")
    with tab3:
        st.info("開發中")
    with tab4:
        render_v2_status()

def render_v1_training():
    st.header("🎯 V1 模型訓練")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("訓練參數")
        symbol = st.selectbox("交易對", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "AVAXUSDT"], key="v1_train_symbol")
        timeframe = st.selectbox("時間框架", ["15m", "1h", "4h"], index=0, key="v1_train_timeframe")
        st.markdown("---")
        lookback = st.slider("回溯週期", 10, 50, 20)
        imbalance_threshold = st.slider("OFI閾值", 0.5, 0.8, 0.6, 0.05)
        st.markdown("---")
        forward_window = st.slider("前瞻窗口", 8, 20, 12)
        profit_threshold = st.slider("盈利目標 %", 0.5, 3.0, 1.0, 0.1) / 100
        stop_loss = st.slider("止損 %", 0.3, 2.0, 0.5, 0.1) / 100
        st.markdown("---")
        n_estimators = st.slider("樹數量", 100, 500, 200, 50)
        max_depth = st.slider("最大深度", 3, 10, 5)
        test_size = st.slider("驗證集比例", 0.1, 0.3, 0.2, 0.05)
        oos_size = st.slider("OOS測試集比例", 0.05, 0.2, 0.1, 0.05)
        st.markdown("---")
        if st.button("開始V1訓練", type="primary", use_container_width=True):
            st.session_state['v1_training_params'] = {'symbol': symbol, 'timeframe': timeframe, 'lookback': lookback, 'imbalance_threshold': imbalance_threshold, 'forward_window': forward_window, 'profit_threshold': profit_threshold, 'stop_loss': stop_loss, 'n_estimators': n_estimators, 'max_depth': max_depth, 'test_size': test_size, 'oos_size': oos_size}
            st.session_state['v1_training_started'] = True
    with col2:
        st.subheader("訓練過程")
        if st.session_state.get('v1_training_started', False):
            train_v1_model_in_gui(st.session_state['v1_training_params'])
            st.session_state['v1_training_started'] = False
        else:
            st.info("### V1訓練流程\n\n1. 加載HuggingFace K線\n2. OFI + 流動性信號\n3. 50+指標特徵\n4. 前瞻盈虧標籤\n5. XGBoost訓練\n6. 三集驗證\n\n**預計**: 5-10分鐘")

def train_v1_model_in_gui(params):
    try:
        config = {'signal_detection': {'lookback': params['lookback'], 'imbalance_threshold': params['imbalance_threshold'], 'liquidity_strength': 1.5, 'microstructure_window': 10}, 'feature_engineering': {'lookback_periods': [5, 10, 20, 30], 'use_price_features': True, 'use_volume_features': True, 'use_microstructure': True}, 'ml_model': {'n_estimators': params['n_estimators'], 'max_depth': params['max_depth'], 'learning_rate': 0.05}, 'label_generation': {'forward_window': params['forward_window'], 'profit_threshold': params['profit_threshold'], 'stop_loss': params['stop_loss']}}
        with st.spinner('步驟 1/5: 加載數據...'):
            loader = HFDataLoader()
            df = loader.load_klines(params['symbol'], params['timeframe'])
            st.success(f"✓ 加載: {len(df)} 筆")
        with st.spinner('步驟 2/5: 信號檢測...'):
            signal_detector = SignalDetector(config['signal_detection'])
            df = signal_detector.detect_signals(df)
            st.success(f"✓ 做多: {df['signal_long'].sum()} | 做空: {df['signal_short'].sum()}")
        with st.spinner('步驟 3/5: 特徵工程...'):
            feature_engineer = FeatureEngineer(config['feature_engineering'])
            df = feature_engineer.create_features(df)
            df = feature_engineer.create_labels(df, forward_window=config['label_generation']['forward_window'], profit_threshold=config['label_generation']['profit_threshold'], stop_loss=config['label_generation']['stop_loss'])
            feature_cols = feature_engineer.get_feature_names()
            st.success(f"✓ 特徵: {len(feature_cols)}")
        with st.spinner('步驟 4/5: 訓練模型...'):
            ml_predictor = MLPredictor(config['ml_model'])
            train_results = ml_predictor.train(df, feature_cols, test_size=params['test_size'], oos_size=params['oos_size'])
            st.success("✓ 訓練完成")
        with st.spinner('步驟 5/5: 保存模型...'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"{params['symbol']}_{params['timeframe']}_v1_{timestamp}"
            model_dir = project_root / 'models' / model_name
            ml_predictor.save(model_dir)
            with open(model_dir / 'model_config.json', 'w') as f:
                json.dump({'symbol': params['symbol'], 'timeframe': params['timeframe'], 'training_date': timestamp, 'model_version': 'v1', 'data_samples': len(df), 'config': config}, f, indent=2)
            st.success(f"✅ V1完成: {model_name}")
        st.session_state['latest_v1_model'] = model_name
    except Exception as e:
        st.error(f"❌ 失敗: {str(e)}")
        import traceback
        with st.expander("詳情"):
            st.code(traceback.format_exc())

def render_v2_training():
    st.header("🧠 V2 Transformer模型訓練")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("訓練參數")
        symbol = st.selectbox("交易對", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"], key="v2_train_symbol")
        timeframe = st.selectbox("時間框架", ["15m", "1h"], index=0, key="v2_train_timeframe")
        st.markdown("---")
        sequence_length = st.slider("序列長度", 50, 200, 100, 10, help="Transformer輸入K線數")
        use_transformer = st.checkbox("Transformer模型", value=True)
        use_lgb = st.checkbox("LightGBM模型", value=True)
        if use_transformer and use_lgb:
            st.caption("✓ 集成模式")
        st.markdown("---")
        if st.button("開始V2訓練", type="primary", use_container_width=True):
            st.session_state['v2_training_params'] = {'symbol': symbol, 'timeframe': timeframe, 'sequence_length': sequence_length, 'use_transformer': use_transformer, 'use_lgb': use_lgb}
            st.session_state['v2_training_started'] = True
    with col2:
        st.subheader("訓練過程")
        if st.session_state.get('v2_training_started', False):
            train_v2_model_in_gui(st.session_state['v2_training_params'])
            st.session_state['v2_training_started'] = False
        else:
            st.info("### V2 Transformer訓練\n\n1. 加載HuggingFace K線\n2. 50+指標 + 微觀結構\n3. 100根K線序列\n4. Transformer + LightGBM\n5. 加權集成\n\n**預計**: 10-20分鐘")

def train_v2_model_in_gui(params):
    try:
        v2_feature_path = v2_root / 'core' / 'feature_engineer.py'
        v2_ensemble_path = v2_root / 'core' / 'ensemble_predictor.py'
        v2_loader_path = v2_root / 'data' / 'hf_loader.py'
        
        v2_feature_module = load_v2_module(v2_feature_path, 'v2_feature_engineer')
        v2_ensemble_module = load_v2_module(v2_ensemble_path, 'v2_ensemble_predictor')
        v2_loader_module = load_v2_module(v2_loader_path, 'v2_hf_loader')
        
        V2FeatureEngineer = v2_feature_module.FeatureEngineer
        EnsemblePredictor = v2_ensemble_module.EnsemblePredictor
        V2HFDataLoader = v2_loader_module.HFDataLoader
        
        with st.spinner('步驟 1/6: 加載數據...'):
            loader = V2HFDataLoader()
            df = loader.load_klines(params['symbol'], params['timeframe'])
            st.success(f"✓ 加載: {len(df)} 筆")
        
        with st.spinner('步驟 2/6: 提取特徵...'):
            feature_config = {'sequence_length': params['sequence_length'], 'use_orderbook_features': False, 'use_microstructure': True, 'use_momentum': True, 'lookback_periods': [5, 10, 20, 50]}
            feature_engineer = V2FeatureEngineer(feature_config)
            df = feature_engineer.create_features(df)
            st.success(f"✓ 特徵: {len(df)} 筆")
        
        with st.spinner('步驟 3/6: 生成標籤...'):
            df = create_v2_labels(df)
            long_signals = (df['label'] == 1).sum()
            short_signals = (df['label'] == -1).sum()
            st.success(f"✓ 做多: {long_signals} | 做空: {short_signals}")
        
        with st.spinner('步驟 4/6: 準備數據...'):
            exclude_cols = ['timestamp', 'label']
            feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
            
            for col in feature_cols:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=feature_cols + ['label'])
            
            seq_len = params['sequence_length']
            train_size = int(len(df) * 0.7)
            val_size = int(len(df) * 0.15)
            
            if train_size < seq_len + 100:
                st.error(f"數據不足! 需要至少 {seq_len + 100} 筆,當前僅 {train_size} 筆")
                return
            
            df_train = df.iloc[:train_size].reset_index(drop=True)
            df_val = df.iloc[train_size:train_size+val_size].reset_index(drop=True)
            
            X_train_seq = feature_engineer.prepare_sequences(df_train, feature_cols)
            X_val_seq = feature_engineer.prepare_sequences(df_val, feature_cols)
            
            X_train = df_train[feature_cols].values[seq_len:]
            y_train = df_train['label'].values[seq_len:]
            X_val = df_val[feature_cols].values[seq_len:]
            y_val = df_val['label'].values[seq_len:]
            
            st.success(f"✓ 訓練: {len(X_train)} | 驗證: {len(X_val)}")
            st.info(f"序列形狀: Train {X_train_seq.shape}, Val {X_val_seq.shape}")
        
        with st.spinner('步驟 5/6: 訓練模型...'):
            ensemble_config = {'use_transformer': params['use_transformer'], 'use_lgb': params['use_lgb'], 'ensemble_method': 'weighted_avg', 'weights': {'transformer': 0.5, 'lgb': 0.5}}
            predictor = EnsemblePredictor(ensemble_config)
            results = predictor.train(X_train, y_train, X_val, y_val, X_train_seq, X_val_seq)
            st.success("✓ 訓練完成")
        
        with st.spinner('步驟 6/6: 保存模型...'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"{params['symbol']}_{params['timeframe']}_v2_{timestamp}"
            model_dir = project_root / 'models' / model_name
            predictor.save(model_dir)
            with open(model_dir / 'model_config.json', 'w') as f:
                json.dump({'symbol': params['symbol'], 'timeframe': params['timeframe'], 'model_version': 'v2', 'training_date': timestamp, 'data_samples': len(df), 'train_samples': len(X_train), 'val_samples': len(X_val), 'feature_count': len(feature_cols), 'sequence_length': params['sequence_length']}, f, indent=2)
            st.success(f"✅ V2完成: {model_name}")
        
        st.session_state['latest_v2_model'] = model_name
    except Exception as e:
        st.error(f"❌ 失敗: {str(e)}")
        import traceback
        with st.expander("詳情"):
            st.code(traceback.format_exc())

def create_v2_labels(df, forward_window=8, profit_threshold=0.004, stop_loss=0.003):
    df = df.copy()
    df['label'] = 0
    for i in range(len(df) - forward_window):
        current_price = df.iloc[i]['close']
        future_prices = df.iloc[i+1:i+forward_window+1]['close']
        max_price = future_prices.max()
        min_price = future_prices.min()
        max_profit = (max_price - current_price) / current_price
        max_loss = (current_price - min_price) / current_price
        if max_profit >= profit_threshold and max_loss < stop_loss:
            df.loc[df.index[i], 'label'] = 1
        short_profit = (current_price - min_price) / current_price
        short_loss = (max_price - current_price) / current_price
        if short_profit >= profit_threshold and short_loss < stop_loss:
            df.loc[df.index[i], 'label'] = -1
    return df

def render_v2_status():
    st.header("⚙️ V2 狀態")
    col1, col2, col3 = st.columns(3)
    models_dir = project_root / 'models'
    v1_count = v2_count = 0
    if models_dir.exists():
        for d in models_dir.iterdir():
            if d.is_dir():
                if '_v1_' in d.name:
                    v1_count += 1
                elif '_v2_' in d.name:
                    v2_count += 1
    with col1:
        st.metric("V1模型", v1_count)
    with col2:
        st.metric("V2模型", v2_count)
    with col3:
        st.metric("總模型", v1_count + v2_count)

if __name__ == "__main__":
    main()
