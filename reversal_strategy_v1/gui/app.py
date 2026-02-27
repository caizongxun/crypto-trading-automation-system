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
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

project_root = Path(__file__).parent.parent.parent
v1_root = project_root / 'reversal_strategy_v1'
v2_root = project_root / 'high_frequency_strategy_v2'

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(v1_root))
sys.path.insert(0, str(v2_root))

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
        strategy_version = st.radio("選擇策略版本", ["V1 - 訂單流反轉策略", "V2 - 高頻Transformer策略"], index=0)
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
        else:
            st.subheader("V2 特點")
            st.caption("LightGBM / Transformer")
            st.info("LightGBM快速訓練\n可選Transformer時序\n多時間框架特徵\n市場狀態自適應")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("月交易目標", "140-150筆")
            with col2:
                st.metric("月報酬目標", "50%+")
        st.markdown("---")
        if st.button("查看V1 vs V2對比", use_container_width=True):
            st.session_state['show_comparison'] = True
    
    if st.session_state.get('show_comparison', False):
        show_strategy_comparison()
    
    if "V1" in strategy_version:
        render_v1_interface()
    else:
        render_v2_interface()

def show_strategy_comparison():
    with st.expander("V1 vs V2 策略對比", expanded=True):
        comparison_df = pd.DataFrame({
            '項目': ['模型架構', '時序學習', '集成學習', '信號過濾', '風險管理', '市場自適應', '月交易量', '月報酬目標', '訓練時間', 'GPU需求'],
            'V1 反轉策略': ['XGBoost', 'X', 'X', '單層', '固定', 'X', '50-80筆', '30-50%', '5-10分鐘', '不需要'],
            'V2 高頻策略': ['LightGBM/Transformer', 'O (可選)', 'O (加權集成)', '三層過濾', '動態調整', 'O', '140-150筆', '50%+', '5-15分鐘', '可選']
        })
        st.table(comparison_df)
        if st.button("關閉對比", use_container_width=True):
            st.session_state['show_comparison'] = False
            st.rerun()

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
    tab1, tab2, tab3, tab4 = st.tabs(["V2 模型訓練", "V2 回測分析", "V2 模擬交易", "V2 系統狀態"])
    with tab1:
        render_v2_training()
    with tab2:
        render_v2_backtest()
    with tab3:
        st.info("開發中")
    with tab4:
        render_v2_status()

def render_v1_training():
    st.header("V1 模型訓練")
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
            st.info("V1訓練流程:\n1. 加載HuggingFace K線\n2. OFI + 流動性信號\n3. 50+指標特徵\n4. 前瞻盈虧標籤\n5. XGBoost訓練\n6. 三集驗證\n\n預計: 5-10分鐘")

def train_v1_model_in_gui(params):
    try:
        config = {'signal_detection': {'lookback': params['lookback'], 'imbalance_threshold': params['imbalance_threshold'], 'liquidity_strength': 1.5, 'microstructure_window': 10}, 'feature_engineering': {'lookback_periods': [5, 10, 20, 30], 'use_price_features': True, 'use_volume_features': True, 'use_microstructure': True}, 'ml_model': {'n_estimators': params['n_estimators'], 'max_depth': params['max_depth'], 'learning_rate': 0.05}, 'label_generation': {'forward_window': params['forward_window'], 'profit_threshold': params['profit_threshold'], 'stop_loss': params['stop_loss']}}
        with st.spinner('步驟 1/5: 加載數據...'):
            loader = HFDataLoader()
            df = loader.load_klines(params['symbol'], params['timeframe'])
            st.success(f"[OK] 加載: {len(df)} 筆")
        with st.spinner('步驟 2/5: 信號檢測...'):
            signal_detector = SignalDetector(config['signal_detection'])
            df = signal_detector.detect_signals(df)
            st.success(f"[OK] 做多: {df['signal_long'].sum()} | 做空: {df['signal_short'].sum()}")
        with st.spinner('步驟 3/5: 特徵工程...'):
            feature_engineer = FeatureEngineer(config['feature_engineering'])
            df = feature_engineer.create_features(df)
            df = feature_engineer.create_labels(df, forward_window=config['label_generation']['forward_window'], profit_threshold=config['label_generation']['profit_threshold'], stop_loss=config['label_generation']['stop_loss'])
            feature_cols = feature_engineer.get_feature_names()
            st.success(f"[OK] 特徵: {len(feature_cols)}")
        with st.spinner('步驟 4/5: 訓練模型...'):
            ml_predictor = MLPredictor(config['ml_model'])
            train_results = ml_predictor.train(df, feature_cols, test_size=params['test_size'], oos_size=params['oos_size'])
            st.success("[OK] 訓練完成")
        with st.spinner('步驟 5/5: 保存模型...'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"{params['symbol']}_{params['timeframe']}_v1_{timestamp}"
            model_dir = project_root / 'models' / model_name
            ml_predictor.save(model_dir)
            with open(model_dir / 'model_config.json', 'w') as f:
                json.dump({'symbol': params['symbol'], 'timeframe': params['timeframe'], 'training_date': timestamp, 'model_version': 'v1', 'data_samples': len(df), 'config': config}, f, indent=2)
            st.success(f"[V1 完成] {model_name}")
        st.session_state['latest_v1_model'] = model_name
    except Exception as e:
        st.error(f"[失敗] {str(e)}")
        import traceback
        with st.expander("詳情"):
            st.code(traceback.format_exc())

def render_v2_training():
    st.header("V2 模型訓練")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("訓練參數")
        symbol = st.selectbox("交易對", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"], key="v2_train_symbol")
        timeframe = st.selectbox("時間框架", ["15m", "1h"], index=0, key="v2_train_timeframe")
        st.markdown("---")
        sequence_length = st.slider("序列長度", 50, 200, 100, 10, help="Transformer輸入K線數")
        st.markdown("**模型選擇**")
        use_lgb = st.checkbox("LightGBM模型", value=True, help="快速訓練，低記憶體")
        use_transformer = st.checkbox("Transformer模型", value=False, help="時序學習，需要更多記憶體")
        if not use_lgb and not use_transformer:
            st.warning("[警告] 至少選擇一個模型")
        elif use_transformer and use_lgb:
            st.info("[模式] 集成 (最佳效果)")
        elif use_lgb:
            st.success("[模式] LightGBM (快速訓練)")
        else:
            st.info("[模式] Transformer (時序學習)")
        st.markdown("---")
        if st.button("開始V2訓練", type="primary", use_container_width=True, disabled=(not use_lgb and not use_transformer)):
            st.session_state['v2_training_params'] = {'symbol': symbol, 'timeframe': timeframe, 'sequence_length': sequence_length, 'use_transformer': use_transformer, 'use_lgb': use_lgb}
            st.session_state['v2_training_started'] = True
    with col2:
        st.subheader("訓練過程")
        if st.session_state.get('v2_training_started', False):
            train_v2_model_in_gui(st.session_state['v2_training_params'])
            st.session_state['v2_training_started'] = False
        else:
            st.info("V2 訓練流程:\n1. 加載HuggingFace K線\n2. 50+指標 + 微觀結構\n3. 100根K線序列\n4. LightGBM / Transformer\n5. 加權集成\n\n預計: \n- LightGBM: 5分鐘\n- Transformer: 15分鐘\n- 集成: 20分鐘")

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
        
        with st.spinner('步驟 1/7: 加載數據...'):
            loader = V2HFDataLoader()
            df = loader.load_klines(params['symbol'], params['timeframe'])
            st.success(f"[OK] 加載: {len(df)} 筆")
        
        with st.spinner('步驟 2/7: 提取特徵...'):
            feature_config = {'sequence_length': params['sequence_length'], 'use_orderbook_features': False, 'use_microstructure': True, 'use_momentum': True, 'lookback_periods': [5, 10, 20, 50]}
            feature_engineer = V2FeatureEngineer(feature_config)
            df = feature_engineer.create_features(df)
            st.success(f"[OK] 特徵: {len(df)} 筆")
        
        with st.spinner('步驟 3/7: 生成標籤...'):
            df = create_v2_labels(df)
            long_signals = (df['label'] == 1).sum()
            short_signals = (df['label'] == -1).sum()
            neutral_signals = (df['label'] == 0).sum()
            st.info(f"[標籤分布] 做多={long_signals} ({long_signals/len(df)*100:.1f}%) | 做空={short_signals} ({short_signals/len(df)*100:.1f}%) | 中立={neutral_signals} ({neutral_signals/len(df)*100:.1f}%)")
            st.success(f"[OK] 做多: {long_signals} | 做空: {short_signals}")
        
        with st.spinner('步驟 4/7: 準備數據...'):
            exclude_cols = ['timestamp', 'label']
            feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
            for col in feature_cols:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=feature_cols + ['label'])
            seq_len = params['sequence_length']
            train_size = int(len(df) * 0.7)
            val_size = int(len(df) * 0.15)
            if train_size < seq_len + 100:
                st.error(f"[錯誤] 數據不足! 需要至少 {seq_len + 100} 筆,當前僅 {train_size} 筆")
                return
            df_train = df.iloc[:train_size].reset_index(drop=True)
            df_val = df.iloc[train_size:train_size+val_size].reset_index(drop=True)
            X_train_seq = None
            X_val_seq = None
            if params['use_transformer']:
                X_train_seq = feature_engineer.prepare_sequences(df_train, feature_cols)
                X_val_seq = feature_engineer.prepare_sequences(df_val, feature_cols)
                st.info(f"[序列形狀] Train {X_train_seq.shape}, Val {X_val_seq.shape}")
            X_train = df_train[feature_cols].values[seq_len:]
            y_train = df_train['label'].values[seq_len:]
            X_val = df_val[feature_cols].values[seq_len:]
            y_val = df_val['label'].values[seq_len:]
            st.success(f"[OK] 訓練: {len(X_train)} | 驗證: {len(X_val)}")
        
        with st.spinner('步驟 5/7: 訓練模型...'):
            ensemble_config = {'use_transformer': params['use_transformer'], 'use_lgb': params['use_lgb'], 'ensemble_method': 'weighted_avg', 'weights': {'transformer': 0.5, 'lgb': 0.5}}
            predictor = EnsemblePredictor(ensemble_config)
            results = predictor.train(X_train, y_train, X_val, y_val, X_train_seq, X_val_seq)
            st.success("[OK] 訓練完成")
        
        with st.spinner('步驟 6/7: 評估模型...'):
            y_pred_train, _ = predictor.predict(X_train, X_train_seq)
            y_pred_val, conf_val = predictor.predict(X_val, X_val_seq)
            
            # 訓練集指標
            train_report = classification_report(y_train, y_pred_train, target_names=['Short', 'Neutral', 'Long'], output_dict=True, zero_division=0)
            # 驗證集指標
            val_report = classification_report(y_val, y_pred_val, target_names=['Short', 'Neutral', 'Long'], output_dict=True, zero_division=0)
            
            # AUC (多分類 One-vs-Rest)
            try:
                from sklearn.preprocessing import label_binarize
                y_val_bin = label_binarize(y_val, classes=[-1, 0, 1])
                y_pred_val_bin = label_binarize(y_pred_val, classes=[-1, 0, 1])
                val_auc = roc_auc_score(y_val_bin, y_pred_val_bin, average='macro', multi_class='ovr')
            except:
                val_auc = 0
            
            st.success("[OK] 評估完成")
            
            # 顯示評估結果
            st.markdown("---")
            st.subheader("訓練集績效")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("準確率", f"{train_report['accuracy']:.3f}")
            with col2:
                st.metric("Macro Precision", f"{train_report['macro avg']['precision']:.3f}")
            with col3:
                st.metric("Macro Recall", f"{train_report['macro avg']['recall']:.3f}")
            
            st.markdown("**分類詳細**")
            train_metrics_df = pd.DataFrame({
                '類別': ['Short (-1)', 'Neutral (0)', 'Long (1)'],
                'Precision': [train_report['Short']['precision'], train_report['Neutral']['precision'], train_report['Long']['precision']],
                'Recall': [train_report['Short']['recall'], train_report['Neutral']['recall'], train_report['Long']['recall']],
                'F1-Score': [train_report['Short']['f1-score'], train_report['Neutral']['f1-score'], train_report['Long']['f1-score']],
                'Support': [train_report['Short']['support'], train_report['Neutral']['support'], train_report['Long']['support']]
            })
            st.dataframe(train_metrics_df, use_container_width=True)
            
            st.markdown("---")
            st.subheader("驗證集績效")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("準確率", f"{val_report['accuracy']:.3f}")
            with col2:
                st.metric("Macro Precision", f"{val_report['macro avg']['precision']:.3f}")
            with col3:
                st.metric("Macro Recall", f"{val_report['macro avg']['recall']:.3f}")
            with col4:
                st.metric("AUC (OvR)", f"{val_auc:.3f}")
            
            st.markdown("**分類詳細**")
            val_metrics_df = pd.DataFrame({
                '類別': ['Short (-1)', 'Neutral (0)', 'Long (1)'],
                'Precision': [val_report['Short']['precision'], val_report['Neutral']['precision'], val_report['Long']['precision']],
                'Recall': [val_report['Short']['recall'], val_report['Neutral']['recall'], val_report['Long']['recall']],
                'F1-Score': [val_report['Short']['f1-score'], val_report['Neutral']['f1-score'], val_report['Long']['f1-score']],
                'Support': [val_report['Short']['support'], val_report['Neutral']['support'], val_report['Long']['support']]
            })
            st.dataframe(val_metrics_df, use_container_width=True)
            
            # 平均信心度
            avg_conf_long = conf_val[y_pred_val == 1].mean() if (y_pred_val == 1).sum() > 0 else 0
            avg_conf_short = conf_val[y_pred_val == -1].mean() if (y_pred_val == -1).sum() > 0 else 0
            st.info(f"[信心度] 做多平均: {avg_conf_long:.1%} | 做空平均: {avg_conf_short:.1%}")
        
        with st.spinner('步驟 7/7: 保存模型...'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"{params['symbol']}_{params['timeframe']}_v2_{timestamp}"
            model_dir = project_root / 'models' / model_name
            predictor.save(model_dir)
            
            # 保存評估指標
            metrics_to_save = {
                'train_accuracy': train_report['accuracy'],
                'val_accuracy': val_report['accuracy'],
                'val_auc': float(val_auc),
                'val_precision': val_report['macro avg']['precision'],
                'val_recall': val_report['macro avg']['recall'],
                'val_f1': val_report['macro avg']['f1-score'],
                'avg_conf_long': float(avg_conf_long),
                'avg_conf_short': float(avg_conf_short)
            }
            
            with open(model_dir / 'model_config.json', 'w') as f:
                json.dump({
                    'symbol': params['symbol'],
                    'timeframe': params['timeframe'],
                    'model_version': 'v2',
                    'training_date': timestamp,
                    'data_samples': len(df),
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'feature_count': len(feature_cols),
                    'sequence_length': params['sequence_length'],
                    'use_transformer': params['use_transformer'],
                    'use_lgb': params['use_lgb'],
                    'metrics': metrics_to_save
                }, f, indent=2)
            st.success(f"[V2 完成] {model_name}")
        
        st.session_state['latest_v2_model'] = model_name
        
    except Exception as e:
        st.error(f"[失敗] {str(e)}")
        import traceback
        with st.expander("詳情"):
            st.code(traceback.format_exc())

def render_v2_backtest():
    st.header("V2 回測分析")
    models_dir = project_root / 'models'
    v2_models = []
    if models_dir.exists():
        v2_models = [d.name for d in models_dir.iterdir() if d.is_dir() and '_v2_' in d.name]
    if not v2_models:
        st.warning("[警告] 沒有V2模型，請先訓練模型")
        return
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("回測參數")
        selected_model = st.selectbox("選擇模型", v2_models, key="v2_backtest_model")
        
        # 顯示模型評估指標
        model_dir = project_root / 'models' / selected_model
        if (model_dir / 'model_config.json').exists():
            with open(model_dir / 'model_config.json', 'r') as f:
                model_config = json.load(f)
                if 'metrics' in model_config:
                    st.markdown("**模型指標**")
                    m = model_config['metrics']
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("驗證準確率", f"{m['val_accuracy']:.3f}")
                        st.metric("Val AUC", f"{m['val_auc']:.3f}")
                    with col_b:
                        st.metric("Val Precision", f"{m['val_precision']:.3f}")
                        st.metric("Val Recall", f"{m['val_recall']:.3f}")
        
        st.markdown("---")
        st.markdown("**資金設定**")
        initial_capital = st.number_input("初始資金 (USDT)", 1000, 100000, 10000, 1000)
        
        st.markdown("**交易設定**")
        commission = st.slider("手續費 %", 0.01, 0.5, 0.1, 0.01) / 100
        slippage = st.slider("滑點 %", 0.01, 0.2, 0.05, 0.01) / 100
        
        st.markdown("**信心度設定**")
        confidence_threshold = st.slider("信心度閾值", 0.1, 0.9, 0.35, 0.05, help="只有預測信心度高於此閾值才交易")
        
        st.markdown("**止盈止損**")
        take_profit = st.slider("止盈 %", 0.5, 5.0, 1.5, 0.1) / 100
        stop_loss = st.slider("止損 %", 0.3, 3.0, 0.8, 0.1) / 100
        
        st.markdown("---")
        if st.button("開始V2回測", type="primary", use_container_width=True):
            st.session_state['v2_backtest_params'] = {
                'model_name': selected_model,
                'initial_capital': initial_capital,
                'commission': commission,
                'slippage': slippage,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'confidence_threshold': confidence_threshold
            }
            st.session_state['v2_backtest_started'] = True
    
    with col2:
        st.subheader("回測結果")
        if st.session_state.get('v2_backtest_started', False):
            run_v2_backtest_in_gui(st.session_state['v2_backtest_params'])
            st.session_state['v2_backtest_started'] = False
        else:
            st.info("選擇模型並設定參數後，點擊「開始V2回測」\n\n建議:\n- 如果模型不產生信號，試著降低信心度閾值至 0.2-0.3")

def run_v2_backtest_in_gui(params):
    try:
        model_dir = project_root / 'models' / params['model_name']
        with open(model_dir / 'model_config.json', 'r') as f:
            model_config = json.load(f)
        symbol = model_config['symbol']
        timeframe = model_config['timeframe']
        sequence_length = model_config.get('sequence_length', 100)
        
        v2_backtest_path = v2_root / 'backtest' / 'engine.py'
        v2_ensemble_path = v2_root / 'core' / 'ensemble_predictor.py'
        v2_feature_path = v2_root / 'core' / 'feature_engineer.py'
        v2_loader_path = v2_root / 'data' / 'hf_loader.py'
        
        v2_backtest_module = load_v2_module(v2_backtest_path, 'v2_backtest_engine')
        v2_ensemble_module = load_v2_module(v2_ensemble_path, 'v2_ensemble_predictor')
        v2_feature_module = load_v2_module(v2_feature_path, 'v2_feature_engineer')
        v2_loader_module = load_v2_module(v2_loader_path, 'v2_hf_loader')
        
        V2BacktestEngine = v2_backtest_module.BacktestEngine
        EnsemblePredictor = v2_ensemble_module.EnsemblePredictor
        V2FeatureEngineer = v2_feature_module.FeatureEngineer
        V2HFDataLoader = v2_loader_module.HFDataLoader
        
        with st.spinner('步驟 1/5: 加載模型...'):
            predictor = EnsemblePredictor({})
            predictor.load(model_dir)
            predictor.confidence_threshold = params['confidence_threshold']
            st.success(f"[OK] 模型: {symbol} {timeframe} | 信心度: {params['confidence_threshold']:.0%}")
        
        with st.spinner('步驟 2/5: 加載數據...'):
            loader = V2HFDataLoader()
            df = loader.load_klines(symbol, timeframe)
            st.success(f"[OK] 數據: {len(df)} 筆")
        
        with st.spinner('步驟 3/5: 特徵工程...'):
            feature_config = {'sequence_length': sequence_length, 'use_orderbook_features': False, 'use_microstructure': True, 'use_momentum': True, 'lookback_periods': [5, 10, 20, 50]}
            feature_engineer = V2FeatureEngineer(feature_config)
            df = feature_engineer.create_features(df)
            exclude_cols = ['timestamp']
            feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
            for col in feature_cols:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=feature_cols)
            st.success(f"[OK] 特徵: {len(feature_cols)}")
        
        with st.spinner('步驟 4/5: 生成預測...'):
            X = df[feature_cols].values
            X_seq = None
            if model_config.get('use_transformer', False):
                X_seq = feature_engineer.prepare_sequences(df, feature_cols)
                X = X[sequence_length:]
            predictions, confidences = predictor.predict(X, X_seq)
            
            long_count = (predictions == 1).sum()
            short_count = (predictions == -1).sum()
            neutral_count = (predictions == 0).sum()
            st.info(f"[信號統計] 做多={long_count} ({long_count/len(predictions)*100:.1f}%) | 做空={short_count} ({short_count/len(predictions)*100:.1f}%) | 中立={neutral_count} ({neutral_count/len(predictions)*100:.1f}%)")
            avg_conf = confidences[predictions != 0].mean() if len(confidences[predictions != 0]) > 0 else 0
            st.info(f"[平均信心度] {avg_conf:.1%}")
            st.success(f"[OK] 預測: {len(predictions)} 筆")
        
        with st.spinner('步驟 5/5: 執行回測...'):
            backtest_config = {
                'initial_capital': params['initial_capital'],
                'commission': params['commission'],
                'slippage': params['slippage'],
                'take_profit': params['take_profit'],
                'stop_loss': params['stop_loss']
            }
            engine = V2BacktestEngine(backtest_config)
            df_backtest = df.iloc[sequence_length:].reset_index(drop=True) if model_config.get('use_transformer', False) else df
            results = engine.run(df_backtest, predictions, confidences)
            st.success("[OK] 回測完成")
        
        metrics = results['metrics']
        
        if 'error' in metrics:
            st.error(f"[警告] {metrics['error']}")
            st.warning("模型不產生交易信號，建議:")
            st.info("1. 降低信心度閾值至 0.2-0.3\n2. 檢查模型評估指標\n3. 重新訓練模型")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("總交易", metrics['total_trades'])
            st.metric("勝率", f"{metrics['win_rate']:.1%}")
        with col2:
            st.metric("總報酬", f"{metrics['total_return']:.1%}")
            st.metric("盈虧因子", f"{metrics['profit_factor']:.2f}")
        with col3:
            st.metric("Sharpe比率", f"{metrics['sharpe_ratio']:.2f}")
            st.metric("最大回撤", f"{metrics['max_drawdown']:.1%}")
        with col4:
            st.metric("平均盈利", f"{metrics['avg_win']:.2f}")
            st.metric("平均虧損", f"{metrics['avg_loss']:.2f}")
        
        st.markdown("---")
        st.subheader("權益曲線")
        equity_df = pd.DataFrame(results['equity_curve'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df['timestamp'],
            y=equity_df['equity'],
            mode='lines',
            name='權益',
            line=dict(color='blue', width=2)
        ))
        fig.update_layout(
            height=400,
            xaxis_title='時間',
            yaxis_title='權益 (USDT)',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if len(results['trades']) > 0:
            st.markdown("---")
            st.subheader("交易明細")
            trades_df = pd.DataFrame(results['trades'])
            trades_df['pnl_pct'] = trades_df['pnl_pct'] * 100
            st.dataframe(trades_df.head(50), use_container_width=True)
        
    except Exception as e:
        st.error(f"[失敗] {str(e)}")
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
    st.header("V2 狀態")
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
