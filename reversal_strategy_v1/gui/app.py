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

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 添加V2路徑
v2_root = project_root.parent / 'high_frequency_strategy_v2'
sys.path.insert(0, str(v2_root))

# V1引用
from core.signal_detector import SignalDetector
from core.feature_engineer import FeatureEngineer
from core.ml_predictor import MLPredictor
from core.risk_manager import RiskManager
from backtest.engine import BacktestEngine
from data.hf_loader import HFDataLoader

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
        
        strategy_version = st.radio(
            "選擇策略版本",
            [
                "V1 - 訂單流反轉策略",
                "V2 - 高頻Transformer策略"
            ],
            index=0,
            help="V1適合中頻交易(50-80筆/月)，V2適合高頻交易(140-150筆/月)"
        )
        
        st.markdown("---")
        
        if "V1" in strategy_version:
            st.subheader("🔵 V1 特點")
            st.caption("訂單流不平衡 + XGBoost")
            st.info(
                "訂單流不平衡檢測\n"
                "流動性掃蕩識別\n"
                "市場微觀結構分析\n"
                "XGBoost機器學習"
            )
            col1, col2 = st.columns(2)
            with col1:
                st.metric("月交易目標", "50-80筆")
            with col2:
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
            col1, col2 = st.columns(2)
            with col1:
                st.metric("月交易目標", "140-150筆")
            with col2:
                st.metric("月報酬目標", "50%+")
        
        st.markdown("---")
        
        if st.button("📊 查看V1 vs V2對比", use_container_width=True):
            st.session_state['show_comparison'] = True
    
    # 顯示對比彈窗
    if st.session_state.get('show_comparison', False):
        show_strategy_comparison()
    
    # 根據選擇渲染不同界面
    if "V1" in strategy_version:
        render_v1_interface()
    else:
        render_v2_interface()

def show_strategy_comparison():
    """顯示策略對比"""
    with st.expander("🏆 V1 vs V2 策略對比", expanded=True):
        comparison_df = pd.DataFrame({
            '項目': [
                '模型架構',
                '時序學習',
                '集成學習',
                '信號過濾',
                '風險管理',
                '市場自適應',
                '月交易量',
                '月報酬目標',
                '訓練時間',
                'GPU需求'
            ],
            'V1 反轉策略': [
                'XGBoost',
                '✘',
                '✘',
                '單層',
                '固定',
                '✘',
                '50-80筆',
                '30-50%',
                '5-10分鐘',
                '不需要'
            ],
            'V2 高頻策略': [
                'Transformer + LightGBM',
                '✓ (100根K線)',
                '✓ (加權集成)',
                '三層過濾',
                '動態調整',
                '✓',
                '140-150筆',
                '50%+',
                '10-20分鐘',
                '建議使用'
            ]
        })
        
        st.table(comparison_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **選擇V1如果:**
            - 初學者
            - 無GPU設備
            - 中低頻交易偏好
            - 追求穩定
            """)
        
        with col2:
            st.info("""
            **選擇V2如果:**
            - 有深度學習經驗
            - 有GPU設備
            - 高頻自動化交易
            - 追求高報酬
            """)
        
        if st.button("關閉對比", use_container_width=True):
            st.session_state['show_comparison'] = False
            st.rerun()

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
        render_paper_trading_tab()
    with tab4:
        render_live_trading_tab()
    with tab5:
        render_v1_analytics()

def render_v2_interface():
    """V2策略界面"""
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 V2模型訓練",
        "📈 V2回測分析",
        "🎮 V2模擬交易",
        "⚙️ 系統狀態"
    ])
    
    with tab1:
        render_v2_training()
    with tab2:
        render_v2_backtest()
    with tab3:
        render_v2_paper_trading()
    with tab4:
        render_v2_status()

def render_v1_training():
    """V1模型訓練 - 在GUI中直接執行"""
    st.header("🎯 V1 模型訓練")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("訓練參數")
        
        symbol = st.selectbox(
            "交易對",
            ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "AVAXUSDT"],
            key="v1_train_symbol"
        )
        
        timeframe = st.selectbox(
            "時間框架",
            ["15m", "1h", "4h"],
            index=0,
            key="v1_train_timeframe"
        )
        
        st.markdown("---")
        
        st.subheader("信號檢測")
        lookback = st.slider("回溯週期", 10, 50, 20)
        imbalance_threshold = st.slider("OFI閾值", 0.5, 0.8, 0.6, 0.05)
        
        st.markdown("---")
        
        st.subheader("標籤生成")
        forward_window = st.slider("前瞻窗口", 8, 20, 12)
        profit_threshold = st.slider("盈利目標 %", 0.5, 3.0, 1.0, 0.1) / 100
        stop_loss = st.slider("止損 %", 0.3, 2.0, 0.5, 0.1) / 100
        
        st.markdown("---")
        
        st.subheader("機器學習模型")
        n_estimators = st.slider("樹數量", 100, 500, 200, 50)
        max_depth = st.slider("最大深度", 3, 10, 5)
        test_size = st.slider("驗證集比例", 0.1, 0.3, 0.2, 0.05)
        oos_size = st.slider("OOS測試集比例", 0.05, 0.2, 0.1, 0.05)
        
        st.markdown("---")
        
        if st.button("開始V1訓練", type="primary", use_container_width=True):
            st.session_state['v1_training_params'] = {
                'symbol': symbol,
                'timeframe': timeframe,
                'lookback': lookback,
                'imbalance_threshold': imbalance_threshold,
                'forward_window': forward_window,
                'profit_threshold': profit_threshold,
                'stop_loss': stop_loss,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'test_size': test_size,
                'oos_size': oos_size
            }
            st.session_state['v1_training_started'] = True
    
    with col2:
        st.subheader("訓練過程")
        
        if st.session_state.get('v1_training_started', False):
            train_v1_model_in_gui(st.session_state['v1_training_params'])
            st.session_state['v1_training_started'] = False
        else:
            st.info("""
            ### V1訓練流程
            
            1. **加載數據**: 從HuggingFace加載歷史K線
            2. **信號檢測**: 訂單流不平衡 + 流動性掃蕩
            3. **特徵工程**: 提取50+個技術指標
            4. **標籤生成**: 前瞻窗口盈虧標籤
            5. **模型訓練**: XGBoost機器學習
            6. **模型驗證**: 訓練集/驗證集/OOS測試
            
            **預計時間**: 5-10分鐘
            
            請配置左側參數後點擊開始訓練
            """)

def train_v1_model_in_gui(params):
    """在GUI中直接執行V1訓練"""
    try:
        config = {
            'signal_detection': {
                'lookback': params['lookback'],
                'imbalance_threshold': params['imbalance_threshold'],
                'liquidity_strength': 1.5,
                'microstructure_window': 10
            },
            'feature_engineering': {
                'lookback_periods': [5, 10, 20, 30],
                'use_price_features': True,
                'use_volume_features': True,
                'use_microstructure': True
            },
            'ml_model': {
                'n_estimators': params['n_estimators'],
                'max_depth': params['max_depth'],
                'learning_rate': 0.05
            },
            'label_generation': {
                'forward_window': params['forward_window'],
                'profit_threshold': params['profit_threshold'],
                'stop_loss': params['stop_loss']
            }
        }
        
        with st.spinner('步驟 1/5: 加載歷史數據...'):
            loader = HFDataLoader()
            df = loader.load_klines(params['symbol'], params['timeframe'])
            st.success(f"✓ 加載完成: {len(df)} 筆數據")
        
        with st.spinner('步驟 2/5: 檢測反轉信號...'):
            signal_detector = SignalDetector(config['signal_detection'])
            df = signal_detector.detect_signals(df)
            long_signals = df['signal_long'].sum()
            short_signals = df['signal_short'].sum()
            st.success(f"✓ 做多信號: {long_signals} | 做空信號: {short_signals}")
        
        with st.spinner('步驟 3/5: 生成ML特徵...'):
            feature_engineer = FeatureEngineer(config['feature_engineering'])
            df = feature_engineer.create_features(df)
            df = feature_engineer.create_labels(
                df,
                forward_window=config['label_generation']['forward_window'],
                profit_threshold=config['label_generation']['profit_threshold'],
                stop_loss=config['label_generation']['stop_loss']
            )
            feature_cols = feature_engineer.get_feature_names()
            st.success(f"✓ 特徵數量: {len(feature_cols)}")
        
        with st.spinner('步驟 4/5: 訓練機器學習模型...'):
            ml_predictor = MLPredictor(config['ml_model'])
            train_results = ml_predictor.train(
                df,
                feature_cols,
                test_size=params['test_size'],
                oos_size=params['oos_size']
            )
            st.success("✓ 模型訓練完成")
        
        with st.spinner('步驟 5/5: 保存模型...'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"{params['symbol']}_{params['timeframe']}_v1_{timestamp}"
            model_dir = Path('models') / model_name
            ml_predictor.save(model_dir)
            
            model_config = {
                'symbol': params['symbol'],
                'timeframe': params['timeframe'],
                'training_date': timestamp,
                'model_version': 'v1',
                'data_samples': len(df),
                'long_signals': int(long_signals),
                'short_signals': int(short_signals),
                'config': config
            }
            
            with open(model_dir / 'model_config.json', 'w') as f:
                json.dump(model_config, f, indent=2)
            
            st.success(f"✅ V1訓練完成: {model_name}")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("訓練樣本數", train_results['train_samples'])
        with col_b:
            st.metric("驗證樣本數", train_results['val_samples'])
        with col_c:
            st.metric("OOS樣本數", train_results['oos_samples'])
        
        st.markdown("---")
        st.subheader("信號分布")
        label_dist = pd.DataFrame({
            '類別': ['做多', '做空', '中性'],
            '數量': [(df['label']==1).sum(), (df['label']==-1).sum(), (df['label']==0).sum()]
        })
        st.bar_chart(label_dist.set_index('類別'))
        
        st.session_state['latest_v1_model'] = model_name
        
    except Exception as e:
        st.error(f"❌ 訓練失敗: {str(e)}")
        import traceback
        with st.expander("錯誤詳情"):
            st.code(traceback.format_exc())

def render_v2_training():
    """V2模型訓練 - 在GUI中直接執行"""
    st.header("🧠 V2 Transformer模型訓練")
    
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
            st.caption("✓ 集成模式: Transformer + LightGBM")
        
        st.markdown("---")
        
        if st.button("開始V2訓練", type="primary", use_container_width=True):
            st.session_state['v2_training_params'] = {
                'symbol': symbol,
                'timeframe': timeframe,
                'sequence_length': sequence_length,
                'use_transformer': use_transformer,
                'use_lgb': use_lgb
            }
            st.session_state['v2_training_started'] = True
    
    with col2:
        st.subheader("訓練過程")
        
        if st.session_state.get('v2_training_started', False):
            train_v2_model_in_gui(st.session_state['v2_training_params'])
            st.session_state['v2_training_started'] = False
        else:
            st.info("""
            ### V2 Transformer訓練流程
            
            1. **加載數據**: HuggingFace歷史K線
            2. **特徵提取**: 
               - 50+技術指標
               - 市場微觀結構
               - 時間特徵
               - 波動率狀態
            3. **時序準備**: 創建100根K線序列
            4. **模型訓練**:
               - Transformer (4層, 8頭注意力)
               - LightGBM (快速決策)
               - 集成學習
            5. **模型驗證**: 訓練/驗證/測試集
            
            **預計時間**: 10-20分鐘 (GPU加速)
            
            請配置左側參數後點擊開始訓練
            """)

def train_v2_model_in_gui(params):
    """在GUI中直接執行V2訓練"""
    try:
        # 使用絕對路徑引用V2模組
        v2_feature_path = v2_root / 'core' / 'feature_engineer.py'
        v2_ensemble_path = v2_root / 'core' / 'ensemble_predictor.py'
        v2_loader_path = v2_root / 'data' / 'hf_loader.py'
        
        # 加載V2模組
        v2_feature_module = load_v2_module(v2_feature_path, 'v2_feature_engineer')
        v2_ensemble_module = load_v2_module(v2_ensemble_path, 'v2_ensemble_predictor')
        v2_loader_module = load_v2_module(v2_loader_path, 'v2_hf_loader')
        
        V2FeatureEngineer = v2_feature_module.FeatureEngineer
        EnsemblePredictor = v2_ensemble_module.EnsemblePredictor
        V2HFDataLoader = v2_loader_module.HFDataLoader
        
        with st.spinner('步驟 1/6: 加載歷史數據...'):
            loader = V2HFDataLoader()
            df = loader.load_klines(params['symbol'], params['timeframe'])
            st.success(f"✓ 加載完成: {len(df)} 筆數據")
        
        with st.spinner('步驟 2/6: 提取特徵...'):
            feature_config = {
                'sequence_length': params['sequence_length'],
                'use_orderbook_features': False,
                'use_microstructure': True,
                'use_momentum': True,
                'lookback_periods': [5, 10, 20, 50]
            }
            
            feature_engineer = V2FeatureEngineer(feature_config)
            df = feature_engineer.create_features(df)
            st.success(f"✓ 特徵提取完成: {len(df)} 筆")
        
        with st.spinner('步驟 3/6: 生成交易標籤...'):
            df = create_v2_labels(df)
            long_signals = (df['label'] == 1).sum()
            short_signals = (df['label'] == -1).sum()
            neutral = (df['label'] == 0).sum()
            st.success(f"✓ 做多: {long_signals} | 做空: {short_signals} | 中性: {neutral}")
        
        with st.spinner('步驟 4/6: 準備訓練數據...'):
            exclude_cols = ['timestamp', 'label', 'bb_upper', 'bb_lower']
            feature_cols = [col for col in df.columns 
                           if col not in exclude_cols and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
            
            train_size = int(len(df) * 0.7)
            val_size = int(len(df) * 0.15)
            
            df_train = df.iloc[:train_size]
            df_val = df.iloc[train_size:train_size+val_size]
            
            X_train = df_train[feature_cols].values
            y_train = df_train['label'].values
            X_val = df_val[feature_cols].values
            y_val = df_val['label'].values
            
            X_train_seq = feature_engineer.prepare_sequences(df_train, feature_cols)
            X_val_seq = feature_engineer.prepare_sequences(df_val, feature_cols)
            y_train_seq = df_train['label'].values[params['sequence_length']:]
            y_val_seq = df_val['label'].values[params['sequence_length']:]
            
            st.success(f"✓ 訓練集: {len(X_train)} | 驗證集: {len(X_val)}")
        
        with st.spinner('步驟 5/6: 訓練集成模型 (需要10-20分鐘)...'):
            ensemble_config = {
                'use_transformer': params['use_transformer'],
                'use_lgb': params['use_lgb'],
                'ensemble_method': 'weighted_avg',
                'weights': {
                    'transformer': 0.5,
                    'lgb': 0.5
                }
            }
            
            predictor = EnsemblePredictor(ensemble_config)
            results = predictor.train(
                X_train, y_train,
                X_val, y_val,
                X_train_seq, X_val_seq
            )
            st.success("✓ 模型訓練完成")
        
        with st.spinner('步驟 6/6: 保存模型...'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"{params['symbol']}_{params['timeframe']}_v2_{timestamp}"
            model_dir = Path('models') / model_name
            
            predictor.save(model_dir)
            
            model_config = {
                'symbol': params['symbol'],
                'timeframe': params['timeframe'],
                'model_version': 'v2',
                'training_date': timestamp,
                'data_samples': len(df),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'long_signals': int(long_signals),
                'short_signals': int(short_signals),
                'feature_count': len(feature_cols),
                'sequence_length': params['sequence_length'],
                'feature_config': feature_config,
                'ensemble_config': ensemble_config
            }
            
            with open(model_dir / 'model_config.json', 'w') as f:
                json.dump(model_config, f, indent=2)
            
            with open(model_dir / 'feature_names.txt', 'w') as f:
                f.write('\n'.join(feature_cols))
            
            st.success(f"✅ V2訓練完成: {model_name}")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("訓練樣本", len(X_train))
        with col_b:
            st.metric("驗證樣本", len(X_val))
        with col_c:
            st.metric("特徵數量", len(feature_cols))
        
        st.session_state['latest_v2_model'] = model_name
        
    except Exception as e:
        st.error(f"❌ V2訓練失敗: {str(e)}")
        import traceback
        with st.expander("錯誤詳情"):
            st.code(traceback.format_exc())

def create_v2_labels(df: pd.DataFrame, 
                    forward_window: int = 8,
                    profit_threshold: float = 0.004,
                    stop_loss: float = 0.003) -> pd.DataFrame:
    """生成V2交易標籤"""
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

# 以下是V1回測和其他功能 (保持不變)
def render_v1_backtest():
    st.header("📈 V1 回測分析")
    st.info("請先完成V1模型訓練")

def render_v2_backtest():
    st.header("📈 V2 回測分析")
    st.info("V2回測功能開發中")

def render_paper_trading_tab():
    st.header("🎮 模擬交易")
    st.info("模擬交易功能開發中")

def render_live_trading_tab():
    st.header("💰 實盤交易")
    st.warning("實盤交易需先完成回測驗證")

def render_v2_paper_trading():
    st.header("🎮 V2 模擬交易")
    st.info("V2模擬交易功能開發中")

def render_v1_analytics():
    st.header("📊 V1 績效分析")
    st.info("請先運行V1回測")

def render_v2_status():
    st.header("⚙️ V2 系統狀態")
    
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

if __name__ == "__main__":
    main()
