"""
Unified Trading System - V1 & V2 Integrated GUI
統一交易系統 - V1和V2整合界面
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

# 添加V2路徑
v2_root = project_root.parent / 'high_frequency_strategy_v2'
sys.path.insert(0, str(v2_root))

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
    """V1模型訓練頁面"""
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
            params = st.session_state['v1_training_params']
            
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
                st.session_state['v1_training_started'] = False
                
            except Exception as e:
                st.error(f"❌ 訓練失敗: {str(e)}")
                import traceback
                with st.expander("錯誤詳情"):
                    st.code(traceback.format_exc())
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
            """)

def render_v2_training():
    """V2模型訓練頁面"""
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
            with st.spinner("訓練V2模型 (需時10-20分鐘)..."):
                try:
                    v2_train_script = project_root.parent / 'high_frequency_strategy_v2' / 'train_model.py'
                    result = subprocess.run([
                        sys.executable,
                        str(v2_train_script),
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
        ### Transformer深度學習訓練
        
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
        
        st.markdown("---")
        st.subheader("🎯 V2性能目標")
        metrics_df = pd.DataFrame({
            '指標': ['月交易數', '月報酬率', '勝率', '最大回撤', 'Sharpe'],
            '目標值': ['140-150', '50%+', '60%+', '<20%', '>2.0']
        })
        st.table(metrics_df)

def render_v1_backtest():
    """V1回測分析 - 保持原有完整功能"""
    st.header("📈 V1 回測分析")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("回測配置")
        
        models_dir = Path('models')
        if models_dir.exists():
            v1_models = [d.name for d in models_dir.iterdir() 
                        if d.is_dir() and '_v1_' in d.name]
        else:
            v1_models = []
        
        if not v1_models:
            v1_models = ["無可用V1模型"]
        
        model_version = st.selectbox(
            "選擇V1模型",
            v1_models,
            key="v1_backtest_model"
        )
        
        st.markdown("---")
        
        st.subheader("回測參數")
        
        data_source = st.radio(
            "數據源",
            ["Binance API (最新)", "HuggingFace (歷史)"],
            index=0
        )
        
        if data_source == "Binance API (最新)":
            backtest_days = st.slider("回測天數", 7, 90, 30)
        else:
            backtest_days = st.slider("回測天數", 30, 365, 90)
        
        initial_capital = st.number_input("初始資金 (USDT)", 10, 10000, 10)
        leverage = st.slider("槓桿倍數", 1, 20, 3)
        
        st.markdown("---")
        
        st.subheader("交易參數")
        min_signal_strength = st.slider("最小信號強度", 1, 5, 1)
        min_confidence = st.slider("最小模型置信度", 0.5, 0.95, 0.55, 0.05)
        
        st.markdown("---")
        
        st.subheader("倉位管理")
        position_size_pct = st.slider("倉位大小 (%當前資金)", 10, 100, 95, 5) / 100
        
        st.markdown("---")
        
        st.subheader("風險管理")
        sltp_mode = st.radio(
            "止損止盈模式",
            ["固定百分比 (推薦)", "ATR倍數"],
            index=0
        )
        
        if sltp_mode == "固定百分比 (推薦)":
            fixed_sl_pct = st.slider("固定止損 %", 0.3, 3.0, 0.5, 0.1) / 100
            fixed_tp_pct = st.slider("固定止盈 %", 0.5, 5.0, 2.0, 0.1) / 100
            atr_multiplier_sl = None
            atr_multiplier_tp = None
        else:
            atr_multiplier_sl = st.slider("ATR止損倍數", 0.5, 3.0, 1.5, 0.1)
            atr_multiplier_tp = st.slider("ATR止盈倍數", 1.0, 5.0, 3.0, 0.5)
            fixed_sl_pct = None
            fixed_tp_pct = None
        
        st.markdown("---")
        
        st.subheader("交易費用")
        maker_fee = st.number_input("Maker手續費 %", 0.01, 0.1, 0.02, 0.01) / 100
        taker_fee = st.number_input("Taker手續費 %", 0.01, 0.1, 0.04, 0.01) / 100
        
        st.markdown("---")
        
        if st.button("運行V1回測", type="primary", use_container_width=True, 
                    disabled=(model_version=="無可用V1模型")):
            st.session_state['v1_backtest_params'] = {
                'model': model_version,
                'data_source': 'binance' if 'Binance' in data_source else 'hf',
                'days': backtest_days,
                'capital': initial_capital,
                'leverage': leverage,
                'min_signal_strength': min_signal_strength,
                'min_confidence': min_confidence,
                'maker_fee': maker_fee,
                'taker_fee': taker_fee,
                'position_size_pct': position_size_pct,
                'sltp_mode': sltp_mode,
                'atr_multiplier_sl': atr_multiplier_sl,
                'atr_multiplier_tp': atr_multiplier_tp,
                'fixed_sl_pct': fixed_sl_pct,
                'fixed_tp_pct': fixed_tp_pct
            }
            st.session_state['v1_backtest_started'] = True
    
    with col2:
        st.subheader("回測結果")
        
        if st.session_state.get('v1_backtest_started', False):
            params = st.session_state['v1_backtest_params']
            
            try:
                with st.spinner('加載訓練模型...'):
                    model_dir = Path('models') / params['model']
                    with open(model_dir / 'model_config.json', 'r') as f:
                        model_config = json.load(f)
                    
                    symbol = model_config['symbol']
                    timeframe = model_config['timeframe']
                    config = model_config['config']
                
                with st.spinner('加載歷史數據...'):
                    if params['data_source'] == 'binance':
                        backtest_engine = BacktestEngine({
                            'initial_capital': params['capital'],
                            'leverage': params['leverage'],
                            'maker_fee': params['maker_fee'],
                            'taker_fee': params['taker_fee'],
                            'slippage': 0.0001,
                            'position_size_pct': params['position_size_pct']
                        })
                        df = backtest_engine.fetch_latest_data(symbol, timeframe, days=params['days'])
                    else:
                        loader = HFDataLoader()
                        df = loader.load_klines(symbol, timeframe)
                        bars_per_day = {'15m': 96, '1h': 24, '4h': 6}.get(timeframe, 96)
                        df = df.tail(params['days'] * bars_per_day)
                
                with st.spinner('檢測交易信號...'):
                    signal_detector = SignalDetector(config['signal_detection'])
                    df = signal_detector.detect_signals(df)
                
                with st.spinner('生成ML特徵...'):
                    feature_engineer = FeatureEngineer(config['feature_engineering'])
                    df = feature_engineer.create_features(df)
                
                with st.spinner('ML信號驗證...'):
                    ml_predictor = MLPredictor(config['ml_model'])
                    ml_predictor.load(model_dir)
                    df = ml_predictor.predict(df)
                
                with st.spinner('計算風險參數...'):
                    if params['sltp_mode'] == "固定百分比 (推薦)":
                        for i in range(len(df)):
                            current_price = df.iloc[i]['close']
                            if df.iloc[i]['signal_long'] == 1:
                                df.loc[df.index[i], 'stop_loss'] = current_price * (1 - params['fixed_sl_pct'])
                                df.loc[df.index[i], 'take_profit'] = current_price * (1 + params['fixed_tp_pct'])
                            elif df.iloc[i]['signal_short'] == 1:
                                df.loc[df.index[i], 'stop_loss'] = current_price * (1 + params['fixed_sl_pct'])
                                df.loc[df.index[i], 'take_profit'] = current_price * (1 - params['fixed_tp_pct'])
                    else:
                        risk_manager = RiskManager({
                            'initial_capital': params['capital'],
                            'max_risk_per_trade': 0.02,
                            'max_leverage': 10,
                            'default_leverage': params['leverage'],
                            'atr_multiplier_sl': params['atr_multiplier_sl'],
                            'atr_multiplier_tp': params['atr_multiplier_tp']
                        })
                        
                        for i in range(len(df)):
                            if df.iloc[i]['signal_long'] == 1:
                                sltp = risk_manager.calculate_stop_loss_take_profit(df.iloc[:i+1], 'LONG')
                                df.loc[df.index[i], 'stop_loss'] = sltp['stop_loss']
                                df.loc[df.index[i], 'take_profit'] = sltp['take_profit']
                            elif df.iloc[i]['signal_short'] == 1:
                                sltp = risk_manager.calculate_stop_loss_take_profit(df.iloc[:i+1], 'SHORT')
                                df.loc[df.index[i], 'stop_loss'] = sltp['stop_loss']
                                df.loc[df.index[i], 'take_profit'] = sltp['take_profit']
                
                with st.spinner('執行回測...'):
                    backtest_engine = BacktestEngine({
                        'initial_capital': params['capital'],
                        'leverage': params['leverage'],
                        'maker_fee': params['maker_fee'],
                        'taker_fee': params['taker_fee'],
                        'slippage': 0.0001,
                        'position_size_pct': params['position_size_pct']
                    })
                    
                    results = backtest_engine.run_backtest(
                        df, 
                        params['min_signal_strength'], 
                        params['min_confidence']
                    )
                
                if 'error' not in results:
                    st.success("✅ V1回測完成")
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("總收益率", f"{results['total_return']:.2f}%")
                    with col_b:
                        st.metric("勝率", f"{results['win_rate']:.2f}%")
                    with col_c:
                        st.metric("交易次數", results['total_trades'])
                    with col_d:
                        st.metric("最大回撤", f"{results['max_drawdown']:.2f}%")
                    
                    col_e, col_f, col_g, col_h = st.columns(4)
                    with col_e:
                        st.metric("最終資金", f"{results['final_capital']:.2f} USDT")
                    with col_f:
                        st.metric("總盈虧", f"{results['total_pnl']:.2f} USDT")
                    with col_g:
                        st.metric("盈虧因子", f"{results['profit_factor']:.2f}")
                    with col_h:
                        st.metric("Sharpe比率", f"{results['sharpe_ratio']:.2f}")
                    
                    st.markdown("---")
                    
                    st.subheader("權益曲線")
                    equity_df = pd.DataFrame(results['equity_curve'])
                    
                    fig = make_subplots(
                        rows=2, cols=1,
                        row_heights=[0.7, 0.3],
                        subplot_titles=('權益曲線', '水下回撤'),
                        vertical_spacing=0.1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=equity_df['time'], y=equity_df['equity'], 
                                  name='權益', line=dict(color='blue', width=2)),
                        row=1, col=1
                    )
                    
                    equity_df['cummax'] = equity_df['equity'].cummax()
                    equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
                    
                    fig.add_trace(
                        go.Scatter(x=equity_df['time'], y=equity_df['drawdown'],
                                  name='回撤', fill='tozeroy', line=dict(color='red', width=1)),
                        row=2, col=1
                    )
                    
                    fig.update_xaxes(title_text="時間", row=2, col=1)
                    fig.update_yaxes(title_text="權益 (USDT)", row=1, col=1)
                    fig.update_yaxes(title_text="回撤 (%)", row=2, col=1)
                    
                    fig.update_layout(height=600, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
                    
                    st.subheader("交易明細")
                    trades_df = pd.DataFrame(results['trades'])
                    st.dataframe(
                        trades_df[['entry_time', 'exit_time', 'type', 'entry_price', 
                                  'exit_price', 'pnl', 'return_pct', 'reason']],
                        use_container_width=True
                    )
                    
                    st.session_state['v1_backtest_results'] = results
                    st.session_state['v1_backtest_started'] = False
                else:
                    st.error(f"❌ 回測失敗: {results['error']}")
                    st.session_state['v1_backtest_started'] = False
                    
            except Exception as e:
                st.error(f"❌ 回測失敗: {str(e)}")
                import traceback
                with st.expander("錯誤詳情"):
                    st.code(traceback.format_exc())
                st.session_state['v1_backtest_started'] = False
        else:
            st.info("請配置回測參數後點擊運行回測")

def render_v2_backtest():
    """V2回測分析"""
    st.header("📈 V2 回測分析")
    st.info("V2回測功能開發中,先完成模型訓練")
    st.caption("預計功能: 高頻交易回測、三層信號過濾、動態風險管理")

def render_paper_trading_tab():
    """模擬交易頁面"""
    st.header("🎮 模擬交易")
    st.info(
        "模擬交易功能開發中\n\n"
        "**功能規劃:**\n"
        "- 使用Demo帳戶\n"
        "- 實時信號監控\n"
        "- 自動下單執行"
    )

def render_live_trading_tab():
    """實盤交易頁面"""
    st.header("💰 實盤交易")
    st.warning("在回測結果驗證通過前,實盤交易功能已禁用")

def render_v2_paper_trading():
    """V2模擬交易"""
    st.header("🎮 V2 模擬交易")
    st.info("V2模擬交易功能開發中")

def render_v1_analytics():
    """V1績效分析"""
    st.header("📊 V1 績效分析")
    
    if 'v1_backtest_results' in st.session_state:
        results = st.session_state['v1_backtest_results']
        
        st.subheader("模型表現")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("總交易次數", results['total_trades'])
            st.metric("獲勝交易", results['winning_trades'])
            st.metric("虧損交易", results['losing_trades'])
        
        with col2:
            st.metric("勝率", f"{results['win_rate']:.2f}%")
            st.metric("平均盈利", f"{results['avg_win']:.2f} USDT")
            st.metric("平均虧損", f"{results['avg_loss']:.2f} USDT")
        
        with col3:
            st.metric("盈虧因子", f"{results['profit_factor']:.2f}")
            st.metric("Sharpe比率", f"{results['sharpe_ratio']:.2f}")
            st.metric("最大回撤", f"{results['max_drawdown']:.2f}%")
        
        st.markdown("---")
        
        st.subheader("交易分布")
        trades_df = pd.DataFrame(results['trades'])
        
        if len(trades_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=trades_df['pnl'], nbinsx=20, name='盈虧分布'))
                fig.update_layout(title='盈虧分布', xaxis_title='盈虧 (USDT)', yaxis_title='頻率')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                long_pnl = trades_df[trades_df['type']=='LONG']['pnl'].sum()
                short_pnl = trades_df[trades_df['type']=='SHORT']['pnl'].sum()
                
                fig = go.Figure(data=[
                    go.Bar(x=['做多', '做空'], y=[long_pnl, short_pnl])
                ])
                fig.update_layout(title='做多 vs 做空總盈虧', yaxis_title='盈虧 (USDT)')
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("請先運行V1回測以查看分析結果")

def render_v2_status():
    """V2系統狀態"""
    st.header("⚙️ V2 系統狀態")
    
    # 模型統計
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
        'XGBoost': '2.0+ (V1)',
        'LightGBM': '4.0+ (V2)',
        'PyTorch': '2.0+ (V2必需)',
        'CUDA': '12.1+ (可選)',
        'RAM': '8GB+',
        'GPU': '4GB+ VRAM (V2建議)'
    }
    
    req_df = pd.DataFrame(list(requirements.items()), columns=['組件', '版本/規格'])
    st.table(req_df)
    
    st.markdown("---")
    
    # 快速開始
    st.subheader("🚀 快速開始")
    
    with st.expander("📖 安裝指南"):
        st.code("""
# 1. 安裝V1依賴
cd reversal_strategy_v1
pip install -r requirements.txt

# 2. 安裝V2依賴
cd ../high_frequency_strategy_v2
pip install -r requirements.txt

# 3. 安裝TA-Lib
# Windows: 下載.whl從 https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# Linux: sudo apt-get install ta-lib
# Mac: brew install ta-lib
        """, language="bash")

if __name__ == "__main__":
    main()
