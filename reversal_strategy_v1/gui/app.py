"""
Reversal Strategy V1 - Streamlit GUI
反轉策略交易系統主界面
"""
import streamlit as st
import sys
from pathlib import Path
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.signal_detector import SignalDetector
from core.feature_engineer import FeatureEngineer
from core.ml_predictor import MLPredictor
from core.risk_manager import RiskManager
from backtest.engine import BacktestEngine
from data.hf_loader import HFDataLoader

st.set_page_config(
    page_title="反轉交易系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("加密貨幣反轉交易系統")
    
    with st.sidebar:
        st.header("系統配置")
        
        version = st.selectbox(
            "策略版本",
            ["V1 - 訂單流反轉策略", "V2 - 即將推出", "V3 - 即將推出"],
            index=0
        )
        
        st.markdown("---")
        st.subheader("當前版本: V1")
        st.caption("訂單流不平衡與流動性區域策略")
        
        st.markdown("---")
        st.info(
            "**V1 策略特點:**\n\n"
            "訂單流不平衡檢測\n"
            "流動性掃蕩識別\n"
            "市場微觀結構分析\n"
            "機器學習信號驗證"
        )
    
    if "V1" in version:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "模型訓練",
            "回測分析",
            "模擬交易", 
            "實盤交易",
            "績效分析"
        ])
        
        with tab1:
            render_training_tab()
        
        with tab2:
            render_backtest_tab()
        
        with tab3:
            render_paper_trading_tab()
        
        with tab4:
            render_live_trading_tab()
        
        with tab5:
            render_analytics_tab()

def render_training_tab():
    """模型訓練頁面"""
    st.header("模型訓練")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("訓練參數")
        
        symbol = st.selectbox(
            "交易對",
            ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "AVAXUSDT"],
            key="train_symbol"
        )
        
        timeframe = st.selectbox(
            "時間框架",
            ["15m", "1h", "4h"],
            index=0,
            key="train_timeframe"
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
        
        if st.button("開始訓練", type="primary", use_container_width=True):
            st.session_state['training_params'] = {
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
            st.session_state['training_started'] = True
    
    with col2:
        st.subheader("訓練過程")
        
        if st.session_state.get('training_started', False):
            params = st.session_state['training_params']
            
            try:
                # 構建配置
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
                
                # 1. 加載數據
                with st.spinner('步驟 1/5: 加載歷史數據...'):
                    loader = HFDataLoader()
                    df = loader.load_klines(params['symbol'], params['timeframe'])
                    st.success(f"加載完成: {len(df)} 筆數據")
                
                # 2. 信號檢測
                with st.spinner('步驟 2/5: 檢測反轉信號...'):
                    signal_detector = SignalDetector(config['signal_detection'])
                    df = signal_detector.detect_signals(df)
                    long_signals = df['signal_long'].sum()
                    short_signals = df['signal_short'].sum()
                    st.success(f"做多信號: {long_signals} | 做空信號: {short_signals}")
                
                # 3. 特徵工程
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
                    st.success(f"特徵數量: {len(feature_cols)}")
                
                # 4. 訓練模型
                with st.spinner('步驟 4/5: 訓練機器學習模型...'):
                    ml_predictor = MLPredictor(config['ml_model'])
                    train_results = ml_predictor.train(
                        df,
                        feature_cols,
                        test_size=params['test_size'],
                        oos_size=params['oos_size']
                    )
                
                # 5. 保存模型
                with st.spinner('步驟 5/5: 保存模型...'):
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    model_name = f"{params['symbol']}_{params['timeframe']}_v1_{timestamp}"
                    model_dir = Path('models') / model_name
                    ml_predictor.save(model_dir)
                    
                    model_config = {
                        'symbol': params['symbol'],
                        'timeframe': params['timeframe'],
                        'training_date': timestamp,
                        'data_samples': len(df),
                        'long_signals': int(long_signals),
                        'short_signals': int(short_signals),
                        'config': config
                    }
                    
                    with open(model_dir / 'model_config.json', 'w') as f:
                        json.dump(model_config, f, indent=2)
                    
                    st.success(f"訓練完成: {model_name}")
                
                # 顯示結果
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
                
                st.session_state['latest_model'] = model_name
                st.session_state['training_started'] = False
                
            except Exception as e:
                st.error(f"訓練失敗: {str(e)}")
                st.session_state['training_started'] = False
        else:
            st.info("請配置參數後點擊開始訓練")

def render_backtest_tab():
    """回測頁面"""
    st.header("回測分析")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("回測配置")
        
        # 獲取可用模型列表
        models_dir = Path('models')
        if models_dir.exists():
            available_models = [d.name for d in models_dir.iterdir() if d.is_dir()]
        else:
            available_models = []
        
        if not available_models:
            available_models = ["無可用模型"]
        
        model_version = st.selectbox(
            "選擇模型",
            available_models,
            key="backtest_model"
        )
        
        st.markdown("---")
        
        st.subheader("回測參數")
        
        data_source = st.radio(
            "數據源",
            ["Binance API (最新)", "HuggingFace (歷史)"],
            index=0
        )
        
        if data_source == "Binance API (最新)":
            backtest_days = st.slider("回測天數", 7, 60, 30)
        else:
            backtest_days = st.slider("回測天數", 30, 365, 90)
        
        initial_capital = st.number_input("初始資金 (USDT)", 10, 10000, 10)
        leverage = st.slider("槓桿倍數", 1, 20, 3)
        
        st.markdown("---")
        
        st.subheader("交易參數")
        min_signal_strength = st.slider("最小信號強度", 1, 5, 2)
        min_confidence = st.slider("最小模型置信度", 0.5, 0.95, 0.6, 0.05)
        
        maker_fee = st.number_input("Maker手續費 %", 0.01, 0.1, 0.02, 0.01) / 100
        taker_fee = st.number_input("Taker手續費 %", 0.01, 0.1, 0.04, 0.01) / 100
        
        st.markdown("---")
        
        if st.button("運行回測", type="primary", use_container_width=True, disabled=(model_version=="無可用模型")):
            st.session_state['backtest_params'] = {
                'model': model_version,
                'data_source': 'binance' if 'Binance' in data_source else 'hf',
                'days': backtest_days,
                'capital': initial_capital,
                'leverage': leverage,
                'min_signal_strength': min_signal_strength,
                'min_confidence': min_confidence,
                'maker_fee': maker_fee,
                'taker_fee': taker_fee
            }
            st.session_state['backtest_started'] = True
    
    with col2:
        st.subheader("回測結果")
        
        if st.session_state.get('backtest_started', False):
            params = st.session_state['backtest_params']
            
            try:
                # 1. 加載模型
                with st.spinner('加載訓練模型...'):
                    model_dir = Path('models') / params['model']
                    with open(model_dir / 'model_config.json', 'r') as f:
                        model_config = json.load(f)
                    
                    symbol = model_config['symbol']
                    timeframe = model_config['timeframe']
                    config = model_config['config']
                
                # 2. 加載數據
                with st.spinner('加載歷史數據...'):
                    if params['data_source'] == 'binance':
                        backtest_engine = BacktestEngine({
                            'initial_capital': params['capital'],
                            'leverage': params['leverage'],
                            'maker_fee': params['maker_fee'],
                            'taker_fee': params['taker_fee'],
                            'slippage': 0.0001
                        })
                        df = backtest_engine.fetch_latest_data(symbol, timeframe, days=params['days'])
                    else:
                        loader = HFDataLoader()
                        df = loader.load_klines(symbol, timeframe)
                        bars_per_day = {'15m': 96, '1h': 24, '4h': 6}.get(timeframe, 96)
                        df = df.tail(params['days'] * bars_per_day)
                
                # 3. 信號檢測
                with st.spinner('檢測交易信號...'):
                    signal_detector = SignalDetector(config['signal_detection'])
                    df = signal_detector.detect_signals(df)
                
                # 4. 特徵工程
                with st.spinner('生成ML特徵...'):
                    feature_engineer = FeatureEngineer(config['feature_engineering'])
                    df = feature_engineer.create_features(df)
                
                # 5. ML預測
                with st.spinner('ML信號驗證...'):
                    ml_predictor = MLPredictor(config['ml_model'])
                    ml_predictor.load(model_dir)
                    df = ml_predictor.predict(df)
                
                # 6. 計算止損止盈
                with st.spinner('計算風險參數...'):
                    risk_manager = RiskManager(config.get('risk_management', {
                        'initial_capital': params['capital'],
                        'max_risk_per_trade': 0.02,
                        'max_leverage': 10,
                        'default_leverage': params['leverage'],
                        'atr_multiplier_sl': 1.5,
                        'atr_multiplier_tp': 3.0
                    }))
                    
                    for i in range(len(df)):
                        if df.iloc[i]['signal_long'] == 1:
                            sltp = risk_manager.calculate_stop_loss_take_profit(df.iloc[:i+1], 'LONG')
                            df.loc[df.index[i], 'stop_loss'] = sltp['stop_loss']
                            df.loc[df.index[i], 'take_profit'] = sltp['take_profit']
                        elif df.iloc[i]['signal_short'] == 1:
                            sltp = risk_manager.calculate_stop_loss_take_profit(df.iloc[:i+1], 'SHORT')
                            df.loc[df.index[i], 'stop_loss'] = sltp['stop_loss']
                            df.loc[df.index[i], 'take_profit'] = sltp['take_profit']
                
                # 7. 執行回測
                with st.spinner('執行回測...'):
                    backtest_engine = BacktestEngine({
                        'initial_capital': params['capital'],
                        'leverage': params['leverage'],
                        'maker_fee': params['maker_fee'],
                        'taker_fee': params['taker_fee'],
                        'slippage': 0.0001
                    })
                    
                    results = backtest_engine.run_backtest(
                        df, 
                        params['min_signal_strength'], 
                        params['min_confidence']
                    )
                
                if 'error' not in results:
                    st.success("回測完成")
                    
                    # 顯示關鍵指標
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
                    
                    # 權益曲線
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
                    
                    # 交易明細
                    st.subheader("交易明細")
                    trades_df = pd.DataFrame(results['trades'])
                    st.dataframe(
                        trades_df[['entry_time', 'exit_time', 'type', 'entry_price', 
                                  'exit_price', 'pnl', 'return_pct', 'reason']],
                        use_container_width=True
                    )
                    
                    st.session_state['backtest_results'] = results
                    st.session_state['backtest_started'] = False
                else:
                    st.error(f"回測失敗: {results['error']}")
                    st.session_state['backtest_started'] = False
                    
            except Exception as e:
                st.error(f"回測失敗: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.session_state['backtest_started'] = False
        else:
            st.info("請配置回測參數後點擊運行回測")

def render_paper_trading_tab():
    """模擬交易頁面"""
    st.header("模擬交易")
    st.info(
        "模擬交易功能將使用Bybit Demo帳戶實現\n\n"
        "**此功能將允許您:**\n"
        "- 使用模擬資金測試策略\n"
        "- 監控實時表現\n"
        "- 在實盤交易前驗證模型"
    )

def render_live_trading_tab():
    """實盤交易頁面"""
    st.header("實盤交易")
    st.warning("在回測結果驗證通過前,實盤交易功能已禁用")
    
    st.markdown(
        "**啟用實盤交易前:**\n"
        "1. 完成回測並獲得滿意的結果\n"
        "2. 使用模擬交易測試至少7天\n"
        "3. 配置Binance API憑證\n"
        "4. 設置風險管理參數"
    )

def render_analytics_tab():
    """分析頁面"""
    st.header("績效分析")
    
    if 'backtest_results' in st.session_state:
        results = st.session_state['backtest_results']
        
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
                # 盈虧分布
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=trades_df['pnl'], nbinsx=20, name='盈虧分布'))
                fig.update_layout(title='盈虧分布', xaxis_title='盈虧 (USDT)', yaxis_title='頻率')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 做多vs做空表現
                long_pnl = trades_df[trades_df['type']=='LONG']['pnl'].sum()
                short_pnl = trades_df[trades_df['type']=='SHORT']['pnl'].sum()
                
                fig = go.Figure(data=[
                    go.Bar(x=['做多', '做空'], y=[long_pnl, short_pnl])
                ])
                fig.update_layout(title='做多 vs 做空總盈虧', yaxis_title='盈虧 (USDT)')
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("請先運行回測以查看分析結果")

if __name__ == "__main__":
    main()
