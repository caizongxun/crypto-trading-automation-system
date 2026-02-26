#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v10 剝頭皮策略回測 Tab - Streamlit GUI
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta


def render():
    st.header("λv10 剝頭皮策略回測")
    
    with st.expander("策略介紹", expanded=False):
        st.markdown("""
        ### v10 高頻剝頭皮策略
        
        **核心特點:**
        - 時間框架: 15分鐘
        - 交易頻率: 每日 40-50 筆
        - TP/SL: 0.5% / 0.3% (1.67:1 RR)
        - 平均持有: 3-5 根K線 (45-75分鐘)
        
        **優化方案:**
        1. 動態 TP/SL - 根據波動性調整
        2. 信號質量分級 - 高信心加大倉位
        3. 移動止損 - 保護盈利
        4. 時段過濾 - 只在高品質時段交易
        5. 嚴格篩選 - 過濾低質量信號
        """)
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "回測配置",
        "報告概覽",
        "詳細分析", 
        "交易明細",
        "參數優化",
        "批次生成"
    ])
    
    with tab1:
        render_backtest_config_tab()
    
    with tab2:
        render_overview_tab()
    
    with tab3:
        render_analysis_tab()
    
    with tab4:
        render_trades_tab()
    
    with tab5:
        render_optimize_tab()
    
    with tab6:
        render_generate_tab()


def render_backtest_config_tab():
    """ Tab 1: 回測配置 """
    st.subheader("互動式回測配置")
    
    st.info("""
    **說明:** 在這裡配置回測參數並執行,結果會即時顯示在其他分頁。
    """)
    
    # 檢查模型
    models_dir = Path('models_output')
    long_models = sorted(models_dir.glob('scalping_long_*_v10_*.pkl'))
    short_models = sorted(models_dir.glob('scalping_short_*_v10_*.pkl'))
    
    if not long_models or not short_models:
        st.error("未找到 v10 模型")
        st.warning("請先訓練 v10 模型: `python train_v10_high_frequency.py`")
        return
    
    st.success(f"找到模型: Long={len(long_models)}, Short={len(short_models)}")
    
    # ============ 基礎配置 ============
    st.markdown("### 基礎配置")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        backtest_days = st.number_input(
            "回測天數",
            min_value=30,
            max_value=365,
            value=90,
            step=10
        )
        
        symbol = st.selectbox(
            "交易對",
            options=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
            index=0
        )
    
    with col2:
        train_ratio = st.slider(
            "訓練集比例",
            min_value=0.5,
            max_value=0.95,
            value=0.9,
            step=0.05
        )
        
        timeframe = st.selectbox(
            "時間框架",
            options=['15m', '5m', '30m', '1h'],
            index=0
        )
    
    with col3:
        initial_capital = st.number_input(
            "起始資金 (USD)",
            min_value=1.0,
            max_value=1000000.0,
            value=10000.0,
            step=100.0
        )
        
        leverage = st.slider(
            "槓桿倍數",
            min_value=1,
            max_value=20,
            value=10,
            step=1
        )
    
    st.divider()
    
    # ============ 交易參數 ============
    st.markdown("### 交易參數")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        position_size = st.slider(
            "基礎倉位 (%)",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5
        ) / 100
    
    with col2:
        threshold = st.slider(
            "信號閾值",
            min_value=0.5,
            max_value=0.9,
            value=0.55,
            step=0.05,
            help="降低可增加交易數"
        )
    
    with col3:
        tp_pct = st.slider(
            "止盈 TP (%)",
            min_value=0.1,
            max_value=2.0,
            value=0.6,
            step=0.1
        ) / 100  # 轉換為小數
    
    with col4:
        sl_pct = st.slider(
            "止損 SL (%)",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.05
        ) / 100  # 轉換為小數
    
    rr_ratio = tp_pct / sl_pct
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("風險報酬比 (RR)", f"{rr_ratio:.2f}:1")
    with col2:
        if rr_ratio < 1.5:
            st.warning("建議 RR 比至少 1.5:1")
        elif rr_ratio >= 2:
            st.success("優秀的 RR 比!")
    
    col1, col2 = st.columns(2)
    with col1:
        long_enabled = st.checkbox("啟用 Long", value=True)
    with col2:
        short_enabled = st.checkbox("啟用 Short", value=True)
    
    st.divider()
    
    # ============ 優化方案 ============
    st.markdown("### 優化方案")
    st.caption("勾選以下方案來提升績效")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 方案1: 動態 TP/SL")
        enable_dynamic_tpsl = st.checkbox(
            "啟用動態 TP/SL",
            value=False,
            help="根據市場波動性自動調整止盈止損"
        )
        
        if enable_dynamic_tpsl:
            with st.expander("調整參數"):
                st.caption("低波動")
                col_a, col_b = st.columns(2)
                with col_a:
                    low_vol_tp = st.number_input("TP%", 0.1, 1.0, 0.3, 0.1, key="low_tp") / 100
                with col_b:
                    low_vol_sl = st.number_input("SL%", 0.1, 0.5, 0.2, 0.05, key="low_sl") / 100
                
                st.caption("中波動")
                col_a, col_b = st.columns(2)
                with col_a:
                    mid_vol_tp = st.number_input("TP%", 0.1, 1.0, 0.5, 0.1, key="mid_tp") / 100
                with col_b:
                    mid_vol_sl = st.number_input("SL%", 0.1, 0.5, 0.25, 0.05, key="mid_sl") / 100
                
                st.caption("高波動")
                col_a, col_b = st.columns(2)
                with col_a:
                    high_vol_tp = st.number_input("TP%", 0.1, 2.0, 0.8, 0.1, key="high_tp") / 100
                with col_b:
                    high_vol_sl = st.number_input("SL%", 0.1, 1.0, 0.35, 0.05, key="high_sl") / 100
        else:
            low_vol_tp, low_vol_sl = 0.003, 0.002
            mid_vol_tp, mid_vol_sl = 0.005, 0.0025
            high_vol_tp, high_vol_sl = 0.008, 0.0035
        
        st.markdown("#### 方案2: 信號質量分級")
        enable_quality_sizing = st.checkbox(
            "啟用質量分級倉位",
            value=False,
            help="高信心信號使用更大倉位"
        )
        
        if enable_quality_sizing:
            with st.expander("調整參數"):
                col_a, col_b = st.columns(2)
                with col_a:
                    high_conf_threshold = st.slider("高信心閾值", 0.7, 0.9, 0.75, 0.05)
                    mid_conf_threshold = st.slider("中信心閾值", 0.6, 0.8, 0.65, 0.05)
                with col_b:
                    high_conf_size = st.slider("高信心倉位%", 2.0, 5.0, 3.0, 0.5) / 100
                    mid_conf_size = st.slider("中信心倉位%", 1.0, 3.0, 2.0, 0.5) / 100
                    low_conf_size = st.slider("低信心倉位%", 0.5, 2.0, 1.0, 0.5) / 100
        else:
            high_conf_threshold, mid_conf_threshold = 0.75, 0.65
            high_conf_size, mid_conf_size, low_conf_size = 0.03, 0.02, 0.01
        
        st.markdown("#### 方案3: 移動止損")
        enable_trailing_stop = st.checkbox(
            "啟用移動止損",
            value=False,
            help="達到一定盈利後跟隨價格移動止損"
        )
        
        if enable_trailing_stop:
            with st.expander("調整參數"):
                trailing_activation = st.slider(
                    "啟動比例 (TP的X%)",
                    0.3, 0.8, 0.5, 0.1,
                    help="獲利達 TP*50% 時啟動"
                )
                trailing_distance = st.slider(
                    "跟隨距離 (%)",
                    0.1, 0.5, 0.15, 0.05,
                    help="回撤 0.15% 就出場"
                ) / 100
        else:
            trailing_activation = 0.5
            trailing_distance = 0.0015
    
    with col2:
        st.markdown("#### 方案4: 時段過濾")
        enable_time_filter = st.checkbox(
            "啟用時段過濾",
            value=False,
            help="只在高流動性時段交易"
        )
        
        if enable_time_filter:
            st.info("""
            過濾條件:
            - 過濾週末
            - 過濾午休 (12-14點)
            - 只在亞/歐/美開盤時段
            """)
        
        st.markdown("#### 方案5: 嚴格篩選")
        enable_strict_filter = st.checkbox(
            "啟用嚴格篩選",
            value=False,
            help="過濾低質量信號"
        )
        
        if enable_strict_filter:
            with st.expander("調整參數"):
                min_volume_ratio = st.slider(
                    "最小量能比例",
                    0.5, 1.5, 0.8, 0.1,
                    help="量能低於平均 80% 不交易"
                )
                max_return_threshold = st.slider(
                    "最大波動 (%)",
                    1.0, 5.0, 2.0, 0.5,
                    help="單根K線波動太大不交易"
                ) / 100
                high_volatility_threshold = st.slider(
                    "高波動時閾值",
                    0.6, 0.8, 0.65, 0.05,
                    help="波動擴大時需要更高機率"
                )
        else:
            min_volume_ratio = 0.8
            max_return_threshold = 0.02
            high_volatility_threshold = 0.65
    
    st.divider()
    
    # ============ 執行按鈕 ============
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        run_button = st.button(
            "執行回測",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        save_config = st.button(
            "保存配置",
            use_container_width=True
        )
    
    with col3:
        load_config = st.button(
            "載入配置",
            use_container_width=True
        )
    
    # 保存/載入配置
    if save_config:
        config = {
            'backtest_days': backtest_days, 'symbol': symbol, 'timeframe': timeframe,
            'train_ratio': train_ratio, 'initial_capital': initial_capital, 'leverage': leverage,
            'position_size': position_size, 'threshold': threshold, 'tp_pct': tp_pct * 100, 'sl_pct': sl_pct * 100,
            'long_enabled': long_enabled, 'short_enabled': short_enabled,
            'enable_dynamic_tpsl': enable_dynamic_tpsl,
            'enable_quality_sizing': enable_quality_sizing,
            'enable_trailing_stop': enable_trailing_stop,
            'enable_time_filter': enable_time_filter,
            'enable_strict_filter': enable_strict_filter
        }
        
        config_dir = Path('backtest_results/v10_configs')
        config_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_path = config_dir / f'config_{timestamp}.json'
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        st.success(f"配置已保存: {config_path.name}")
    
    if load_config:
        config_dir = Path('backtest_results/v10_configs')
        if config_dir.exists():
            configs = sorted(config_dir.glob('config_*.json'))
            if configs:
                with open(configs[-1], 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                st.success(f"已載入: {configs[-1].name}")
                st.json(loaded)
            else:
                st.warning("無已保存配置")
    
    # ============ 執行回測 ============
    if run_button:
        if not long_enabled and not short_enabled:
            st.error("請至少啟用一個方向")
            return
        
        with st.spinner("正在執行回測..."):
            try:
                from utils.hf_data_loader import load_klines
                from backtest_v10_scalping_advanced import AdvancedScalpingBacktester
                
                end_date = datetime.now()
                start_date = end_date - timedelta(days=backtest_days)
                
                st.info(f"載入 {symbol} {timeframe} 數據...")
                df = load_klines(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                if df is None or len(df) == 0:
                    st.error("數據載入失敗")
                    return
                
                st.success(f"成功載入 {len(df)} 根K線")
                
                oos_start = int(len(df) * train_ratio)
                
                long_path = str(long_models[-1])
                short_path = str(short_models[-1])
                
                # 注意: tp_pct 和 sl_pct 已經是小數形式,不需再除以 100
                backtester = AdvancedScalpingBacktester(
                    long_model_path=long_path,
                    short_model_path=short_path,
                    initial_capital=initial_capital,
                    position_size=position_size,
                    leverage=leverage,
                    threshold=threshold,
                    tp_pct=tp_pct,  # 已經是小數
                    sl_pct=sl_pct,  # 已經是小數
                    # 優化方案
                    enable_dynamic_tpsl=enable_dynamic_tpsl,
                    enable_quality_sizing=enable_quality_sizing,
                    enable_trailing_stop=enable_trailing_stop,
                    enable_time_filter=enable_time_filter,
                    enable_strict_filter=enable_strict_filter,
                    # 動態 TP/SL (已經是小數)
                    low_vol_tp=low_vol_tp,
                    low_vol_sl=low_vol_sl,
                    mid_vol_tp=mid_vol_tp,
                    mid_vol_sl=mid_vol_sl,
                    high_vol_tp=high_vol_tp,
                    high_vol_sl=high_vol_sl,
                    # 質量分級
                    high_conf_threshold=high_conf_threshold,
                    mid_conf_threshold=mid_conf_threshold,
                    high_conf_size=high_conf_size,
                    mid_conf_size=mid_conf_size,
                    low_conf_size=low_conf_size,
                    # 移動止損
                    trailing_activation=trailing_activation,
                    trailing_distance=trailing_distance,
                    # 嚴格篩選
                    min_volume_ratio=min_volume_ratio,
                    max_return_threshold=max_return_threshold,
                    high_volatility_threshold=high_volatility_threshold
                )
                
                report = backtester.run_backtest(
                    df,
                    start_idx=oos_start,
                    long_enabled=long_enabled,
                    short_enabled=short_enabled
                )
                
                if not report:
                    st.error("回測失敗或無交易")
                    return
                
                # 保存結果
                output_dir = Path('backtest_results/v10_detailed')
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                trades_df = report['trades']
                equity_df = report['equity']
                summary = report['summary']
                
                trades_df.to_csv(output_dir / f'trades_{timestamp}.csv', index=False, encoding='utf-8-sig')
                equity_df.to_csv(output_dir / f'equity_curve_{timestamp}.csv', index=False, encoding='utf-8-sig')
                
                # 保存 JSON
                report_json = {
                    'timestamp': timestamp,
                    'config': {
                        'strategy': 'v10_scalping_advanced',
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'backtest_days': backtest_days,
                        'threshold': threshold,
                        'tp_pct': tp_pct * 100,  # 轉回百分比顯示
                        'sl_pct': sl_pct * 100,  # 轉回百分比顯示
                        'leverage': leverage,
                        'position_size': position_size * 100,
                        'initial_capital': initial_capital,
                        'optimizations': {
                            'dynamic_tpsl': enable_dynamic_tpsl,
                            'quality_sizing': enable_quality_sizing,
                            'trailing_stop': enable_trailing_stop,
                            'time_filter': enable_time_filter,
                            'strict_filter': enable_strict_filter
                        }
                    },
                    'summary': {
                        'total_trades': int(summary['total_trades']),
                        'win_rate': float(summary['win_rate']),
                        'total_return_pct': float(summary['total_return_pct']),
                        'total_pnl': float(summary['total_pnl']),
                        'profit_factor': float(summary['profit_factor']),
                        'sharpe_ratio': float(summary['sharpe_ratio']),
                        'max_drawdown': float(summary['max_drawdown']),
                        'avg_win': float(summary['avg_win']),
                        'avg_loss': float(summary['avg_loss'])
                    }
                }
                
                with open(output_dir / f'summary_{timestamp}.json', 'w', encoding='utf-8') as f:
                    json.dump(report_json, f, indent=2, ensure_ascii=False)
                
                st.success("回測完成!")
                st.balloons()
                
                # 結果摘要
                st.markdown("### 回測結果")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("總交易數", summary['total_trades'])
                with col2:
                    st.metric("勝率", f"{summary['win_rate']*100:.2f}%")
                with col3:
                    st.metric("總報酬", f"{summary['total_return_pct']*100:.2f}%")
                with col4:
                    st.metric("Sharpe", f"{summary['sharpe_ratio']:.2f}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("盈虧比", f"{summary['profit_factor']:.2f}")
                with col2:
                    st.metric("最大回撤", f"{summary['max_drawdown']*100:.2f}%")
                with col3:
                    st.metric("平均獲利", f"${summary['avg_win']:.2f}")
                with col4:
                    st.metric("平均虧損", f"${summary['avg_loss']:.2f}")
                
                # 優化效果
                if any([enable_dynamic_tpsl, enable_quality_sizing, enable_trailing_stop, enable_time_filter, enable_strict_filter]):
                    st.markdown("### 啟用的優化方案")
                    enabled = []
                    if enable_dynamic_tpsl: enabled.append("動態 TP/SL")
                    if enable_quality_sizing: enabled.append("質量分級倉位")
                    if enable_trailing_stop: enabled.append("移動止損")
                    if enable_time_filter: enabled.append("時段過濾")
                    if enable_strict_filter: enabled.append("嚴格篩選")
                    st.info(f"已啟用: {', '.join(enabled)}")
                
                st.info("切換到其他分頁查看詳細結果")
                
            except Exception as e:
                st.error(f"執行錯誤: {e}")
                import traceback
                st.code(traceback.format_exc())


def render_optimize_tab():
    """ Tab 5: 參數優化 """
    st.subheader("自動參數優化")
    
    st.info("""
    **說明:** 
    
    自動搜索最佳參數組合,找出高報酬配置。
    
    優化方式:
    - **快速優化**: 網格搜索 TP/SL + Threshold (約 5-10 分鐘)
    - **完整優化**: 包含所有參數 + 優化方案 (約 30-60 分鐘)
    """)
    
    # 檢查模型
    models_dir = Path('models_output')
    long_models = sorted(models_dir.glob('scalping_long_*_v10_*.pkl'))
    short_models = sorted(models_dir.glob('scalping_short_*_v10_*.pkl'))
    
    if not long_models or not short_models:
        st.error("未找到 v10 模型")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        opt_days = st.number_input("優化天數", 30, 180, 90, 30)
    
    with col2:
        opt_symbol = st.selectbox("交易對", ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
    
    with col3:
        opt_timeframe = st.selectbox("時間框架", ['15m', '5m', '30m'])
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("快速優化 (5-10分鐘)", type="primary", use_container_width=True):
            with st.spinner("正在執行快速優化..."):
                import subprocess
                import sys
                
                try:
                    result = subprocess.run(
                        [sys.executable, 'optimize_v10_parameters.py',
                         '--symbol', opt_symbol,
                         '--timeframe', opt_timeframe,
                         '--days', str(opt_days),
                         '--mode', 'quick'],
                        capture_output=True,
                        text=True,
                        timeout=600
                    )
                    
                    if result.returncode == 0:
                        st.success("優化完成!")
                        
                        # 讀取結果
                        opt_dir = Path('backtest_results/v10_optimization')
                        if opt_dir.exists():
                            config_files = sorted(opt_dir.glob('best_config_*.json'))
                            if config_files:
                                with open(config_files[-1], 'r', encoding='utf-8') as f:
                                    best_config = json.load(f)
                                
                                st.markdown("### 最佳配置")
                                st.json(best_config)
                                
                                # 顯示可復製的參數
                                params = best_config['best_params']
                                st.code(f"""
# 複製以下參數到回測配置:
Threshold: {params['threshold']:.2f}
TP: {params['tp_pct']*100:.1f}%
SL: {params['sl_pct']*100:.1f}%
倉位: {params['position_size']*100:.1f}%
""")
                        
                        with st.expander("查看執行輸出"):
                            st.code(result.stdout, language='text')
                    else:
                        st.error("優化失敗")
                        st.code(result.stderr, language='text')
                        
                except subprocess.TimeoutExpired:
                    st.error("優化超時 (10分鐘)")
                except Exception as e:
                    st.error(f"發生錯誤: {e}")
    
    with col2:
        if st.button("完整優化 (30-60分鐘)", use_container_width=True):
            st.warning("功能還未實現,請使用快速優化")
    
    st.divider()
    
    # 顯示歷史優化結果
    st.subheader("歷史優化結果")
    
    opt_dir = Path('backtest_results/v10_optimization')
    if opt_dir.exists():
        result_files = sorted(opt_dir.glob('optimization_results_*.csv'), reverse=True)
        
        if result_files:
            st.success(f"找到 {len(result_files)} 個優化結果")
            
            selected_file = st.selectbox(
                "選擇優化結果",
                options=[f.name for f in result_files]
            )
            
            if selected_file:
                selected_path = opt_dir / selected_file
                df = pd.read_csv(selected_path)
                
                st.markdown("### Top 10 配置")
                st.dataframe(
                    df.head(10)[[
                        'position_size', 'threshold', 'tp_pct', 'sl_pct',
                        'total_return_pct', 'sharpe_ratio', 'win_rate',
                        'total_trades', 'max_drawdown'
                    ]],
                    use_container_width=True
                )
                
                # 絡制帕雷托前線
                if len(df) > 10:
                    st.markdown("### Pareto 最優解")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=df['total_return_pct'] * 100,
                        y=df['sharpe_ratio'],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=df['win_rate'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="勝率")
                        ),
                        text=[f"TP: {row['tp_pct']*100:.1f}%, SL: {row['sl_pct']*100:.1f}%"
                              for _, row in df.iterrows()],
                        hovertemplate='<b>Return: %{x:.2f}%</b><br>Sharpe: %{y:.2f}<br>%{text}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title="報酬 vs Sharpe 比率",
                        xaxis_title="總報酬 (%)",
                        yaxis_title="Sharpe 比率",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("尚無優化結果")
    else:
        st.warning("尚無優化結果")


# 保留其他 Tab 的函數 (render_overview_tab, render_analysis_tab, render_trades_tab, render_generate_tab, load_latest_report)
# 這裡檢略了這些函數的重複代碼,它們與之前一樣
