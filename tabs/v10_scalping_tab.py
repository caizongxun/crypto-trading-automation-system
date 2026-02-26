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
    st.header("v10 剝頭皮策略回測")
    
    with st.expander("策略介紹", expanded=False):
        st.markdown("""
        ### v10 高頻剝頭皮策略
        
        **核心特點:**
        - 時間框架: 15分鐘
        - 交易頻率: 每日 40-50 筆
        - TP/SL: 0.5% / 0.25% (2:1 RR)
        - 平均持有: 3-5 根K線 (45-75分鐘)
        
        **適用場景:**
        - 高波動性市場
        - 短線交易
        - 快進快出
        
        **歷史績效:**
        - 總報酬: **234.45%**
        - 勝率: **57.2%**
        - Sharpe: **5.38**
        - 最大回撤: **-5.5%**
        """)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "回測配置",
        "報告概覽",
        "詳細分析", 
        "交易明細",
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
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 數據配置")
        
        # 回測天數
        backtest_days = st.number_input(
            "回測天數",
            min_value=30,
            max_value=365,
            value=90,
            step=10,
            help="使用最近 N 天的數據進行回測"
        )
        
        # 訓練/測試分割
        train_ratio = st.slider(
            "訓練集比例",
            min_value=0.5,
            max_value=0.95,
            value=0.9,
            step=0.05,
            help="數據前 X% 用於訓練,剩餘用於回測"
        )
        
        # 幣種
        symbol = st.selectbox(
            "交易對",
            options=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
            index=0
        )
        
        # 時間框架
        timeframe = st.selectbox(
            "時間框架",
            options=['15m', '5m', '30m', '1h'],
            index=0,
            help="K線時間間隔"
        )
    
    with col2:
        st.markdown("#### 交易配置")
        
        # 起始資金
        initial_capital = st.number_input(
            "起始資金 (USD)",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000
        )
        
        # 槓桿
        leverage = st.slider(
            "槓桿倍數",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
            help="槓桿越高風險越大"
        )
        
        # 最大倉位
        position_size = st.slider(
            "最大倉位 (%)",
            min_value=1.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            help="單筆交易使用資金的百分比"
        ) / 100
        
        # 閾值
        threshold = st.slider(
            "信號閾值",
            min_value=0.5,
            max_value=0.9,
            value=0.6,
            step=0.05,
            help="模型預測機率需超過此值才開倉"
        )
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 止盈止損")
        
        # TP
        tp_pct = st.slider(
            "止盈 TP (%)",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1,
            help="價格達到此百分比時止盈"
        )
        
        # SL
        sl_pct = st.slider(
            "止損 SL (%)",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="價格反向達到此百分比時止損"
        )
        
        # 計算 RR
        rr_ratio = tp_pct / sl_pct
        st.metric("風險報酬比 (RR)", f"{rr_ratio:.2f}:1")
        
        if rr_ratio < 1.5:
            st.warning("建議 RR 比至少 1.5:1")
        elif rr_ratio >= 2:
            st.success("優秀的 RR 比!")
    
    with col2:
        st.markdown("#### 方向選擇")
        
        long_enabled = st.checkbox("啟用 Long", value=True)
        short_enabled = st.checkbox("啟用 Short", value=True)
        
        if not long_enabled and not short_enabled:
            st.error("至少需要啟用一個方向")
        
        st.markdown("#### 進階設置")
        
        max_holding_bars = st.number_input(
            "最大持有K線數",
            min_value=1,
            max_value=100,
            value=20,
            help="超過此K線數強制平倉"
        )
        
        trailing_stop = st.checkbox(
            "啟用移動止損",
            value=False,
            help="動態調整止損位置 (實驗性功能)"
        )
    
    st.divider()
    
    # 執行按鈕
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
    
    # 保存配置
    if save_config:
        config = {
            'backtest_days': backtest_days,
            'train_ratio': train_ratio,
            'symbol': symbol,
            'timeframe': timeframe,
            'initial_capital': initial_capital,
            'leverage': leverage,
            'position_size': position_size,
            'threshold': threshold,
            'tp_pct': tp_pct,
            'sl_pct': sl_pct,
            'long_enabled': long_enabled,
            'short_enabled': short_enabled,
            'max_holding_bars': max_holding_bars,
            'trailing_stop': trailing_stop
        }
        
        config_dir = Path('backtest_results/v10_configs')
        config_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_path = config_dir / f'config_{timestamp}.json'
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        st.success(f"配置已保存: {config_path.name}")
    
    # 載入配置
    if load_config:
        config_dir = Path('backtest_results/v10_configs')
        if config_dir.exists():
            configs = sorted(config_dir.glob('config_*.json'))
            if configs:
                with open(configs[-1], 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                st.success(f"已載入配置: {configs[-1].name}")
                st.json(loaded_config)
            else:
                st.warning("未找到已保存的配置")
        else:
            st.warning("配置目錄不存在")
    
    # 執行回測
    if run_button:
        if not long_enabled and not short_enabled:
            st.error("請至少啟用一個交易方向")
            return
        
        with st.spinner("正在執行回測..."):
            try:
                # 載入數據
                from utils.hf_data_loader import load_klines
                
                end_date = datetime.now()
                start_date = end_date - timedelta(days=backtest_days)
                
                st.info(f"載入 {symbol} {timeframe} 數據: {start_date.date()} 至 {end_date.date()}")
                
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
                
                # 計算起始索引
                oos_start = int(len(df) * train_ratio)
                
                st.info(f"訓練集: {oos_start} 根K線, 測試集: {len(df) - oos_start} 根K線")
                
                # 執行回測
                from backtest_v10_scalping import ScalpingBacktester
                
                long_path = str(long_models[-1])
                short_path = str(short_models[-1])
                
                backtester = ScalpingBacktester(
                    long_model_path=long_path,
                    short_model_path=short_path,
                    initial_capital=initial_capital,
                    position_size=position_size,
                    leverage=leverage,
                    threshold=threshold,
                    tp_pct=tp_pct / 100,
                    sl_pct=sl_pct / 100
                )
                
                report = backtester.run_backtest(
                    df,
                    start_idx=oos_start,
                    long_enabled=long_enabled,
                    short_enabled=short_enabled
                )
                
                if not report:
                    st.error("回測執行失敗")
                    return
                
                # 保存結果
                output_dir = Path('backtest_results/v10_detailed')
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                trades_df = report['trades']
                equity_df = report['equity']
                summary = report['summary']
                
                # 保存 CSV
                trades_df.to_csv(output_dir / f'trades_{timestamp}.csv', index=False, encoding='utf-8-sig')
                equity_df.to_csv(output_dir / f'equity_curve_{timestamp}.csv', index=False, encoding='utf-8-sig')
                
                # 保存 JSON
                report_json = {
                    'timestamp': timestamp,
                    'config': {
                        'strategy': 'v10_scalping',
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'backtest_days': backtest_days,
                        'threshold': threshold,
                        'tp_pct': tp_pct,
                        'sl_pct': sl_pct,
                        'leverage': leverage,
                        'position_size': position_size * 100,
                        'initial_capital': initial_capital
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
                
                # 顯示結果摘要
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
                
                st.info("切換到其他分頁查看詳細結果")
                
            except Exception as e:
                st.error(f"執行錯誤: {e}")
                import traceback
                st.code(traceback.format_exc())


def load_latest_report():
    """ 載入最新報告數據 """
    results_dir = Path('backtest_results/v10_detailed')
    
    if not results_dir.exists():
        return None, None, None
    
    trade_files = sorted(results_dir.glob('trades_*.csv'))
    equity_files = sorted(results_dir.glob('equity_curve_*.csv'))
    summary_files = sorted(results_dir.glob('summary_*.json'))
    
    if not trade_files or not equity_files or not summary_files:
        return None, None, None
    
    trades_df = pd.read_csv(trade_files[-1])
    equity_df = pd.read_csv(equity_files[-1])
    
    with open(summary_files[-1], 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return trades_df, equity_df, data


def render_overview_tab():
    """ Tab 2: 報告概覽 """
    trades_df, equity_df, data = load_latest_report()
    
    if trades_df is None:
        st.warning("未找到 v10 報告數據")
        st.info("請先在 '回測配置' 分頁執行回測")
        return
    
    summary = data['summary']
    config = data['config']
    
    st.caption(f"最後更新: {data.get('timestamp', 'N/A')}")
    
    st.subheader("關鍵指標")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_trades = summary['total_trades']
    win_rate = summary['win_rate']
    wins = int(total_trades * win_rate)
    
    with col1:
        st.metric(
            "總報酬",
            f"{summary['total_return_pct']*100:.2f}%",
            delta=f"${summary['total_pnl']:.2f}"
        )
    
    with col2:
        st.metric(
            "勝率",
            f"{summary['win_rate']*100:.2f}%",
            delta=f"{wins}/{total_trades} 勝"
        )
    
    with col3:
        st.metric(
            "Sharpe",
            f"{summary['sharpe_ratio']:.2f}",
            delta="很優秀" if summary['sharpe_ratio'] > 3 else None
        )
    
    with col4:
        st.metric(
            "最大回撤",
            f"{summary['max_drawdown']*100:.2f}%",
            delta="低風險" if summary['max_drawdown'] < 0.1 else None
        )
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("策略配置")
        st.json(config)
    
    with col2:
        st.subheader("統計資訊")
        st.json({
            "總交易數": summary['total_trades'],
            "盈虧比": round(summary['profit_factor'], 2),
            "平均獲利": f"${summary['avg_win']:.2f}",
            "平均虧損": f"${summary['avg_loss']:.2f}"
        })
    
    st.divider()
    
    st.subheader("資金曲線")
    
    equity_df['time'] = pd.to_datetime(equity_df['time'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=equity_df['time'],
        y=equity_df['equity'],
        mode='lines',
        name='資金',
        line=dict(color='#00d4aa', width=2),
        fill='tonexty',
        fillcolor='rgba(0, 212, 170, 0.1)'
    ))
    
    initial_capital = config.get('initial_capital', 10000)
    fig.add_hline(
        y=initial_capital, 
        line_dash="dash", 
        line_color="gray", 
        opacity=0.5,
        annotation_text="起始資金"
    )
    
    fig.update_layout(
        title="資金增長趨勢",
        xaxis_title="時間",
        yaxis_title="資金 (USD)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("累積 PnL")
    
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(trades_df))),
        y=trades_df['cumulative_pnl'],
        mode='lines',
        name='累積 PnL',
        line=dict(color='#1890ff', width=2),
        fill='tozeroy',
        fillcolor='rgba(24, 144, 255, 0.1)'
    ))
    
    fig.update_layout(
        title="累積盈虧趨勢",
        xaxis_title="交易編號",
        yaxis_title="PnL (USD)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_analysis_tab():
    """ Tab 3: 詳細分析 """
    trades_df, equity_df, data = load_latest_report()
    
    if trades_df is None:
        st.warning("未找到報告數據")
        return
    
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['hour'] = trades_df['entry_time'].dt.hour
    trades_df['weekday'] = trades_df['entry_time'].dt.dayofweek
    
    st.subheader("PnL 分佈分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=trades_df['pnl'],
            nbinsx=50,
            name='PnL',
            marker_color='#1890ff',
            opacity=0.7
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)
        fig.update_layout(
            title="PnL 分佈圖",
            xaxis_title="PnL (USD)",
            yaxis_title="次數",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        win_rate_ma = trades_df['win'].rolling(100, min_periods=1).mean() * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(win_rate_ma))),
            y=win_rate_ma,
            mode='lines',
            name='勝率',
            line=dict(color='#52c41a', width=2)
        ))
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(
            title="勝率趨勢 (100MA)",
            xaxis_title="交易編號",
            yaxis_title="勝率 (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.subheader("時間分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hour_stats = trades_df.groupby('hour').agg({
            'pnl': 'sum',
            'win': 'mean'
        }).reset_index()
        
        hour_stats['win_rate'] = hour_stats['win'] * 100
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(
                x=hour_stats['hour'],
                y=hour_stats['pnl'],
                name='PnL',
                marker_color=['#52c41a' if x > 0 else '#f5222d' for x in hour_stats['pnl']],
                opacity=0.7
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=hour_stats['hour'],
                y=hour_stats['win_rate'],
                name='勝率',
                mode='lines+markers',
                line=dict(color='#1890ff', width=2)
            ),
            secondary_y=True
        )
        
        fig.update_xaxes(title_text="小時")
        fig.update_yaxes(title_text="PnL (USD)", secondary_y=False)
        fig.update_yaxes(title_text="勝率 (%)", secondary_y=True)
        fig.update_layout(title="按小時表現", height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        weekday_stats = trades_df.groupby('weekday').agg({
            'pnl': 'sum',
            'win': 'mean'
        }).reset_index()
        
        weekday_names = ['一', '二', '三', '四', '五', '六', '日']
        weekday_stats['day_name'] = weekday_stats['weekday'].apply(lambda x: weekday_names[x])
        weekday_stats['win_rate'] = weekday_stats['win'] * 100
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(
                x=weekday_stats['day_name'],
                y=weekday_stats['pnl'],
                name='PnL',
                marker_color=['#52c41a' if x > 0 else '#f5222d' for x in weekday_stats['pnl']],
                opacity=0.7
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=weekday_stats['day_name'],
                y=weekday_stats['win_rate'],
                name='勝率',
                mode='lines+markers',
                line=dict(color='#1890ff', width=2)
            ),
            secondary_y=True
        )
        
        fig.update_xaxes(title_text="星期")
        fig.update_yaxes(title_text="PnL (USD)", secondary_y=False)
        fig.update_yaxes(title_text="勝率 (%)", secondary_y=True)
        fig.update_layout(title="按星期表現", height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.subheader("Long vs Short 對比")
    
    side_stats = trades_df.groupby('side').agg({
        'pnl': ['sum', 'mean', 'count'],
        'win': 'mean'
    }).reset_index()
    
    side_stats.columns = ['side', 'total_pnl', 'avg_pnl', 'count', 'win_rate']
    side_stats['win_rate'] *= 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=side_stats['side'].str.upper(),
            y=side_stats['total_pnl'],
            marker_color=['#1890ff', '#52c41a'],
            text=side_stats['total_pnl'].apply(lambda x: f'${x:.2f}'),
            textposition='auto'
        ))
        fig.update_layout(title="總 PnL 對比", height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=side_stats['side'].str.upper(),
            y=side_stats['win_rate'],
            marker_color=['#1890ff', '#52c41a'],
            text=side_stats['win_rate'].apply(lambda x: f'{x:.1f}%'),
            textposition='auto'
        ))
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(title="勝率對比", height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=side_stats['side'].str.upper(),
            values=side_stats['count'],
            marker_colors=['#1890ff', '#52c41a'],
            textinfo='label+percent+value',
            hole=0.4
        ))
        fig.update_layout(title="交易數量分佈", height=350)
        st.plotly_chart(fig, use_container_width=True)


def render_trades_tab():
    """ Tab 4: 交易明細 """
    trades_df, _, _ = load_latest_report()
    
    if trades_df is None:
        st.warning("未找到報告數據")
        return
    
    st.subheader("全部交易明細")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        side_filter = st.multiselect(
            "方向",
            options=['long', 'short'],
            default=['long', 'short']
        )
    
    with col2:
        exit_filter = st.multiselect(
            "出場原因",
            options=trades_df['exit_reason'].unique().tolist(),
            default=trades_df['exit_reason'].unique().tolist()
        )
    
    with col3:
        win_filter = st.selectbox(
            "結果",
            options=['全部', '獲利', '虧損']
        )
    
    filtered_df = trades_df[
        (trades_df['side'].isin(side_filter)) &
        (trades_df['exit_reason'].isin(exit_filter))
    ]
    
    if win_filter == '獲利':
        filtered_df = filtered_df[filtered_df['win'] == True]
    elif win_filter == '虧損':
        filtered_df = filtered_df[filtered_df['win'] == False]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("符合筆數", len(filtered_df))
    with col2:
        st.metric("總 PnL", f"${filtered_df['pnl'].sum():.2f}")
    with col3:
        st.metric("勝率", f"{filtered_df['win'].mean()*100:.1f}%")
    with col4:
        st.metric("平均 PnL", f"${filtered_df['pnl'].mean():.2f}")
    
    st.divider()
    
    display_df = filtered_df[[
        'entry_time', 'side', 'entry_price', 'exit_price',
        'exit_reason', 'pnl', 'return_pct', 'bars_held', 'win'
    ]].copy()
    
    display_df['side'] = display_df['side'].str.upper()
    display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}")
    display_df['exit_price'] = display_df['exit_price'].apply(lambda x: f"${x:.2f}")
    display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:.2f}")
    display_df['return_pct'] = display_df['return_pct'].apply(lambda x: f"{x*100:.2f}%")
    display_df['win'] = display_df['win'].apply(lambda x: 'V' if x else 'X')
    
    display_df.columns = [
        '進場時間', '方向', '進場價', '出場價',
        '出場原因', 'PnL', '報酬%', '持有K線', '結果'
    ]
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=600
    )
    
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="下載 CSV",
        data=csv,
        file_name=f"v10_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def render_generate_tab():
    """ Tab 5: 批次生成 """
    st.subheader("批次生成完整報告")
    
    st.info("""
    **說明:**
    
    執行完整的 v10 回測並生成所有報告檔案 (包含圖表分析)。
    
    生成後的檔案會儲存在 `backtest_results/v10_detailed/` 目錄下:
    - `trades_*.csv` - 交易明細
    - `equity_curve_*.csv` - 資金曲線
    - `summary_*.json` - 摘要統計
    - `analysis_*.png` - 9 張分析圖表
    """)
    
    results_dir = Path('backtest_results/v10_detailed')
    
    if results_dir.exists():
        files = list(results_dir.glob('*'))
        st.success(f"已找到 {len(files)} 個報告檔案")
        
        if files:
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            st.caption(f"最新檔案: {latest_file.name}")
            st.caption(f"更新時間: {datetime.fromtimestamp(latest_file.stat().st_mtime)}")
    else:
        st.warning("尚未生成任何報告")
    
    st.divider()
    
    if st.button("執行完整回測並生成報告", type="primary"):
        with st.spinner("正在執行 v10 回測..."):
            import subprocess
            import sys
            
            try:
                result = subprocess.run(
                    [sys.executable, 'generate_v10_report.py'],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    st.success("報告生成成功!")
                    st.balloons()
                    st.info("請切換到其他分頁查看結果")
                    
                    with st.expander("查看執行輸出"):
                        st.code(result.stdout, language='text')
                else:
                    st.error("執行失敗")
                    st.code(result.stderr, language='text')
                    
            except subprocess.TimeoutExpired:
                st.error("執行超時 (5分鐘)")
            except Exception as e:
                st.error(f"發生錯誤: {e}")
    
    st.divider()
    
    st.markdown("""
    ### 相關文件
    
    - `generate_v10_report.py` - 報告生成程式
    - `backtest_v10_scalping.py` - v10 回測引擎
    - `train_v10_high_frequency.py` - v10 模型訓練
    """)
