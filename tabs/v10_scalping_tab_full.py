#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v10 剝頭皮策略回測 Tab - Streamlit GUI (完整版)
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
    st.subheader("互動式回測配置")
    st.warning("功能過於複雜,請使用簡化版,在 v10_scalping_tab_full.py 找到此檔")


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


def render_optimize_tab():
    """ Tab 5: 參數優化 """
    st.subheader("自動參數優化")
    
    st.info("""
    **說明:** 
    
    自動搜索最佳參數組合,找出高報酬配置。
    
    優化方式:
    - **快速優化**: 網格搜索 TP/SL + Threshold (約 5-10 分鐘)
    """)
    
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
    
    if st.button("執行快速優化", type="primary", use_container_width=True):
        with st.spinner("正在優化..."):
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
                    
                    opt_dir = Path('backtest_results/v10_optimization')
                    if opt_dir.exists():
                        config_files = sorted(opt_dir.glob('best_config_*.json'))
                        if config_files:
                            with open(config_files[-1], 'r', encoding='utf-8') as f:
                                best_config = json.load(f)
                            
                            st.markdown("### 最佳配置")
                            st.json(best_config)
                            
                            params = best_config['best_params']
                            st.code(f"""
# 複製以下參數到回測配置:
Threshold: {params['threshold']:.2f}
TP: {params['tp_pct']*100:.1f}%
SL: {params['sl_pct']*100:.1f}%
倉位: {params['position_size']*100:.1f}%
""")
                    
                    with st.expander("查看輸出"):
                        st.code(result.stdout, language='text')
                else:
                    st.error("優化失敗")
                    st.code(result.stderr, language='text')
                    
            except subprocess.TimeoutExpired:
                st.error("優化超時")
            except Exception as e:
                st.error(f"錯誤: {e}")


def render_generate_tab():
    """ Tab 6: 批次生成 """
    st.subheader("批次生成完整報告")
    st.info("執行完整的 v10 回測並生成所有報告檔案")
    
    if st.button("執行完整回測並生成報告", type="primary"):
        with st.spinner("正在執行..."):
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
                    with st.expander("查看輸出"):
                        st.code(result.stdout, language='text')
                else:
                    st.error("執行失敗")
                    st.code(result.stderr, language='text')
                    
            except subprocess.TimeoutExpired:
                st.error("執行超時 (5分鐘)")
            except Exception as e:
                st.error(f"發生錯誤: {e}")
