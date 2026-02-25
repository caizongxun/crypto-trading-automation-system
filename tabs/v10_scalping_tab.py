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
from datetime import datetime


def render():
    st.header("📈 v10 剝頭皮策略回測")
    
    # 說明
    with st.expander("📊 策略介紹", expanded=False):
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
    
    # 分頁
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 報告概覽",
        "📈 詳細分析", 
        "📝 交易明細",
        "⚙️ 生成報告"
    ])
    
    with tab1:
        render_overview_tab()
    
    with tab2:
        render_analysis_tab()
    
    with tab3:
        render_trades_tab()
    
    with tab4:
        render_generate_tab()


def load_latest_report():
    """ 載入最新報告數據 """
    results_dir = Path('backtest_results/v10_detailed')
    
    if not results_dir.exists():
        return None, None, None
    
    # 找最新檔案
    trade_files = sorted(results_dir.glob('trades_*.csv'))
    equity_files = sorted(results_dir.glob('equity_curve_*.csv'))
    summary_files = sorted(results_dir.glob('summary_*.json'))
    
    if not trade_files or not equity_files or not summary_files:
        return None, None, None
    
    # 載入數據
    trades_df = pd.read_csv(trade_files[-1])
    equity_df = pd.read_csv(equity_files[-1])
    
    with open(summary_files[-1], 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return trades_df, equity_df, data


def render_overview_tab():
    """ Tab 1: 報告概覽 """
    trades_df, equity_df, data = load_latest_report()
    
    if trades_df is None:
        st.warning("⚠️ 未找到 v10 報告數據")
        st.info("請先在 '生成報告' 分頁執行回測")
        return
    
    summary = data['summary']
    config = data['config']
    
    # 最新更新時間
    st.caption(f"最後更新: {data.get('timestamp', 'N/A')}")
    
    # 關鍵指標
    st.subheader("🎯 關鍵指標")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 總報酬",
            f"{summary['total_return_pct']*100:.2f}%",
            delta=f"${summary['total_pnl']:.2f}"
        )
    
    with col2:
        st.metric(
            "🎯 勝率",
            f"{summary['win_rate']*100:.2f}%",
            delta=f"{summary['wins']}/{summary['total_trades']} 勝"
        )
    
    with col3:
        st.metric(
            "📈 Sharpe",
            f"{summary['sharpe_ratio']:.2f}",
            delta="很優秀" if summary['sharpe_ratio'] > 3 else None
        )
    
    with col4:
        st.metric(
            "🚨 最大回撤",
            f"{summary['max_drawdown']*100:.2f}%",
            delta="低風險" if summary['max_drawdown'] < 0.1 else None
        )
    
    st.divider()
    
    # 策略配置
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ 策略配置")
        st.json({
            "策略": config['strategy'],
            "時間框架": config['timeframe'],
            "Threshold": config['threshold'],
            "TP%": config['tp_pct'],
            "SL%": config['sl_pct'],
            "RR比": round(config['tp_pct'] / config['sl_pct'], 2)
        })
    
    with col2:
        st.subheader("📊 統計資訊")
        
        annual_return = summary['total_return_pct'] * (365 / 234)
        trades_per_day = summary['total_trades'] / 234
        
        st.json({
            "總交易數": summary['total_trades'],
            "每日交易": round(trades_per_day, 1),
            "年化報酬": f"{annual_return*100:.1f}%",
            "盈虧比": round(summary['profit_factor'], 2),
            "平均獲利": f"${summary['avg_win']:.2f}",
            "平均虧損": f"${summary['avg_loss']:.2f}"
        })
    
    st.divider()
    
    # 資金曲線
    st.subheader("📈 資金曲線")
    
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
    
    fig.add_hline(
        y=10000, 
        line_dash="dash", 
        line_color="gray", 
        opacity=0.5,
        annotation_text="起始資金"
    )
    
    fig.update_layout(
        title="資金増長趋勢",
        xaxis_title="時間",
        yaxis_title="資金 (USD)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 累積 PnL
    st.subheader("💵 累積 PnL")
    
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
        title="累積盈虧趋勢",
        xaxis_title="交易編號",
        yaxis_title="PnL (USD)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_analysis_tab():
    """ Tab 2: 詳細分析 """
    trades_df, equity_df, data = load_latest_report()
    
    if trades_df is None:
        st.warning("⚠️ 未找到報告數據")
        return
    
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['hour'] = trades_df['entry_time'].dt.hour
    trades_df['weekday'] = trades_df['entry_time'].dt.dayofweek
    
    # PnL 分佈
    st.subheader("📊 PnL 分佈分析")
    
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
        # 勝率移動平均
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
            title="勝率趋勢 (100MA)",
            xaxis_title="交易編號",
            yaxis_title="勝率 (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # 時間分析
    st.subheader("⏰ 時間分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 按小時
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
        # 按星期
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
    
    # Long vs Short
    st.subheader("🔄 Long vs Short 對比")
    
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
    
    st.divider()
    
    # 出場原因
    st.subheader("🚺 出場原因分析")
    
    exit_stats = trades_df.groupby('exit_reason').agg({
        'pnl': ['sum', 'mean', 'count']
    }).reset_index()
    
    exit_stats.columns = ['exit_reason', 'total_pnl', 'avg_pnl', 'count']
    exit_stats['percentage'] = exit_stats['count'] / len(trades_df) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=exit_stats['exit_reason'],
            y=exit_stats['count'],
            marker_color=['#52c41a', '#f5222d', '#faad14'],
            text=exit_stats['percentage'].apply(lambda x: f'{x:.1f}%'),
            textposition='auto'
        ))
        fig.update_layout(title="出場原因次數", height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=exit_stats['exit_reason'],
            y=exit_stats['total_pnl'],
            marker_color=['#52c41a', '#f5222d', '#faad14'],
            text=exit_stats['total_pnl'].apply(lambda x: f'${x:.2f}'),
            textposition='auto'
        ))
        fig.update_layout(title="按出場原因 PnL", height=350)
        st.plotly_chart(fig, use_container_width=True)


def render_trades_tab():
    """ Tab 3: 交易明細 """
    trades_df, _, _ = load_latest_report()
    
    if trades_df is None:
        st.warning("⚠️ 未找到報告數據")
        return
    
    st.subheader("📝 全部交易明細")
    
    # 篩選選項
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
    
    # 應用篩選
    filtered_df = trades_df[
        (trades_df['side'].isin(side_filter)) &
        (trades_df['exit_reason'].isin(exit_filter))
    ]
    
    if win_filter == '獲利':
        filtered_df = filtered_df[filtered_df['win'] == True]
    elif win_filter == '虧損':
        filtered_df = filtered_df[filtered_df['win'] == False]
    
    # 顯示統計
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
    
    # 顯示表格
    display_df = filtered_df[[
        'entry_time', 'side', 'entry_price', 'exit_price',
        'exit_reason', 'pnl', 'return_pct', 'bars_held', 'win'
    ]].copy()
    
    display_df['side'] = display_df['side'].str.upper()
    display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}")
    display_df['exit_price'] = display_df['exit_price'].apply(lambda x: f"${x:.2f}")
    display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:.2f}")
    display_df['return_pct'] = display_df['return_pct'].apply(lambda x: f"{x*100:.2f}%")
    display_df['win'] = display_df['win'].apply(lambda x: '✅' if x else '❌')
    
    display_df.columns = [
        '進場時間', '方向', '進場價', '出場價',
        '出場原因', 'PnL', '報酬%', '持有K線', '結果'
    ]
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=600
    )
    
    # 下載按鈕
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="💾 下載 CSV",
        data=csv,
        file_name=f"v10_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def render_generate_tab():
    """ Tab 4: 生成報告 """
    st.subheader("⚙️ 生成 v10 回測報告")
    
    st.info("""
    💡 **說明:**
    
    這裡可以重新執行 v10 回測並生成詳細報告。
    
    生成後的檔案會儲存在 `backtest_results/v10_detailed/` 目錄下：
    - `trades_*.csv` - 交易明細
    - `equity_curve_*.csv` - 資金曲線
    - `summary_*.json` - 摘要統計
    - `analysis_*.png` - 9 張分析圖表
    """)
    
    # 檢查檔案是否存在
    results_dir = Path('backtest_results/v10_detailed')
    
    if results_dir.exists():
        files = list(results_dir.glob('*'))
        st.success(f"✅ 已找到 {len(files)} 個報告檔案")
        
        # 顯示最新檔案
        if files:
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            st.caption(f"最新檔案: {latest_file.name}")
            st.caption(f"更新時間: {datetime.fromtimestamp(latest_file.stat().st_mtime)}")
    else:
        st.warning("⚠️ 尚未生成任何報告")
    
    st.divider()
    
    # 生成按鈕
    if st.button("🚀 執行回測並生成報告", type="primary"):
        with st.spinner("正在執行 v10 回測..."):
            import subprocess
            import sys
            
            try:
                # 執行 generate_v10_report.py
                result = subprocess.run(
                    [sys.executable, 'generate_v10_report.py'],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    st.success("✅ 報告生成成功!")
                    st.balloons()
                    st.info("🔄 請切換到其他分頁查看結果")
                    
                    # 顯示輸出
                    with st.expander("📝 查看執行輸出"):
                        st.code(result.stdout, language='text')
                else:
                    st.error("❌ 執行失敗")
                    st.code(result.stderr, language='text')
                    
            except subprocess.TimeoutExpired:
                st.error("❌ 執行超時 (5分鐘)")
            except Exception as e:
                st.error(f"❌ 發生錯誤: {e}")
    
    st.divider()
    
    st.markdown("""
    ### 📚 相關文件
    
    - `generate_v10_report.py` - 報告生成程式
    - `backtest_v10_scalping.py` - v10 回測引擎
    - `train_v10_high_frequency.py` - v10 模型訓練
    """)
