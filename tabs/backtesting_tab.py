import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger
from utils.backtester import EventDrivenBacktester

logger = setup_logger('backtesting_tab', 'logs/backtesting_tab.log')

class BacktestingTab:
    def __init__(self):
        logger.info("Initializing BacktestingTab")
    
    def render(self):
        logger.info("Rendering Backtesting Tab")
        st.header("策略回測")
        
        st.markdown("""
        ### 事件驅動回測引擎 - 機構級別驗證
        
        **三大防線**:
        - 🛡️ **T+1 執行**: 訊號在 t 時刻產生，在 t+1 開盤價執行
        - 🛡️ **悉觀假設**: 同時觸及 SL/TP 時，假設先觸及 SL
        - 🛡️ **摩擦成本**: 手續費 + 滑價 = 0.10-0.12% / 交易
        """)
        
        st.markdown("---")
        
        # 選擇模型與特徵
        col1, col2 = st.columns(2)
        
        with col1:
            models_dir = Path("models_output")
            if not models_dir.exists():
                st.warning("請先訓練模型")
                return
            
            model_files = list(models_dir.glob("lgb_model*.txt"))
            if not model_files:
                st.warning("沒有找到模型檔案")
                return
            
            selected_model = st.selectbox(
                "選擇模型",
                [f.name for f in model_files]
            )
        
        with col2:
            features_dir = Path("features_output")
            if not features_dir.exists():
                st.warning("請先生成特徵")
                return
            
            feature_files = list(features_dir.glob("features_*.parquet"))
            if not feature_files:
                st.warning("沒有特徵檔案")
                return
            
            selected_features = st.selectbox(
                "選擇特徵檔案",
                [f.name for f in feature_files]
            )
        
        st.markdown("---")
        
        # 回測參數
        st.subheader("回測參數")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            initial_capital = st.number_input(
                "初始資金 (USD)",
                min_value=1000,
                max_value=1000000,
                value=10000
            )
        
        with col2:
            risk_reward = st.number_input(
                "盈虧比 (TP:SL)",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.5
            )
        
        with col3:
            stop_loss_pct = st.number_input(
                "停損百分比 (%)",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1
            ) / 100
        
        with col4:
            threshold = st.number_input(
                "預測閉值",
                min_value=0.3,
                max_value=0.8,
                value=0.5,
                step=0.05
            )
        
        st.markdown("---")
        
        # 執行回測
        if st.button("執行回測", use_container_width=True):
            self.run_backtest(
                models_dir / selected_model,
                features_dir / selected_features,
                initial_capital,
                risk_reward,
                stop_loss_pct,
                threshold
            )
    
    def run_backtest(self, model_path: Path, features_path: Path,
                    initial_capital: float, risk_reward: float,
                    stop_loss_pct: float, threshold: float):
        logger.info(f"Starting backtest with model={model_path}, features={features_path}")
        
        with st.spinner("載入資料..."):
            try:
                features_df = pd.read_parquet(features_path)
                st.success(f"載入 {len(features_df):,} 筆特徵")
            except Exception as e:
                st.error(f"載入特徵失敗: {str(e)}")
                return
        
        # 切分測試集 (OOS - 後 20%)
        split_idx = int(len(features_df) * 0.8)
        test_df = features_df.iloc[split_idx:].copy()
        
        st.info(f"使用 OOS 測試集: {len(test_df):,} 筆 (後 20%)")
        
        # 初始化回測引擎
        backtester = EventDrivenBacktester(
            initial_capital=initial_capital,
            risk_reward_ratio=risk_reward,
            stop_loss_pct=stop_loss_pct
        )
        
        # 執行回測
        with st.spinner("執行事件驅動回測... (可能需要幾分鐘)"):
            try:
                results = backtester.run_backtest(
                    test_df,
                    str(model_path),
                    threshold
                )
                
                if results:
                    self.display_results(results)
                else:
                    st.error("回測失敗")
            
            except Exception as e:
                logger.error(f"Backtest error: {str(e)}", exc_info=True)
                st.error(f"回測錯誤: {str(e)}")
    
    def display_results(self, results: dict):
        logger.info("Displaying backtest results")
        
        st.success("回測完成")
        
        # 核心指標
        st.subheader("核心指標")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "總交易數",
                results['total_trades']
            )
        
        with col2:
            st.metric(
                "實盤勝率",
                f"{results['win_rate']*100:.2f}%",
                delta="目標: 33%+"
            )
        
        with col3:
            st.metric(
                "總報酬",
                f"{results['total_return']*100:.2f}%"
            )
        
        with col4:
            st.metric(
                "最終資金",
                f"${results['final_capital']:.2f}"
            )
        
        with col5:
            st.metric(
                "最大回撤",
                f"{results['max_drawdown']*100:.2f}%"
            )
        
        # 進階指標
        st.markdown("---")
        st.subheader("進階指標")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("獲利交易", results['winning_trades'])
        
        with col2:
            st.metric("虧損交易", results['losing_trades'])
        
        with col3:
            st.metric("平均獲利", f"${results['avg_win']:.2f}")
        
        with col4:
            st.metric("平均虧損", f"${results['avg_loss']:.2f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("夏普比率", f"{results['sharpe_ratio']:.2f}")
        
        with col2:
            st.metric("盈虧比 (Profit Factor)", f"{results['profit_factor']:.2f}")
        
        # 資金曲線
        st.markdown("---")
        st.subheader("資金曲線")
        
        equity_df = results['equity_df']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df['time'],
            y=equity_df['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="資金曲線 (Equity Curve)",
            xaxis_title="時間",
            yaxis_title="資金 (USD)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 交易記錄
        st.markdown("---")
        st.subheader("交易記錄")
        
        trades_df = results['trades_df']
        
        # 顯示前 20 筆交易
        display_df = trades_df[[
            'entry_time', 'exit_time', 'entry_price', 'exit_price',
            'exit_reason', 'net_return', 'pnl', 'capital'
        ]].head(20)
        
        display_df['net_return'] = display_df['net_return'].apply(lambda x: f"{x*100:.2f}%")
        display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:.2f}")
        display_df['capital'] = display_df['capital'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(display_df)
        
        # 下載報告
        st.markdown("---")
        
        # 保存報告
        report_dir = Path("backtest_reports")
        report_dir.mkdir(exist_ok=True)
        
        report_path = report_dir / f"backtest_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(report_path, index=False)
        
        st.success(f"交易記錄已保存: {report_path}")
        logger.info(f"Backtest report saved to {report_path}")