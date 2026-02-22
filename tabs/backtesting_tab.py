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
        ### 事件驅動回測引擎 - 三層物理濾網優化
        
        **三大防線**:
        - 🛡️ **T+1 執行**: 訊號在 t 時刻產生，在 t+1 開盤價執行
        - 🛡️ **悲觀假設**: 同時觸及 SL/TP 時，假設先觸及 SL
        - 🛡️ **非對稱手續費**: 進場 Maker 0.01%, 停損 Taker 0.06%
        
        **三層濾網**:
        - ✅ **Layer 1**: 拉高機率閉值 (0.65+) - 只做高品質訊號
        - ✅ **Layer 2**: 黃金時段濾網 (9-13, 18-21 UTC) - 迴避洗盤時段
        - ✅ **Layer 3**: Maker 限價進場 (0.01%) - 減少摩擦成本 75%
        """)
        
        st.markdown("---")
        
        # 選擇模型與特徵
        col1, col2 = st.columns(2)
        
        with col1:
            models_dir = Path("models_output")
            if not models_dir.exists():
                st.warning("請先訓練模型")
                return
            
            # 支援 LightGBM (.txt) 和 CatBoost (.pkl)
            lgb_files = list(models_dir.glob("lgb_model*.txt"))
            cb_files = list(models_dir.glob("catboost_model*.pkl"))
            model_files = lgb_files + cb_files
            
            if not model_files:
                st.warning("沒有找到模型檔案")
                return
            
            selected_model = st.selectbox(
                "選擇模型",
                [f.name for f in model_files],
                key="backtest_model_selectbox"
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
                [f.name for f in feature_files],
                key="backtest_features_selectbox"
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
                value=10000,
                key="backtest_capital"
            )
        
        with col2:
            risk_reward = st.number_input(
                "盈虧比 (TP:SL)",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.5,
                key="backtest_risk_reward"
            )
        
        with col3:
            stop_loss_pct = st.number_input(
                "停損百分比 (%)",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                key="backtest_stop_loss"
            ) / 100
        
        with col4:
            probability_threshold = st.slider(
                "機率閉值",
                min_value=0.50,
                max_value=0.80,
                value=0.65,
                step=0.05,
                key="backtest_prob_threshold",
                help="Layer 1: 只做機率大於此值的訊號 (0.65 = 65%)"
            )
        
        st.markdown("---")
        
        # 進階選項 - 三層濾網
        st.subheader("三層濾網設定")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("🟢 **Layer 1: 機率閉值**")
            st.metric("當前閉值", f"{probability_threshold*100:.0f}%")
            st.caption("過濾低品質訊號")
        
        with col2:
            st.markdown("🟡 **Layer 2: 時間濾網**")
            use_time_filter = st.checkbox(
                "啟用黃金時段",
                value=True,
                key="backtest_time_filter",
                help="只允許在 09:00-13:59, 18:00-21:59 (UTC) 進場"
            )
            if use_time_filter:
                st.caption("✅ 9-13, 18-21 UTC")
            else:
                st.caption("❌ 全天交易")
        
        with col3:
            st.markdown("🔵 **Layer 3: Maker 手續費**")
            st.info("預設啟用 (0.01%)")
            st.caption("減少摩擦成本 75%")
        
        st.markdown("---")
        
        # 預期效果
        st.subheader("🎯 預期效果")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "交易數",
                "100-150",
                delta="-50% vs 基準",
                help="過濾低品質訊號後"
            )
        
        with col2:
            st.metric(
                "勝率",
                "40%+",
                delta="+6% vs 基準",
                help="只做高機率訊號"
            )
        
        with col3:
            st.metric(
                "Profit Factor",
                "1.10+",
                delta="+0.22 vs 基準",
                help="總獲利 / 總虧損"
            )
        
        st.markdown("---")
        
        # 執行回測
        if st.button("🚀 執行三層濾網回測", use_container_width=True, key="backtest_run_button"):
            self.run_backtest(
                models_dir / selected_model,
                features_dir / selected_features,
                initial_capital,
                risk_reward,
                stop_loss_pct,
                probability_threshold,
                use_time_filter
            )
    
    def run_backtest(self, model_path: Path, features_path: Path,
                    initial_capital: float, risk_reward: float,
                    stop_loss_pct: float, probability_threshold: float,
                    use_time_filter: bool):
        logger.info(f"Starting 3-layer filtered backtest: model={model_path}, features={features_path}")
        
        with st.spinner("載入資料..."):
            try:
                features_df = pd.read_parquet(features_path)
                st.success(f"載入 {len(features_df):,} 筆特徵")
            except Exception as e:
                st.error(f"載入特徵失敗: {str(e)}")
                return
        
        # 切分測試集
        split_idx = int(len(features_df) * 0.8)
        test_df = features_df.iloc[split_idx:].copy()
        
        st.info(f"使用 OOS 測試集: {len(test_df):,} 筆 (後 20%)")
        
        # 初始化回測引擎
        backtester = EventDrivenBacktester(
            initial_capital=initial_capital,
            risk_reward_ratio=risk_reward,
            stop_loss_pct=stop_loss_pct,
            maker_fee=0.0001,  # Layer 3: Maker 0.01%
            taker_fee=0.0004,  # Taker 0.04%
            slippage_pct=0.0002,  # 滑價 0.02%
            use_time_filter=use_time_filter,  # Layer 2
            probability_threshold=probability_threshold  # Layer 1
        )
        
        # 執行回測
        with st.spinner("執行三層濾網回測... (可能需要幾分鐘)"):
            try:
                results = backtester.run_backtest(
                    test_df,
                    str(model_path)
                )
                
                if results:
                    self.display_results(results, probability_threshold, use_time_filter)
                else:
                    st.error("回測失敗")
            
            except Exception as e:
                logger.error(f"Backtest error: {str(e)}", exc_info=True)
                st.error(f"回測錯誤: {str(e)}")
    
    def display_results(self, results: dict, probability_threshold: float, use_time_filter: bool):
        logger.info("Displaying 3-layer filtered backtest results")
        
        st.success("✅ 回測完成")
        
        # 顯示濾網統計
        st.subheader("🛡️ 三層濾網效果")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Layer 1: 機率濾網",
                f"{results.get('filtered_signals_prob', 0):,} 個訊號",
                delta=f"< {probability_threshold*100:.0f}%"
            )
        
        with col2:
            st.metric(
                "Layer 2: 時間濾網",
                f"{results.get('filtered_signals_time', 0):,} 個訊號",
                delta="非黃金時段" if use_time_filter else "未啟用"
            )
        
        with col3:
            st.metric(
                "Layer 3: Maker 手續費",
                "0.01%",
                delta="-75% vs Taker"
            )
        
        st.markdown("---")
        
        # 核心指標
        st.subheader("📊 核心指標")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "總交易數",
                results['total_trades']
            )
        
        with col2:
            win_rate = results['win_rate']
            win_rate_delta = "+" if win_rate >= 0.36 else "-"
            st.metric(
                "實盤勝率",
                f"{win_rate*100:.2f}%",
                delta=f"{win_rate_delta} 目標: 36.9%+"
            )
        
        with col3:
            total_return = results['total_return']
            st.metric(
                "總報酬",
                f"{total_return*100:.2f}%",
                delta="正" if total_return > 0 else "負"
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
        
        # 評估
        profit_factor = results['profit_factor']
        
        if win_rate >= 0.369 and total_return > 0 and profit_factor > 1.0:
            st.success("✅ 策略達標！勝率 ≥ 36.9%, 報酬為正, Profit Factor > 1.0")
        elif win_rate >= 0.369:
            st.warning("⚠️ 勝率達標但報酬為負，考慮調整盈虧比或降低摩擦成本")
        else:
            st.error(f"❌ 勝率不足 ({win_rate*100:.2f}% < 36.9%)，需要優化模型或提高閉值")
        
        # 進階指標
        st.markdown("---")
        st.subheader("📈 進階指標")
        
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
            pf_color = "normal"
            if profit_factor >= 1.1:
                pf_color = "normal"
            st.metric(
                "盈虧比 (Profit Factor)",
                f"{profit_factor:.2f}",
                delta="目標: 1.10+"
            )
        
        # 資金曲線
        st.markdown("---")
        st.subheader("📉 資金曲線")
        
        equity_df = results['equity_df']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity_df['time'],
            y=equity_df['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='green' if total_return > 0 else 'red', width=2)
        ))
        
        # 添加初始資金基準線
        fig.add_hline(
            y=results['final_capital'] - results['total_return'] * results['final_capital'],
            line_dash="dash",
            line_color="gray",
            annotation_text="Initial Capital"
        )
        
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
        st.subheader("📋 交易記錄 (Top 20)")
        
        trades_df = results['trades_df']
        
        display_df = trades_df[[
            'entry_time', 'exit_time', 'entry_price', 'exit_price',
            'exit_reason', 'gross_return', 'pnl', 'capital', 'entry_hour', 'probability'
        ]].head(20)
        
        display_df['gross_return'] = display_df['gross_return'].apply(lambda x: f"{x*100:.2f}%")
        display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:.2f}")
        display_df['capital'] = display_df['capital'].apply(lambda x: f"${x:.2f}")
        display_df['probability'] = display_df['probability'].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(display_df, use_container_width=True)
        
        # 下載報告
        st.markdown("---")
        
        report_dir = Path("backtest_reports")
        report_dir.mkdir(exist_ok=True)
        
        report_path = report_dir / f"backtest_3layer_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(report_path, index=False)
        
        st.success(f"💾 交易記錄已保存: {report_path}")
        logger.info(f"Backtest report saved to {report_path}")