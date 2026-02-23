import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from huggingface_hub import hf_hub_download

sys.path.append(str(Path(__file__).parent.parent))
from config import Config
from utils.logger import setup_logger
from utils.agent_backtester import BidirectionalAgentBacktester
from utils.feature_engineering import FeatureEngineer

logger = setup_logger('backtesting_tab', 'logs/backtesting_tab.log')

class BacktestingTab:
    def __init__(self):
        logger.info("Initializing BacktestingTab")
        self.feature_engineer = FeatureEngineer()
    
    def render(self):
        logger.info("Rendering Backtesting Tab")
        st.header("策略回測")
        
        # 回測模式選擇
        backtest_mode = st.radio(
            "選擇回測模式",
            ['bidirectional', 'unidirectional'],
            format_func=lambda x: '雙向智能體 (Long + Short)' if x == 'bidirectional' else '單向 (Long Only)',
            horizontal=True,
            key="backtest_mode_selector"
        )
        
        if backtest_mode == 'bidirectional':
            self.render_bidirectional()
        else:
            self.render_unidirectional()
    
    def load_klines(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """從 HuggingFace 載入 K 線數據"""
        try:
            repo_id = Config.HF_REPO_ID
            base = symbol.replace("USDT", "")
            filename = f"{base}_{timeframe}.parquet"
            path_in_repo = f"klines/{symbol}/{filename}"
            
            logger.info(f"Loading {symbol} {timeframe} from HuggingFace")
            
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=path_in_repo,
                repo_type="dataset",
                token=Config.HF_TOKEN
            )
            df = pd.read_parquet(local_path)
            logger.info(f"Loaded {len(df):,} records for {symbol} {timeframe}")
            return df
        
        except Exception as e:
            logger.error(f"Failed to load {symbol} {timeframe}: {str(e)}")
            return pd.DataFrame()
    
    def render_bidirectional(self):
        """雙向智能體回測介面"""
        st.markdown("""
        ### 雙向智能體回測 - 事件驅動狀態機
        
        **核心特性**:
        - 事件驅動: 逐根 1m K 線處理
        - 狀態機: 5 種狀態 (IDLE, HUNTING_LONG, HUNTING_SHORT, LONG_POS, SHORT_POS)
        - 悉觀成交: 限價單必須嚴格穿越
        - 薫丁格處理: 同時觸及 TP+SL 則 SL 優先
        - 不對稱成本: Maker 0.01%, Taker 0.04% + 滑價 0.02%
        - 訂單過期: 15 分鐘未成交自動取消
        """)
        
        st.markdown("---")
        
        # 選擇模型
        col1, col2 = st.columns(2)
        
        models_dir = Path("models_output")
        if not models_dir.exists():
            st.warning("請先訓練模型")
            return
        
        with col1:
            long_models = list(models_dir.glob("catboost_long_*.pkl"))
            if not long_models:
                st.warning("沒有找到 Long Oracle 模型")
                return
            
            selected_long_model = st.selectbox(
                "Long Oracle 模型",
                [f.name for f in sorted(long_models, key=lambda x: x.stat().st_mtime, reverse=True)],
                key="backtest_long_model"
            )
        
        with col2:
            short_models = list(models_dir.glob("catboost_short_*.pkl"))
            if not short_models:
                st.warning("沒有找到 Short Oracle 模型")
                return
            
            selected_short_model = st.selectbox(
                "Short Oracle 模型",
                [f.name for f in sorted(short_models, key=lambda x: x.stat().st_mtime, reverse=True)],
                key="backtest_short_model"
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
                key="backtest_capital_bid"
            )
        
        with col2:
            position_size_pct = st.slider(
                "仓位比例 (%)",
                min_value=10,
                max_value=100,
                value=95,
                step=5,
                key="backtest_position_size"
            ) / 100
        
        with col3:
            tp_pct = st.number_input(
                "停利 (%)",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.1,
                key="backtest_tp"
            ) / 100
        
        with col4:
            sl_pct = st.number_input(
                "停損 (%)",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                key="backtest_sl"
            ) / 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            prob_threshold_long = st.slider(
                "Long 機率閾值",
                min_value=0.50,
                max_value=0.80,
                value=0.65,
                step=0.05,
                key="backtest_prob_long"
            )
        
        with col2:
            prob_threshold_short = st.slider(
                "Short 機率閾值",
                min_value=0.50,
                max_value=0.80,
                value=0.65,
                step=0.05,
                key="backtest_prob_short"
            )
        
        with col3:
            hunting_expire_bars = st.number_input(
                "訂單過期 (分鐘)",
                min_value=5,
                max_value=60,
                value=15,
                step=5,
                key="backtest_expire"
            )
        
        st.markdown("---")
        
        # 交易時段設定
        st.subheader("交易時段")
        
        use_24_7 = st.checkbox(
            "24/7 交易 (全天候)",
            value=True,
            key="backtest_24_7"
        )
        
        trading_hours = [(0, 24)] if use_24_7 else [(9, 14), (18, 22)]
        
        if not use_24_7:
            st.info("黃金時段: 09:00-13:59, 18:00-21:59 (UTC)")
        
        st.markdown("---")
        
        # 執行回測
        if st.button("執行雙向智能體回測", use_container_width=True, key="backtest_run_bid"):
            self.run_bidirectional_backtest(
                long_model_path=models_dir / selected_long_model,
                short_model_path=models_dir / selected_short_model,
                initial_capital=initial_capital,
                position_size_pct=position_size_pct,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                prob_threshold_long=prob_threshold_long,
                prob_threshold_short=prob_threshold_short,
                hunting_expire_bars=hunting_expire_bars,
                trading_hours=trading_hours
            )
    
    def run_bidirectional_backtest(self, long_model_path: Path, short_model_path: Path,
                                   initial_capital: float, position_size_pct: float,
                                   tp_pct: float, sl_pct: float,
                                   prob_threshold_long: float, prob_threshold_short: float,
                                   hunting_expire_bars: int, trading_hours: list):
        logger.info("Starting bidirectional agent backtest")
        
        # Step 1: 載入 1m K 線
        with st.spinner("載入 1m K 線..."):
            df_1m = self.load_klines("BTCUSDT", "1m")
            
            if df_1m.empty:
                st.error("無法載入數據")
                return
            
            if 'open_time' in df_1m.columns:
                df_1m['open_time'] = pd.to_datetime(df_1m['open_time'])
                df_1m.set_index('open_time', inplace=True)
            
            st.success(f"載入 {len(df_1m):,} 筆 1m K 線")
        
        # Step 2: 生成特徵
        with st.spinner("生成特徵 (滾動視窗)..."):
            df_features = self.feature_engineer.create_features_from_1m(
                df_1m,
                use_micro_structure=True,
                label_type='both'
            )
            st.success(f"特徵生成完成: {len(df_features):,} 筆")
        
        # Step 3: 切分測試集
        split_idx = int(len(df_features) * 0.8)
        df_test = df_features.iloc[split_idx:].copy()
        
        st.info(f"測試集: {len(df_test):,} 筆 ({df_test.index[0]} ~ {df_test.index[-1]})")
        
        # Step 4: 初始化回測器
        with st.spinner("初始化回測器..."):
            backtester = BidirectionalAgentBacktester(
                model_long_path=str(long_model_path),
                model_short_path=str(short_model_path),
                initial_capital=initial_capital,
                position_size_pct=position_size_pct,
                prob_threshold_long=prob_threshold_long,
                prob_threshold_short=prob_threshold_short,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                hunting_expire_bars=hunting_expire_bars,
                trading_hours=trading_hours,
                maker_fee=0.0001,
                taker_fee=0.0004,
                slippage=0.0002
            )
        
        # Step 5: 執行回測
        feature_cols = ['efficiency_ratio', 'extreme_time_diff', 'vol_imbalance_ratio',
                       'z_score', 'bb_width_pct', 'rsi', 'atr_pct', 'z_score_1h', 'atr_pct_1d']
        available_features = [col for col in feature_cols if col in df_test.columns]
        
        with st.spinner("執行回測... (可能需要 5-10 分鐘)"):
            try:
                results = backtester.run(df_test, available_features)
                
                if results:
                    self.display_bidirectional_results(backtester, results)
                else:
                    st.warning("回測完成但沒有交易")
            
            except Exception as e:
                logger.error(f"Backtest error: {str(e)}", exc_info=True)
                st.error(f"回測錯誤: {str(e)}")
    
    def display_bidirectional_results(self, backtester: BidirectionalAgentBacktester, results: dict):
        logger.info("Displaying bidirectional backtest results")
        
        st.success("回測完成")
        
        # 核心指標
        st.markdown("---")
        st.subheader("核心指標")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("總交易數", results.get('total_trades', 0))
        
        with col2:
            long_trades = results.get('long_trades', 0)
            short_trades = results.get('short_trades', 0)
            st.metric("Long / Short", f"{long_trades} / {short_trades}")
        
        with col3:
            win_rate = results.get('win_rate', 0)
            st.metric("勝率", f"{win_rate*100:.2f}%")
        
        with col4:
            total_return = results.get('total_return_pct', 0)
            st.metric("總報酬", f"{total_return*100:+.2f}%", 
                     delta="正" if total_return > 0 else "負")
        
        with col5:
            final_capital = results.get('final_capital', 0)
            st.metric("最終資金", f"${final_capital:,.2f}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("獲利交易", results.get('winning_trades', 0))
        
        with col2:
            st.metric("虧損交易", results.get('losing_trades', 0))
        
        with col3:
            profit_factor = results.get('profit_factor', 0)
            st.metric("Profit Factor", f"{profit_factor:.2f}")
        
        with col4:
            tp_rate = results.get('tp_rate', 0)
            sl_rate = results.get('sl_rate', 0)
            st.metric("TP / SL 比例", f"{tp_rate*100:.0f}% / {sl_rate*100:.0f}%")
        
        # 評估
        if win_rate >= 0.35 and total_return > 0 and profit_factor > 1.0:
            st.success("策略達標: 勝率 ≥ 35%, 報酬為正, Profit Factor > 1.0")
        elif total_return > 0:
            st.info("策略可用: 報酬為正但有提升空間")
        else:
            st.warning("策略需要優化: 報酬為負")
        
        # 權益曲線
        st.markdown("---")
        st.subheader("資金曲線 & 狀態時間軸")
        
        equity_df = backtester.get_equity_curve()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('資金權益曲線', '智能體狀態時間軸'),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4]
        )
        
        # 子圖 1: 資金曲線
        fig.add_trace(
            go.Scatter(
                x=equity_df['timestamp'],
                y=equity_df['capital'],
                mode='lines',
                name='Capital',
                line=dict(color='#2E86AB', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_hline(
            y=backtester.initial_capital,
            line_dash="dash",
            line_color="gray",
            annotation_text="Initial",
            row=1, col=1
        )
        
        # 子圖 2: 狀態時間軸
        state_mapping = {
            'IDLE': 0,
            'HUNTING_LONG': 1,
            'HUNTING_SHORT': 2,
            'LONG_POSITION': 3,
            'SHORT_POSITION': 4
        }
        equity_df['state_num'] = equity_df['state'].map(state_mapping)
        
        fig.add_trace(
            go.Scatter(
                x=equity_df['timestamp'],
                y=equity_df['state_num'],
                mode='markers',
                name='State',
                marker=dict(size=2, color='#A23B72', opacity=0.5)
            ),
            row=2, col=1
        )
        
        fig.update_yaxes(title_text="Capital ($)", row=1, col=1)
        fig.update_yaxes(
            title_text="State",
            ticktext=['IDLE', 'HUNT_L', 'HUNT_S', 'LONG', 'SHORT'],
            tickvals=[0, 1, 2, 3, 4],
            row=2, col=1
        )
        fig.update_xaxes(title_text="Time", row=2, col=1)
        
        fig.update_layout(height=800, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 交易記錄
        st.markdown("---")
        st.subheader("交易記錄")
        
        trades_df = backtester.get_trades_df()
        
        if not trades_df.empty:
            # 顯示統計
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("平均獲利", f"${results.get('avg_win', 0):+.2f}")
            
            with col2:
                st.metric("平均虧損", f"${results.get('avg_loss', 0):+.2f}")
            
            # 顯示交易表格
            display_df = trades_df[[
                'entry_time', 'entry_price', 'exit_time', 'exit_price',
                'direction', 'exit_reason', 'pnl_pct', 'pnl_net', 'fees'
            ]].head(50)
            
            display_df['pnl_pct'] = display_df['pnl_pct'].apply(lambda x: f"{x*100:+.2f}%")
            display_df['pnl_net'] = display_df['pnl_net'].apply(lambda x: f"${x:+.2f}")
            display_df['fees'] = display_df['fees'].apply(lambda x: f"${x:.2f}")
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # 下載報告
            report_dir = Path("backtest_results")
            report_dir.mkdir(exist_ok=True)
            
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            trades_path = report_dir / f"bidirectional_trades_{timestamp}.csv"
            trades_df.to_csv(trades_path, index=False)
            
            st.success(f"交易記錄已儲存: {trades_path}")
        else:
            st.info("沒有交易記錄")
    
    def render_unidirectional(self):
        """單向回測介面 (保留)"""
        st.info("單向回測功能保留,建議使用雙向智能體回測")
        st.markdown("""
        **建議使用雙向回測**:
        - 更高交易頻率
        - 市場中性策略
        - 牛熊市都能獲利
        - 四大死亡陷阱處理
        """)