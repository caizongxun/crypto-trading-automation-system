import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from huggingface_hub import hf_hub_download

sys.path.append(str(Path(__file__).parent.parent))
from config import Config
from utils.logger import setup_logger
from utils.agent_backtester import BidirectionalAgentBacktester
from utils.adaptive_backtester import AdaptiveBacktester
from utils.backtest_analyzer import BacktestAnalyzer
from utils.feature_engineering import FeatureEngineer

# 試圖載入 V2
try:
    from utils.feature_engineering_v2 import FeatureEngineerV2
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False

logger = setup_logger('backtesting_tab', 'logs/backtesting_tab.log')

class BacktestingTab:
    def __init__(self):
        logger.info("Initializing BacktestingTab")
        self.feature_engineer_v1 = FeatureEngineer()
        if V2_AVAILABLE:
            self.feature_engineer_v2 = FeatureEngineerV2(
                enable_advanced_features=True,
                enable_ml_features=True
            )
    
    def detect_model_version(self, model_name: str) -> str:
        """偵測模型版本"""
        if '_v2_' in model_name:
            return 'v2'
        return 'v1'
    
    def render(self):
        logger.info("Rendering Backtesting Tab")
        st.header("策略回測 - 多維度優化系統")
        
        # 檢查 V2 狀態
        if V2_AVAILABLE:
            st.success("V2 系統已啟用 - 支持 V1 和 V2 模型")
        else:
            st.info("當前僅支持 V1 模型")
        
        st.markdown("---")
        
        # 回測引擎選擇
        st.markdown("### 回測引擎選擇")
        
        backtest_engine = st.radio(
            "選擇回測引擎",
            ['standard', 'adaptive'],
            format_func=lambda x: '標準雙向智能體' if x == 'standard' else '進階自適應智能體 (推薦)',
            horizontal=True,
            key="backtest_engine"
        )
        
        if backtest_engine == 'standard':
            st.info("標準引擎: 固定參數回測,適合建立基線")
            self.render_standard_backtest()
        else:
            st.success("自適應引擎: 動態參數調整,根據市場狀態優化")
            self.render_adaptive_backtest()
    
    def load_klines(self, symbol: str, timeframe: str, backtest_days: int = None) -> pd.DataFrame:
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
            
            # 根據 backtest_days 限制數據
            if backtest_days is not None:
                # 1440 = 1天的分鐘數
                total_minutes = backtest_days * 1440
                df = df.tail(total_minutes)
                logger.info(f"Limited to last {backtest_days} days ({len(df):,} records)")
            else:
                logger.info(f"Loaded all data ({len(df):,} records)")
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            return pd.DataFrame()
    
    def render_model_selector(self, prefix: str, direction: str) -> tuple:
        """模型選擇器 - 分離 V1/V2"""
        models_dir = Path("models_output")
        
        if direction == "long":
            v1_models = list(models_dir.glob("catboost_long_[0-9]*.pkl"))
            v2_models = list(models_dir.glob("catboost_long_v2_*.pkl"))
        else:
            v1_models = list(models_dir.glob("catboost_short_[0-9]*.pkl"))
            v2_models = list(models_dir.glob("catboost_short_v2_*.pkl"))
        
        # 排序
        v1_models = sorted(v1_models, key=lambda x: x.stat().st_mtime, reverse=True)
        v2_models = sorted(v2_models, key=lambda x: x.stat().st_mtime, reverse=True)
        
        # 版本選擇
        if V2_AVAILABLE and v2_models:
            version_choice = st.radio(
                f"{direction.upper()} 模型版本",
                options=["V2 (推薦)", "V1"],
                index=0,
                key=f"{prefix}_{direction}_version",
                horizontal=True
            )
            
            use_v2 = "V2" in version_choice
        else:
            use_v2 = False
            if V2_AVAILABLE:
                st.info(f"沒有 {direction.upper()} V2 模型,使用 V1")
        
        # 選擇模型
        if use_v2:
            if not v2_models:
                st.warning(f"沒有找到 {direction.upper()} V2 模型")
                return None, 'v2'
            
            selected_model = st.selectbox(
                f"{direction.upper()} Oracle (V2)",
                [f.name for f in v2_models],
                key=f"{prefix}_{direction}_model"
            )
            return selected_model, 'v2'
        else:
            if not v1_models:
                st.warning(f"沒有找到 {direction.upper()} V1 模型")
                return None, 'v1'
            
            selected_model = st.selectbox(
                f"{direction.upper()} Oracle (V1)",
                [f.name for f in v1_models],
                key=f"{prefix}_{direction}_model"
            )
            return selected_model, 'v1'
    
    def render_standard_backtest(self):
        st.markdown("---")
        st.subheader("標準回測配置")
        
        # 模型選擇
        models_dir = Path("models_output")
        if not models_dir.exists():
            st.warning("請先訓練模型")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_long_model, long_version = self.render_model_selector("std", "long")
            if selected_long_model is None:
                return
        
        with col2:
            selected_short_model, short_version = self.render_model_selector("std", "short")
            if selected_short_model is None:
                return
        
        # 版本一致性檢查
        if long_version != short_version:
            st.warning("Long 和 Short 模型版本不一致,建議使用相同版本")
        
        model_version = long_version
        
        # 顯示版本資訊
        if model_version == 'v2':
            st.success("使用 V2 特徵工程 (44-54 個特徵)")
        else:
            st.info("使用 V1 特徵工程 (9 個特徵)")
        
        st.markdown("---")
        
        # 新增: 回測期間和槓桿設定
        st.markdown("#### 回測設定")
        col1, col2 = st.columns(2)
        
        with col1:
            backtest_days = st.select_slider(
                "回測天數",
                options=[7, 30, 60, 90, 180, 365],
                value=180,
                key="std_backtest_days",
                help="選擇回測數據的天數 (從最新數據往回算)"
            )
            st.caption(f"約 {backtest_days * 1440:,} 根 1m K線")
        
        with col2:
            leverage = st.slider(
                "槓桿倍數",
                min_value=1,
                max_value=10,
                value=1,
                step=1,
                key="std_leverage",
                help="報酬率會乘以槓桿倍數 (風險也同步放大)"
            )
            if leverage > 1:
                st.warning(f"⚠️ 使用 {leverage}x 槓桿,風險提高 {leverage} 倍")
        
        st.markdown("---")
        
        # 資金管理
        st.markdown("#### 資金管理")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            initial_capital = st.number_input(
                "初始資金 (USD)",
                min_value=1000,
                max_value=1000000,
                value=10000,
                key="std_capital"
            )
        
        with col2:
            position_size_pct = st.slider(
                "倉位比例 (%)",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
                key="std_position"
            ) / 100
        
        with col3:
            tp_pct = st.number_input(
                "停利 (%)",
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.1,
                key="std_tp"
            ) / 100
        
        with col4:
            sl_pct = st.number_input(
                "停損 (%)",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                key="std_sl"
            ) / 100
        
        st.markdown("#### 閾值設定")
        col1, col2 = st.columns(2)
        
        with col1:
            prob_threshold_long = st.slider(
                "Long 機率閾值",
                min_value=0.10,
                max_value=0.50,
                value=0.16,
                step=0.01,
                key="std_prob_long",
                help="建議 0.16 以獲得 100+ 筆交易"
            )
        
        with col2:
            prob_threshold_short = st.slider(
                "Short 機率閾值",
                min_value=0.10,
                max_value=0.50,
                value=0.16,
                step=0.01,
                key="std_prob_short"
            )
        
        st.info(
            f"機率解讀: {prob_threshold_long:.2f} = {prob_threshold_long/0.05:.1f}x Lift (基礎勝率 5%)"
        )
        
        # 交易時段
        st.markdown("#### 交易時段")
        use_24_7 = st.checkbox(
            "24/7 交易",
            value=False,
            key="std_24_7",
            help="不勾選則只在黃金時段交易 (09-13, 18-21 UTC)"
        )
        
        trading_hours = [(0, 24)] if use_24_7 else [(9, 14), (18, 22)]
        
        st.markdown("---")
        
        if st.button("執行標準回測", use_container_width=True, type="primary", key="std_run"):
            self.run_standard_backtest(
                long_model_path=models_dir / selected_long_model,
                short_model_path=models_dir / selected_short_model,
                model_version=model_version,
                initial_capital=initial_capital,
                position_size_pct=position_size_pct,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                prob_threshold_long=prob_threshold_long,
                prob_threshold_short=prob_threshold_short,
                trading_hours=trading_hours,
                backtest_days=backtest_days,
                leverage=leverage
            )
    
    def render_adaptive_backtest(self):
        st.markdown("---")
        st.subheader("自適應回測配置")
        
        # 模型選擇
        models_dir = Path("models_output")
        if not models_dir.exists():
            st.warning("請先訓練模型")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_long_model, long_version = self.render_model_selector("adp", "long")
            if selected_long_model is None:
                return
        
        with col2:
            selected_short_model, short_version = self.render_model_selector("adp", "short")
            if selected_short_model is None:
                return
        
        model_version = long_version
        
        if model_version == 'v2':
            st.success("使用 V2 特徵工程")
        else:
            st.info("使用 V1 特徵工程")
        
        st.markdown("---")
        
        # 新增: 回測期間和槓桿
        st.markdown("#### 回測設定")
        col1, col2 = st.columns(2)
        
        with col1:
            backtest_days = st.select_slider(
                "回測天數",
                options=[7, 30, 60, 90, 180, 365],
                value=180,
                key="adp_backtest_days"
            )
            st.caption(f"約 {backtest_days * 1440:,} 根 1m K線")
        
        with col2:
            leverage = st.slider(
                "槓桿倍數",
                min_value=1,
                max_value=10,
                value=1,
                step=1,
                key="adp_leverage"
            )
            if leverage > 1:
                st.warning(f"⚠️ {leverage}x 槓桿")
        
        st.markdown("---")
        
        # 基礎參數
        st.markdown("#### 基礎參數")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            initial_capital = st.number_input(
                "初始資金",
                min_value=1000,
                max_value=1000000,
                value=10000,
                key="adp_capital"
            )
        
        with col2:
            base_position_size = st.slider(
                "基礎倉位 (%)",
                min_value=5,
                max_value=20,
                value=10,
                key="adp_position"
            ) / 100
        
        with col3:
            prob_threshold = st.slider(
                "基礎閾值",
                min_value=0.10,
                max_value=0.30,
                value=0.16,
                step=0.01,
                key="adp_threshold"
            )
        
        with col4:
            base_tp_sl_ratio = st.selectbox(
                "基礎 TP:SL",
                ['2:1', '1.5:1', '1:1'],
                key="adp_ratio"
            )
        
        if base_tp_sl_ratio == '2:1':
            base_tp, base_sl = 0.02, 0.01
        elif base_tp_sl_ratio == '1.5:1':
            base_tp, base_sl = 0.015, 0.01
        else:
            base_tp, base_sl = 0.015, 0.015
        
        st.markdown("---")
        
        # 自適應功能
        st.markdown("#### 自適應功能開關")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            enable_vol_adapt = st.checkbox(
                "波動率自適應",
                value=True,
                key="adp_vol",
                help="根據 ATR 動態調整 TP/SL"
            )
        
        with col2:
            enable_prob_layer = st.checkbox(
                "機率分層倉位",
                value=True,
                key="adp_prob",
                help="高機率時加大倉位"
            )
        
        with col3:
            enable_time_based = st.checkbox(
                "時段差異化",
                value=True,
                key="adp_time",
                help="不同時段不同策略"
            )
        
        with col4:
            enable_risk_ctrl = st.checkbox(
                "風控強化",
                value=True,
                key="adp_risk",
                help="日內虧損限制 + 連續停損保護"
            )
        
        # 風控參數
        if enable_risk_ctrl:
            st.markdown("#### 風控參數")
            col1, col2 = st.columns(2)
            
            with col1:
                max_daily_loss = st.slider(
                    "最大日內虧損 (%)",
                    min_value=1,
                    max_value=10,
                    value=3,
                    key="adp_daily_loss"
                ) / 100
            
            with col2:
                max_consec_loss = st.number_input(
                    "最大連續停損",
                    min_value=3,
                    max_value=10,
                    value=5,
                    key="adp_consec"
                )
        else:
            max_daily_loss = 0.05
            max_consec_loss = 10
        
        # 交易時段
        st.markdown("#### 交易時段")
        use_24_7 = st.checkbox(
            "24/7 交易",
            value=False,
            key="adp_24_7"
        )
        
        trading_hours = [(0, 24)] if use_24_7 else [(9, 14), (18, 22)]
        
        st.markdown("---")
        
        if st.button("執行自適應回測", use_container_width=True, type="primary", key="adp_run"):
            self.run_adaptive_backtest(
                long_model_path=models_dir / selected_long_model,
                short_model_path=models_dir / selected_short_model,
                model_version=model_version,
                initial_capital=initial_capital,
                base_position_size_pct=base_position_size,
                prob_threshold=prob_threshold,
                base_tp_pct=base_tp,
                base_sl_pct=base_sl,
                trading_hours=trading_hours,
                enable_volatility_adaptation=enable_vol_adapt,
                enable_probability_layering=enable_prob_layer,
                enable_time_based_strategy=enable_time_based,
                enable_risk_controls=enable_risk_ctrl,
                max_daily_loss_pct=max_daily_loss,
                max_consecutive_losses=max_consec_loss,
                backtest_days=backtest_days,
                leverage=leverage
            )
    
    def run_standard_backtest(self, model_version='v1', backtest_days=180, leverage=1, **params):
        logger.info(f"Starting standard backtest with {model_version} features, {backtest_days} days, {leverage}x leverage")
        
        with st.spinner("載入 1m K線..."):
            df_1m = self.load_klines("BTCUSDT", "1m", backtest_days=backtest_days)
            if df_1m.empty:
                st.error("無法載入數據")
                return
            
            if 'open_time' in df_1m.columns:
                df_1m['open_time'] = pd.to_datetime(df_1m['open_time'])
                df_1m.set_index('open_time', inplace=True)
        
        # 根據版本選擇 feature engineer
        if model_version == 'v2' and V2_AVAILABLE:
            with st.spinner("生成 V2 特徵 (44-54個)..."):
                df_features = self.feature_engineer_v2.create_features_from_1m(
                    df_1m, 
                    use_adaptive_labels=False,
                    label_type='both'
                )
                feature_cols = self.feature_engineer_v2.get_feature_list()
        else:
            with st.spinner("生成 V1 特徵 (9個)..."):
                df_features = self.feature_engineer_v1.create_features_from_1m(
                    df_1m, use_micro_structure=True, label_type='both'
                )
                feature_cols = [
                    'efficiency_ratio', 'extreme_time_diff', 'vol_imbalance_ratio',
                    'z_score', 'bb_width_pct', 'rsi', 'atr_pct', 'z_score_1h', 'atr_pct_1d'
                ]
        
        # 只保留必要的特徵
        df_features_filtered = df_features[feature_cols].copy()
        
        # 加入 OHLCV
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df_1m.columns:
                df_features_filtered[col] = df_1m[col]
        
        split_idx = int(len(df_features_filtered) * 0.8)
        df_test = df_features_filtered.iloc[split_idx:].copy()
        
        actual_backtest_days = len(df_test) / 1440
        
        st.info(
            f"測試集: {len(df_test):,} 筆 ({actual_backtest_days:.1f} 天) | "
            f"版本: {model_version.upper()} | 特徵數: {len(feature_cols)} | "
            f"槓桿: {leverage}x"
        )
        
        with st.spinner("執行回測..."):
            backtester = BidirectionalAgentBacktester(
                model_long_path=str(params['long_model_path']),
                model_short_path=str(params['short_model_path']),
                initial_capital=params['initial_capital'],
                position_size_pct=params['position_size_pct'],
                prob_threshold_long=params['prob_threshold_long'],
                prob_threshold_short=params['prob_threshold_short'],
                tp_pct=params['tp_pct'],
                sl_pct=params['sl_pct'],
                trading_hours=params['trading_hours']
            )
            
            results = backtester.run(df_test, feature_cols)
            
            if results.get('total_trades', 0) > 0:
                # 應用槓桿到報酬率
                results['total_return_pct_leveraged'] = results['total_return_pct'] * leverage
                results['leverage'] = leverage
                results['backtest_days'] = actual_backtest_days
                
                self.display_results_with_analysis(
                    backtester, results, params['long_model_path'], 
                    params['short_model_path'], model_version
                )
            else:
                st.warning("沒有交易。請檢查 logs/agent_backtester.log")
    
    def run_adaptive_backtest(self, model_version='v1', backtest_days=180, leverage=1, **params):
        logger.info(f"Starting adaptive backtest with {model_version} features, {backtest_days} days, {leverage}x leverage")
        
        with st.spinner("載入數據..."):
            df_1m = self.load_klines("BTCUSDT", "1m", backtest_days=backtest_days)
            if df_1m.empty:
                st.error("無法載入數據")
                return
            
            if 'open_time' in df_1m.columns:
                df_1m['open_time'] = pd.to_datetime(df_1m['open_time'])
                df_1m.set_index('open_time', inplace=True)
        
        # 根據版本選擇 feature engineer
        if model_version == 'v2' and V2_AVAILABLE:
            with st.spinner("生成 V2 特徵..."):
                df_features = self.feature_engineer_v2.create_features_from_1m(
                    df_1m, 
                    use_adaptive_labels=False,
                    label_type='both'
                )
                feature_cols = self.feature_engineer_v2.get_feature_list()
        else:
            with st.spinner("生成 V1 特徵..."):
                df_features = self.feature_engineer_v1.create_features_from_1m(
                    df_1m, use_micro_structure=True, label_type='both'
                )
                feature_cols = [
                    'efficiency_ratio', 'extreme_time_diff', 'vol_imbalance_ratio',
                    'z_score', 'bb_width_pct', 'rsi', 'atr_pct', 'z_score_1h', 'atr_pct_1d'
                ]
        
        # 只保留特徵
        df_features_filtered = df_features[feature_cols].copy()
        
        # 加入 OHLCV
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df_1m.columns:
                df_features_filtered[col] = df_1m[col]
        
        split_idx = int(len(df_features_filtered) * 0.8)
        df_test = df_features_filtered.iloc[split_idx:].copy()
        
        actual_backtest_days = len(df_test) / 1440
        
        st.info(f"測試集: {len(df_test):,} 筆 ({actual_backtest_days:.1f} 天) | 槓桿: {leverage}x")
        
        with st.spinner("執行自適應回測..."):
            backtester = AdaptiveBacktester(
                model_long_path=str(params['long_model_path']),
                model_short_path=str(params['short_model_path']),
                initial_capital=params['initial_capital'],
                base_position_size_pct=params['base_position_size_pct'],
                prob_threshold_long=params['prob_threshold'],
                prob_threshold_short=params['prob_threshold'],
                base_tp_pct=params['base_tp_pct'],
                base_sl_pct=params['base_sl_pct'],
                trading_hours=params['trading_hours'],
                enable_volatility_adaptation=params['enable_volatility_adaptation'],
                enable_probability_layering=params['enable_probability_layering'],
                enable_time_based_strategy=params['enable_time_based_strategy'],
                enable_risk_controls=params['enable_risk_controls'],
                max_daily_loss_pct=params['max_daily_loss_pct'],
                max_consecutive_losses=params['max_consecutive_losses']
            )
            
            results = backtester.run(df_test, feature_cols)
            
            if results.get('total_trades', 0) > 0:
                # 應用槓桿
                results['total_return_pct_leveraged'] = results['total_return_pct'] * leverage
                results['leverage'] = leverage
                results['backtest_days'] = actual_backtest_days
                
                self.display_results_with_analysis(
                    backtester, results, params['long_model_path'], 
                    params['short_model_path'], model_version
                )
            else:
                st.warning("沒有交易")
    
    def display_results_with_analysis(self, backtester, results: dict, 
                                     long_model_path, short_model_path,
                                     model_version='v1'):
        st.success("回測完成")
        
        # 顯示版本
        if model_version == 'v2':
            st.info("本次回測使用 V2 特徵 (44-54個)")
        else:
            st.info("本次回測使用 V1 特徵 (9個)")
        
        st.markdown("---")
        
        # 獲取槓桿和天數
        leverage = results.get('leverage', 1)
        backtest_days = results.get('backtest_days', 180)
        
        # 基本指標
        st.markdown("### 核心指標")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("總交易數", results['total_trades'])
        
        with col2:
            st.metric("Long / Short", f"{results['long_trades']} / {results['short_trades']}")
        
        with col3:
            win_rate = results['win_rate']
            st.metric("勝率", f"{win_rate*100:.2f}%")
        
        with col4:
            total_return = results['total_return_pct']
            st.metric(
                "總報酬 (無槓桿)", 
                f"{total_return*100:+.2f}%",
                delta="正" if total_return > 0 else "負"
            )
        
        with col5:
            # 顯示槓桿後報酬
            leveraged_return = results.get('total_return_pct_leveraged', total_return)
            st.metric(
                f"總報酬 ({leverage}x槓桿)",
                f"{leveraged_return*100:+.2f}%",
                delta=f"{leverage}x" if leverage > 1 else None
            )
        
        with col6:
            pf = results['profit_factor']
            st.metric("Profit Factor", f"{pf:.2f}")
        
        # 新增: 每日平均績效
        st.markdown("### 績效分析")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            trades_per_day = results['total_trades'] / backtest_days
            st.metric("每日交易數", f"{trades_per_day:.2f}")
        
        with col2:
            return_per_day = (leveraged_return * 100) / backtest_days
            st.metric("每日平均報酬", f"{return_per_day:+.3f}%")
        
        with col3:
            st.metric("回測天數", f"{backtest_days:.1f} 天")
        
        with col4:
            if backtest_days > 0:
                annualized_return = ((1 + leveraged_return) ** (365 / backtest_days) - 1) * 100
                st.metric("年化報酬", f"{annualized_return:+.1f}%")
        
        # 績效診斷
        st.markdown("---")
        st.markdown("### 績效診斷")
        
        # 統計顯著性檢查
        if results['total_trades'] < 30:
            st.error("❌ 樣本數不足 (<30), 統計不顯著! 請降低閾值增加交易數量")
        else:
            st.success("✅ 樣本數充足, 統計顯著")
        
        # 勝率檢查
        if win_rate < 0.40:
            st.warning(f"⚠️ 勝率偏低 ({win_rate*100:.1f}%), 建議 > 45%")
        elif win_rate > 0.55:
            st.success(f"✅ 勝率優異 ({win_rate*100:.1f}%)")
        
        # Profit Factor 檢查
        if pf < 1.2:
            st.error(f"❌ Profit Factor 過低 ({pf:.2f}), 建議 > 1.5")
        elif pf > 1.5:
            st.success(f"✅ Profit Factor 健康 ({pf:.2f})")
        
        # 報酬率檢查
        if leveraged_return * 100 < 5:
            st.warning(f"⚠️ 總報酬過低 ({leveraged_return*100:.2f}%), 考慮提高槓桿或優化策略")
        
        st.markdown("---")
        
        # 執行完整診斷分析
        st.markdown("### 詳細診斷報告")
        
        with st.spinner("生成診斷分析..."):
            trades_df = backtester.get_trades_df()
            equity_df = backtester.get_equity_curve()
            
            analyzer = BacktestAnalyzer(
                trades_df=trades_df,
                equity_df=equity_df,
                model_long_path=str(long_model_path),
                model_short_path=str(short_model_path)
            )
            
            analysis_results = analyzer.analyze_all()
            
            # 顯示優化建議
            st.markdown("#### 💡 優化建議")
            recommendations = analysis_results.get('recommendations', [])
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    st.info(f"{i}. {rec}")
            else:
                st.success("當前策略表現良好,無需特別調整")
            
            # 顯示時段分析
            st.markdown("#### 📊 時段績效分析")
            hourly = analysis_results.get('hourly_performance', pd.DataFrame())
            if not hourly.empty:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=hourly.index,
                    y=hourly['total_pnl'],
                    name='時段 PnL',
                    marker_color=['green' if x > 0 else 'red' for x in hourly['total_pnl']]
                ))
                fig.update_layout(
                    title="逐小時盈虧分布 (UTC)", 
                    xaxis_title="Hour", 
                    yaxis_title="PnL ($)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 顯示機率分層
            st.markdown("#### 🎯 機率分層分析")
            layers = analysis_results.get('probability_layers', pd.DataFrame())
            if not layers.empty:
                st.dataframe(layers, use_container_width=True)
            
            # 生成 HTML 報告
            report_dir = Path("backtest_results")
            report_dir.mkdir(exist_ok=True)
            
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            html_path = report_dir / f"diagnostic_report_{model_version}_{leverage}x_{timestamp}.html"
            
            analyzer.generate_html_report(html_path)
            
            st.success(f"✅ 完整診斷報告已保存: {html_path}")
            
            # 下載按鈕
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                trades_csv = trades_df.to_csv(index=False)
                st.download_button(
                    "📥 下載交易記錄 CSV",
                    trades_csv,
                    f"trades_{model_version}_{leverage}x_{timestamp}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                st.download_button(
                    "📥 下載診斷報告 HTML",
                    html_content,
                    f"report_{model_version}_{leverage}x_{timestamp}.html",
                    "text/html",
                    use_container_width=True
                )