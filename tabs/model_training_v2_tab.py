import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import subprocess
import time
import json
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(str(Path(__file__).parent.parent))
from config import Config
from utils.logger import setup_logger

try:
    from utils.feature_engineering_v2 import FeatureEngineerV2
    from huggingface_hub import hf_hub_download
    import catboost
    V2_AVAILABLE = True
except ImportError as e:
    V2_AVAILABLE = False
    IMPORT_ERROR = str(e)

logger = setup_logger('model_training_v2_tab', 'logs/model_training_v2_tab.log')

class ModelTrainingV2Tab:
    def __init__(self):
        logger.info("Initializing ModelTrainingV2Tab")
        if V2_AVAILABLE:
            self.feature_engineer = FeatureEngineerV2(
                enable_advanced_features=True,
                enable_ml_features=True
            )
    
    def render(self):
        logger.info("Rendering Model Training V2 Tab")
        st.header("V2 模型訓練 - 進階特徵系統")
        
        # 檢查依賴
        if not V2_AVAILABLE:
            st.error(f"V2 系統不可用: {IMPORT_ERROR}")
            st.info("請執行: pip install optuna")
            return
        
        st.success("V2 系統已就緒")
        st.markdown("---")
        
        # 訓練模式選擇
        st.subheader("訓練模式")
        
        training_mode = st.radio(
            "選擇訓練模式",
            options=['quick_test', 'full_training', 'custom'],
            format_func=lambda x: {
                'quick_test': '快速測試 (30-60分鐘)',
                'full_training': '完整訓練 (2-4小時)',
                'custom': '自定義配置'
            }[x],
            key="v2_training_mode"
        )
        
        # 模式說明
        if training_mode == 'quick_test':
            st.info(
                "快速測試模式:\n"
                "- 數據量: 最近 7 天\n"
                "- Optuna trials: 20\n"
                "- Walk-Forward: 3 folds\n"
                "- 適合驗證可行性"
            )
        elif training_mode == 'full_training':
            st.info(
                "完整訓練模式:\n"
                "- 數據量: 全部數據\n"
                "- Optuna trials: 50\n"
                "- Walk-Forward: 5 folds\n"
                "- 適合正式部署"
            )
        
        st.markdown("---")
        
        # 特徵配置
        st.subheader("特徵配置")
        
        col1, col2 = st.columns(2)
        
        with col1:
            enable_advanced = st.checkbox(
                "啟用進階特徵 (25個)",
                value=True,
                key="v2_advanced",
                help="包含: 微觀結構 + 多時間框架特徵"
            )
        
        with col2:
            enable_ml = st.checkbox(
                "啟用 ML 特徵 (10個)",
                value=True,
                key="v2_ml",
                help="包含: 特徵交互 + 聚類 + 統計特徵"
            )
        
        # 計算總特徵數
        base_features = 9 + 10  # 基礎 + 訂單流
        total_features = base_features
        if enable_advanced:
            total_features += 25
        if enable_ml:
            total_features += 10
        
        st.info(f"預計生成 {total_features} 個特徵")
        
        st.markdown("---")
        
        # 自定義配置
        if training_mode == 'custom':
            st.subheader("自定義參數")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                data_days = st.number_input(
                    "數據天數",
                    min_value=1,
                    max_value=365,
                    value=30,
                    key="v2_data_days"
                )
            
            with col2:
                optuna_trials = st.number_input(
                    "Optuna Trials",
                    min_value=10,
                    max_value=100,
                    value=30,
                    key="v2_optuna_trials"
                )
            
            with col3:
                wf_folds = st.number_input(
                    "Walk-Forward Folds",
                    min_value=2,
                    max_value=10,
                    value=4,
                    key="v2_wf_folds"
                )
        else:
            # 自動配置
            if training_mode == 'quick_test':
                data_days = 7
                optuna_trials = 20
                wf_folds = 3
            else:  # full_training
                data_days = 365
                optuna_trials = 50
                wf_folds = 5
        
        st.markdown("---")
        
        # 訓練按鈕
        if st.button("開始訓練", use_container_width=True, type="primary", key="v2_start_training"):
            self.run_training(
                enable_advanced=enable_advanced,
                enable_ml=enable_ml,
                data_days=data_days,
                optuna_trials=optuna_trials,
                wf_folds=wf_folds
            )
    
    def load_klines(self, symbol: str, timeframe: str, days: int = None) -> pd.DataFrame:
        """
        載入 K 線數據
        """
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
            
            # 限制天數
            if days is not None:
                df = df.tail(days * 1440)  # 1440 = 1天的分鐘數
            
            logger.info(f"Loaded {len(df):,} records")
            return df
        
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            st.error(f"載入數據失敗: {str(e)}")
            return pd.DataFrame()
    
    def run_training(self, enable_advanced: bool, enable_ml: bool,
                    data_days: int, optuna_trials: int, wf_folds: int):
        """
        執行完整訓練流程
        """
        logger.info("Starting V2 training process")
        
        # 創建進度容器
        progress_container = st.container()
        
        with progress_container:
            st.markdown("### 訓練進度")
            
            # 步驟 1: 載入數據
            st.markdown("**步驟 1/5: 載入數據**")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("載入 1m K線...")
            df_1m = self.load_klines("BTCUSDT", "1m", days=data_days)
            
            if df_1m.empty:
                st.error("數據載入失敗")
                return
            
            progress_bar.progress(20)
            status_text.text(f"已載入 {len(df_1m):,} 筆數據")
            
            # 步驟 2: 特徵工程
            st.markdown("**步驟 2/5: 特徵工程**")
            
            # 更新特徵配置
            self.feature_engineer.enable_advanced = enable_advanced
            self.feature_engineer.enable_ml = enable_ml
            
            status_text.text("生成特徵...")
            
            if 'open_time' in df_1m.columns:
                df_1m['open_time'] = pd.to_datetime(df_1m['open_time'])
                df_1m.set_index('open_time', inplace=True)
            
            df_features = self.feature_engineer.create_features_from_1m(
                df_1m,
                use_adaptive_labels=True,
                label_type='both'
            )
            
            progress_bar.progress(40)
            
            feature_cols = self.feature_engineer.get_feature_list()
            status_text.text(f"已生成 {len(feature_cols)} 個特徵")
            
            # 步驟 3: 調用 train_v2.py
            st.markdown("**步驟 3/5: 模型訓練 (Optuna 優化)**")
            status_text.text(f"執行 {optuna_trials} trials 超參數搜索...")
            
            # 準備參數
            cmd = [
                sys.executable,
                "train_v2.py",
                "--enable_advanced", str(enable_advanced),
                "--enable_ml", str(enable_ml),
                "--optuna_trials", str(optuna_trials),
                "--wf_folds", str(wf_folds),
                "--data_days", str(data_days)
            ]
            
            # 執行訓練
            try:
                # 創建日誌顯示區域
                log_expander = st.expander("訓練日誌", expanded=True)
                log_placeholder = log_expander.empty()
                
                # 執行訓練腳本
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                # 實時顯示輸出
                output_lines = []
                for line in process.stdout:
                    output_lines.append(line)
                    # 只顯示最後 20 行
                    log_placeholder.code(''.join(output_lines[-20:]))
                
                process.wait()
                
                if process.returncode != 0:
                    st.error(f"訓練失敗,返回碼: {process.returncode}")
                    return
                
                progress_bar.progress(80)
                status_text.text("訓練完成,分析結果...")
                
            except Exception as e:
                st.error(f"訓練過程出錯: {str(e)}")
                logger.error(f"Training failed: {str(e)}")
                return
            
            # 步驟 4: 載入結果
            st.markdown("**步驟 4/5: 載入訓練結果**")
            
            models_dir = Path("models_output")
            
            # 找到最新的 V2 模型
            v2_models = list(models_dir.glob("catboost_*_v2_*.pkl"))
            if v2_models:
                latest_model = max(v2_models, key=lambda x: x.stat().st_mtime)
                status_text.text(f"找到模型: {latest_model.name}")
            else:
                st.warning("未找到訓練好的模型")
                return
            
            progress_bar.progress(90)
            
            # 步驟 5: 顯示結果
            st.markdown("**步驟 5/5: 結果分析**")
            
            # 載入訓練報告
            reports_dir = Path("training_reports")
            if reports_dir.exists():
                report_files = list(reports_dir.glob("v2_training_report_*.json"))
                if report_files:
                    latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
                    
                    with open(latest_report, 'r') as f:
                        report = json.load(f)
                    
                    self.display_training_results(report)
            
            progress_bar.progress(100)
            status_text.text("訓練流程完成!")
            
            st.success("V2 模型訓練完成!")
            st.info("現在可以前往回測標籤測試模型性能")
    
    def display_training_results(self, report: dict):
        """
        顯示訓練結果
        """
        st.markdown("---")
        st.subheader("訓練結果")
        
        # 基本信息
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("特徵數", report.get('total_features', 'N/A'))
        
        with col2:
            st.metric("訓練樣本", f"{report.get('train_samples', 0):,}")
        
        with col3:
            st.metric("測試樣本", f"{report.get('test_samples', 0):,}")
        
        with col4:
            training_time = report.get('training_time_minutes', 0)
            st.metric("訓練時間", f"{training_time:.1f} 分鐘")
        
        st.markdown("---")
        
        # Long 和 Short 性能
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Long Oracle")
            long_metrics = report.get('long_metrics', {})
            
            st.metric("AUC", f"{long_metrics.get('auc', 0):.4f}")
            st.metric("Precision@0.16", f"{long_metrics.get('precision_at_016', 0)*100:.2f}%")
            st.metric("Recall@0.16", f"{long_metrics.get('recall_at_016', 0)*100:.2f}%")
        
        with col2:
            st.markdown("#### Short Oracle")
            short_metrics = report.get('short_metrics', {})
            
            st.metric("AUC", f"{short_metrics.get('auc', 0):.4f}")
            st.metric("Precision@0.16", f"{short_metrics.get('precision_at_016', 0)*100:.2f}%")
            st.metric("Recall@0.16", f"{short_metrics.get('recall_at_016', 0)*100:.2f}%")
        
        st.markdown("---")
        
        # Walk-Forward 結果
        st.markdown("#### Walk-Forward 驗證")
        
        wf_results = report.get('walk_forward_results', {})
        if wf_results:
            long_wf = wf_results.get('long', {})
            short_wf = wf_results.get('short', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Long WF AUC**")
                if 'fold_scores' in long_wf:
                    for i, score in enumerate(long_wf['fold_scores'], 1):
                        st.text(f"Fold {i}: {score:.4f}")
                    st.text(f"Average: {long_wf.get('mean_auc', 0):.4f} ± {long_wf.get('std_auc', 0):.4f}")
            
            with col2:
                st.markdown("**Short WF AUC**")
                if 'fold_scores' in short_wf:
                    for i, score in enumerate(short_wf['fold_scores'], 1):
                        st.text(f"Fold {i}: {score:.4f}")
                    st.text(f"Average: {short_wf.get('mean_auc', 0):.4f} ± {short_wf.get('std_auc', 0):.4f}")
        
        st.markdown("---")
        
        # 最佳超參數
        st.markdown("#### Optuna 最佳超參數")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Long Oracle**")
            long_params = report.get('long_best_params', {})
            for key, value in long_params.items():
                st.text(f"{key}: {value}")
        
        with col2:
            st.markdown("**Short Oracle**")
            short_params = report.get('short_best_params', {})
            for key, value in short_params.items():
                st.text(f"{key}: {value}")
        
        st.markdown("---")
        
        # 特徵重要性 Top 10
        st.markdown("#### 特徵重要性 Top 10")
        
        long_importance = report.get('long_feature_importance', {})
        short_importance = report.get('short_feature_importance', {})
        
        if long_importance or short_importance:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Long Oracle**")
                if long_importance:
                    # 取前10個
                    top_features = dict(sorted(
                        long_importance.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:10])
                    
                    fig = go.Figure(go.Bar(
                        x=list(top_features.values()),
                        y=list(top_features.keys()),
                        orientation='h'
                    ))
                    fig.update_layout(
                        height=400,
                        margin=dict(l=150, r=20, t=20, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Short Oracle**")
                if short_importance:
                    top_features = dict(sorted(
                        short_importance.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:10])
                    
                    fig = go.Figure(go.Bar(
                        x=list(top_features.values()),
                        y=list(top_features.keys()),
                        orientation='h'
                    ))
                    fig.update_layout(
                        height=400,
                        margin=dict(l=150, r=20, t=20, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)