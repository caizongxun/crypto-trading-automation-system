import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import threading
import queue

sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from utils.logger import setup_logger

try:
    from train_v2 import AdvancedTrainer
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False

logger = setup_logger('model_training_v2_tab', 'logs/model_training_v2_tab.log')

class ModelTrainingV2Tab:
    """
    V2 模型訓練標籤頁
    """
    
    def __init__(self):
        if 'training_status' not in st.session_state:
            st.session_state.training_status = 'idle'
        if 'training_results' not in st.session_state:
            st.session_state.training_results = None
        if 'training_logs' not in st.session_state:
            st.session_state.training_logs = []
    
    def render(self):
        st.header("🧠 V2 模型訓練 (進階版)")
        
        if not V2_AVAILABLE:
            st.error("⚠️ V2 系統未安裝")
            st.info("請執行: python upgrade_to_v2.py")
            return
        
        # 左右分欄
        col1, col2 = st.columns([1, 2])
        
        with col1:
            self.render_config_panel()
        
        with col2:
            self.render_results_panel()
    
    def render_config_panel(self):
        """配置面板"""
        st.subheader("⚙️ 訓練配置")
        
        # 訓練模式
        training_mode = st.radio(
            "🎯 訓練模式",
            options=[
                "🚀 快速測試 (30-60分鐘)",
                "🏆 完整訓練 (2-4小時)",
                "💡 自定義"
            ],
            index=0,
            help="快速測試不含超參優化,適合初步驗證"
        )
        
        st.markdown("---")
        
        # 根據模式顯示不同配置
        if training_mode == "🚀 快速測試 (30-60分鐘)":
            st.info("👍 快速模式配置")
            st.markdown("""
            - 超參優化: 關閉
            - 集成學習: 啟用
            - Walk-Forward: 關閉
            - 預計時間: 30-60分鐘
            """)
            
            enable_hyperopt = False
            enable_ensemble = True
            enable_walk_forward = False
            n_trials = 0
        
        elif training_mode == "🏆 完整訓練 (2-4小時)":
            st.success("🌟 完整模式配置")
            st.markdown("""
            - 超參優化: 啟用 (50 trials)
            - 集成學習: 啟用
            - Walk-Forward: 啟用 (5-fold)
            - 預計時間: 2-4小時
            """)
            
            enable_hyperopt = True
            enable_ensemble = True
            enable_walk_forward = True
            n_trials = 50
        
        else:  # 自定義
            st.warning("🔧 自定義配置")
            
            enable_hyperopt = st.checkbox(
                "🔍 啟用 Optuna 超參優化",
                value=True,
                help="自動搜索最佳參數,需要較長時間"
            )
            
            if enable_hyperopt:
                n_trials = st.slider(
                    "Optuna Trials",
                    min_value=10,
                    max_value=100,
                    value=50,
                    step=10
                )
            else:
                n_trials = 0
            
            enable_ensemble = st.checkbox(
                "🤝 啟用集成學習",
                value=True,
                help="CatBoost + XGBoost 集成"
            )
            
            enable_walk_forward = st.checkbox(
                "📊 啟用 Walk-Forward 驗證",
                value=True,
                help="5-fold 時序驗證,確保穩定性"
            )
        
        st.markdown("---")
        
        # 特徵配置
        st.subheader("🎁 特徵配置")
        
        enable_advanced = st.checkbox(
            "🚀 啟用進階特徵",
            value=True,
            help="包含微觀結構 + MTF (额外 25 個特徵)"
        )
        
        enable_ml = st.checkbox(
            "🧠 啟用 ML 特徵",
            value=True,
            help="包含特徵交互 + 聚類 (额外 10 個特徵)"
        )
        
        total_features = 19  # 基礎 + 訂單流
        if enable_advanced:
            total_features += 25
        if enable_ml:
            total_features += 10
        
        st.info(f"📊 預計特徵數: **{total_features}** 個")
        
        st.markdown("---")
        
        # 開始訓練按鈕
        if st.session_state.training_status == 'idle':
            if st.button("🚀 開始訓練", type="primary", use_container_width=True):
                self.start_training(
                    enable_hyperopt=enable_hyperopt,
                    enable_ensemble=enable_ensemble,
                    enable_walk_forward=enable_walk_forward,
                    n_trials=n_trials,
                    enable_advanced=enable_advanced,
                    enable_ml=enable_ml
                )
        
        elif st.session_state.training_status == 'running':
            st.warning("⏳ 訓練進行中...")
            if st.button("⛔ 停止訓練", use_container_width=True):
                st.session_state.training_status = 'stopped'
                st.rerun()
        
        elif st.session_state.training_status == 'completed':
            st.success("✅ 訓練完成!")
            if st.button("🔄 重新訓練", use_container_width=True):
                st.session_state.training_status = 'idle'
                st.session_state.training_results = None
                st.session_state.training_logs = []
                st.rerun()
    
    def render_results_panel(self):
        """結果面板"""
        
        if st.session_state.training_status == 'idle':
            st.info("👈 請在左側配置並開始訓練")
            
            # 顯示 V2 優勢
            st.subheader("🌟 V2 系統優勢")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "特徵數",
                    "44-54",
                    "+388%",
                    help="V1: 9 個"
                )
            
            with col2:
                st.metric(
                    "交易數",
                    "80-120",
                    "+116%",
                    help="V1: 37 筆"
                )
            
            with col3:
                st.metric(
                    "勝率",
                    "42-48%",
                    "+11%",
                    help="V1: 37.84%"
                )
            
            with col4:
                st.metric(
                    "Profit Factor",
                    "1.45-1.65",
                    "+19%",
                    help="V1: 1.22"
                )
            
            st.markdown("---")
            
            # 特徵對比表
            st.subheader("📊 特徵對比")
            
            comparison_df = pd.DataFrame({
                '特徵類型': [
                    '基礎技術指標',
                    '訂單流特徵',
                    '微觀結構',
                    '多時間框架',
                    'ML衍生特徵'
                ],
                'V1': [9, 0, 0, 0, 0],
                'V2': [9, 10, 10, 15, 10]
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='V1', x=comparison_df['特徵類型'], y=comparison_df['V1']))
            fig.add_trace(go.Bar(name='V2', x=comparison_df['特徵類型'], y=comparison_df['V2']))
            fig.update_layout(barmode='group', height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            return
        
        elif st.session_state.training_status == 'running':
            st.subheader("🔄 訓練進度")
            
            # 進度條
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 實時日誌
            st.subheader("📝 實時日誌")
            log_container = st.container()
            
            with log_container:
                if st.session_state.training_logs:
                    for log in st.session_state.training_logs[-20:]:
                        st.text(log)
            
            return
        
        elif st.session_state.training_status == 'completed':
            results = st.session_state.training_results
            
            if results is None:
                st.error("⚠️ 訓練結果遺失")
                return
            
            # 顯示結果
            self.render_training_results(results)
    
    def render_training_results(self, results):
        """顯示訓練結果"""
        st.subheader("🏆 訓練結果")
        
        # 1. 模型路徑
        st.success("✅ 模型已保存")
        col1, col2 = st.columns(2)
        
        with col1:
            st.code(f"Long: {results['long_path'].name}", language="text")
        
        with col2:
            st.code(f"Short: {results['short_path'].name}", language="text")
        
        st.markdown("---")
        
        # 2. 模型性能
        st.subheader("📊 模型性能")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Long Oracle")
            st.metric("AUC", f"{results['eval_long']['auc']:.4f}")
            
            # Threshold 分析
            threshold_metrics = results['eval_long']['threshold_metrics']
            
            threshold_df = pd.DataFrame([
                {
                    'Threshold': th,
                    'Precision': f"{m['precision']*100:.1f}%",
                    'Recall': f"{m['recall']:.2f}%",
                    'Samples': m['samples']
                }
                for th, m in threshold_metrics.items()
            ])
            
            st.dataframe(threshold_df, use_container_width=True)
        
        with col2:
            st.markdown("### Short Oracle")
            st.metric("AUC", f"{results['eval_short']['auc']:.4f}")
            
            threshold_metrics = results['eval_short']['threshold_metrics']
            
            threshold_df = pd.DataFrame([
                {
                    'Threshold': th,
                    'Precision': f"{m['precision']*100:.1f}%",
                    'Recall': f"{m['recall']:.2f}%",
                    'Samples': m['samples']
                }
                for th, m in threshold_metrics.items()
            ])
            
            st.dataframe(threshold_df, use_container_width=True)
        
        st.markdown("---")
        
        # 3. Walk-Forward 結果
        if results.get('walk_forward') is not None:
            st.subheader("📊 Walk-Forward 驗證")
            
            wf_df = results['walk_forward']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Average AUC",
                    f"{wf_df['auc'].mean():.4f}",
                    f"±{wf_df['auc'].std():.4f}"
                )
            
            with col2:
                st.metric(
                    "Average Precision@0.16",
                    f"{wf_df['precision_016'].mean()*100:.1f}%"
                )
            
            # 結果表格
            st.dataframe(wf_df, use_container_width=True)
            
            # AUC 趋勢圖
            fig = px.line(
                wf_df,
                x='fold',
                y='auc',
                title='Walk-Forward AUC 趋勢',
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 4. 下一步
        st.subheader("🚀 下一步")
        
        st.info("""
        🎯 **建議流程**:
        
        1. ✅ 前往 "策略回測" 標籤頁
        2. 📊 選擇剛才訓練的 V2 模型
        3. 🚀 執行回測驗證
        4. 📊 對比 V1 vs V2 性能
        5. ✅ 如果 Profit Factor > 1.4 -> Paper Trading
        """)
        
        if st.button("📊 前往回測頁面", type="primary", use_container_width=True):
            st.switch_page("pages/backtesting.py")
    
    def start_training(self, enable_hyperopt, enable_ensemble, 
                      enable_walk_forward, n_trials,
                      enable_advanced, enable_ml):
        """開始訓練"""
        st.session_state.training_status = 'running'
        st.session_state.training_logs = []
        
        try:
            # 初始化 trainer
            from utils.feature_engineering_v2 import FeatureEngineerV2
            
            # 更新 FeatureEngineer 配置
            FeatureEngineerV2.__init__.__defaults__ = (
                enable_advanced,
                enable_ml
            )
            
            trainer = AdvancedTrainer(
                enable_hyperopt=enable_hyperopt,
                enable_ensemble=enable_ensemble,
                enable_walk_forward=enable_walk_forward,
                n_trials=n_trials
            )
            
            # 執行訓練
            results = trainer.run()
            
            # 保存結果
            st.session_state.training_results = results
            st.session_state.training_status = 'completed'
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            st.session_state.training_status = 'error'
            st.error(f"⚠️ 訓練失敗: {str(e)}")
        
        st.rerun()