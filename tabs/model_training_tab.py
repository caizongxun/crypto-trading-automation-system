import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger
from utils.model_trainer import ModelTrainer

logger = setup_logger('model_training_tab', 'logs/model_training_tab.log')

class ModelTrainingTab:
    def __init__(self):
        logger.info("Initializing ModelTrainingTab")
        self.trainer = ModelTrainer()
    
    def render(self):
        logger.info("Rendering Model Training Tab")
        st.header("模型訓練")
        
        st.markdown("""
        ### LightGBM 訓練架構 - 防過擬合與時間序列切分
        
        **關鍵特性**:
        - 嚴格 80/20 時間序列切分 (絕不打亂)
        - Scale_pos_weight 自動計算 (處理不平衡)
        - 特徵重要性監控
        - AUC + Recall 雙指標追蹤
        - OOS 樣本外驗證
        """)
        
        st.markdown("---")
        
        # 選擇特徵檔案
        features_dir = Path("features_output")
        if not features_dir.exists():
            st.warning("請先在 '特徵工程' Tab 中生成特徵檔案")
            return
        
        feature_files = list(features_dir.glob("features_*_multi_tf.parquet"))
        if not feature_files:
            st.warning("沒有找到特徵檔案")
            return
        
        file_options = [f.name for f in feature_files]
        selected_file = st.selectbox("選擇特徵檔案", file_options)
        
        st.markdown("---")
        
        # 超參數設定
        st.subheader("超參數設定")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
                value=0.02
            )
            
            max_depth = st.slider(
                "Max Depth",
                min_value=3,
                max_value=10,
                value=6
            )
        
        with col2:
            num_leaves = st.slider(
                "Num Leaves",
                min_value=15,
                max_value=127,
                value=31
            )
            
            min_child_samples = st.slider(
                "Min Child Samples",
                min_value=20,
                max_value=200,
                value=50
            )
        
        with col3:
            n_estimators = st.slider(
                "N Estimators",
                min_value=100,
                max_value=1000,
                value=500
            )
            
            early_stopping = st.slider(
                "Early Stopping Rounds",
                min_value=20,
                max_value=100,
                value=50
            )
        
        st.markdown("---")
        
        # 訓練按鈕
        if st.button("開始訓練模型", use_container_width=True):
            self.train_model(
                features_dir / selected_file,
                learning_rate,
                max_depth,
                num_leaves,
                min_child_samples,
                n_estimators,
                early_stopping
            )
    
    def train_model(self, feature_file: Path, learning_rate: float, max_depth: int,
                   num_leaves: int, min_child_samples: int, n_estimators: int,
                   early_stopping: int):
        logger.info(f"Starting model training with {feature_file}")
        
        with st.spinner("載入特徵檔案..."):
            try:
                features_df = pd.read_parquet(feature_file)
                st.success(f"載入 {len(features_df):,} 筆資料")
                logger.info(f"Loaded {len(features_df)} records from {feature_file}")
            except Exception as e:
                st.error(f"載入檔案失敗: {str(e)}")
                logger.error(f"Failed to load {feature_file}: {str(e)}")
                return
        
        # 配置參數
        params = {
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'min_child_samples': min_child_samples,
            'n_estimators': n_estimators,
            'early_stopping_rounds': early_stopping
        }
        
        with st.spinner("訓練模型中... (可能需要幾分鐘)"):
            try:
                results = self.trainer.train(
                    features_df,
                    params
                )
                
                if results:
                    self.display_results(results, feature_file.stem)
                else:
                    st.error("模型訓練失敗")
            
            except Exception as e:
                logger.error(f"Training error: {str(e)}", exc_info=True)
                st.error(f"訓練失敗: {str(e)}")
    
    def display_results(self, results: dict, model_name: str):
        logger.info("Displaying training results")
        
        st.success("模型訓練完成")
        
        # 核心指標
        st.subheader("核心指標")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            train_auc = results['metrics']['train_auc']
            st.metric(
                "訓練集 AUC",
                f"{train_auc:.4f}",
                delta="目標: 0.75+"
            )
        
        with col2:
            test_auc = results['metrics']['test_auc']
            auc_diff = train_auc - test_auc
            st.metric(
                "OOS 測試集 AUC",
                f"{test_auc:.4f}",
                delta=f"-{auc_diff:.4f}" if auc_diff > 0 else f"+{abs(auc_diff):.4f}"
            )
        
        with col3:
            recall = results['metrics']['test_recall']
            st.metric(
                "OOS Recall",
                f"{recall:.4f}",
                delta="目標: 0.55+"
            )
        
        with col4:
            precision = results['metrics']['test_precision']
            st.metric(
                "OOS Precision",
                f"{precision:.4f}"
            )
        
        # 評估
        st.markdown("---")
        st.subheader("模型評估")
        
        if test_auc >= 0.75:
            st.success(f"AUC 達標: {test_auc:.4f} >= 0.75")
        else:
            st.warning(f"AUC 未達標: {test_auc:.4f} < 0.75 (差距: {0.75 - test_auc:.4f})")
        
        if recall >= 0.55:
            st.success(f"Recall 達標: {recall:.4f} >= 0.55")
        else:
            st.warning(f"Recall 未達標: {recall:.4f} < 0.55 (差距: {0.55 - recall:.4f})")
        
        # 過擬合檢測
        if auc_diff > 0.05:
            st.error(f"警告: 過擬合風險高 (AUC 差異: {auc_diff:.4f})")
        elif auc_diff > 0.02:
            st.warning(f"注意: 輕度過擬合 (AUC 差異: {auc_diff:.4f})")
        else:
            st.success(f"模型泛化良好 (AUC 差異: {auc_diff:.4f})")
        
        # 特徵重要性
        st.markdown("---")
        st.subheader("特徵重要性 Top 15")
        
        importance_df = pd.DataFrame({
            '特徵': results['feature_importance']['features'],
            '重要性': results['feature_importance']['importances']
        }).head(15)
        
        st.bar_chart(importance_df.set_index('特徵'))
        st.dataframe(importance_df)
        
        # 混淆矩陣
        st.markdown("---")
        st.subheader("混淆矩陣 (OOS)")
        
        cm = results['confusion_matrix']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("True Negative (TN)", f"{cm['tn']:,}")
            st.metric("False Positive (FP)", f"{cm['fp']:,}")
        with col2:
            st.metric("False Negative (FN)", f"{cm['fn']:,}")
            st.metric("True Positive (TP)", f"{cm['tp']:,}")
        
        # 保存模型
        st.markdown("---")
        
        model_dir = Path("models_output")
        model_path = results['model_path']
        
        st.success(f"模型已保存: {model_path}")
        
        # 保存訓練報告
        report_path = model_dir / f"{model_name}_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        st.info(f"訓練報告已保存: {report_path}")
        
        logger.info(f"Training completed: AUC={test_auc:.4f}, Recall={recall:.4f}")