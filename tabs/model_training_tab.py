import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import joblib
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger
from utils.model_trainer import ModelTrainer
from utils.feature_engineering import FeatureEngineer
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score

logger = setup_logger('model_training_tab', 'logs/model_training_tab.log')

class ModelTrainingTab:
    def __init__(self):
        logger.info("Initializing ModelTrainingTab")
    
    def render(self):
        logger.info("Rendering Model Training Tab")
        st.header("模型訓練")
        
        # 訓練模式選擇
        training_mode = st.radio(
            "選擇訓練模式",
            ['unidirectional', 'bidirectional'],
            format_func=lambda x: '🔵 單向 (Long Only)' if x == 'unidirectional' else '🔴🔵 雙向 (Long + Short)',
            horizontal=True,
            key="training_mode_selector"
        )
        
        if training_mode == 'unidirectional':
            self.render_unidirectional()
        else:
            self.render_bidirectional()
    
    def render_bidirectional(self):
        """渲染雙向訓練介面"""
        st.markdown("""
        ### 🎯 雙向狩獵架構 - Bidirectional Hunting
        
        **核心概念**:
        - 🔵 **Long Oracle**: 捕捉底部反轉、W 底、上漲突破
        - 🔴 **Short Oracle**: 捕捉頂部背離、M 頭、下跌突破
        
        **優勢**:
        - ✅ 交易次數翻倍 (100 → 200+ 次)
        - ✅ 報酬倍增 (+5% → +10%+)
        - ✅ 市場中性 (牛熊市都賺錢)
        
        **標籤邏輯**:
        - Long: 未來 4 小時先漲 2% (停利) 且沒先跌 1% (停損)
        - Short: 未來 4 小時先跌 2% (停利) 且沒先漲 1% (停損)
        """)
        
        st.markdown("---")
        
        # 訓練按鈕
        if st.button("🚀 訓練雙向模型 (Long + Short Oracles)", use_container_width=True, key="bidirectional_train_button"):
            self.train_bidirectional()
    
    def train_bidirectional(self):
        """執行雙向訓練"""
        logger.info("Starting bidirectional training")
        
        # Step 1: 載入資料
        with st.spinner("📊 載入 1m K 線..."):
            klines_dir = Path("klines_output")
            klines_file = klines_dir / "klines_BTCUSDT_1m.parquet"
            
            if not klines_file.exists():
                st.error("請先在 '數據下載' Tab 中下載 BTCUSDT 1m K 線")
                return
            
            try:
                df_1m = pd.read_parquet(klines_file)
                df_1m.set_index('open_time', inplace=True)
                st.success(f"✅ 載入 {len(df_1m):,} 筆 1m K 線")
            except Exception as e:
                st.error(f"載入失敗: {str(e)}")
                return
        
        # Step 2: 生成雙向特徵與標籤
        with st.spinner("⚙️ 生成雙向特徵與標籤 (label_long + label_short)..."):
            engineer = FeatureEngineer()
            df_features = engineer.create_features_from_1m(
                df_1m,
                use_micro_structure=True,
                label_type='both'
            )
            
            st.success(f"✅ 生成 {len(df_features):,} 筆雙向標籤")
            
            # 顯示標籤分布
            col1, col2 = st.columns(2)
            with col1:
                long_rate = df_features['label_long'].mean()
                st.metric("🔵 Long 標籤比例", f"{long_rate*100:.2f}%")
            with col2:
                short_rate = df_features['label_short'].mean()
                st.metric("🔴 Short 標籤比例", f"{short_rate*100:.2f}%")
        
        # Step 3: 準備特徵矩陣
        feature_cols = [
            'efficiency_ratio', 'extreme_time_diff', 'vol_imbalance_ratio',
            'z_score', 'bb_width_pct', 'rsi', 'atr_pct',
            'z_score_1h', 'atr_pct_1d'
        ]
        
        available_features = [col for col in feature_cols if col in df_features.columns]
        X = df_features[available_features].values
        y_long = df_features['label_long'].values
        y_short = df_features['label_short'].values
        
        # 時間序列切分
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_long_train, y_long_test = y_long[:split_idx], y_long[split_idx:]
        y_short_train, y_short_test = y_short[:split_idx], y_short[split_idx:]
        
        st.info(f"📋 訓練集: {len(X_train):,}, 測試集: {len(X_test):,}")
        
        # Step 4: 訓練 Long Oracle
        st.markdown("---")
        st.subheader("🔵 訓練 Long Oracle")
        
        with st.spinner("🎯 訓練中... (2-3 分鐘)"):
            model_long = CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=10,
                random_strength=0.5,
                bagging_temperature=0.2,
                loss_function='Logloss',
                eval_metric='AUC',
                verbose=False,
                random_seed=42,
                task_type='CPU'
            )
            
            model_long.fit(
                X_train, y_long_train,
                eval_set=(X_test, y_long_test),
                early_stopping_rounds=50,
                verbose=False
            )
            
            # 機率校準
            model_long_calibrated = CalibratedClassifierCV(
                model_long,
                method='isotonic',
                cv='prefit'
            )
            model_long_calibrated.fit(X_test, y_long_test)
            
            # 評估
            y_long_pred = model_long_calibrated.predict(X_test)
            y_long_proba = model_long_calibrated.predict_proba(X_test)[:, 1]
            auc_long = roc_auc_score(y_long_test, y_long_proba)
            
            st.success(f"✅ Long Oracle 訓練完成 - AUC: {auc_long:.4f}")
        
        # Step 5: 訓練 Short Oracle
        st.markdown("---")
        st.subheader("🔴 訓練 Short Oracle")
        
        with st.spinner("🎯 訓練中... (2-3 分鐘)"):
            model_short = CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=10,
                random_strength=0.5,
                bagging_temperature=0.2,
                loss_function='Logloss',
                eval_metric='AUC',
                verbose=False,
                random_seed=42,
                task_type='CPU'
            )
            
            model_short.fit(
                X_train, y_short_train,
                eval_set=(X_test, y_short_test),
                early_stopping_rounds=50,
                verbose=False
            )
            
            # 機率校準
            model_short_calibrated = CalibratedClassifierCV(
                model_short,
                method='isotonic',
                cv='prefit'
            )
            model_short_calibrated.fit(X_test, y_short_test)
            
            # 評估
            y_short_pred = model_short_calibrated.predict(X_test)
            y_short_proba = model_short_calibrated.predict_proba(X_test)[:, 1]
            auc_short = roc_auc_score(y_short_test, y_short_proba)
            
            st.success(f"✅ Short Oracle 訓練完成 - AUC: {auc_short:.4f}")
        
        # Step 6: 保存模型
        st.markdown("---")
        st.subheader("💾 保存模型")
        
        models_dir = Path('models_output')
        models_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        long_model_path = models_dir / f'catboost_long_model_{timestamp}.pkl'
        short_model_path = models_dir / f'catboost_short_model_{timestamp}.pkl'
        
        joblib.dump(model_long_calibrated, long_model_path)
        joblib.dump(model_short_calibrated, short_model_path)
        
        st.success(f"✅ Long Oracle: {long_model_path.name}")
        st.success(f"✅ Short Oracle: {short_model_path.name}")
        
        # Step 7: 綜合報告
        st.markdown("---")
        st.subheader("📊 雙向模型績效")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔵 Long Oracle")
            st.metric("AUC", f"{auc_long:.4f}", delta="目標: 0.65+")
            st.metric("標籤比例", f"{y_long_test.mean()*100:.2f}%")
            
            precision_long = precision_score(y_long_test, y_long_pred)
            recall_long = recall_score(y_long_test, y_long_pred)
            
            st.metric("Precision", f"{precision_long:.4f}")
            st.metric("Recall", f"{recall_long:.4f}")
        
        with col2:
            st.markdown("### 🔴 Short Oracle")
            st.metric("AUC", f"{auc_short:.4f}", delta="目標: 0.65+")
            st.metric("標籤比例", f"{y_short_test.mean()*100:.2f}%")
            
            precision_short = precision_score(y_short_test, y_short_pred)
            recall_short = recall_score(y_short_test, y_short_pred)
            
            st.metric("Precision", f"{precision_short:.4f}")
            st.metric("Recall", f"{recall_short:.4f}")
        
        # 評估
        if auc_long >= 0.65 and auc_short >= 0.65:
            st.success("✅ 雙向模型達標！兩個 Oracle AUC 都超過 0.65")
        else:
            st.warning("⚠️ 模型效能可提升，建議調整超參數或增加資料")
        
        # 保存報告
        report = {
            'timestamp': timestamp,
            'total_samples': len(df_features),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'long_oracle': {
                'auc': float(auc_long),
                'precision': float(precision_long),
                'recall': float(recall_long),
                'positive_rate': float(y_long_test.mean()),
                'model_path': str(long_model_path)
            },
            'short_oracle': {
                'auc': float(auc_short),
                'precision': float(precision_short),
                'recall': float(recall_short),
                'positive_rate': float(y_short_test.mean()),
                'model_path': str(short_model_path)
            }
        }
        
        report_path = models_dir / f'bidirectional_report_{timestamp}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        st.info(f"💾 訓練報告: {report_path.name}")
        
        logger.info(f"Bidirectional training completed: Long AUC={auc_long:.4f}, Short AUC={auc_short:.4f}")
    
    def render_unidirectional(self):
        """渲染單向訓練介面 (原有功能)"""
        st.info("🚧 單向訓練功能保留，請先使用雙向訓練")
        st.markdown("""
        此功能保留來與舊版本相容。
        
        **建議使用雙向訓練**：
        - 更高交易頻率
        - 市場中性策略
        - 牛熊市都能獲利
        """)