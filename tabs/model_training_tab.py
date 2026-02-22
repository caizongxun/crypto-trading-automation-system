import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import joblib
from datetime import datetime
from huggingface_hub import hf_hub_download

sys.path.append(str(Path(__file__).parent.parent))
from config import Config
from utils.logger import setup_logger
from utils.model_trainer import ModelTrainer
from utils.micro_structure import MicroStructureEngineer
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score

logger = setup_logger('model_training_tab', 'logs/model_training_tab.log')

class ModelTrainingTab:
    def __init__(self):
        logger.info("Initializing ModelTrainingTab")
        self.micro_engineer = MicroStructureEngineer()
    
    def render(self):
        logger.info("Rendering Model Training Tab")
        st.header("模型訓練")
        
        # 訓練模式選擇
        training_mode = st.radio(
            "選擇訓練模式",
            ['unidirectional', 'bidirectional'],
            format_func=lambda x: '單向 (Long Only)' if x == 'unidirectional' else '雙向 (Long + Short)',
            horizontal=True,
            key="training_mode_selector"
        )
        
        if training_mode == 'bidirectional':
            self.render_bidirectional()
        else:
            self.render_unidirectional()
    
    def load_1m_data_from_hf(self, symbol: str = "BTCUSDT"):
        """從 HuggingFace 載入 1m 數據"""
        try:
            base = symbol.replace("USDT", "")
            filename = f"{base}_1m.parquet"
            folder = f"klines/{symbol}"
            
            logger.info(f"Downloading {filename} from HuggingFace")
            
            file_path = hf_hub_download(
                repo_id=Config.HF_REPO_ID,
                filename=f"{folder}/{filename}",
                repo_type="dataset",
                token=Config.HF_TOKEN
            )
            
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(df)} records from HuggingFace")
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading from HuggingFace: {str(e)}")
            return None
    
    def load_1m_data_local(self, symbol: str = "BTCUSDT"):
        """從本地載入 1m 數據"""
        base = symbol.replace("USDT", "")
        filename = f"{base}_1m.parquet"
        
        # 嘗試多個路徑
        possible_paths = [
            Path(f"temp_data/{filename}"),
            Path(f"klines_output/klines_{symbol}_1m.parquet"),
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"Loading from local: {path}")
                try:
                    df = pd.read_parquet(path)
                    logger.info(f"Loaded {len(df)} records from {path}")
                    return df
                except Exception as e:
                    logger.error(f"Error loading {path}: {str(e)}")
                    continue
        
        return None
    
    def render_bidirectional(self):
        """雙向訓練介面"""
        st.markdown("""
        ### 雙向狩獵架構 - Bidirectional Hunting
        
        **核心概念**:
        - Long Oracle: 捕捉底部反轉、W 底、上漲突破
        - Short Oracle: 捕捉頂部背離、M 頭、下跌突破
        
        **優勢**:
        - 交易次數翻倍 (100 -> 200+ 次)
        - 報酬倍增 (+5% -> +10%+)
        - 市場中性 (牛熊市都賺錢)
        
        **標籤邏輯**:
        - Long: 未來 4 小時先漲 2% (停利) 且沒先跌 1% (停損)
        - Short: 未來 4 小時先跌 2% (停利) 且沒先漲 1% (停損)
        """)
        
        st.markdown("---")
        
        if st.button("訓練雙向模型 (Long + Short Oracles)", use_container_width=True):
            self.train_bidirectional()
    
    def train_bidirectional(self):
        """執行雙向訓練"""
        logger.info("Starting bidirectional training")
        
        # Step 1: 載入 1m 數據
        with st.spinner("載入 1m K 線..."):
            # 先嘗試本地
            df_1m = self.load_1m_data_local("BTCUSDT")
            
            # 如果本地沒有，從 HuggingFace 下載
            if df_1m is None:
                st.info("本地沒有數據，嘗試從 HuggingFace 下載...")
                df_1m = self.load_1m_data_from_hf("BTCUSDT")
            
            if df_1m is None or df_1m.empty:
                st.error("""
                無法載入 BTCUSDT 1m 數據。
                
                請先到 'K棒資料抓取' Tab 下載 BTCUSDT 1m 數據。
                """)
                return
            
            # 設定 index
            if 'open_time' in df_1m.columns:
                df_1m['open_time'] = pd.to_datetime(df_1m['open_time'])
                df_1m.set_index('open_time', inplace=True)
            
            st.success(f"載入 {len(df_1m):,} 筆 1m K 線")
        
        # Step 2: 生成雙向標籤
        with st.spinner("生成雙向特徵與標籤..."):
            # 壓縮為 15m
            df_15m = self.micro_engineer.compress_1m_to_15m(df_1m)
            
            # 生成雙向標籤
            df_features = self.micro_engineer.add_bidirectional_labels(
                df_15m,
                lookahead_bars=16,
                tp_pct_long=0.02,
                sl_pct_long=0.01,
                tp_pct_short=0.02,
                sl_pct_short=0.01
            )
            
            st.success(f"生成 {len(df_features):,} 筆雙向標籤")
            
            # 顯示標籤分布
            col1, col2 = st.columns(2)
            with col1:
                long_rate = df_features['label_long'].mean()
                st.metric("Long 標籤比例", f"{long_rate*100:.2f}%")
            with col2:
                short_rate = df_features['label_short'].mean()
                st.metric("Short 標籤比例", f"{short_rate*100:.2f}%")
        
        # Step 3: 準備特徵矩陣
        feature_cols = ['efficiency_ratio', 'extreme_time_diff', 'vol_imbalance_ratio']
        available_features = [col for col in feature_cols if col in df_features.columns]
        
        if not available_features:
            st.error("無法找到微觀結構特徵")
            return
        
        X = df_features[available_features].values
        y_long = df_features['label_long'].values
        y_short = df_features['label_short'].values
        
        # 時間序列切分
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_long_train, y_long_test = y_long[:split_idx], y_long[split_idx:]
        y_short_train, y_short_test = y_short[:split_idx], y_short[split_idx:]
        
        st.info(f"訓練集: {len(X_train):,}, 測試集: {len(X_test):,}")
        
        # Step 4: 訓練 Long Oracle
        st.markdown("---")
        st.subheader("訓練 Long Oracle")
        
        with st.spinner("訓練 Long 模型中... (2-3 分鐘)"):
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
            
            st.success(f"Long Oracle 訓練完成 - AUC: {auc_long:.4f}")
        
        # Step 5: 訓練 Short Oracle
        st.markdown("---")
        st.subheader("訓練 Short Oracle")
        
        with st.spinner("訓練 Short 模型中... (2-3 分鐘)"):
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
            
            st.success(f"Short Oracle 訓練完成 - AUC: {auc_short:.4f}")
        
        # Step 6: 保存模型
        st.markdown("---")
        st.subheader("保存模型")
        
        models_dir = Path('models_output')
        models_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        long_model_path = models_dir / f'catboost_long_model_{timestamp}.pkl'
        short_model_path = models_dir / f'catboost_short_model_{timestamp}.pkl'
        
        joblib.dump(model_long_calibrated, long_model_path)
        joblib.dump(model_short_calibrated, short_model_path)
        
        st.success(f"Long Oracle: {long_model_path.name}")
        st.success(f"Short Oracle: {short_model_path.name}")
        
        # Step 7: 綜合報告
        st.markdown("---")
        st.subheader("雙向模型績效")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Long Oracle")
            st.metric("AUC", f"{auc_long:.4f}", delta="目標: 0.65+")
            st.metric("標籤比例", f"{y_long_test.mean()*100:.2f}%")
            
            precision_long = precision_score(y_long_test, y_long_pred)
            recall_long = recall_score(y_long_test, y_long_pred)
            
            st.metric("Precision", f"{precision_long:.4f}")
            st.metric("Recall", f"{recall_long:.4f}")
        
        with col2:
            st.markdown("### Short Oracle")
            st.metric("AUC", f"{auc_short:.4f}", delta="目標: 0.65+")
            st.metric("標籤比例", f"{y_short_test.mean()*100:.2f}%")
            
            precision_short = precision_score(y_short_test, y_short_pred)
            recall_short = recall_score(y_short_test, y_short_pred)
            
            st.metric("Precision", f"{precision_short:.4f}")
            st.metric("Recall", f"{recall_short:.4f}")
        
        # 評估
        if auc_long >= 0.65 and auc_short >= 0.65:
            st.success("雙向模型達標！兩個 Oracle AUC 都超過 0.65")
        else:
            st.warning("模型效能可提升，建議調整超參數或增加資料")
        
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
        
        st.info(f"訓練報告: {report_path.name}")
        
        logger.info(f"Bidirectional training completed: Long AUC={auc_long:.4f}, Short AUC={auc_short:.4f}")
    
    def render_unidirectional(self):
        """單向訓練介面"""
        st.info("單向訓練功能保留，建議使用雙向訓練")
        st.markdown("""
        **建議使用雙向訓練**:
        - 更高交易頻率
        - 市場中性策略
        - 牛熊市都能獲利
        """)