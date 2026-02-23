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
from utils.feature_engineering import FeatureEngineer
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score

logger = setup_logger('model_training_tab', 'logs/model_training_tab.log')

class ModelTrainingTab:
    def __init__(self):
        logger.info("Initializing ModelTrainingTab")
        self.feature_engineer = FeatureEngineer()
    
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
    
    def load_klines(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """載入 HuggingFace 資料 - 與 feature_engineer.py 完全相同"""
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
            logger.info(f"Loaded {len(df)} records for {symbol} {timeframe}")
            return df
        
        except Exception as e:
            logger.error(f"Failed to load {symbol} {timeframe}: {str(e)}")
            return pd.DataFrame()
    
    def render_bidirectional(self):
        """雙向訓練介面"""
        st.markdown("""
        ### 雙向狩獵架構 - Bidirectional Hunting (高雜訊對抗版)
        
        **核心概念**:
        - Long Oracle: 捕捉底部反轉、W 底、上漲突破
        - Short Oracle: 捕捉頂部背離、M 頭、下跌突破
        
        **高雜訊對抗三大武器**:
        - L2 Regularization: 25.0 (強制權重分散)
        - Bagging Temperature: 2.0 (破解同質性)
        - Random Strength: 1.5 (防止單一特徵壟斷)
        
        **訓練警告**:
        - 深度 7 + 1500 樹 = 訓練時間 30-60 分鐘 (CPU)
        - 建議使用 GPU 加速 (10x)
        - 目標: Long AUC 0.70+, Short AUC 0.71+
        """)
        
        st.markdown("---")
        
        if st.button("訓練雙向模型 (Long + Short Oracles) - 硬核版", use_container_width=True):
            self.train_bidirectional()
    
    def train_bidirectional(self):
        """執行雙向訓練 - 高雜訊對抗版"""
        logger.info("="*80)
        logger.info("BIDIRECTIONAL TRAINING - HARDCORE ANTI-NOISE VERSION")
        logger.info("="*80)
        
        # Step 1: 載入 1m 數據
        with st.spinner("載入 1m K 線 (from HuggingFace)..."):
            df_1m = self.load_klines("BTCUSDT", "1m")
            
            if df_1m.empty:
                st.error("""
                無法從 HuggingFace 載入 BTCUSDT 1m 數據。
                
                請確認:
                1. Config.HF_REPO_ID 設定正確
                2. HuggingFace dataset 中存在 klines/BTCUSDT/BTC_1m.parquet
                3. 先到 'K棒資料抓取' Tab 上傳數據到 HuggingFace
                """)
                return
            
            # 設定 index
            if 'open_time' in df_1m.columns:
                df_1m['open_time'] = pd.to_datetime(df_1m['open_time'])
                df_1m.set_index('open_time', inplace=True)
            
            st.success(f"載入 {len(df_1m):,} 筆 1m K 線")
            logger.info(f"Loaded {len(df_1m):,} 1m bars")
        
        # Step 2: 使用 FeatureEngineer 生成完整特徵 (滾動視窗)
        with st.spinner("生成特徵 (滾動視窗架構)..."):
            df_features = self.feature_engineer.create_features_from_1m(
                df_1m,
                use_micro_structure=True,
                label_type='both'
            )
            st.success(f"特徵生成完成: {len(df_features):,} 筆")
        
        # 顯示標籤分布
        col1, col2 = st.columns(2)
        with col1:
            long_rate = df_features['label_long'].mean()
            st.metric("Long 標籤比例", f"{long_rate*100:.2f}%")
        with col2:
            short_rate = df_features['label_short'].mean()
            st.metric("Short 標籤比例", f"{short_rate*100:.2f}%")
        
        # Step 3: 準備特徵矩陣
        feature_cols = ['efficiency_ratio', 'extreme_time_diff', 'vol_imbalance_ratio',
                       'z_score', 'bb_width_pct', 'rsi', 'atr_pct', 'z_score_1h', 'atr_pct_1d']
        available_features = [col for col in feature_cols if col in df_features.columns]
        
        if not available_features:
            st.error("無法找到特徵")
            return
        
        logger.info(f"Using features: {available_features}")
        st.info(f"使用 {len(available_features)} 個特徵: {', '.join(available_features)}")
        
        X = df_features[available_features].values
        y_long = df_features['label_long'].values
        y_short = df_features['label_short'].values
        
        # 時間序列切分
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_long_train, y_long_test = y_long[:split_idx], y_long[split_idx:]
        y_short_train, y_short_test = y_short[:split_idx], y_short[split_idx:]
        
        st.info(f"訓練集: {len(X_train):,}, 測試集: {len(X_test):,}")
        logger.info(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
        
        # Step 4: 設定 TimeSeriesSplit (防止洩漏)
        tscv = TimeSeriesSplit(n_splits=5, gap=240)
        logger.info("TimeSeriesSplit configured: n_splits=5, gap=240 (4 hours)")
        
        # Step 5: 訓練 Long Oracle - 硬核版
        st.markdown("---")
        st.subheader("訓練 Long Oracle - 高雜訊對抗版")
        st.warning("訓練時間: 30-60 分鐘 (CPU) / 3-6 分鐘 (GPU)")
        
        with st.spinner("訓練 Long 模型中... (這會花一些時間)"):
            # 動態計算 scale_pos_weight
            pos_rate_long = y_long_train.mean()
            scale_pos_weight_long = (1 - pos_rate_long) / pos_rate_long if pos_rate_long > 0 else 1.0
            
            logger.info("="*80)
            logger.info("HARDCORE PARAMETER CONFIG - LONG ORACLE")
            logger.info("="*80)
            logger.info(f"Positive rate: {pos_rate_long*100:.2f}%")
            logger.info(f"Scale pos weight: {scale_pos_weight_long:.2f}")
            logger.info(f"Iterations: 1500")
            logger.info(f"Learning rate: 0.03")
            logger.info(f"Depth: 7")
            logger.info(f"L2 leaf reg: 25.0 (HIGH REGULARIZATION)")
            logger.info(f"Bagging temperature: 2.0 (HIGH DIVERSITY)")
            logger.info(f"Random strength: 1.5 (ANTI-DOMINANCE)")
            logger.info("="*80)
            
            st.info(f"Long scale_pos_weight: {scale_pos_weight_long:.2f}")
            
            # 硬核版參數配置
            model_long = CatBoostClassifier(
                # 1. 深度與廣度學習
                iterations=1500,
                learning_rate=0.03,
                depth=7,
                
                # 2. 對抗滾動重疊的「三大核心武器」
                l2_leaf_reg=25.0,           # 極度關鍵: 大幅拉高 L2 正則化
                bagging_temperature=2.0,    # 極度關鍵: 貝葉斯抽樣溫度
                random_strength=1.5,        # 增加特徵分裂隨機分數
                
                # 3. 基礎配置
                scale_pos_weight=float(scale_pos_weight_long),
                loss_function='Logloss',
                eval_metric='AUC',
                verbose=100,
                random_seed=42,
                task_type='CPU'
            )
            
            # 正確的校準: 在 X_train 上使用 CV
            logger.info("Training and calibrating Long Oracle (on X_train only)...")
            model_long_calibrated = CalibratedClassifierCV(
                estimator=model_long,
                method='isotonic',
                cv=tscv
            )
            model_long_calibrated.fit(X_train, y_long_train)
            
            # 在完全乾淨的測試集上評估
            y_long_proba = model_long_calibrated.predict_proba(X_test)[:, 1]
            auc_long = roc_auc_score(y_long_test, y_long_proba)
            
            logger.info(f"Long Oracle OOS AUC: {auc_long:.4f}")
            st.success(f"Long Oracle 訓練完成 - AUC: {auc_long:.4f}")
        
        # Step 6: 訓練 Short Oracle - 硬核版
        st.markdown("---")
        st.subheader("訓練 Short Oracle - 高雜訊對抗版")
        st.warning("訓練時間: 30-60 分鐘 (CPU) / 3-6 分鐘 (GPU)")
        
        with st.spinner("訓練 Short 模型中... (這會花一些時間)"):
            # 動態計算 scale_pos_weight
            pos_rate_short = y_short_train.mean()
            scale_pos_weight_short = (1 - pos_rate_short) / pos_rate_short if pos_rate_short > 0 else 1.0
            
            logger.info("="*80)
            logger.info("HARDCORE PARAMETER CONFIG - SHORT ORACLE")
            logger.info("="*80)
            logger.info(f"Positive rate: {pos_rate_short*100:.2f}%")
            logger.info(f"Scale pos weight: {scale_pos_weight_short:.2f}")
            logger.info("="*80)
            
            st.info(f"Short scale_pos_weight: {scale_pos_weight_short:.2f}")
            
            # 硬核版參數配置
            model_short = CatBoostClassifier(
                # 1. 深度與廣度學習
                iterations=1500,
                learning_rate=0.03,
                depth=7,
                
                # 2. 對抗滾動重疊的「三大核心武器」
                l2_leaf_reg=25.0,
                bagging_temperature=2.0,
                random_strength=1.5,
                
                # 3. 基礎配置
                scale_pos_weight=float(scale_pos_weight_short),
                loss_function='Logloss',
                eval_metric='AUC',
                verbose=100,
                random_seed=42,
                task_type='CPU'
            )
            
            # 正確的校準: 在 X_train 上使用 CV
            logger.info("Training and calibrating Short Oracle (on X_train only)...")
            model_short_calibrated = CalibratedClassifierCV(
                estimator=model_short,
                method='isotonic',
                cv=tscv
            )
            model_short_calibrated.fit(X_train, y_short_train)
            
            # 在完全乾淨的測試集上評估
            y_short_proba = model_short_calibrated.predict_proba(X_test)[:, 1]
            auc_short = roc_auc_score(y_short_test, y_short_proba)
            
            logger.info(f"Short Oracle OOS AUC: {auc_short:.4f}")
            st.success(f"Short Oracle 訓練完成 - AUC: {auc_short:.4f}")
        
        # Step 7: 計算動態閾值的 Precision/Recall
        logger.info("Calculating metrics with dynamic threshold...")
        
        # 使用相對基礎率 2x 作為閾值
        threshold_long = pos_rate_long * 2
        threshold_short = pos_rate_short * 2
        
        y_long_pred_dynamic = (y_long_proba >= threshold_long).astype(int)
        y_short_pred_dynamic = (y_short_proba >= threshold_short).astype(int)
        
        precision_long = precision_score(y_long_test, y_long_pred_dynamic, zero_division=0)
        recall_long = recall_score(y_long_test, y_long_pred_dynamic, zero_division=0)
        
        precision_short = precision_score(y_short_test, y_short_pred_dynamic, zero_division=0)
        recall_short = recall_score(y_short_test, y_short_pred_dynamic, zero_division=0)
        
        # Step 8: 保存模型
        st.markdown("---")
        st.subheader("保存模型")
        
        models_dir = Path('models_output')
        models_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        long_model_path = models_dir / f'catboost_long_hardcore_{timestamp}.pkl'
        short_model_path = models_dir / f'catboost_short_hardcore_{timestamp}.pkl'
        
        joblib.dump(model_long_calibrated, long_model_path)
        joblib.dump(model_short_calibrated, short_model_path)
        
        st.success(f"Long Oracle: {long_model_path.name}")
        st.success(f"Short Oracle: {short_model_path.name}")
        
        # Step 9: 綜合報告
        st.markdown("---")
        st.subheader("雙向模型績效 (硬核版)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Long Oracle")
            st.metric("AUC", f"{auc_long:.4f}", delta="目標: 0.70+")
            st.metric("標籤比例", f"{pos_rate_long*100:.2f}%")
            st.metric("Threshold", f"{threshold_long:.4f}")
            st.metric("Precision", f"{precision_long:.4f}")
            st.metric("Recall", f"{recall_long:.4f}")
        
        with col2:
            st.markdown("### Short Oracle")
            st.metric("AUC", f"{auc_short:.4f}", delta="目標: 0.70+")
            st.metric("標籤比例", f"{pos_rate_short*100:.2f}%")
            st.metric("Threshold", f"{threshold_short:.4f}")
            st.metric("Precision", f"{precision_short:.4f}")
            st.metric("Recall", f"{recall_short:.4f}")
        
        # 評估
        if auc_long >= 0.70 and auc_short >= 0.70:
            st.success("聖杯達標,兩個 Oracle AUC 都突破 0.70")
        elif auc_long >= 0.68 and auc_short >= 0.68:
            st.info("接近聖杯,再調整一次可能突破")
        else:
            st.warning("需要再微調參數")
        
        # 保存報告
        report = {
            'timestamp': timestamp,
            'version': 'hardcore_anti_noise',
            'architecture': 'rolling_window',
            'calibration_method': 'isotonic_with_tscv',
            'parameters': {
                'iterations': 1500,
                'learning_rate': 0.03,
                'depth': 7,
                'l2_leaf_reg': 25.0,
                'bagging_temperature': 2.0,
                'random_strength': 1.5
            },
            'features': available_features,
            'total_samples': len(df_features),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'long_oracle': {
                'auc': float(auc_long),
                'precision': float(precision_long),
                'recall': float(recall_long),
                'positive_rate': float(pos_rate_long),
                'scale_pos_weight': float(scale_pos_weight_long),
                'threshold': float(threshold_long),
                'model_path': str(long_model_path)
            },
            'short_oracle': {
                'auc': float(auc_short),
                'precision': float(precision_short),
                'recall': float(recall_short),
                'positive_rate': float(pos_rate_short),
                'scale_pos_weight': float(scale_pos_weight_short),
                'threshold': float(threshold_short),
                'model_path': str(short_model_path)
            }
        }
        
        report_path = models_dir / f'bidirectional_hardcore_report_{timestamp}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        st.info(f"訓練報告: {report_path.name}")
        
        logger.info("="*80)
        logger.info(f"HARDCORE TRAINING COMPLETED")
        logger.info(f"Long AUC: {auc_long:.4f} (Target: 0.70+)")
        logger.info(f"Short AUC: {auc_short:.4f} (Target: 0.70+)")
        logger.info("="*80)
    
    def render_unidirectional(self):
        """單向訓練介面"""
        st.info("單向訓練功能保留,建議使用雙向訓練")
        st.markdown("""
        **建議使用雙向訓練**:
        - 更高交易頻率
        - 市場中性策略
        - 牛熊市都能獲利
        """)