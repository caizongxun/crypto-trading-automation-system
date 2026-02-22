#!/usr/bin/env python3
"""
雙向狩獵架構 - 訓練 Long/Short 雙大腦

此腳本會同時訓練兩個獨立模型:
1. Long Oracle - 捕捉底部反轉與上漲突破
2. Short Oracle - 捕捉頂部背離與下跌突破

Usage:
    python train_bidirectional.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
from datetime import datetime

sys.path.append(str(Path(__file__).parent))
from utils.logger import setup_logger
from utils.data_loader import DataLoader
from utils.feature_engineering import FeatureEngineer
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score

logger = setup_logger('train_bidirectional', 'logs/train_bidirectional.log')

def main():
    logger.info("="*80)
    logger.info("雙向狩獵架構 - BIDIRECTIONAL HUNTING ARCHITECTURE")
    logger.info("="*80)
    
    # ========================================
    # Step 1: 載入資料
    # ========================================
    logger.info("Step 1: Loading data")
    
    data_loader = DataLoader()
    df_1m = data_loader.load_klines('BTCUSDT', '1m')
    
    if df_1m is None or df_1m.empty:
        logger.error("Failed to load 1m klines")
        return
    
    logger.info(f"Loaded {len(df_1m):,} 1m bars")
    
    # ========================================
    # Step 2: 生成雙向特徵與標籤
    # ========================================
    logger.info("Step 2: Feature engineering with bidirectional labels")
    
    engineer = FeatureEngineer()
    df_features = engineer.create_features_from_1m(
        df_1m,
        use_micro_structure=True,
        label_type='both'  # 雙向標籤
    )
    
    logger.info(f"Generated {len(df_features):,} samples with bidirectional labels")
    
    # ========================================
    # Step 3: 準備特徵矩陣
    # ========================================
    logger.info("Step 3: Preparing feature matrix")
    
    # 選擇特徵
    feature_cols = [
        # 微觀結構
        'efficiency_ratio', 'extreme_time_diff', 'vol_imbalance_ratio',
        # 15m 技術指標
        'z_score', 'bb_width_pct', 'rsi', 'atr_pct',
        # 跨週期特徵
        'z_score_1h', 'atr_pct_1d'
    ]
    
    available_features = [col for col in feature_cols if col in df_features.columns]
    logger.info(f"Using {len(available_features)} features: {available_features}")
    
    X = df_features[available_features].values
    y_long = df_features['label_long'].values
    y_short = df_features['label_short'].values
    
    # 時間序列切分 (80/20)
    split_idx = int(len(X) * 0.8)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_long_train, y_long_test = y_long[:split_idx], y_long[split_idx:]
    y_short_train, y_short_test = y_short[:split_idx], y_short[split_idx:]
    
    logger.info(f"Train set: {len(X_train):,} samples")
    logger.info(f"Test set:  {len(X_test):,} samples")
    logger.info(f"Long  positive rate (train): {y_long_train.mean()*100:.2f}%")
    logger.info(f"Short positive rate (train): {y_short_train.mean()*100:.2f}%")
    
    # ========================================
    # Step 4: 訓練 Long Oracle
    # ========================================
    logger.info("="*80)
    logger.info("🔵 Training Long Oracle (Bottom Reversal Detector)")
    logger.info("="*80)
    
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
        verbose=100
    )
    
    # 機率校準
    logger.info("Calibrating Long Oracle probabilities...")
    model_long_calibrated = CalibratedClassifierCV(
        model_long,
        method='isotonic',
        cv='prefit'
    )
    model_long_calibrated.fit(X_test, y_long_test)
    
    # 評估
    y_long_pred = model_long_calibrated.predict(X_test)
    y_long_proba = model_long_calibrated.predict_proba(X_test)[:, 1]
    
    logger.info("\nLong Oracle Performance:")
    logger.info(classification_report(y_long_test, y_long_pred, target_names=['Negative', 'Positive']))
    logger.info(f"AUC: {roc_auc_score(y_long_test, y_long_proba):.4f}")
    
    # ========================================
    # Step 5: 訓練 Short Oracle
    # ========================================
    logger.info("="*80)
    logger.info("🔴 Training Short Oracle (Top Reversal Detector)")
    logger.info("="*80)
    
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
        verbose=100
    )
    
    # 機率校準
    logger.info("Calibrating Short Oracle probabilities...")
    model_short_calibrated = CalibratedClassifierCV(
        model_short,
        method='isotonic',
        cv='prefit'
    )
    model_short_calibrated.fit(X_test, y_short_test)
    
    # 評估
    y_short_pred = model_short_calibrated.predict(X_test)
    y_short_proba = model_short_calibrated.predict_proba(X_test)[:, 1]
    
    logger.info("\nShort Oracle Performance:")
    logger.info(classification_report(y_short_test, y_short_pred, target_names=['Negative', 'Positive']))
    logger.info(f"AUC: {roc_auc_score(y_short_test, y_short_proba):.4f}")
    
    # ========================================
    # Step 6: 保存模型
    # ========================================
    logger.info("="*80)
    logger.info("💾 Saving bidirectional models")
    logger.info("="*80)
    
    models_dir = Path('models_output')
    models_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存 Long Oracle
    long_model_path = models_dir / f'catboost_long_model_{timestamp}.pkl'
    joblib.dump(model_long_calibrated, long_model_path)
    logger.info(f"Long Oracle saved: {long_model_path}")
    
    # 保存 Short Oracle
    short_model_path = models_dir / f'catboost_short_model_{timestamp}.pkl'
    joblib.dump(model_short_calibrated, short_model_path)
    logger.info(f"Short Oracle saved: {short_model_path}")
    
    # ========================================
    # Step 7: 生成綜合報告
    # ========================================
    logger.info("="*80)
    logger.info("📊 BIDIRECTIONAL TRAINING SUMMARY")
    logger.info("="*80)
    
    report = {
        'timestamp': timestamp,
        'total_samples': len(df_features),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'long_oracle': {
            'positive_rate': float(y_long_test.mean()),
            'auc': float(roc_auc_score(y_long_test, y_long_proba)),
            'model_path': str(long_model_path)
        },
        'short_oracle': {
            'positive_rate': float(y_short_test.mean()),
            'auc': float(roc_auc_score(y_short_test, y_short_proba)),
            'model_path': str(short_model_path)
        }
    }
    
    # 保存報告
    report_path = models_dir / f'bidirectional_report_{timestamp}.json'
    import json
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    logger.info(f"\nLong Oracle:")
    logger.info(f"  - Positive rate: {report['long_oracle']['positive_rate']*100:.2f}%")
    logger.info(f"  - AUC: {report['long_oracle']['auc']:.4f}")
    logger.info(f"  - Model: {report['long_oracle']['model_path']}")
    
    logger.info(f"\nShort Oracle:")
    logger.info(f"  - Positive rate: {report['short_oracle']['positive_rate']*100:.2f}%")
    logger.info(f"  - AUC: {report['short_oracle']['auc']:.4f}")
    logger.info(f"  - Model: {report['short_oracle']['model_path']}")
    
    logger.info(f"\nReport saved: {report_path}")
    logger.info("="*80)
    logger.info("✅ Bidirectional training completed successfully!")
    logger.info("="*80)

if __name__ == '__main__':
    main()