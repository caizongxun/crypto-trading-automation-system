#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 模型訓練 - 根本解決方案

核心改進:
1. 重新設計特徵工程 (只用有用的特徵)
2. 重新定義目標 (可操作的目標)
3. Walk-forward 驗證 (避免過括合)
4. 加入 Imbalance 處理
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from datetime import datetime, timedelta
import json
import pickle
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Fix encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_v4.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def calculate_better_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    重新設計特徵工程
    
    只保留有用的特徵:
    1. Price momentum (價格動量)
    2. Volume profile (成交量特徵)
    3. Volatility (波動率)
    4. Market microstructure (市場微觀結構)
    """
    features = pd.DataFrame(index=df.index)
    
    # 1. Price Momentum (5個)
    features['ret_1'] = df['close'].pct_change(1)
    features['ret_3'] = df['close'].pct_change(3)
    features['ret_6'] = df['close'].pct_change(6)
    features['ret_12'] = df['close'].pct_change(12)
    features['ret_24'] = df['close'].pct_change(24)
    
    # 2. Volume Profile (4個)
    vol_ma = df['volume'].rolling(24).mean()
    features['vol_ratio'] = df['volume'] / (vol_ma + 1e-8)
    features['vol_surge'] = (df['volume'] > vol_ma * 2).astype(int)
    
    # Volume-weighted price change
    vwap = (df['close'] * df['volume']).rolling(6).sum() / (df['volume'].rolling(6).sum() + 1e-8)
    features['price_vs_vwap'] = (df['close'] - vwap) / vwap
    
    # Buy/Sell pressure
    range_val = df['high'] - df['low'] + 1e-8
    buy_pressure = (df['close'] - df['low']) / range_val
    features['buy_pressure_ma'] = buy_pressure.rolling(6).mean()
    
    # 3. Volatility (3個)
    features['volatility_6h'] = df['close'].pct_change().rolling(6).std()
    features['volatility_24h'] = df['close'].pct_change().rolling(24).std()
    features['volatility_ratio'] = features['volatility_6h'] / (features['volatility_24h'] + 1e-8)
    
    # 4. Market Microstructure (3個)
    # Candle body ratio
    body = abs(df['close'] - df['open'])
    features['body_ratio'] = body / range_val
    
    # Direction consistency
    direction = (df['close'] > df['open']).astype(int)
    features['direction_ma3'] = direction.rolling(3).mean()
    features['direction_ma6'] = direction.rolling(6).mean()
    
    # 5. Time features (2個)
    if 'open_time' in df.columns:
        hour = pd.to_datetime(df['open_time']).dt.hour
    elif isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
    else:
        hour = pd.Series(12, index=df.index)
    
    features['is_asia'] = ((hour >= 0) & (hour < 8)).astype(int)
    features['is_high_vol_time'] = ((hour >= 8) & (hour < 16)).astype(int)
    
    return features


def create_better_labels(df: pd.DataFrame, horizon: int = 4) -> pd.Series:
    """
    重新定義目標
    
    目標: 未來 horizon 小時內有明顯漲幅 (>0.8%)
    
    這是可交易的目標:
    - 漲幅足夠覆蓋手續費 + 滑點
    - 時間窗口合理 (4h)
    """
    future_max = df['high'].rolling(horizon).max().shift(-horizon)
    future_return = (future_max - df['close']) / df['close']
    
    # Long signal: 未來4h內最高點漲超過 0.8%
    long_label = (future_return > 0.008).astype(int)
    
    return long_label


def create_short_labels(df: pd.DataFrame, horizon: int = 4) -> pd.Series:
    """
    Short 目標
    """
    future_min = df['low'].rolling(horizon).min().shift(-horizon)
    future_return = (df['close'] - future_min) / df['close']
    
    # Short signal: 未來4h內最低點跌超過 0.8%
    short_label = (future_return > 0.008).astype(int)
    
    return short_label


def train_v4_model(
    symbol: str = 'BTCUSDT',
    timeframe: str = '1h',
    days: int = 180,
    horizon: int = 4
):
    """
    訓練 v4 模型
    """
    logger.info("="*80)
    logger.info("[START] Training v4 Model")
    logger.info("="*80)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Days: {days}")
    logger.info(f"Horizon: {horizon}h")
    logger.info("")
    
    # 1. 載入數據
    logger.info("Step 1/6: Loading data...")
    from utils.hf_data_loader import load_klines
    
    # 計算開始日期
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = load_klines(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    logger.info(f"Loaded {len(df)} candles")
    
    # 2. 計算特徵
    logger.info("\nStep 2/6: Engineering features...")
    features = calculate_better_features(df)
    logger.info(f"Created {len(features.columns)} features")
    logger.info(f"Features: {features.columns.tolist()}")
    
    # 3. 創建目標
    logger.info("\nStep 3/6: Creating labels...")
    long_labels = create_better_labels(df, horizon)
    short_labels = create_short_labels(df, horizon)
    
    logger.info(f"Long positive rate: {long_labels.mean():.2%}")
    logger.info(f"Short positive rate: {short_labels.mean():.2%}")
    
    # 4. 清理數據
    logger.info("\nStep 4/6: Cleaning data...")
    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)
    
    # 移除無效數據
    valid_idx = ~(long_labels.isna() | short_labels.isna())
    features = features[valid_idx]
    long_labels = long_labels[valid_idx]
    short_labels = short_labels[valid_idx]
    df = df[valid_idx]
    
    logger.info(f"Valid samples: {len(features)}")
    
    # 5. Walk-forward 切分
    logger.info("\nStep 5/6: Walk-forward validation...")
    
    # 70% 訓練, 15% 驗證, 15% 測試
    n = len(features)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    X_train = features.iloc[:train_end]
    y_train_long = long_labels.iloc[:train_end]
    y_train_short = short_labels.iloc[:train_end]
    
    X_val = features.iloc[train_end:val_end]
    y_val_long = long_labels.iloc[train_end:val_end]
    y_val_short = short_labels.iloc[train_end:val_end]
    
    X_test = features.iloc[val_end:]
    y_test_long = long_labels.iloc[val_end:]
    y_test_short = short_labels.iloc[val_end:]
    
    logger.info(f"Train: {len(X_train)} samples")
    logger.info(f"Val: {len(X_val)} samples")
    logger.info(f"Test: {len(X_test)} samples")
    
    # 6. 訓練模型
    logger.info("\nStep 6/6: Training models...")
    
    # Long model
    logger.info("\n[LONG] Training...")
    model_long = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=3,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        auto_class_weights='Balanced'  # 處理不平衡
    )
    
    model_long.fit(
        X_train, y_train_long,
        eval_set=(X_val, y_val_long),
        use_best_model=True
    )
    
    # Short model
    logger.info("\n[SHORT] Training...")
    model_short = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=3,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        auto_class_weights='Balanced'
    )
    
    model_short.fit(
        X_train, y_train_short,
        eval_set=(X_val, y_val_short),
        use_best_model=True
    )
    
    # 7. 評估
    logger.info("\n" + "="*80)
    logger.info("[EVALUATION] Test Set Performance")
    logger.info("="*80)
    
    # Long
    pred_long = model_long.predict_proba(X_test)[:, 1]
    pred_long_binary = (pred_long > 0.5).astype(int)
    
    auc_long = roc_auc_score(y_test_long, pred_long)
    precision_long = precision_score(y_test_long, pred_long_binary, zero_division=0)
    recall_long = recall_score(y_test_long, pred_long_binary, zero_division=0)
    f1_long = f1_score(y_test_long, pred_long_binary, zero_division=0)
    
    logger.info(f"\n[LONG] Test Metrics:")
    logger.info(f"  AUC: {auc_long:.4f}")
    logger.info(f"  Precision: {precision_long:.4f}")
    logger.info(f"  Recall: {recall_long:.4f}")
    logger.info(f"  F1: {f1_long:.4f}")
    logger.info(f"  Positive rate: {y_test_long.mean():.2%}")
    
    # Short
    pred_short = model_short.predict_proba(X_test)[:, 1]
    pred_short_binary = (pred_short > 0.5).astype(int)
    
    auc_short = roc_auc_score(y_test_short, pred_short)
    precision_short = precision_score(y_test_short, pred_short_binary, zero_division=0)
    recall_short = recall_score(y_test_short, pred_short_binary, zero_division=0)
    f1_short = f1_score(y_test_short, pred_short_binary, zero_division=0)
    
    logger.info(f"\n[SHORT] Test Metrics:")
    logger.info(f"  AUC: {auc_short:.4f}")
    logger.info(f"  Precision: {precision_short:.4f}")
    logger.info(f"  Recall: {recall_short:.4f}")
    logger.info(f"  F1: {f1_short:.4f}")
    logger.info(f"  Positive rate: {y_test_short.mean():.2%}")
    
    # 8. Feature Importance
    logger.info("\n[FEATURE IMPORTANCE] Top 10:")
    importance = model_long.get_feature_importance()
    feature_names = features.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.2f}")
    
    # 9. 儲存模型
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    models_dir = Path('models_output')
    models_dir.mkdir(exist_ok=True)
    
    long_path = models_dir / f'catboost_long_v4_{timestamp}.pkl'
    short_path = models_dir / f'catboost_short_v4_{timestamp}.pkl'
    
    with open(long_path, 'wb') as f:
        pickle.dump({
            'model': model_long,
            'features': features.columns.tolist(),
            'metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'horizon': horizon,
                'train_date': timestamp
            }
        }, f)
    
    with open(short_path, 'wb') as f:
        pickle.dump({
            'model': model_short,
            'features': features.columns.tolist(),
            'metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'horizon': horizon,
                'train_date': timestamp
            }
        }, f)
    
    logger.info(f"\n[SAVE] Models saved:")
    logger.info(f"  Long: {long_path}")
    logger.info(f"  Short: {short_path}")
    
    # 10. 儲存報告
    report = {
        'version': 'v4',
        'timestamp': timestamp,
        'symbol': symbol,
        'timeframe': timeframe,
        'horizon': horizon,
        'long_model': {
            'auc': float(auc_long),
            'precision': float(precision_long),
            'recall': float(recall_long),
            'f1': float(f1_long),
            'positive_rate': float(y_test_long.mean())
        },
        'short_model': {
            'auc': float(auc_short),
            'precision': float(precision_short),
            'recall': float(recall_short),
            'f1': float(f1_short),
            'positive_rate': float(y_test_short.mean())
        },
        'features': features.columns.tolist(),
        'feature_count': len(features.columns),
        'top_features': importance_df.head(10).to_dict('records')
    }
    
    report_path = models_dir / f'v4_training_report_{timestamp}.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n[SAVE] Report saved: {report_path}")
    logger.info("\n" + "="*80)
    logger.info("[DONE] Training complete!")
    logger.info("="*80)
    
    return report


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train v4 model')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='1h')
    parser.add_argument('--days', type=int, default=180)
    parser.add_argument('--horizon', type=int, default=4, help='Prediction horizon in hours')
    
    args = parser.parse_args()
    
    report = train_v4_model(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        horizon=args.horizon
    )
    
    print(f"\n✅ Training complete!")
    print(f"Long AUC: {report['long_model']['auc']:.4f}")
    print(f"Short AUC: {report['short_model']['auc']:.4f}")
