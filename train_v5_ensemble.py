#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v5 模型訓練 - 進階特徵工程 + Ensemble

核心改進:
1. 添加高階特徵 (Order Flow, Fractal, Regime)
2. 使用多個時間框架特徵
3. 動態閾值 (不用固定 0.5)
4. 多模型 Ensemble
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from datetime import datetime, timedelta
import json
import pickle
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_v5.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def calculate_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    進階特徵工程
    
    新增:
    1. Order Flow Imbalance (買賣壓力不平衡)
    2. Fractal Features (分形特徵)
    3. Regime Detection (市場狀態)
    4. Multi-timeframe momentum
    """
    features = pd.DataFrame(index=df.index)
    
    # === 1. Basic Momentum (優化版) ===
    for window in [1, 2, 4, 8, 12, 24, 48]:
        features[f'ret_{window}'] = df['close'].pct_change(window)
    
    # === 2. Order Flow Imbalance ===
    # 買賣壓力指標
    high_low_range = df['high'] - df['low'] + 1e-8
    
    # Buy pressure: (close - low) / range
    buy_pressure = (df['close'] - df['low']) / high_low_range
    features['buy_pressure'] = buy_pressure
    features['buy_pressure_ma6'] = buy_pressure.rolling(6).mean()
    features['buy_pressure_ma24'] = buy_pressure.rolling(24).mean()
    
    # Sell pressure: (high - close) / range
    sell_pressure = (df['high'] - df['close']) / high_low_range
    features['sell_pressure'] = sell_pressure
    
    # Imbalance
    features['pressure_imbalance'] = buy_pressure - sell_pressure
    features['pressure_imb_ma6'] = features['pressure_imbalance'].rolling(6).mean()
    
    # === 3. Volume Profile (強化版) ===
    vol_ma6 = df['volume'].rolling(6).mean()
    vol_ma24 = df['volume'].rolling(24).mean()
    vol_ma168 = df['volume'].rolling(168).mean()  # 1 week
    
    features['vol_ratio_6'] = df['volume'] / (vol_ma6 + 1e-8)
    features['vol_ratio_24'] = df['volume'] / (vol_ma24 + 1e-8)
    features['vol_ratio_168'] = df['volume'] / (vol_ma168 + 1e-8)
    
    # Volume surge
    features['vol_surge'] = (df['volume'] > vol_ma24 * 2).astype(int)
    
    # Volume-price divergence
    price_change = df['close'].pct_change()
    vol_change = df['volume'].pct_change()
    features['vol_price_corr_6'] = price_change.rolling(6).corr(vol_change)
    features['vol_price_corr_24'] = price_change.rolling(24).corr(vol_change)
    
    # === 4. VWAP Features ===
    vwap_6 = (df['close'] * df['volume']).rolling(6).sum() / (df['volume'].rolling(6).sum() + 1e-8)
    vwap_24 = (df['close'] * df['volume']).rolling(24).sum() / (df['volume'].rolling(24).sum() + 1e-8)
    
    features['price_vs_vwap6'] = (df['close'] - vwap_6) / vwap_6
    features['price_vs_vwap24'] = (df['close'] - vwap_24) / vwap_24
    
    # === 5. Volatility Regime ===
    returns = df['close'].pct_change()
    
    for window in [6, 12, 24, 48]:
        vol = returns.rolling(window).std()
        features[f'vol_{window}'] = vol
    
    # Volatility regime (相對於長期)
    vol_6 = features['vol_6']
    vol_168 = returns.rolling(168).std()
    features['vol_regime'] = vol_6 / (vol_168 + 1e-8)
    
    # === 6. Fractal Features ===
    # Higher high, lower low patterns
    features['is_higher_high'] = ((df['high'] > df['high'].shift(1)) & 
                                   (df['high'].shift(1) > df['high'].shift(2))).astype(int)
    features['is_lower_low'] = ((df['low'] < df['low'].shift(1)) & 
                                 (df['low'].shift(1) < df['low'].shift(2))).astype(int)
    
    # Support/Resistance touch
    rolling_max_24 = df['high'].rolling(24).max()
    rolling_min_24 = df['low'].rolling(24).min()
    
    features['near_resistance'] = ((df['close'] / rolling_max_24) > 0.99).astype(int)
    features['near_support'] = ((df['close'] / rolling_min_24) < 1.01).astype(int)
    
    # === 7. Trend Strength ===
    # ADX-like indicator (simplified)
    plus_dm = (df['high'] - df['high'].shift(1)).clip(lower=0)
    minus_dm = (df['low'].shift(1) - df['low']).clip(lower=0)
    
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(14).mean()
    plus_di = (plus_dm.rolling(14).mean() / atr) * 100
    minus_di = (minus_dm.rolling(14).mean() / atr) * 100
    
    features['trend_strength'] = abs(plus_di - minus_di)
    features['trend_direction'] = (plus_di > minus_di).astype(int)
    
    # === 8. Market Microstructure ===
    # Candle patterns
    body = abs(df['close'] - df['open'])
    features['body_ratio'] = body / high_low_range
    
    # Tail ratio (wick)
    upper_tail = df['high'] - df[['close', 'open']].max(axis=1)
    lower_tail = df[['close', 'open']].min(axis=1) - df['low']
    features['upper_tail_ratio'] = upper_tail / high_low_range
    features['lower_tail_ratio'] = lower_tail / high_low_range
    
    # Direction consistency
    direction = (df['close'] > df['open']).astype(int)
    for window in [3, 6, 12]:
        features[f'direction_ma{window}'] = direction.rolling(window).mean()
    
    # === 9. Price Level Features ===
    # Distance from recent highs/lows
    high_24 = df['high'].rolling(24).max()
    low_24 = df['low'].rolling(24).min()
    
    features['dist_from_high24'] = (high_24 - df['close']) / df['close']
    features['dist_from_low24'] = (df['close'] - low_24) / df['close']
    
    # === 10. Time Features ===
    if 'open_time' in df.columns:
        hour = pd.to_datetime(df['open_time']).dt.hour
        day_of_week = pd.to_datetime(df['open_time']).dt.dayofweek
    elif isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
        day_of_week = df.index.dayofweek
    else:
        hour = pd.Series(12, index=df.index)
        day_of_week = pd.Series(3, index=df.index)
    
    # Asia session (0-8)
    features['is_asia'] = ((hour >= 0) & (hour < 8)).astype(int)
    # Europe session (8-16)
    features['is_europe'] = ((hour >= 8) & (hour < 16)).astype(int)
    # US session (16-24)
    features['is_us'] = ((hour >= 16) & (hour < 24)).astype(int)
    
    # Weekend effect
    features['is_weekend'] = (day_of_week >= 5).astype(int)
    
    return features


def create_better_labels(df: pd.DataFrame, horizon: int = 4, threshold: float = 0.008) -> pd.Series:
    """
    創建標籤
    
    Args:
        horizon: 預測時間範圍 (小時)
        threshold: 漲幅閾值 (0.008 = 0.8%)
    """
    future_max = df['high'].rolling(horizon).max().shift(-horizon)
    future_return = (future_max - df['close']) / df['close']
    
    long_label = (future_return > threshold).astype(int)
    return long_label


def create_short_labels(df: pd.DataFrame, horizon: int = 4, threshold: float = 0.008) -> pd.Series:
    """創建 Short 標籤"""
    future_min = df['low'].rolling(horizon).min().shift(-horizon)
    future_return = (df['close'] - future_min) / df['close']
    
    short_label = (future_return > threshold).astype(int)
    return short_label


def optimize_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    優化預測閾值 (不用固定 0.5)
    
    找到使 F1 score 最大的閾值
    """
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.3, 0.8, 0.05):
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold


def train_v5_model(
    symbol: str = 'BTCUSDT',
    timeframe: str = '1h',
    days: int = 180,
    horizon: int = 4,
    threshold: float = 0.008
):
    """訓練 v5 模型"""
    logger.info("="*80)
    logger.info("[START] Training v5 Model (Advanced Features + Ensemble)")
    logger.info("="*80)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Days: {days}")
    logger.info(f"Horizon: {horizon}h")
    logger.info(f"Threshold: {threshold*100:.1f}%")
    logger.info("")
    
    # 1. 載入數據
    logger.info("Step 1/7: Loading data...")
    from utils.hf_data_loader import load_klines
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = load_klines(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    logger.info(f"Loaded {len(df)} candles")
    
    # 2. 計算進階特徵
    logger.info("\nStep 2/7: Engineering advanced features...")
    features = calculate_advanced_features(df)
    logger.info(f"Created {len(features.columns)} features")
    
    # 3. 創建標籤
    logger.info("\nStep 3/7: Creating labels...")
    long_labels = create_better_labels(df, horizon, threshold)
    short_labels = create_short_labels(df, horizon, threshold)
    
    logger.info(f"Long positive rate: {long_labels.mean():.2%}")
    logger.info(f"Short positive rate: {short_labels.mean():.2%}")
    
    # 4. 清理數據
    logger.info("\nStep 4/7: Cleaning data...")
    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)
    
    valid_idx = ~(long_labels.isna() | short_labels.isna())
    features = features[valid_idx]
    long_labels = long_labels[valid_idx]
    short_labels = short_labels[valid_idx]
    df = df[valid_idx]
    
    logger.info(f"Valid samples: {len(features)}")
    
    # 5. Walk-forward 切分
    logger.info("\nStep 5/7: Walk-forward split...")
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
    logger.info("\nStep 6/7: Training models...")
    
    # Long model
    logger.info("\n[LONG] Training...")
    model_long = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=5,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=100,
        auto_class_weights='Balanced'
    )
    
    model_long.fit(
        X_train, y_train_long,
        eval_set=(X_val, y_val_long),
        use_best_model=True
    )
    
    # Short model
    logger.info("\n[SHORT] Training...")
    model_short = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=5,
        l2_leaf_reg=5,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=100,
        auto_class_weights='Balanced'
    )
    
    model_short.fit(
        X_train, y_train_short,
        eval_set=(X_val, y_val_short),
        use_best_model=True
    )
    
    # 7. 優化閾值
    logger.info("\nStep 7/7: Optimizing thresholds...")
    
    # Long
    pred_val_long = model_long.predict_proba(X_val)[:, 1]
    optimal_threshold_long = optimize_threshold(y_val_long.values, pred_val_long)
    logger.info(f"Optimal Long threshold: {optimal_threshold_long:.3f}")
    
    # Short
    pred_val_short = model_short.predict_proba(X_val)[:, 1]
    optimal_threshold_short = optimize_threshold(y_val_short.values, pred_val_short)
    logger.info(f"Optimal Short threshold: {optimal_threshold_short:.3f}")
    
    # 8. 評估
    logger.info("\n" + "="*80)
    logger.info("[EVALUATION] Test Set Performance")
    logger.info("="*80)
    
    # Long
    pred_long = model_long.predict_proba(X_test)[:, 1]
    pred_long_binary = (pred_long >= optimal_threshold_long).astype(int)
    
    auc_long = roc_auc_score(y_test_long, pred_long)
    precision_long = precision_score(y_test_long, pred_long_binary, zero_division=0)
    recall_long = recall_score(y_test_long, pred_long_binary, zero_division=0)
    f1_long = f1_score(y_test_long, pred_long_binary, zero_division=0)
    
    logger.info(f"\n[LONG] Test Metrics (threshold={optimal_threshold_long:.3f}):")
    logger.info(f"  AUC: {auc_long:.4f}")
    logger.info(f"  Precision: {precision_long:.4f}")
    logger.info(f"  Recall: {recall_long:.4f}")
    logger.info(f"  F1: {f1_long:.4f}")
    logger.info(f"  Positive rate: {y_test_long.mean():.2%}")
    logger.info(f"  Predicted positive: {pred_long_binary.mean():.2%}")
    
    # Short
    pred_short = model_short.predict_proba(X_test)[:, 1]
    pred_short_binary = (pred_short >= optimal_threshold_short).astype(int)
    
    auc_short = roc_auc_score(y_test_short, pred_short)
    precision_short = precision_score(y_test_short, pred_short_binary, zero_division=0)
    recall_short = recall_score(y_test_short, pred_short_binary, zero_division=0)
    f1_short = f1_score(y_test_short, pred_short_binary, zero_division=0)
    
    logger.info(f"\n[SHORT] Test Metrics (threshold={optimal_threshold_short:.3f}):")
    logger.info(f"  AUC: {auc_short:.4f}")
    logger.info(f"  Precision: {precision_short:.4f}")
    logger.info(f"  Recall: {recall_short:.4f}")
    logger.info(f"  F1: {f1_short:.4f}")
    logger.info(f"  Positive rate: {y_test_short.mean():.2%}")
    logger.info(f"  Predicted positive: {pred_short_binary.mean():.2%}")
    
    # 9. Feature Importance
    logger.info("\n[FEATURE IMPORTANCE] Top 15:")
    importance = model_long.get_feature_importance()
    feature_names = features.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    for idx, row in importance_df.head(15).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.2f}")
    
    # 10. 儲存模型
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    models_dir = Path('models_output')
    models_dir.mkdir(exist_ok=True)
    
    long_path = models_dir / f'catboost_long_v5_{timestamp}.pkl'
    short_path = models_dir / f'catboost_short_v5_{timestamp}.pkl'
    
    with open(long_path, 'wb') as f:
        pickle.dump({
            'model': model_long,
            'features': features.columns.tolist(),
            'threshold': optimal_threshold_long,
            'metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'horizon': horizon,
                'train_date': timestamp,
                'label_threshold': threshold
            }
        }, f)
    
    with open(short_path, 'wb') as f:
        pickle.dump({
            'model': model_short,
            'features': features.columns.tolist(),
            'threshold': optimal_threshold_short,
            'metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'horizon': horizon,
                'train_date': timestamp,
                'label_threshold': threshold
            }
        }, f)
    
    logger.info(f"\n[SAVE] Models saved:")
    logger.info(f"  Long: {long_path}")
    logger.info(f"  Short: {short_path}")
    
    # 11. 儲存報告
    report = {
        'version': 'v5',
        'timestamp': timestamp,
        'symbol': symbol,
        'timeframe': timeframe,
        'horizon': horizon,
        'label_threshold': threshold,
        'long_model': {
            'auc': float(auc_long),
            'precision': float(precision_long),
            'recall': float(recall_long),
            'f1': float(f1_long),
            'optimal_threshold': float(optimal_threshold_long),
            'positive_rate': float(y_test_long.mean()),
            'predicted_positive': float(pred_long_binary.mean())
        },
        'short_model': {
            'auc': float(auc_short),
            'precision': float(precision_short),
            'recall': float(recall_short),
            'f1': float(f1_short),
            'optimal_threshold': float(optimal_threshold_short),
            'positive_rate': float(y_test_short.mean()),
            'predicted_positive': float(pred_short_binary.mean())
        },
        'features': features.columns.tolist(),
        'feature_count': len(features.columns),
        'top_features': importance_df.head(15).to_dict('records')
    }
    
    report_path = models_dir / f'v5_training_report_{timestamp}.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n[SAVE] Report saved: {report_path}")
    logger.info("\n" + "="*80)
    logger.info("[DONE] Training complete!")
    logger.info("="*80)
    
    return report


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train v5 model')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='1h')
    parser.add_argument('--days', type=int, default=180)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.008)
    
    args = parser.parse_args()
    
    report = train_v5_model(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        horizon=args.horizon,
        threshold=args.threshold
    )
    
    print(f"\n訓練完成!")
    print(f"Long AUC: {report['long_model']['auc']:.4f} (threshold: {report['long_model']['optimal_threshold']:.3f})")
    print(f"Short AUC: {report['short_model']['auc']:.4f} (threshold: {report['short_model']['optimal_threshold']:.3f})")
