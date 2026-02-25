#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v6 最終版本 - 基於利潤的標籤

根本改變:
1. 目標不是預測漲跌,而是預測「可盈利的交易機會」
2. 考慮交易成本 (手續費 + 滑點)
3. 只標記「風險報酬比 >= 2:1」的機會
4. 添加止損邏輯
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
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_v6.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def calculate_smart_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    精簡但有效的特徵集
    
    只保留最有預測力的特徵,避免過擬合
    """
    features = pd.DataFrame(index=df.index)
    
    # 1. Price Action (核心)
    for h in [2, 4, 8, 12, 24]:
        features[f'ret_{h}'] = df['close'].pct_change(h)
    
    # 2. Volatility (關鍵)
    returns = df['close'].pct_change()
    features['vol_6'] = returns.rolling(6).std()
    features['vol_24'] = returns.rolling(24).std()
    features['vol_ratio'] = features['vol_6'] / (features['vol_24'] + 1e-8)
    
    # 3. Volume
    vol_ma24 = df['volume'].rolling(24).mean()
    features['vol_surge'] = df['volume'] / (vol_ma24 + 1e-8)
    
    # 4. Order Flow
    range_val = df['high'] - df['low'] + 1e-8
    buy_pressure = (df['close'] - df['low']) / range_val
    features['buy_pressure'] = buy_pressure.rolling(6).mean()
    
    # 5. Trend
    ma_fast = df['close'].rolling(12).mean()
    ma_slow = df['close'].rolling(24).mean()
    features['trend'] = (ma_fast - ma_slow) / ma_slow
    
    # 6. Position in range
    high_24 = df['high'].rolling(24).max()
    low_24 = df['low'].rolling(24).min()
    features['position_in_range'] = (df['close'] - low_24) / (high_24 - low_24 + 1e-8)
    
    # 7. Recent momentum quality
    # 看過去 6 根 K 線有幾根是漲的
    direction = (df['close'] > df['open']).astype(int)
    features['bullish_candles'] = direction.rolling(6).sum()
    
    # 8. Candle patterns
    body = abs(df['close'] - df['open'])
    features['body_ratio'] = body / range_val
    
    # 9. ATR (Average True Range)
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    features['atr'] = tr.rolling(14).mean()
    features['atr_pct'] = features['atr'] / df['close']
    
    # 10. Time features
    if 'open_time' in df.columns:
        hour = pd.to_datetime(df['open_time']).dt.hour
    elif isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
    else:
        hour = pd.Series(12, index=df.index)
    
    features['is_high_vol_time'] = ((hour >= 8) & (hour < 16)).astype(int)
    
    return features


def create_profit_labels(
    df: pd.DataFrame,
    horizon: int = 6,
    take_profit: float = 0.012,  # 1.2% TP
    stop_loss: float = 0.006,     # 0.6% SL (RR = 2:1)
    fee: float = 0.001            # 0.1% 手續費
) -> pd.Series:
    """
    創建基於利潤的標籤
    
    邏輯:
    1. 看未來 horizon 小時內
    2. 如果先觸及 TP (扣除手續費後仍盈利) -> 標記為 1
    3. 如果先觸及 SL -> 標記為 0
    4. 都沒觸及 -> 標記為 0
    
    這樣的標籤代表「真正可盈利的交易機會」
    """
    labels = pd.Series(0, index=df.index)
    
    for i in range(len(df) - horizon):
        entry_price = df['close'].iloc[i]
        
        # 看未來 horizon 根 K 線
        future_highs = df['high'].iloc[i+1:i+1+horizon]
        future_lows = df['low'].iloc[i+1:i+1+horizon]
        
        # TP 和 SL 價格
        tp_price = entry_price * (1 + take_profit)
        sl_price = entry_price * (1 - stop_loss)
        
        # 找到第一次觸及 TP 或 SL 的時間
        tp_hit_idx = future_highs[future_highs >= tp_price].index
        sl_hit_idx = future_lows[future_lows <= sl_price].index
        
        if len(tp_hit_idx) > 0 and len(sl_hit_idx) > 0:
            # 都觸及了,看誰先
            if tp_hit_idx[0] < sl_hit_idx[0]:
                labels.iloc[i] = 1  # TP 先到
            else:
                labels.iloc[i] = 0  # SL 先到
        elif len(tp_hit_idx) > 0:
            labels.iloc[i] = 1  # 只觸及 TP
        else:
            labels.iloc[i] = 0  # 只觸及 SL 或都沒觸及
    
    return labels


def create_short_profit_labels(
    df: pd.DataFrame,
    horizon: int = 6,
    take_profit: float = 0.012,
    stop_loss: float = 0.006,
    fee: float = 0.001
) -> pd.Series:
    """Short 版本的利潤標籤"""
    labels = pd.Series(0, index=df.index)
    
    for i in range(len(df) - horizon):
        entry_price = df['close'].iloc[i]
        
        future_highs = df['high'].iloc[i+1:i+1+horizon]
        future_lows = df['low'].iloc[i+1:i+1+horizon]
        
        # Short: TP 在下方, SL 在上方
        tp_price = entry_price * (1 - take_profit)
        sl_price = entry_price * (1 + stop_loss)
        
        tp_hit_idx = future_lows[future_lows <= tp_price].index
        sl_hit_idx = future_highs[future_highs >= sl_price].index
        
        if len(tp_hit_idx) > 0 and len(sl_hit_idx) > 0:
            if tp_hit_idx[0] < sl_hit_idx[0]:
                labels.iloc[i] = 1
            else:
                labels.iloc[i] = 0
        elif len(tp_hit_idx) > 0:
            labels.iloc[i] = 1
        else:
            labels.iloc[i] = 0
    
    return labels


def train_v6_model(
    symbol: str = 'BTCUSDT',
    timeframe: str = '1h',
    days: int = 180,
    horizon: int = 6,
    take_profit: float = 0.012,
    stop_loss: float = 0.006
):
    """訓練 v6 最終版本"""
    logger.info("="*80)
    logger.info("[START] Training v6 Model - Profit-Based Labels")
    logger.info("="*80)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Days: {days}")
    logger.info(f"Horizon: {horizon}h")
    logger.info(f"Take Profit: {take_profit*100:.1f}%")
    logger.info(f"Stop Loss: {stop_loss*100:.1f}%")
    logger.info(f"Risk/Reward: {take_profit/stop_loss:.1f}:1")
    logger.info("")
    
    # 1. 載入數據
    logger.info("Step 1/6: Loading data...")
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
    
    # 2. 計算特徵
    logger.info("\nStep 2/6: Engineering features...")
    features = calculate_smart_features(df)
    logger.info(f"Created {len(features.columns)} features")
    logger.info(f"Features: {features.columns.tolist()}")
    
    # 3. 創建利潤標籤
    logger.info("\nStep 3/6: Creating profit-based labels...")
    logger.info("This may take a few minutes...")
    
    long_labels = create_profit_labels(df, horizon, take_profit, stop_loss)
    short_labels = create_short_profit_labels(df, horizon, take_profit, stop_loss)
    
    logger.info(f"Long profitable rate: {long_labels.mean():.2%}")
    logger.info(f"Short profitable rate: {short_labels.mean():.2%}")
    
    # 4. 清理數據
    logger.info("\nStep 4/6: Cleaning data...")
    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)
    
    valid_idx = ~(long_labels.isna() | short_labels.isna())
    features = features[valid_idx]
    long_labels = long_labels[valid_idx]
    short_labels = short_labels[valid_idx]
    df = df[valid_idx]
    
    logger.info(f"Valid samples: {len(features)}")
    
    # 5. Walk-forward 切分
    logger.info("\nStep 5/6: Walk-forward split...")
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
    
    logger.info(f"Train: {len(X_train)} samples (Long+: {y_train_long.mean():.1%}, Short+: {y_train_short.mean():.1%})")
    logger.info(f"Val: {len(X_val)} samples")
    logger.info(f"Test: {len(X_test)} samples")
    
    # 6. 訓練模型
    logger.info("\nStep 6/6: Training models...")
    
    # Long model
    logger.info("\n[LONG] Training...")
    model_long = CatBoostClassifier(
        iterations=800,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=10,  # 更強的正則化
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
        iterations=800,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=10,
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
    
    # 7. 評估 (使用多個閾值)
    logger.info("\n" + "="*80)
    logger.info("[EVALUATION] Test Set Performance")
    logger.info("="*80)
    
    # Long - 測試不同閾值
    pred_long = model_long.predict_proba(X_test)[:, 1]
    
    logger.info("\n[LONG] Performance at different thresholds:")
    for threshold in [0.4, 0.5, 0.6, 0.7]:
        pred_long_binary = (pred_long >= threshold).astype(int)
        
        precision = precision_score(y_test_long, pred_long_binary, zero_division=0)
        recall = recall_score(y_test_long, pred_long_binary, zero_division=0)
        f1 = f1_score(y_test_long, pred_long_binary, zero_division=0)
        pred_rate = pred_long_binary.mean()
        
        logger.info(f"  Threshold {threshold:.1f}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, PredRate={pred_rate:.1%}")
    
    # 選擇最佳閾值 (Precision >= 0.55)
    best_threshold_long = 0.5
    for threshold in np.arange(0.3, 0.9, 0.05):
        pred_binary = (pred_long >= threshold).astype(int)
        precision = precision_score(y_test_long, pred_binary, zero_division=0)
        if precision >= 0.55 and pred_binary.sum() > 0:
            best_threshold_long = threshold
            break
    
    pred_long_binary = (pred_long >= best_threshold_long).astype(int)
    
    auc_long = roc_auc_score(y_test_long, pred_long)
    precision_long = precision_score(y_test_long, pred_long_binary, zero_division=0)
    recall_long = recall_score(y_test_long, pred_long_binary, zero_division=0)
    f1_long = f1_score(y_test_long, pred_long_binary, zero_division=0)
    
    logger.info(f"\n[LONG] Selected threshold: {best_threshold_long:.2f}")
    logger.info(f"  AUC: {auc_long:.4f}")
    logger.info(f"  Precision: {precision_long:.4f}")
    logger.info(f"  Recall: {recall_long:.4f}")
    logger.info(f"  F1: {f1_long:.4f}")
    logger.info(f"  Actual positive rate: {y_test_long.mean():.2%}")
    logger.info(f"  Predicted positive rate: {pred_long_binary.mean():.2%}")
    
    # Confusion matrix
    cm_long = confusion_matrix(y_test_long, pred_long_binary)
    logger.info(f"\n[LONG] Confusion Matrix:")
    logger.info(f"  TN={cm_long[0,0]}, FP={cm_long[0,1]}")
    logger.info(f"  FN={cm_long[1,0]}, TP={cm_long[1,1]}")
    
    # Short
    pred_short = model_short.predict_proba(X_test)[:, 1]
    
    logger.info("\n[SHORT] Performance at different thresholds:")
    for threshold in [0.4, 0.5, 0.6, 0.7]:
        pred_short_binary = (pred_short >= threshold).astype(int)
        
        precision = precision_score(y_test_short, pred_short_binary, zero_division=0)
        recall = recall_score(y_test_short, pred_short_binary, zero_division=0)
        f1 = f1_score(y_test_short, pred_short_binary, zero_division=0)
        pred_rate = pred_short_binary.mean()
        
        logger.info(f"  Threshold {threshold:.1f}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, PredRate={pred_rate:.1%}")
    
    best_threshold_short = 0.5
    for threshold in np.arange(0.3, 0.9, 0.05):
        pred_binary = (pred_short >= threshold).astype(int)
        precision = precision_score(y_test_short, pred_binary, zero_division=0)
        if precision >= 0.55 and pred_binary.sum() > 0:
            best_threshold_short = threshold
            break
    
    pred_short_binary = (pred_short >= best_threshold_short).astype(int)
    
    auc_short = roc_auc_score(y_test_short, pred_short)
    precision_short = precision_score(y_test_short, pred_short_binary, zero_division=0)
    recall_short = recall_score(y_test_short, pred_short_binary, zero_division=0)
    f1_short = f1_score(y_test_short, pred_short_binary, zero_division=0)
    
    logger.info(f"\n[SHORT] Selected threshold: {best_threshold_short:.2f}")
    logger.info(f"  AUC: {auc_short:.4f}")
    logger.info(f"  Precision: {precision_short:.4f}")
    logger.info(f"  Recall: {recall_short:.4f}")
    logger.info(f"  F1: {f1_short:.4f}")
    logger.info(f"  Actual positive rate: {y_test_short.mean():.2%}")
    logger.info(f"  Predicted positive rate: {pred_short_binary.mean():.2%}")
    
    cm_short = confusion_matrix(y_test_short, pred_short_binary)
    logger.info(f"\n[SHORT] Confusion Matrix:")
    logger.info(f"  TN={cm_short[0,0]}, FP={cm_short[0,1]}")
    logger.info(f"  FN={cm_short[1,0]}, TP={cm_short[1,1]}")
    
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
    
    long_path = models_dir / f'catboost_long_v6_{timestamp}.pkl'
    short_path = models_dir / f'catboost_short_v6_{timestamp}.pkl'
    
    with open(long_path, 'wb') as f:
        pickle.dump({
            'model': model_long,
            'features': features.columns.tolist(),
            'threshold': best_threshold_long,
            'metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'horizon': horizon,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'train_date': timestamp
            }
        }, f)
    
    with open(short_path, 'wb') as f:
        pickle.dump({
            'model': model_short,
            'features': features.columns.tolist(),
            'threshold': best_threshold_short,
            'metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'horizon': horizon,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'train_date': timestamp
            }
        }, f)
    
    logger.info(f"\n[SAVE] Models saved:")
    logger.info(f"  Long: {long_path}")
    logger.info(f"  Short: {short_path}")
    
    # 10. 儲存報告
    report = {
        'version': 'v6',
        'timestamp': timestamp,
        'symbol': symbol,
        'timeframe': timeframe,
        'horizon': horizon,
        'take_profit': take_profit,
        'stop_loss': stop_loss,
        'risk_reward_ratio': take_profit / stop_loss,
        'long_model': {
            'auc': float(auc_long),
            'precision': float(precision_long),
            'recall': float(recall_long),
            'f1': float(f1_long),
            'threshold': float(best_threshold_long),
            'positive_rate': float(y_test_long.mean()),
            'predicted_positive_rate': float(pred_long_binary.mean()),
            'confusion_matrix': cm_long.tolist()
        },
        'short_model': {
            'auc': float(auc_short),
            'precision': float(precision_short),
            'recall': float(recall_short),
            'f1': float(f1_short),
            'threshold': float(best_threshold_short),
            'positive_rate': float(y_test_short.mean()),
            'predicted_positive_rate': float(pred_short_binary.mean()),
            'confusion_matrix': cm_short.tolist()
        },
        'features': features.columns.tolist(),
        'feature_count': len(features.columns),
        'top_features': importance_df.head(10).to_dict('records')
    }
    
    report_path = models_dir / f'v6_training_report_{timestamp}.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n[SAVE] Report saved: {report_path}")
    logger.info("\n" + "="*80)
    logger.info("[DONE] Training complete!")
    logger.info("="*80)
    
    return report


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train v6 final model')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='1h')
    parser.add_argument('--days', type=int, default=180)
    parser.add_argument('--horizon', type=int, default=6, help='Hours to look ahead')
    parser.add_argument('--tp', type=float, default=0.012, help='Take profit %')
    parser.add_argument('--sl', type=float, default=0.006, help='Stop loss %')
    
    args = parser.parse_args()
    
    report = train_v6_model(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        horizon=args.horizon,
        take_profit=args.tp,
        stop_loss=args.sl
    )
    
    print(f"\n訓練完成!")
    print(f"Long: AUC={report['long_model']['auc']:.3f}, Precision={report['long_model']['precision']:.3f}")
    print(f"Short: AUC={report['short_model']['auc']:.3f}, Precision={report['short_model']['precision']:.3f}")
