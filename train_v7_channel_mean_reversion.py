#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v7 通道均值回歸模型

核心概念:
1. 觸碰上軌 -> 預測是否會回到中軌 (不是固定%數)
2. 觸碰下軌 -> 預測是否會回到中軌
3. 成功 = 到達中軌且沒有先觸及反向止損
4. 失敗 = 繼續突破或觸及止損

這是真正的通道交易邏輯!
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
        logging.FileHandler('logs/train_v7_mr.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def calculate_keltner_channels(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 14, multiplier: float = 2.0):
    """計算肯特納通道"""
    middle = df['close'].ewm(span=ema_period, adjust=False).mean()
    
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(atr_period).mean()
    upper = middle + (atr * multiplier)
    lower = middle - (atr * multiplier)
    
    return upper, middle, lower


def identify_touch_events(df: pd.DataFrame, upper: pd.Series, lower: pd.Series, threshold: float = 0.003):
    """識別觸碰通道邊界的事件"""
    dist_to_upper = (upper - df['close']) / df['close']
    dist_to_lower = (df['close'] - lower) / df['close']
    
    upper_touch = (dist_to_upper <= threshold) & (dist_to_upper >= -threshold)
    lower_touch = (dist_to_lower <= threshold) & (dist_to_lower >= -threshold)
    
    return upper_touch, lower_touch


def calculate_bounce_features(df: pd.DataFrame, upper: pd.Series, middle: pd.Series, lower: pd.Series) -> pd.DataFrame:
    """計算反彈相關特徵"""
    features = pd.DataFrame(index=df.index)
    
    # 通道位置
    channel_width = upper - lower
    features['position_in_channel'] = (df['close'] - lower) / (channel_width + 1e-8)
    features['dist_to_upper'] = (upper - df['close']) / df['close']
    features['dist_to_lower'] = (df['close'] - lower) / df['close']
    features['dist_to_middle'] = (df['close'] - middle) / middle
    
    # 動量
    features['momentum_1'] = df['close'].pct_change(1)
    features['momentum_2'] = df['close'].pct_change(2)
    features['momentum_3'] = df['close'].pct_change(3)
    features['acceleration'] = features['momentum_1'] - features['momentum_1'].shift(1)
    
    # K線形態
    high_low_range = df['high'] - df['low'] + 1e-8
    body = abs(df['close'] - df['open'])
    features['body_ratio'] = body / high_low_range
    
    upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
    lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
    features['upper_shadow_ratio'] = upper_shadow / high_low_range
    features['lower_shadow_ratio'] = lower_shadow / high_low_range
    features['buy_pressure'] = (df['close'] - df['low']) / high_low_range
    
    # 成交量
    vol_ma6 = df['volume'].rolling(6).mean()
    vol_ma24 = df['volume'].rolling(24).mean()
    features['volume_ratio_6'] = df['volume'] / (vol_ma6 + 1e-8)
    features['volume_ratio_24'] = df['volume'] / (vol_ma24 + 1e-8)
    
    price_change = df['close'].pct_change()
    features['volume_price_corr'] = price_change.rolling(6).corr(df['volume'].pct_change())
    
    # 波動率
    features['volatility'] = df['close'].pct_change().rolling(6).std()
    features['volatility_24h'] = df['close'].pct_change().rolling(24).std()
    features['volatility_expanding'] = (features['volatility'] > features['volatility_24h']).astype(int)
    
    # 通道特徵
    features['channel_width'] = (upper - lower) / middle
    features['channel_width_change'] = features['channel_width'].pct_change(3)
    features['middle_slope'] = middle.pct_change(3)
    features['deviation_from_middle'] = abs(df['close'] - middle) / middle
    
    # 歷史觸碰
    dist_to_upper_abs = abs((upper - df['close']) / df['close'])
    dist_to_lower_abs = abs((df['close'] - lower) / df['close'])
    near_upper = (dist_to_upper_abs < 0.005).astype(int)
    near_lower = (dist_to_lower_abs < 0.005).astype(int)
    features['upper_touches_24h'] = near_upper.rolling(24).sum()
    features['lower_touches_24h'] = near_lower.rolling(24).sum()
    
    # RSI
    price_changes = df['close'].diff()
    gains = price_changes.where(price_changes > 0, 0).rolling(14).mean()
    losses = -price_changes.where(price_changes < 0, 0).rolling(14).mean()
    features['rsi_like'] = 100 - (100 / (1 + gains / (losses + 1e-8)))
    
    # 時間
    if 'open_time' in df.columns:
        hour = pd.to_datetime(df['open_time']).dt.hour
    elif isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
    else:
        hour = pd.Series(12, index=df.index)
    features['is_high_vol_time'] = ((hour >= 8) & (hour < 16)).astype(int)
    
    return features


def create_mean_reversion_labels(
    df: pd.DataFrame,
    upper: pd.Series,
    middle: pd.Series,
    lower: pd.Series,
    upper_touch: pd.Series,
    lower_touch: pd.Series,
    horizon: int = 12,
    sl_multiplier: float = 0.5
):
    """
    創建均值回歸標籤
    
    上軌觸碰 (Short 訊號):
    - 成功: 未來 horizon 內價格到達中軌,且沒有先觸及止損
    - 止損: 上軌 + (channel_width * sl_multiplier)
    
    下軌觸碰 (Long 訊號):
    - 成功: 未來 horizon 內價格到達中軌,且沒有先觸及止損
    - 止損: 下軌 - (channel_width * sl_multiplier)
    
    Args:
        horizon: 看未來幾根K線
        sl_multiplier: 止損距離 = 通道寬度 * sl_multiplier (0.5 = 通道的一半)
    """
    upper_labels = pd.Series(0, index=df.index)
    lower_labels = pd.Series(0, index=df.index)
    
    channel_width = upper - lower
    
    for i in range(len(df) - horizon):
        # === 上軌觸碰 (Short) ===
        if upper_touch.iloc[i]:
            entry_price = df['close'].iloc[i]
            tp_price = middle.iloc[i]  # 目標 = 中軌
            sl_price = upper.iloc[i] + (channel_width.iloc[i] * sl_multiplier)  # 止損
            
            # 看未來 horizon 根K線
            future_highs = df['high'].iloc[i+1:i+1+horizon]
            future_lows = df['low'].iloc[i+1:i+1+horizon]
            
            if len(future_highs) > 0:
                # 檢查是否先觸及止損
                hit_sl = (future_highs >= sl_price).any()
                
                if not hit_sl:
                    # 沒觸及止損,檢查是否到達目標
                    hit_tp = (future_lows <= tp_price).any()
                    if hit_tp:
                        upper_labels.iloc[i] = 1
        
        # === 下軌觸碰 (Long) ===
        if lower_touch.iloc[i]:
            entry_price = df['close'].iloc[i]
            tp_price = middle.iloc[i]  # 目標 = 中軌
            sl_price = lower.iloc[i] - (channel_width.iloc[i] * sl_multiplier)  # 止損
            
            future_highs = df['high'].iloc[i+1:i+1+horizon]
            future_lows = df['low'].iloc[i+1:i+1+horizon]
            
            if len(future_lows) > 0:
                # 檢查是否先觸及止損
                hit_sl = (future_lows <= sl_price).any()
                
                if not hit_sl:
                    # 沒觸及止損,檢查是否到達目標
                    hit_tp = (future_highs >= tp_price).any()
                    if hit_tp:
                        lower_labels.iloc[i] = 1
    
    return upper_labels, lower_labels


def train_v7_model(
    symbol: str = 'BTCUSDT',
    timeframe: str = '15m',
    days: int = 9999,
    ema_period: int = 20,
    atr_period: int = 14,
    multiplier: float = 2.0,
    horizon: int = 12,
    sl_multiplier: float = 0.5,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
):
    """訓練 v7 均值回歸模型"""
    logger.info("="*80)
    logger.info("[START] Training v7 Model - Channel Mean Reversion")
    logger.info("="*80)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Days: {days}")
    logger.info(f"Keltner: EMA={ema_period}, ATR={atr_period}, Multiplier={multiplier}")
    logger.info(f"Horizon: {horizon} bars")
    logger.info(f"SL Multiplier: {sl_multiplier}x channel width")
    logger.info(f"Data split: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={1-train_ratio-val_ratio:.0%}")
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
    
    # 2. 計算肯特納通道
    logger.info("\nStep 2/7: Calculating Keltner Channels...")
    upper, middle, lower = calculate_keltner_channels(df, ema_period, atr_period, multiplier)
    
    df['keltner_upper'] = upper
    df['keltner_middle'] = middle
    df['keltner_lower'] = lower
    
    # 3. 識別觸碰事件
    logger.info("\nStep 3/7: Identifying touch events...")
    upper_touch, lower_touch = identify_touch_events(df, upper, lower)
    
    logger.info(f"Upper band touches: {upper_touch.sum()}")
    logger.info(f"Lower band touches: {lower_touch.sum()}")
    
    # 4. 計算特徵
    logger.info("\nStep 4/7: Engineering features...")
    features = calculate_bounce_features(df, upper, middle, lower)
    logger.info(f"Created {len(features.columns)} features")
    
    # 5. 創建均值回歸標籤
    logger.info("\nStep 5/7: Creating mean reversion labels...")
    logger.info("  Target: Reach middle band without hitting stop loss")
    
    upper_labels, lower_labels = create_mean_reversion_labels(
        df, upper, middle, lower, upper_touch, lower_touch, horizon, sl_multiplier
    )
    
    upper_samples = upper_touch
    lower_samples = lower_touch
    
    upper_success_rate = upper_labels[upper_samples].mean()
    lower_success_rate = lower_labels[lower_samples].mean()
    
    logger.info(f"Upper touch samples: {upper_samples.sum()}")
    logger.info(f"  - Successful mean reversion: {upper_labels[upper_samples].sum()} ({upper_success_rate:.1%})")
    logger.info(f"Lower touch samples: {lower_samples.sum()}")
    logger.info(f"  - Successful mean reversion: {lower_labels[lower_samples].sum()} ({lower_success_rate:.1%})")
    
    # 6. 準備訓練數據
    logger.info("\nStep 6/7: Preparing training data...")
    
    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    
    X_upper = features[upper_samples].copy()
    y_upper = upper_labels[upper_samples].copy()
    X_lower = features[lower_samples].copy()
    y_lower = lower_labels[lower_samples].copy()
    
    logger.info(f"Upper model samples: {len(X_upper)}")
    logger.info(f"Lower model samples: {len(X_lower)}")
    
    def time_split(X, y, train_r, val_r):
        n = len(X)
        train_end = int(n * train_r)
        val_end = int(n * (train_r + val_r))
        return (
            X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:],
            y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]
        )
    
    # 7. 訓練模型
    logger.info("\nStep 7/7: Training models...")
    results = {}
    
    # Upper band model (Short)
    if len(X_upper) >= 100:
        logger.info("\n[UPPER BAND] Training short signal model...")
        X_train, X_val, X_test, y_train, y_val, y_test = time_split(X_upper, y_upper, train_ratio, val_ratio)
        
        logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"  Train positive rate: {y_train.mean():.2%}")
        
        model_upper = CatBoostClassifier(
            iterations=800,
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
        
        model_upper.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        
        pred_proba = model_upper.predict_proba(X_test)[:, 1]
        pred_binary = (pred_proba >= 0.5).astype(int)
        
        auc = roc_auc_score(y_test, pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        precision = precision_score(y_test, pred_binary, zero_division=0)
        recall = recall_score(y_test, pred_binary, zero_division=0)
        f1 = f1_score(y_test, pred_binary, zero_division=0)
        
        logger.info(f"\n  Test Results:")
        logger.info(f"    AUC: {auc:.4f}")
        logger.info(f"    Precision: {precision:.4f}")
        logger.info(f"    Recall: {recall:.4f}")
        logger.info(f"    F1: {f1:.4f}")
        logger.info(f"    Actual positive: {y_test.mean():.2%}")
        logger.info(f"    Predicted positive: {pred_binary.mean():.2%}")
        
        results['upper'] = {'model': model_upper, 'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1}
    else:
        logger.info(f"\n[UPPER BAND] Insufficient samples ({len(X_upper)})")
        results['upper'] = None
    
    # Lower band model (Long)
    if len(X_lower) >= 100:
        logger.info("\n[LOWER BAND] Training long signal model...")
        X_train, X_val, X_test, y_train, y_val, y_test = time_split(X_lower, y_lower, train_ratio, val_ratio)
        
        logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"  Train positive rate: {y_train.mean():.2%}")
        
        model_lower = CatBoostClassifier(
            iterations=800,
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
        
        model_lower.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        
        pred_proba = model_lower.predict_proba(X_test)[:, 1]
        pred_binary = (pred_proba >= 0.5).astype(int)
        
        auc = roc_auc_score(y_test, pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        precision = precision_score(y_test, pred_binary, zero_division=0)
        recall = recall_score(y_test, pred_binary, zero_division=0)
        f1 = f1_score(y_test, pred_binary, zero_division=0)
        
        logger.info(f"\n  Test Results:")
        logger.info(f"    AUC: {auc:.4f}")
        logger.info(f"    Precision: {precision:.4f}")
        logger.info(f"    Recall: {recall:.4f}")
        logger.info(f"    F1: {f1:.4f}")
        logger.info(f"    Actual positive: {y_test.mean():.2%}")
        logger.info(f"    Predicted positive: {pred_binary.mean():.2%}")
        
        results['lower'] = {'model': model_lower, 'auc': auc, 'precision': precision, 'recall': recall, 'f1': f1}
    else:
        logger.info(f"\n[LOWER BAND] Insufficient samples ({len(X_lower)})")
        results['lower'] = None
    
    # 8. 儲存模型
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    models_dir = Path('models_output')
    models_dir.mkdir(exist_ok=True)
    
    if results['upper']:
        path = models_dir / f'keltner_upper_{timeframe}_v7mr_{timestamp}.pkl'
        with open(path, 'wb') as f:
            pickle.dump({
                'model': results['upper']['model'],
                'features': features.columns.tolist(),
                'metadata': {
                    'symbol': symbol, 'timeframe': timeframe, 'type': 'upper_band_short_mr',
                    'keltner_params': {'ema_period': ema_period, 'atr_period': atr_period, 'multiplier': multiplier},
                    'strategy': 'mean_reversion_to_middle', 'sl_multiplier': sl_multiplier,
                    'train_samples': len(X_upper), 'success_rate': upper_success_rate
                }
            }, f)
        logger.info(f"\n[SAVE] Upper: {path}")
    
    if results['lower']:
        path = models_dir / f'keltner_lower_{timeframe}_v7mr_{timestamp}.pkl'
        with open(path, 'wb') as f:
            pickle.dump({
                'model': results['lower']['model'],
                'features': features.columns.tolist(),
                'metadata': {
                    'symbol': symbol, 'timeframe': timeframe, 'type': 'lower_band_long_mr',
                    'keltner_params': {'ema_period': ema_period, 'atr_period': atr_period, 'multiplier': multiplier},
                    'strategy': 'mean_reversion_to_middle', 'sl_multiplier': sl_multiplier,
                    'train_samples': len(X_lower), 'success_rate': lower_success_rate
                }
            }, f)
        logger.info(f"[SAVE] Lower: {path}")
    
    logger.info("\n" + "="*80)
    logger.info("[DONE] Training complete!")
    logger.info("="*80)
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train v7 mean reversion model')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='15m')
    parser.add_argument('--days', type=int, default=9999)
    parser.add_argument('--ema', type=int, default=20)
    parser.add_argument('--atr', type=int, default=14)
    parser.add_argument('--multiplier', type=float, default=2.0)
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--sl-multiplier', type=float, default=0.5)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    
    args = parser.parse_args()
    
    results = train_v7_model(
        symbol=args.symbol, timeframe=args.timeframe, days=args.days,
        ema_period=args.ema, atr_period=args.atr, multiplier=args.multiplier,
        horizon=args.horizon, sl_multiplier=args.sl_multiplier,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )
    
    print("\n訓練完成!")
    if results['upper']:
        print(f"Upper (Short): AUC={results['upper']['auc']:.3f}, Precision={results['upper']['precision']:.3f}")
    if results['lower']:
        print(f"Lower (Long): AUC={results['lower']['auc']:.3f}, Precision={results['lower']['precision']:.3f}")
