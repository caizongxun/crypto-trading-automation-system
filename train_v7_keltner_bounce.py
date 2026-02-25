#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v7 肯特納通道反彈預測系統

核心概念:
1. 只在價格觸碰肯特納通道邊界時才預測
2. 學習「有效反彈」vs「假突破」的特徵
3. 目標: 預測從通道邊界反彈的機會

這是一個更精準、更少訊號的策略
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
        logging.FileHandler('logs/train_v7.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def calculate_keltner_channels(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 14, multiplier: float = 2.0):
    """
    計算肯特納通道
    
    Args:
        ema_period: EMA 週期
        atr_period: ATR 週期
        multiplier: ATR 倍數
    
    Returns:
        upper, middle, lower bands
    """
    # 計算 EMA
    middle = df['close'].ewm(span=ema_period, adjust=False).mean()
    
    # 計算 ATR
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(atr_period).mean()
    
    # 計算通道
    upper = middle + (atr * multiplier)
    lower = middle - (atr * multiplier)
    
    return upper, middle, lower


def identify_touch_events(df: pd.DataFrame, upper: pd.Series, lower: pd.Series, threshold: float = 0.003):
    """
    識別觸碰通道邊界的事件
    
    Args:
        threshold: 距離閾值 (0.3% 內算觸碰)
    
    Returns:
        upper_touch, lower_touch (boolean Series)
    """
    # 計算距離
    dist_to_upper = (upper - df['close']) / df['close']
    dist_to_lower = (df['close'] - lower) / df['close']
    
    # 觸碰上軌 (接近上軌)
    upper_touch = (dist_to_upper <= threshold) & (dist_to_upper >= -threshold)
    
    # 觸碰下軌 (接近下軌)
    lower_touch = (dist_to_lower <= threshold) & (dist_to_lower >= -threshold)
    
    return upper_touch, lower_touch


def calculate_bounce_features(df: pd.DataFrame, upper: pd.Series, middle: pd.Series, lower: pd.Series) -> pd.DataFrame:
    """
    計算反彈相關特徵
    
    關鍵特徵:
    1. 當前在通道中的位置
    2. 接近邊界的速度和角度
    3. 成交量特徵
    4. 市場動能
    5. 波動率狀態
    """
    features = pd.DataFrame(index=df.index)
    
    # === 1. 通道位置特徵 ===
    channel_width = upper - lower
    features['position_in_channel'] = (df['close'] - lower) / (channel_width + 1e-8)
    features['dist_to_upper'] = (upper - df['close']) / df['close']
    features['dist_to_lower'] = (df['close'] - lower) / df['close']
    features['dist_to_middle'] = (df['close'] - middle) / middle
    
    # === 2. 接近速度 (動量) ===
    # 最近 3 根 K 線的價格變化
    features['momentum_1'] = df['close'].pct_change(1)
    features['momentum_2'] = df['close'].pct_change(2)
    features['momentum_3'] = df['close'].pct_change(3)
    
    # 加速度 (momentum 的變化)
    features['acceleration'] = features['momentum_1'] - features['momentum_1'].shift(1)
    
    # === 3. K 線形態 ===
    high_low_range = df['high'] - df['low'] + 1e-8
    
    # 實體大小
    body = abs(df['close'] - df['open'])
    features['body_ratio'] = body / high_low_range
    
    # 上下影線
    upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
    lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
    features['upper_shadow_ratio'] = upper_shadow / high_low_range
    features['lower_shadow_ratio'] = lower_shadow / high_low_range
    
    # 買賣壓力
    features['buy_pressure'] = (df['close'] - df['low']) / high_low_range
    
    # === 4. 成交量特徵 ===
    vol_ma6 = df['volume'].rolling(6).mean()
    vol_ma24 = df['volume'].rolling(24).mean()
    
    features['volume_ratio_6'] = df['volume'] / (vol_ma6 + 1e-8)
    features['volume_ratio_24'] = df['volume'] / (vol_ma24 + 1e-8)
    
    # 價量配合
    price_change = df['close'].pct_change()
    features['volume_price_corr'] = price_change.rolling(6).corr(df['volume'].pct_change())
    
    # === 5. 波動率 ===
    features['volatility'] = df['close'].pct_change().rolling(6).std()
    features['volatility_24h'] = df['close'].pct_change().rolling(24).std()
    features['volatility_expanding'] = (features['volatility'] > features['volatility_24h']).astype(int)
    
    # === 6. 通道寬度變化 ===
    features['channel_width'] = (upper - lower) / middle
    features['channel_width_change'] = features['channel_width'].pct_change(3)
    
    # === 7. 趨勢強度 ===
    # 中軌斜率
    features['middle_slope'] = middle.pct_change(3)
    
    # 價格距離中軌的偏離度
    features['deviation_from_middle'] = abs(df['close'] - middle) / middle
    
    # === 8. 歷史反彈成功率特徵 ===
    # 過去 24 小時內觸碰次數
    dist_to_upper_abs = abs((upper - df['close']) / df['close'])
    dist_to_lower_abs = abs((df['close'] - lower) / df['close'])
    
    near_upper = (dist_to_upper_abs < 0.005).astype(int)
    near_lower = (dist_to_lower_abs < 0.005).astype(int)
    
    features['upper_touches_24h'] = near_upper.rolling(24).sum()
    features['lower_touches_24h'] = near_lower.rolling(24).sum()
    
    # === 9. RSI-like 指標 ===
    # 簡化版 RSI
    price_changes = df['close'].diff()
    gains = price_changes.where(price_changes > 0, 0).rolling(14).mean()
    losses = -price_changes.where(price_changes < 0, 0).rolling(14).mean()
    features['rsi_like'] = 100 - (100 / (1 + gains / (losses + 1e-8)))
    
    # === 10. 時間特徵 ===
    if 'open_time' in df.columns:
        hour = pd.to_datetime(df['open_time']).dt.hour
    elif isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
    else:
        hour = pd.Series(12, index=df.index)
    
    features['is_high_vol_time'] = ((hour >= 8) & (hour < 16)).astype(int)
    
    return features


def create_bounce_labels(
    df: pd.DataFrame,
    upper_touch: pd.Series,
    lower_touch: pd.Series,
    horizon: int = 4,
    bounce_threshold: float = 0.008  # 反彈至少 0.8%
):
    """
    創建反彈標籤
    
    邏輯:
    - 上軌觸碰: 如果未來 horizon 小時內價格回落 >= bounce_threshold,標記為 1 (有效反彈向下)
    - 下軌觸碰: 如果未來 horizon 小時內價格反彈 >= bounce_threshold,標記為 1 (有效反彈向上)
    
    Returns:
        upper_bounce_labels, lower_bounce_labels
    """
    upper_bounce_labels = pd.Series(0, index=df.index)
    lower_bounce_labels = pd.Series(0, index=df.index)
    
    for i in range(len(df) - horizon):
        # 上軌觸碰: 看是否反彈向下
        if upper_touch.iloc[i]:
            entry_price = df['close'].iloc[i]
            future_lows = df['low'].iloc[i+1:i+1+horizon]
            
            if len(future_lows) > 0:
                lowest = future_lows.min()
                drop_pct = (entry_price - lowest) / entry_price
                
                if drop_pct >= bounce_threshold:
                    upper_bounce_labels.iloc[i] = 1
        
        # 下軌觸碰: 看是否反彈向上
        if lower_touch.iloc[i]:
            entry_price = df['close'].iloc[i]
            future_highs = df['high'].iloc[i+1:i+1+horizon]
            
            if len(future_highs) > 0:
                highest = future_highs.max()
                rise_pct = (highest - entry_price) / entry_price
                
                if rise_pct >= bounce_threshold:
                    lower_bounce_labels.iloc[i] = 1
    
    return upper_bounce_labels, lower_bounce_labels


def train_v7_model(
    symbol: str = 'BTCUSDT',
    timeframe: str = '1h',
    days: int = 180,
    ema_period: int = 20,
    atr_period: int = 14,
    multiplier: float = 2.0,
    horizon: int = 4
):
    """訓練 v7 肯特納反彈模型"""
    logger.info("="*80)
    logger.info("[START] Training v7 Model - Keltner Channel Bounce Prediction")
    logger.info("="*80)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Days: {days}")
    logger.info(f"Keltner: EMA={ema_period}, ATR={atr_period}, Multiplier={multiplier}")
    logger.info(f"Horizon: {horizon}h")
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
    logger.info("\nStep 4/7: Engineering bounce features...")
    features = calculate_bounce_features(df, upper, middle, lower)
    logger.info(f"Created {len(features.columns)} features")
    
    # 5. 創建反彈標籤
    logger.info("\nStep 5/7: Creating bounce labels...")
    upper_bounce_labels, lower_bounce_labels = create_bounce_labels(
        df, upper_touch, lower_touch, horizon
    )
    
    # 只保留有觸碰事件的樣本
    upper_samples = upper_touch
    lower_samples = lower_touch
    
    logger.info(f"Upper touch samples: {upper_samples.sum()}")
    logger.info(f"  - Successful bounces: {upper_bounce_labels[upper_samples].sum()} ({upper_bounce_labels[upper_samples].mean():.1%})")
    logger.info(f"Lower touch samples: {lower_samples.sum()}")
    logger.info(f"  - Successful bounces: {lower_bounce_labels[lower_samples].sum()} ({lower_bounce_labels[lower_samples].mean():.1%})")
    
    # 6. 準備訓練數據
    logger.info("\nStep 6/7: Preparing training data...")
    
    # 清理特徵
    features = features.fillna(0)
    features = features.replace([np.inf, -np.inf], 0)
    
    # Upper band model (預測從上軌反彈向下)
    X_upper = features[upper_samples].copy()
    y_upper = upper_bounce_labels[upper_samples].copy()
    
    # Lower band model (預測從下軌反彈向上)
    X_lower = features[lower_samples].copy()
    y_lower = lower_bounce_labels[lower_samples].copy()
    
    logger.info(f"Upper model training samples: {len(X_upper)}")
    logger.info(f"Lower model training samples: {len(X_lower)}")
    
    # 時間序列切分
    def time_split(X, y, train_ratio=0.7, val_ratio=0.15):
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        y_val = y.iloc[train_end:val_end]
        X_test = X.iloc[val_end:]
        y_test = y.iloc[val_end:]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    # 7. 訓練模型
    logger.info("\nStep 7/7: Training models...")
    
    results = {}
    
    # Upper band model (Short signal)
    if len(X_upper) >= 100:  # 至少需要 100 個樣本
        logger.info("\n[UPPER BAND] Training short signal model...")
        X_train, X_val, X_test, y_train, y_val, y_test = time_split(X_upper, y_upper)
        
        logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        model_upper = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=4,
            l2_leaf_reg=5,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=42,
            verbose=50,
            early_stopping_rounds=50,
            auto_class_weights='Balanced'
        )
        
        model_upper.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )
        
        # 評估
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
        logger.info(f"    Positive rate: {y_test.mean():.2%}")
        logger.info(f"    Predicted positive: {pred_binary.mean():.2%}")
        
        results['upper'] = {
            'model': model_upper,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    else:
        logger.info(f"\n[UPPER BAND] Insufficient samples ({len(X_upper)}), skipping...")
        results['upper'] = None
    
    # Lower band model (Long signal)
    if len(X_lower) >= 100:
        logger.info("\n[LOWER BAND] Training long signal model...")
        X_train, X_val, X_test, y_train, y_val, y_test = time_split(X_lower, y_lower)
        
        logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        model_lower = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=4,
            l2_leaf_reg=5,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=42,
            verbose=50,
            early_stopping_rounds=50,
            auto_class_weights='Balanced'
        )
        
        model_lower.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )
        
        # 評估
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
        logger.info(f"    Positive rate: {y_test.mean():.2%}")
        logger.info(f"    Predicted positive: {pred_binary.mean():.2%}")
        
        results['lower'] = {
            'model': model_lower,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    else:
        logger.info(f"\n[LOWER BAND] Insufficient samples ({len(X_lower)}), skipping...")
        results['lower'] = None
    
    # 8. 儲存模型
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    models_dir = Path('models_output')
    models_dir.mkdir(exist_ok=True)
    
    if results['upper']:
        upper_path = models_dir / f'keltner_upper_v7_{timestamp}.pkl'
        with open(upper_path, 'wb') as f:
            pickle.dump({
                'model': results['upper']['model'],
                'features': features.columns.tolist(),
                'metadata': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'type': 'upper_band_short',
                    'keltner_params': {
                        'ema_period': ema_period,
                        'atr_period': atr_period,
                        'multiplier': multiplier
                    }
                }
            }, f)
        logger.info(f"\n[SAVE] Upper model: {upper_path}")
    
    if results['lower']:
        lower_path = models_dir / f'keltner_lower_v7_{timestamp}.pkl'
        with open(lower_path, 'wb') as f:
            pickle.dump({
                'model': results['lower']['model'],
                'features': features.columns.tolist(),
                'metadata': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'type': 'lower_band_long',
                    'keltner_params': {
                        'ema_period': ema_period,
                        'atr_period': atr_period,
                        'multiplier': multiplier
                    }
                }
            }, f)
        logger.info(f"[SAVE] Lower model: {lower_path}")
    
    logger.info("\n" + "="*80)
    logger.info("[DONE] Training complete!")
    logger.info("="*80)
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train v7 Keltner bounce model')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='1h')
    parser.add_argument('--days', type=int, default=180)
    parser.add_argument('--ema', type=int, default=20)
    parser.add_argument('--atr', type=int, default=14)
    parser.add_argument('--multiplier', type=float, default=2.0)
    parser.add_argument('--horizon', type=int, default=4)
    
    args = parser.parse_args()
    
    results = train_v7_model(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        ema_period=args.ema,
        atr_period=args.atr,
        multiplier=args.multiplier,
        horizon=args.horizon
    )
    
    print("\n訓練完成!")
    if results['upper']:
        print(f"Upper band (Short): AUC={results['upper']['auc']:.3f}, Precision={results['upper']['precision']:.3f}")
    if results['lower']:
        print(f"Lower band (Long): AUC={results['lower']['auc']:.3f}, Precision={results['lower']['precision']:.3f}")
