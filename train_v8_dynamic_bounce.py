#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v8 動態反彈目標模型

核心改進:
1. 反彈目標 = 最近 N 根 K 線的平均波動 × 倍數
2. 止損距離 = ATR × 倍數 (動態調整)
3. 考慮波動性環境,高波動 = 高目標

標籤定義:
- 成功 = 反彈幅度 >= avg_recent_move × bounce_multiplier
- 失敗 = 觸及止損或反彈不足

這才是真正適應市場的策略!
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
        logging.FileHandler('logs/train_v8.log', encoding='utf-8'),
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
    
    return upper, middle, lower, atr


def identify_touch_events(df: pd.DataFrame, upper: pd.Series, lower: pd.Series, threshold: float = 0.003):
    """識別觸碰通道邊界的事件"""
    dist_to_upper = (upper - df['close']) / df['close']
    dist_to_lower = (df['close'] - lower) / df['close']
    
    upper_touch = (dist_to_upper <= threshold) & (dist_to_upper >= -threshold)
    lower_touch = (dist_to_lower <= threshold) & (dist_to_lower >= -threshold)
    
    return upper_touch, lower_touch


def calculate_dynamic_targets(
    df: pd.DataFrame,
    atr: pd.Series,
    lookback: int = 10,
    bounce_multiplier: float = 1.5,
    sl_atr_multiplier: float = 1.0
):
    """
    計算動態目標
    
    Args:
        df: K線數據
        atr: ATR 序列
        lookback: 回看幾根K線計算平均波動
        bounce_multiplier: 反彈目標 = 平均波動 × 此倍數
        sl_atr_multiplier: 止損 = ATR × 此倍數
    
    Returns:
        bounce_target_pct: 反彈目標百分比
        sl_distance_pct: 止損距離百分比
    """
    # 計算最近 N 根K線的平均波動
    recent_moves = df['close'].pct_change().abs().rolling(lookback).mean()
    
    # 反彈目標 = 最近平均波動 × 倍數
    bounce_target_pct = recent_moves * bounce_multiplier
    
    # 止損距離 = ATR × 倍數 / 價格
    sl_distance_pct = (atr * sl_atr_multiplier) / df['close']
    
    return bounce_target_pct, sl_distance_pct


def create_dynamic_bounce_labels(
    df: pd.DataFrame,
    upper_touch: pd.Series,
    lower_touch: pd.Series,
    bounce_target_pct: pd.Series,
    sl_distance_pct: pd.Series,
    horizon: int = 12
):
    """
    創建動態反彈標籤
    
    上軌觸碰 (Short):
    - 成功: 未來 horizon 內,價格下跌 >= bounce_target_pct,且沒先觸及止損
    - 止損: 價格上漲 >= sl_distance_pct
    
    下軌觸碰 (Long):
    - 成功: 未來 horizon 內,價格上漲 >= bounce_target_pct,且沒先觸及止損
    - 止損: 價格下跌 >= sl_distance_pct
    """
    upper_labels = pd.Series(0, index=df.index)
    lower_labels = pd.Series(0, index=df.index)
    
    for i in range(len(df) - horizon):
        # === 上軌觸碰 (Short) ===
        if upper_touch.iloc[i] and not pd.isna(bounce_target_pct.iloc[i]):
            entry_price = df['close'].iloc[i]
            target_move = bounce_target_pct.iloc[i]
            sl_move = sl_distance_pct.iloc[i]
            
            target_price = entry_price * (1 - target_move)  # Short: 往下
            sl_price = entry_price * (1 + sl_move)  # Short: 往上是SL
            
            future_highs = df['high'].iloc[i+1:i+1+horizon]
            future_lows = df['low'].iloc[i+1:i+1+horizon]
            
            if len(future_highs) > 0:
                hit_sl = (future_highs >= sl_price).any()
                
                if not hit_sl:
                    hit_target = (future_lows <= target_price).any()
                    if hit_target:
                        upper_labels.iloc[i] = 1
        
        # === 下軌觸碰 (Long) ===
        if lower_touch.iloc[i] and not pd.isna(bounce_target_pct.iloc[i]):
            entry_price = df['close'].iloc[i]
            target_move = bounce_target_pct.iloc[i]
            sl_move = sl_distance_pct.iloc[i]
            
            target_price = entry_price * (1 + target_move)  # Long: 往上
            sl_price = entry_price * (1 - sl_move)  # Long: 往下是SL
            
            future_highs = df['high'].iloc[i+1:i+1+horizon]
            future_lows = df['low'].iloc[i+1:i+1+horizon]
            
            if len(future_lows) > 0:
                hit_sl = (future_lows <= sl_price).any()
                
                if not hit_sl:
                    hit_target = (future_highs >= target_price).any()
                    if hit_target:
                        lower_labels.iloc[i] = 1
    
    return upper_labels, lower_labels


def calculate_bounce_features(df: pd.DataFrame, upper: pd.Series, middle: pd.Series, lower: pd.Series, atr: pd.Series) -> pd.DataFrame:
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
    
    # 波動性 (關鍵!)
    features['atr_pct'] = atr / df['close']  # ATR 佔價格百分比
    features['volatility_5'] = df['close'].pct_change().abs().rolling(5).mean()
    features['volatility_10'] = df['close'].pct_change().abs().rolling(10).mean()
    features['volatility_20'] = df['close'].pct_change().abs().rolling(20).mean()
    features['volatility_expanding'] = (features['volatility_5'] > features['volatility_20']).astype(int)
    features['volatility_ratio'] = features['volatility_5'] / (features['volatility_20'] + 1e-8)
    
    # K線形態
    high_low_range = df['high'] - df['low'] + 1e-8
    body = abs(df['close'] - df['open'])
    features['body_ratio'] = body / high_low_range
    features['candle_size_pct'] = high_low_range / df['close']
    
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
    
    # 通道特徵
    features['channel_width_pct'] = (upper - lower) / middle
    features['channel_width_change'] = features['channel_width_pct'].pct_change(3)
    features['middle_slope'] = middle.pct_change(3)
    
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


def train_v8_model(
    symbol: str = 'BTCUSDT',
    timeframe: str = '15m',
    days: int = 9999,
    ema_period: int = 20,
    atr_period: int = 14,
    multiplier: float = 2.0,
    lookback: int = 10,
    bounce_multiplier: float = 1.5,
    sl_atr_multiplier: float = 1.0,
    horizon: int = 12,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
):
    """訓練 v8 動態反彈模型"""
    logger.info("="*80)
    logger.info("[START] Training v8 Model - Dynamic Bounce Targets")
    logger.info("="*80)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Keltner: EMA={ema_period}, ATR={atr_period}, Multiplier={multiplier}")
    logger.info(f"Dynamic targets:")
    logger.info(f"  - Lookback: {lookback} bars")
    logger.info(f"  - Bounce target = avg_move × {bounce_multiplier}")
    logger.info(f"  - Stop loss = ATR × {sl_atr_multiplier}")
    logger.info(f"  - Horizon: {horizon} bars")
    logger.info(f"")
    
    # 載入數據
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
    
    # 計算通道和 ATR
    logger.info("\nStep 2/7: Calculating indicators...")
    upper, middle, lower, atr = calculate_keltner_channels(df, ema_period, atr_period, multiplier)
    
    # 計算動態目標
    logger.info("\nStep 3/7: Calculating dynamic targets...")
    bounce_target_pct, sl_distance_pct = calculate_dynamic_targets(
        df, atr, lookback, bounce_multiplier, sl_atr_multiplier
    )
    
    logger.info(f"Average bounce target: {bounce_target_pct.mean():.2%}")
    logger.info(f"Average SL distance: {sl_distance_pct.mean():.2%}")
    logger.info(f"Median bounce target: {bounce_target_pct.median():.2%}")
    logger.info(f"Median SL distance: {sl_distance_pct.median():.2%}")
    
    # 識別觸碰
    logger.info("\nStep 4/7: Identifying touch events...")
    upper_touch, lower_touch = identify_touch_events(df, upper, lower)
    logger.info(f"Upper touches: {upper_touch.sum()}")
    logger.info(f"Lower touches: {lower_touch.sum()}")
    
    # 創建標籤
    logger.info("\nStep 5/7: Creating dynamic bounce labels...")
    upper_labels, lower_labels = create_dynamic_bounce_labels(
        df, upper_touch, lower_touch, bounce_target_pct, sl_distance_pct, horizon
    )
    
    upper_success_rate = upper_labels[upper_touch].mean()
    lower_success_rate = lower_labels[lower_touch].mean()
    
    logger.info(f"Upper: {upper_labels[upper_touch].sum()} / {upper_touch.sum()} = {upper_success_rate:.1%}")
    logger.info(f"Lower: {lower_labels[lower_touch].sum()} / {lower_touch.sum()} = {lower_success_rate:.1%}")
    
    # 計算特徵
    logger.info("\nStep 6/7: Engineering features...")
    features = calculate_bounce_features(df, upper, middle, lower, atr)
    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    logger.info(f"Created {len(features.columns)} features")
    
    # 訓練模型
    logger.info("\nStep 7/7: Training models...")
    results = {}
    
    X_upper = features[upper_touch].copy()
    y_upper = upper_labels[upper_touch].copy()
    X_lower = features[lower_touch].copy()
    y_lower = lower_labels[lower_touch].copy()
    
    def time_split(X, y, train_r, val_r):
        n = len(X)
        train_end = int(n * train_r)
        val_end = int(n * (train_r + val_r))
        return (
            X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:],
            y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]
        )
    
    # Upper model
    if len(X_upper) >= 100:
        logger.info("\n[UPPER] Training...")
        X_train, X_val, X_test, y_train, y_val, y_test = time_split(X_upper, y_upper, train_ratio, val_ratio)
        
        logger.info(f"  Train positive: {y_train.mean():.2%}")
        
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
        
        logger.info(f"\n  Test: AUC={auc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        results['upper'] = {'model': model_upper, 'auc': auc, 'precision': precision}
    
    # Lower model
    if len(X_lower) >= 100:
        logger.info("\n[LOWER] Training...")
        X_train, X_val, X_test, y_train, y_val, y_test = time_split(X_lower, y_lower, train_ratio, val_ratio)
        
        logger.info(f"  Train positive: {y_train.mean():.2%}")
        
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
        
        logger.info(f"\n  Test: AUC={auc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        results['lower'] = {'model': model_lower, 'auc': auc, 'precision': precision}
    
    # 保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    models_dir = Path('models_output')
    models_dir.mkdir(exist_ok=True)
    
    metadata = {
        'symbol': symbol, 'timeframe': timeframe, 'strategy': 'dynamic_bounce',
        'keltner_params': {'ema_period': ema_period, 'atr_period': atr_period, 'multiplier': multiplier},
        'dynamic_params': {'lookback': lookback, 'bounce_multiplier': bounce_multiplier, 'sl_atr_multiplier': sl_atr_multiplier},
        'horizon': horizon
    }
    
    if results.get('upper'):
        path = models_dir / f'keltner_upper_{timeframe}_v8_{timestamp}.pkl'
        with open(path, 'wb') as f:
            pickle.dump({
                'model': results['upper']['model'],
                'features': features.columns.tolist(),
                'metadata': {**metadata, 'type': 'upper', 'success_rate': upper_success_rate}
            }, f)
        logger.info(f"\n[SAVE] Upper: {path}")
    
    if results.get('lower'):
        path = models_dir / f'keltner_lower_{timeframe}_v8_{timestamp}.pkl'
        with open(path, 'wb') as f:
            pickle.dump({
                'model': results['lower']['model'],
                'features': features.columns.tolist(),
                'metadata': {**metadata, 'type': 'lower', 'success_rate': lower_success_rate}
            }, f)
        logger.info(f"[SAVE] Lower: {path}")
    
    logger.info("\n" + "="*80)
    logger.info("[DONE]")
    logger.info("="*80)
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train v8 dynamic bounce model')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='15m')
    parser.add_argument('--days', type=int, default=9999)
    parser.add_argument('--lookback', type=int, default=10, help='回看幾根K線計算平均波動')
    parser.add_argument('--bounce-multiplier', type=float, default=1.5, help='反彈目標 = 平均波動 × 此倍數')
    parser.add_argument('--sl-atr-multiplier', type=float, default=1.0, help='止損 = ATR × 此倍數')
    parser.add_argument('--horizon', type=int, default=12)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    
    args = parser.parse_args()
    
    results = train_v8_model(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        lookback=args.lookback,
        bounce_multiplier=args.bounce_multiplier,
        sl_atr_multiplier=args.sl_atr_multiplier,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    
    print("\n訓練完成!")
    if results.get('upper'):
        print(f"Upper: AUC={results['upper']['auc']:.3f}, Precision={results['upper']['precision']:.3f}")
    if results.get('lower'):
        print(f"Lower: AUC={results['lower']['auc']:.3f}, Precision={results['lower']['precision']:.3f}")
