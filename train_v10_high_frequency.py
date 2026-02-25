#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v10 高頻剝頭皮策略

核心理念:
1. 5m 時間框架 - 更多交易機會
2. 小目標小止損 - TP=0.3%, SL=0.2%
3. 快進快出 - 持有 1-3 根K線
4. 高頻率 - 每天 10-20 筆交易
5. 勝率優先 - 目標 55-60% 勝率

標籤定義:
- 成功: 未來 3 根K線內上漲/下跌 >= 0.3%, 且未觸及 0.2% 止損
- 失敗: 觸及止損或 3 根內未達目標

特徵重點:
- 短期動量 (1-3 根)
- 微觀結構 (買賣壓力)
- 成交量突變
- 價格位置
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from datetime import datetime, timedelta
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
        logging.FileHandler('logs/train_v10.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def create_scalping_labels(
    df: pd.DataFrame,
    tp_pct: float = 0.003,
    sl_pct: float = 0.002,
    horizon: int = 3
):
    """
    創建剝頭皮標籤
    
    Long 標籤:
    - 成功: 未來 horizon 內上漲 >= tp_pct, 且未先觸及 sl_pct
    
    Short 標籤:
    - 成功: 未來 horizon 內下跌 >= tp_pct, 且未先觸及 sl_pct
    """
    long_labels = pd.Series(0, index=df.index)
    short_labels = pd.Series(0, index=df.index)
    
    for i in range(len(df) - horizon):
        entry_price = df['close'].iloc[i]
        
        # Long
        tp_long = entry_price * (1 + tp_pct)
        sl_long = entry_price * (1 - sl_pct)
        
        future_highs = df['high'].iloc[i+1:i+1+horizon]
        future_lows = df['low'].iloc[i+1:i+1+horizon]
        
        if len(future_lows) > 0:
            hit_sl_long = (future_lows <= sl_long).any()
            
            if not hit_sl_long:
                hit_tp_long = (future_highs >= tp_long).any()
                if hit_tp_long:
                    long_labels.iloc[i] = 1
        
        # Short
        tp_short = entry_price * (1 - tp_pct)
        sl_short = entry_price * (1 + sl_pct)
        
        if len(future_highs) > 0:
            hit_sl_short = (future_highs >= sl_short).any()
            
            if not hit_sl_short:
                hit_tp_short = (future_lows <= tp_short).any()
                if hit_tp_short:
                    short_labels.iloc[i] = 1
    
    return long_labels, short_labels


def calculate_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算微觀結構特徵 - 適合高頻交易
    """
    features = pd.DataFrame(index=df.index)
    
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    volume = df['volume']
    
    # === 超短期動量 (1-3根) ===
    features['return_1'] = close.pct_change(1)
    features['return_2'] = close.pct_change(2)
    features['return_3'] = close.pct_change(3)
    
    # 加速度
    features['accel_1'] = features['return_1'].diff()
    features['accel_2'] = features['return_2'].diff()
    
    # 動量強度
    features['momentum_strength'] = features['return_1'].abs() / (features['return_2'].abs() + 1e-8)
    
    # === K線結構 ===
    body = close - open_price
    hl_range = high - low + 1e-8
    
    features['body_pct'] = body / close
    features['body_ratio'] = abs(body) / hl_range
    features['range_pct'] = hl_range / close
    
    # 影線
    upper_wick = high - pd.DataFrame({'close': close, 'open': open_price}).max(axis=1)
    lower_wick = pd.DataFrame({'close': close, 'open': open_price}).min(axis=1) - low
    
    features['upper_wick_ratio'] = upper_wick / hl_range
    features['lower_wick_ratio'] = lower_wick / hl_range
    features['wick_imbalance'] = (upper_wick - lower_wick) / hl_range
    
    # K線方向
    features['bullish_candle'] = (body > 0).astype(int)
    features['bearish_candle'] = (body < 0).astype(int)
    
    # 買賣壓力
    features['buy_pressure'] = (close - low) / hl_range
    features['sell_pressure'] = (high - close) / hl_range
    features['pressure_diff'] = features['buy_pressure'] - features['sell_pressure']
    
    # === 成交量特徵 ===
    vol_ma_3 = volume.rolling(3).mean()
    vol_ma_10 = volume.rolling(10).mean()
    vol_ma_20 = volume.rolling(20).mean()
    
    features['volume_ratio_3'] = volume / (vol_ma_3 + 1e-8)
    features['volume_ratio_10'] = volume / (vol_ma_10 + 1e-8)
    features['volume_spike'] = (volume > vol_ma_3 * 1.5).astype(int)
    features['volume_dry'] = (volume < vol_ma_10 * 0.7).astype(int)
    
    # 量價配合
    features['volume_price_alignment'] = (
        ((features['return_1'] > 0) & (features['volume_ratio_3'] > 1.2)) |
        ((features['return_1'] < 0) & (features['volume_ratio_3'] > 1.2))
    ).astype(int)
    
    # === 波動性 ===
    features['volatility_3'] = close.pct_change().rolling(3).std()
    features['volatility_10'] = close.pct_change().rolling(10).std()
    features['volatility_expanding'] = (features['volatility_3'] > features['volatility_10']).astype(int)
    
    # === 價格位置 ===
    high_3 = high.rolling(3).max()
    low_3 = low.rolling(3).min()
    high_10 = high.rolling(10).max()
    low_10 = low.rolling(10).min()
    
    features['position_in_3bar_range'] = (close - low_3) / (high_3 - low_3 + 1e-8)
    features['position_in_10bar_range'] = (close - low_10) / (high_10 - low_10 + 1e-8)
    
    features['near_3bar_high'] = (close >= high_3 * 0.998).astype(int)
    features['near_3bar_low'] = (close <= low_3 * 1.002).astype(int)
    
    # === 連續方向 ===
    returns = close.pct_change()
    features['consecutive_up'] = (returns > 0).rolling(3).sum()
    features['consecutive_down'] = (returns < 0).rolling(3).sum()
    
    # === 反轉信號 ===
    # Pin bar (長影線)
    features['bullish_pin'] = (
        (features['lower_wick_ratio'] > 0.5) & 
        (features['body_ratio'] < 0.3) &
        (body > 0)
    ).astype(int)
    
    features['bearish_pin'] = (
        (features['upper_wick_ratio'] > 0.5) & 
        (features['body_ratio'] < 0.3) &
        (body < 0)
    ).astype(int)
    
    # 吞噬形態
    prev_body = abs(body.shift(1))
    curr_body = abs(body)
    features['bullish_engulf'] = (
        (body > 0) &
        (body.shift(1) < 0) &
        (curr_body > prev_body * 1.5)
    ).astype(int)
    
    features['bearish_engulf'] = (
        (body < 0) &
        (body.shift(1) > 0) &
        (curr_body > prev_body * 1.5)
    ).astype(int)
    
    # === 短期指標 ===
    # 快速 RSI
    price_changes = close.diff()
    gains = price_changes.where(price_changes > 0, 0).rolling(5).mean()
    losses = -price_changes.where(price_changes < 0, 0).rolling(5).mean()
    rs = gains / (losses + 1e-8)
    features['rsi_5'] = 100 - (100 / (1 + rs))
    
    # 快速 EMA
    ema_5 = close.ewm(span=5, adjust=False).mean()
    ema_10 = close.ewm(span=10, adjust=False).mean()
    
    features['price_vs_ema5'] = (close - ema_5) / ema_5
    features['ema_cross'] = ((ema_5 > ema_10) & (ema_5.shift(1) <= ema_10.shift(1))).astype(int) - \
                            ((ema_5 < ema_10) & (ema_5.shift(1) >= ema_10.shift(1))).astype(int)
    
    # === 時間特徵 ===
    if 'open_time' in df.columns:
        hour = pd.to_datetime(df['open_time']).dt.hour
        minute = pd.to_datetime(df['open_time']).dt.minute
    elif isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
        minute = df.index.minute
    else:
        hour = pd.Series(12, index=df.index)
        minute = pd.Series(0, index=df.index)
    
    # 高流動性時段
    features['high_liquidity_hour'] = (
        ((hour >= 8) & (hour < 10)) |  # 亞洲開盤
        ((hour >= 14) & (hour < 16)) |  # 歐洲開盤
        ((hour >= 21) & (hour < 23))    # 美國開盤
    ).astype(int)
    
    return features


def train_v10_model(
    symbol: str = 'BTCUSDT',
    timeframe: str = '5m',
    days: int = 365,
    tp_pct: float = 0.003,
    sl_pct: float = 0.002,
    horizon: int = 3,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
):
    """
    訓練 v10 高頻剝頭皮模型
    """
    logger.info("="*80)
    logger.info("[START] Training v10 Model - High Frequency Scalping")
    logger.info("="*80)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Parameters:")
    logger.info(f"  - TP: {tp_pct:.2%}")
    logger.info(f"  - SL: {sl_pct:.2%}")
    logger.info(f"  - RR ratio: {tp_pct / sl_pct:.2f}")
    logger.info(f"  - Horizon: {horizon} bars")
    logger.info("")
    
    # 載入數據
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
    
    # 創建標籤
    logger.info("\nStep 2/6: Creating scalping labels...")
    long_labels, short_labels = create_scalping_labels(df, tp_pct, sl_pct, horizon)
    
    long_success_rate = long_labels.mean()
    short_success_rate = short_labels.mean()
    
    logger.info(f"Long signals: {long_labels.sum()} / {len(df)} = {long_success_rate:.1%}")
    logger.info(f"Short signals: {short_labels.sum()} / {len(df)} = {short_success_rate:.1%}")
    
    # 計算特徵
    logger.info("\nStep 3/6: Engineering microstructure features...")
    features = calculate_microstructure_features(df)
    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    logger.info(f"Created {len(features.columns)} features")
    
    # 訓練模型
    logger.info("\nStep 4/6: Training models...")
    results = {}
    
    def time_split(X, y, train_r, val_r):
        n = len(X)
        train_end = int(n * train_r)
        val_end = int(n * (train_r + val_r))
        return (
            X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:],
            y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]
        )
    
    # Long model
    if long_labels.sum() >= 100:
        logger.info("\n[LONG] Training...")
        X_train, X_val, X_test, y_train, y_val, y_test = time_split(features, long_labels, train_ratio, val_ratio)
        
        logger.info(f"  Train positive: {y_train.mean():.2%}")
        
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
            auto_class_weights='Balanced'
        )
        
        model_long.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        
        pred_proba = model_long.predict_proba(X_test)[:, 1]
        pred_binary = (pred_proba >= 0.5).astype(int)
        
        auc = roc_auc_score(y_test, pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        precision = precision_score(y_test, pred_binary, zero_division=0)
        recall = recall_score(y_test, pred_binary, zero_division=0)
        f1 = f1_score(y_test, pred_binary, zero_division=0)
        
        logger.info(f"\n  Test: AUC={auc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        results['long'] = {'model': model_long, 'auc': auc, 'precision': precision}
    
    # Short model
    if short_labels.sum() >= 100:
        logger.info("\n[SHORT] Training...")
        X_train, X_val, X_test, y_train, y_val, y_test = time_split(features, short_labels, train_ratio, val_ratio)
        
        logger.info(f"  Train positive: {y_train.mean():.2%}")
        
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
        
        model_short.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        
        pred_proba = model_short.predict_proba(X_test)[:, 1]
        pred_binary = (pred_proba >= 0.5).astype(int)
        
        auc = roc_auc_score(y_test, pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        precision = precision_score(y_test, pred_binary, zero_division=0)
        recall = recall_score(y_test, pred_binary, zero_division=0)
        f1 = f1_score(y_test, pred_binary, zero_division=0)
        
        logger.info(f"\n  Test: AUC={auc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        results['short'] = {'model': model_short, 'auc': auc, 'precision': precision}
    
    # 保存
    logger.info("\nStep 5/6: Saving models...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    models_dir = Path('models_output')
    models_dir.mkdir(exist_ok=True)
    
    metadata = {
        'symbol': symbol, 'timeframe': timeframe, 'strategy': 'high_frequency_scalping',
        'params': {'tp_pct': tp_pct, 'sl_pct': sl_pct, 'horizon': horizon}
    }
    
    if results.get('long'):
        path = models_dir / f'scalping_long_{timeframe}_v10_{timestamp}.pkl'
        with open(path, 'wb') as f:
            pickle.dump({
                'model': results['long']['model'],
                'features': features.columns.tolist(),
                'metadata': {**metadata, 'type': 'long', 'success_rate': long_success_rate}
            }, f)
        logger.info(f"[SAVE] Long: {path}")
    
    if results.get('short'):
        path = models_dir / f'scalping_short_{timeframe}_v10_{timestamp}.pkl'
        with open(path, 'wb') as f:
            pickle.dump({
                'model': results['short']['model'],
                'features': features.columns.tolist(),
                'metadata': {**metadata, 'type': 'short', 'success_rate': short_success_rate}
            }, f)
        logger.info(f"[SAVE] Short: {path}")
    
    logger.info("\n" + "="*80)
    logger.info("[DONE]")
    logger.info("="*80)
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train v10 high frequency scalping model')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='5m')
    parser.add_argument('--days', type=int, default=365)
    parser.add_argument('--tp-pct', type=float, default=0.003, help='Take profit %')
    parser.add_argument('--sl-pct', type=float, default=0.002, help='Stop loss %')
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    
    args = parser.parse_args()
    
    results = train_v10_model(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    
    print("\n訓練完成!")
    if results.get('long'):
        print(f"Long: AUC={results['long']['auc']:.3f}, Precision={results['long']['precision']:.3f}")
    if results.get('short'):
        print(f"Short: AUC={results['short']['auc']:.3f}, Precision={results['short']['precision']:.3f}")
