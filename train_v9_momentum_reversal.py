#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v9 動能反轉預測模型

核心概念:
1. 識別強勢上漲/下跌後的動能衰竭
2. 預測價格是否即將反轉
3. 結合通道位置、RSI、成交量等多維度特徵

標籤定義:
- 看多反轉: 連續下跌後 → 強勢反彈 (上漲 > threshold)
- 看空反轉: 連續上漲後 → 強勢回調 (下跌 > threshold)

這是捕捉轉折點的策略,不是固定 TP/SL!
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
        logging.FileHandler('logs/train_v9.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def calculate_indicators(df: pd.DataFrame):
    """
    計算技術指標
    
    Returns:
        dict of indicators
    """
    indicators = {}
    
    # EMA
    indicators['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    indicators['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    indicators['atr'] = tr.rolling(14).mean()
    
    # Keltner Channels
    middle = indicators['ema_20']
    indicators['keltner_upper'] = middle + (indicators['atr'] * 2.0)
    indicators['keltner_lower'] = middle - (indicators['atr'] * 2.0)
    
    # RSI
    price_changes = df['close'].diff()
    gains = price_changes.where(price_changes > 0, 0).rolling(14).mean()
    losses = -price_changes.where(price_changes < 0, 0).rolling(14).mean()
    rs = gains / (losses + 1e-8)
    indicators['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    indicators['macd'] = ema_12 - ema_26
    indicators['macd_signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
    indicators['macd_hist'] = indicators['macd'] - indicators['macd_signal']
    
    # Bollinger Bands
    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    indicators['bb_upper'] = sma_20 + (std_20 * 2)
    indicators['bb_lower'] = sma_20 - (std_20 * 2)
    indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / sma_20
    
    return indicators


def identify_momentum_exhaustion(
    df: pd.DataFrame,
    indicators: dict,
    consecutive_bars: int = 3,
    min_move_pct: float = 0.005
):
    """
    識別動能衰竭點
    
    條件:
    1. 連續 N 根同方向
    2. 累計漲跌幅 >= min_move_pct
    3. 在通道邊界附近 OR RSI 極端
    
    Returns:
        bullish_exhaustion: 看空衰竭點 (連續上漲後)
        bearish_exhaustion: 看多衰竭點 (連續下跌後)
    """
    close = df['close']
    rsi = indicators['rsi']
    keltner_upper = indicators['keltner_upper']
    keltner_lower = indicators['keltner_lower']
    
    # 計算連續上漲/下跌
    returns = close.pct_change()
    
    bullish_exhaustion = pd.Series(False, index=df.index)
    bearish_exhaustion = pd.Series(False, index=df.index)
    
    for i in range(consecutive_bars, len(df)):
        # 檢查連續上漲
        recent_returns = returns.iloc[i-consecutive_bars+1:i+1]
        cumulative_return = (1 + recent_returns).prod() - 1
        
        is_consecutive_up = (recent_returns > 0).all()
        is_consecutive_down = (recent_returns < 0).all()
        
        # 看空衰竭 (連續上漲後可能反轉)
        if is_consecutive_up and cumulative_return >= min_move_pct:
            # 額外條件: 接近上軌 OR RSI 超買
            near_upper = (close.iloc[i] >= keltner_upper.iloc[i] * 0.997)
            rsi_overbought = (rsi.iloc[i] >= 70)
            
            if near_upper or rsi_overbought:
                bullish_exhaustion.iloc[i] = True
        
        # 看多衰竭 (連續下跌後可能反轉)
        if is_consecutive_down and abs(cumulative_return) >= min_move_pct:
            # 額外條件: 接近下軌 OR RSI 超賣
            near_lower = (close.iloc[i] <= keltner_lower.iloc[i] * 1.003)
            rsi_oversold = (rsi.iloc[i] <= 30)
            
            if near_lower or rsi_oversold:
                bearish_exhaustion.iloc[i] = True
    
    return bullish_exhaustion, bearish_exhaustion


def create_reversal_labels(
    df: pd.DataFrame,
    bullish_exhaustion: pd.Series,
    bearish_exhaustion: pd.Series,
    indicators: dict,
    horizon: int = 8,
    reversal_threshold_multiplier: float = 1.5
):
    """
    創建反轉標籤
    
    看空反轉 (bullish_exhaustion 後):
    - 成功: 未來 horizon 內下跌 >= ATR × multiplier
    - 失敗: 繼續上漲或小幅下跌
    
    看多反轉 (bearish_exhaustion 後):
    - 成功: 未來 horizon 內上漲 >= ATR × multiplier
    - 失敗: 繼續下跌或小幅上漲
    """
    bearish_reversal_labels = pd.Series(0, index=df.index)  # Short signal
    bullish_reversal_labels = pd.Series(0, index=df.index)  # Long signal
    
    atr = indicators['atr']
    
    for i in range(len(df) - horizon):
        # === 看空反轉 (做空) ===
        if bullish_exhaustion.iloc[i]:
            entry_price = df['close'].iloc[i]
            atr_value = atr.iloc[i]
            target_move = (atr_value / entry_price) * reversal_threshold_multiplier
            
            future_lows = df['low'].iloc[i+1:i+1+horizon]
            
            if len(future_lows) > 0:
                min_price = future_lows.min()
                actual_drop = (entry_price - min_price) / entry_price
                
                if actual_drop >= target_move:
                    bearish_reversal_labels.iloc[i] = 1
        
        # === 看多反轉 (做多) ===
        if bearish_exhaustion.iloc[i]:
            entry_price = df['close'].iloc[i]
            atr_value = atr.iloc[i]
            target_move = (atr_value / entry_price) * reversal_threshold_multiplier
            
            future_highs = df['high'].iloc[i+1:i+1+horizon]
            
            if len(future_highs) > 0:
                max_price = future_highs.max()
                actual_rise = (max_price - entry_price) / entry_price
                
                if actual_rise >= target_move:
                    bullish_reversal_labels.iloc[i] = 1
    
    return bearish_reversal_labels, bullish_reversal_labels


def calculate_reversal_features(df: pd.DataFrame, indicators: dict) -> pd.DataFrame:
    """
    計算反轉相關特徵
    """
    features = pd.DataFrame(index=df.index)
    
    close = df['close']
    volume = df['volume']
    atr = indicators['atr']
    rsi = indicators['rsi']
    macd = indicators['macd']
    macd_hist = indicators['macd_hist']
    
    # === 動能特徵 ===
    features['momentum_1'] = close.pct_change(1)
    features['momentum_2'] = close.pct_change(2)
    features['momentum_3'] = close.pct_change(3)
    features['momentum_5'] = close.pct_change(5)
    
    # 加速度
    features['acceleration_1'] = features['momentum_1'].diff()
    features['acceleration_2'] = features['momentum_2'].diff()
    
    # 動能衰竭信號
    features['momentum_weakening'] = (
        (features['momentum_1'].abs() < features['momentum_2'].abs()) &
        (features['momentum_2'].abs() < features['momentum_3'].abs())
    ).astype(int)
    
    # === RSI 特徵 ===
    features['rsi'] = rsi
    features['rsi_overbought'] = (rsi >= 70).astype(int)
    features['rsi_oversold'] = (rsi <= 30).astype(int)
    features['rsi_extreme'] = ((rsi >= 75) | (rsi <= 25)).astype(int)
    features['rsi_change'] = rsi.diff()
    features['rsi_divergence'] = ((close > close.shift(5)) & (rsi < rsi.shift(5))).astype(int)
    
    # === MACD 特徵 ===
    features['macd'] = macd
    features['macd_hist'] = macd_hist
    features['macd_hist_decreasing'] = (macd_hist < macd_hist.shift(1)).astype(int)
    features['macd_cross'] = ((macd > 0) & (macd.shift(1) <= 0)).astype(int) - ((macd < 0) & (macd.shift(1) >= 0)).astype(int)
    
    # === 通道位置 ===
    keltner_upper = indicators['keltner_upper']
    keltner_lower = indicators['keltner_lower']
    keltner_middle = indicators['ema_20']
    
    features['dist_to_upper'] = (keltner_upper - close) / close
    features['dist_to_lower'] = (close - keltner_lower) / close
    features['dist_to_middle'] = (close - keltner_middle) / close
    features['at_upper_band'] = (close >= keltner_upper * 0.997).astype(int)
    features['at_lower_band'] = (close <= keltner_lower * 1.003).astype(int)
    
    # === 波動性 ===
    features['atr_pct'] = atr / close
    features['volatility_5'] = close.pct_change().rolling(5).std()
    features['volatility_10'] = close.pct_change().rolling(10).std()
    features['volatility_expanding'] = (features['volatility_5'] > features['volatility_10']).astype(int)
    
    # BB Width
    features['bb_width'] = indicators['bb_width']
    features['bb_squeeze'] = (features['bb_width'] < features['bb_width'].rolling(20).mean()).astype(int)
    
    # === 成交量 ===
    vol_ma_5 = volume.rolling(5).mean()
    vol_ma_20 = volume.rolling(20).mean()
    features['volume_ratio'] = volume / (vol_ma_20 + 1e-8)
    features['volume_spike'] = (volume > vol_ma_5 * 1.5).astype(int)
    features['volume_declining'] = (vol_ma_5 < vol_ma_20).astype(int)
    
    # === K線形態 ===
    high_low_range = df['high'] - df['low'] + 1e-8
    body = abs(close - df['open'])
    features['body_ratio'] = body / high_low_range
    features['candle_size'] = high_low_range / close
    
    upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
    lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
    features['upper_shadow_ratio'] = upper_shadow / high_low_range
    features['lower_shadow_ratio'] = lower_shadow / high_low_range
    
    # 錘子線 / 射擊之星
    features['hammer_like'] = ((features['lower_shadow_ratio'] > 0.6) & (features['body_ratio'] < 0.3)).astype(int)
    features['shooting_star_like'] = ((features['upper_shadow_ratio'] > 0.6) & (features['body_ratio'] < 0.3)).astype(int)
    
    # === 趨勢強度 ===
    ema_20 = indicators['ema_20']
    ema_50 = indicators['ema_50']
    features['ema_trend'] = ((ema_20 > ema_50).astype(int) * 2 - 1)  # 1=上升, -1=下降
    features['price_vs_ema20'] = (close - ema_20) / ema_20
    features['ema_slope'] = ema_20.pct_change(3)
    
    # === 連續漲跌 ===
    returns = close.pct_change()
    features['consecutive_up'] = (returns > 0).rolling(3).sum()
    features['consecutive_down'] = (returns < 0).rolling(3).sum()
    
    return features


def train_v9_model(
    symbol: str = 'BTCUSDT',
    timeframe: str = '15m',
    days: int = 9999,
    consecutive_bars: int = 3,
    min_move_pct: float = 0.005,
    horizon: int = 8,
    reversal_threshold_multiplier: float = 1.5,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
):
    """
    訓練 v9 動能反轉模型
    """
    logger.info("="*80)
    logger.info("[START] Training v9 Model - Momentum Reversal")
    logger.info("="*80)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Parameters:")
    logger.info(f"  - Consecutive bars: {consecutive_bars}")
    logger.info(f"  - Min move: {min_move_pct:.2%}")
    logger.info(f"  - Reversal target: ATR × {reversal_threshold_multiplier}")
    logger.info(f"  - Horizon: {horizon} bars")
    logger.info("")
    
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
    
    # 計算指標
    logger.info("\nStep 2/7: Calculating indicators...")
    indicators = calculate_indicators(df)
    
    # 識別動能衰竭點
    logger.info("\nStep 3/7: Identifying momentum exhaustion...")
    bullish_exhaustion, bearish_exhaustion = identify_momentum_exhaustion(
        df, indicators, consecutive_bars, min_move_pct
    )
    
    logger.info(f"Bullish exhaustion points: {bullish_exhaustion.sum()} (看空機會)")
    logger.info(f"Bearish exhaustion points: {bearish_exhaustion.sum()} (看多機會)")
    
    # 創建標籤
    logger.info("\nStep 4/7: Creating reversal labels...")
    bearish_reversal_labels, bullish_reversal_labels = create_reversal_labels(
        df, bullish_exhaustion, bearish_exhaustion, indicators, horizon, reversal_threshold_multiplier
    )
    
    bearish_success_rate = bearish_reversal_labels[bullish_exhaustion].mean()
    bullish_success_rate = bullish_reversal_labels[bearish_exhaustion].mean()
    
    logger.info(f"Bearish reversal (Short): {bearish_reversal_labels[bullish_exhaustion].sum()} / {bullish_exhaustion.sum()} = {bearish_success_rate:.1%}")
    logger.info(f"Bullish reversal (Long): {bullish_reversal_labels[bearish_exhaustion].sum()} / {bearish_exhaustion.sum()} = {bullish_success_rate:.1%}")
    
    # 計算特徵
    logger.info("\nStep 5/7: Engineering features...")
    features = calculate_reversal_features(df, indicators)
    features = features.fillna(0).replace([np.inf, -np.inf], 0)
    logger.info(f"Created {len(features.columns)} features")
    
    # 訓練模型
    logger.info("\nStep 6/7: Training models...")
    results = {}
    
    X_bearish = features[bullish_exhaustion].copy()
    y_bearish = bearish_reversal_labels[bullish_exhaustion].copy()
    X_bullish = features[bearish_exhaustion].copy()
    y_bullish = bullish_reversal_labels[bearish_exhaustion].copy()
    
    def time_split(X, y, train_r, val_r):
        n = len(X)
        train_end = int(n * train_r)
        val_end = int(n * (train_r + val_r))
        return (
            X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:],
            y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]
        )
    
    # Bearish reversal model (Short)
    if len(X_bearish) >= 100:
        logger.info("\n[BEARISH REVERSAL] Training short signal model...")
        X_train, X_val, X_test, y_train, y_val, y_test = time_split(X_bearish, y_bearish, train_ratio, val_ratio)
        
        logger.info(f"  Train positive: {y_train.mean():.2%}")
        
        model_bearish = CatBoostClassifier(
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
        
        model_bearish.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        
        pred_proba = model_bearish.predict_proba(X_test)[:, 1]
        pred_binary = (pred_proba >= 0.5).astype(int)
        
        auc = roc_auc_score(y_test, pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        precision = precision_score(y_test, pred_binary, zero_division=0)
        recall = recall_score(y_test, pred_binary, zero_division=0)
        f1 = f1_score(y_test, pred_binary, zero_division=0)
        
        logger.info(f"\n  Test: AUC={auc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        results['bearish'] = {'model': model_bearish, 'auc': auc, 'precision': precision}
    
    # Bullish reversal model (Long)
    if len(X_bullish) >= 100:
        logger.info("\n[BULLISH REVERSAL] Training long signal model...")
        X_train, X_val, X_test, y_train, y_val, y_test = time_split(X_bullish, y_bullish, train_ratio, val_ratio)
        
        logger.info(f"  Train positive: {y_train.mean():.2%}")
        
        model_bullish = CatBoostClassifier(
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
        
        model_bullish.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        
        pred_proba = model_bullish.predict_proba(X_test)[:, 1]
        pred_binary = (pred_proba >= 0.5).astype(int)
        
        auc = roc_auc_score(y_test, pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        precision = precision_score(y_test, pred_binary, zero_division=0)
        recall = recall_score(y_test, pred_binary, zero_division=0)
        f1 = f1_score(y_test, pred_binary, zero_division=0)
        
        logger.info(f"\n  Test: AUC={auc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        results['bullish'] = {'model': model_bullish, 'auc': auc, 'precision': precision}
    
    # 保存
    logger.info("\nStep 7/7: Saving models...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    models_dir = Path('models_output')
    models_dir.mkdir(exist_ok=True)
    
    metadata = {
        'symbol': symbol, 'timeframe': timeframe, 'strategy': 'momentum_reversal',
        'params': {
            'consecutive_bars': consecutive_bars,
            'min_move_pct': min_move_pct,
            'reversal_threshold_multiplier': reversal_threshold_multiplier,
            'horizon': horizon
        }
    }
    
    if results.get('bearish'):
        path = models_dir / f'reversal_bearish_{timeframe}_v9_{timestamp}.pkl'
        with open(path, 'wb') as f:
            pickle.dump({
                'model': results['bearish']['model'],
                'features': features.columns.tolist(),
                'metadata': {**metadata, 'type': 'bearish_reversal', 'success_rate': bearish_success_rate}
            }, f)
        logger.info(f"[SAVE] Bearish: {path}")
    
    if results.get('bullish'):
        path = models_dir / f'reversal_bullish_{timeframe}_v9_{timestamp}.pkl'
        with open(path, 'wb') as f:
            pickle.dump({
                'model': results['bullish']['model'],
                'features': features.columns.tolist(),
                'metadata': {**metadata, 'type': 'bullish_reversal', 'success_rate': bullish_success_rate}
            }, f)
        logger.info(f"[SAVE] Bullish: {path}")
    
    logger.info("\n" + "="*80)
    logger.info("[DONE]")
    logger.info("="*80)
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train v9 momentum reversal model')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='15m')
    parser.add_argument('--days', type=int, default=9999)
    parser.add_argument('--consecutive-bars', type=int, default=3, help='連續幾根同方向')
    parser.add_argument('--min-move', type=float, default=0.005, help='最小累計漲跌幅')
    parser.add_argument('--reversal-multiplier', type=float, default=1.5, help='反轉目標 = ATR × 此倍數')
    parser.add_argument('--horizon', type=int, default=8)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    
    args = parser.parse_args()
    
    results = train_v9_model(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        consecutive_bars=args.consecutive_bars,
        min_move_pct=args.min_move,
        reversal_threshold_multiplier=args.reversal_multiplier,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )
    
    print("\n訓練完成!")
    if results.get('bearish'):
        print(f"Bearish (Short): AUC={results['bearish']['auc']:.3f}, Precision={results['bearish']['precision']:.3f}")
    if results.get('bullish'):
        print(f"Bullish (Long): AUC={results['bullish']['auc']:.3f}, Precision={results['bullish']['precision']:.3f}")
