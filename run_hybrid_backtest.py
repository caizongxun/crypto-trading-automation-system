#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
執行混合激進回測

Chronos + XGBoost v3 + 激進策略
目標: 30天翻倉

使用方法:
    python run_hybrid_backtest.py --days 30 --aggressive
    python run_hybrid_backtest.py --quick  # 快速測試
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import glob

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hybrid_backtest.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def find_latest_models() -> tuple:
    """
    自動尋找最新的 XGBoost v3 模型
    
    Returns:
        (long_model_path, short_model_path) or (None, None)
    """
    models_dir = Path('models_output')
    if not models_dir.exists():
        return None, None
    
    # 尋找所有 v3 模型
    long_models = list(models_dir.glob('*long_v3*.pkl'))
    short_models = list(models_dir.glob('*short_v3*.pkl'))
    
    if not long_models or not short_models:
        return None, None
    
    # 取最新的
    long_model = max(long_models, key=lambda p: p.stat().st_mtime)
    short_model = max(short_models, key=lambda p: p.stat().st_mtime)
    
    logger.info(f"[AUTO] Found Long model: {long_model.name}")
    logger.info(f"[AUTO] Found Short model: {short_model.name}")
    
    return str(long_model), str(short_model)


def parse_args():
    parser = argparse.ArgumentParser(description='Hybrid Aggressive Backtest')
    
    # 資料參數
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='交易對')
    parser.add_argument('--timeframe', type=str, default='1h', 
                       choices=['15m', '1h', '1d'],
                       help='時間週期')
    parser.add_argument('--days', type=int, default=30, help='回測天數')
    
    # 策略參數
    parser.add_argument('--initial-capital', type=float, default=10000,
                       help='初始資金')
    parser.add_argument('--target-multiplier', type=float, default=2.0,
                       help='目標倍數 (2.0 = 翻倍)')
    parser.add_argument('--base-position', type=float, default=20.0,
                       help='基礎倉位 (%)')
    parser.add_argument('--max-position', type=float, default=50.0,
                       help='最大倉位 (%)')
    parser.add_argument('--leverage', type=float, default=2.0,
                       help='槓桿倍數')
    
    # 模型路徑 (自動檢測)
    parser.add_argument('--xgb-long', type=str, default='',
                       help='XGBoost Long 模型路徑 (留空自動檢測)')
    parser.add_argument('--xgb-short', type=str, default='',
                       help='XGBoost Short 模型路徑 (留空自動檢測)')
    parser.add_argument('--chronos-model', type=str,
                       default='amazon/chronos-t5-small',
                       help='Chronos 模型')
    
    # 模式
    parser.add_argument('--quick', action='store_true',
                       help='快速測試 (7天)')
    parser.add_argument('--aggressive', action='store_true', default=True,
                       help='激進模式')
    parser.add_argument('--use-martingale', action='store_true', default=True,
                       help='使用倒金字塔')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 快速測試
    if args.quick:
        args.days = 7
        args.chronos_model = 'amazon/chronos-t5-tiny'
        logger.info("[QUICK] Quick test mode: 7 days, tiny model")
    
    logger.info("="*80)
    logger.info("[START] HYBRID AGGRESSIVE BACKTEST")
    logger.info("="*80)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Days: {args.days}")
    logger.info(f"Target: ${args.initial_capital:,.0f} -> ${args.initial_capital * args.target_multiplier:,.0f}")
    logger.info(f"Strategy: {'AGGRESSIVE' if args.aggressive else 'CONSERVATIVE'}")
    logger.info(f"Martingale: {'YES' if args.use_martingale else 'NO'}")
    logger.info(f"Leverage: {args.leverage}x")
    logger.info("="*80)
    
    # Step 1: 載入資料
    logger.info(f"\nStep 1/5: Loading data...")
    from utils.hf_data_loader import load_klines
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days + 30)  # 額外 30 天用於特徵計算
    
    df = load_klines(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    logger.info(f"[OK] Loaded {len(df)} bars")
    
    # Step 2: Chronos 預測
    logger.info(f"\nStep 2/5: Running Chronos predictions...")
    from models.chronos_predictor import ChronosPredictor
    
    predictor = ChronosPredictor(
        model_name=args.chronos_model,
        device="cpu"
    )
    
    df_chronos = predictor.predict_batch(
        df=df,
        lookback=168,
        horizon=1,
        num_samples=50,
        tp_pct=1.2,
        sl_pct=0.6
    )
    
    logger.info("[OK] Chronos predictions complete")
    
    # Step 3: XGBoost 特徵工程
    logger.info(f"\nStep 3/5: Engineering features for XGBoost...")
    
    # 自動檢測模型
    if not args.xgb_long or not args.xgb_short:
        logger.info("[AUTO] Auto-detecting XGBoost v3 models...")
        xgb_long_path, xgb_short_path = find_latest_models()
        
        if not xgb_long_path or not xgb_short_path:
            logger.error(f"[ERROR] No XGBoost v3 models found in models_output/")
            logger.error(f"\nPlease train XGBoost v3 models first:")
            logger.error(f"  python train_v3.py --symbol {args.symbol} --timeframe {args.timeframe}")
            sys.exit(1)
    else:
        xgb_long_path = args.xgb_long
        xgb_short_path = args.xgb_short
    
    xgb_long_path = Path(xgb_long_path)
    xgb_short_path = Path(xgb_short_path)
    
    if not xgb_long_path.exists() or not xgb_short_path.exists():
        logger.error(f"[ERROR] Model files not found!")
        logger.error(f"Long: {xgb_long_path}")
        logger.error(f"Short: {xgb_short_path}")
        sys.exit(1)
    
    from utils.feature_engineering_v3 import engineer_features_v3
    
    df_features = engineer_features_v3(
        df,
        symbol=args.symbol,
        timeframe=args.timeframe
    )
    
    logger.info(f"[OK] Features engineered: {len(df_features.columns)} columns")
    
    # Step 4: 初始化回測引擎
    logger.info(f"\nStep 4/5: Initializing backtest engine...")
    from utils.hybrid_backtester import HybridBacktester
    
    backtester = HybridBacktester(
        initial_capital=args.initial_capital,
        leverage=args.leverage,
        fee_rate=0.0004
    )
    
    backtester.load_models(
        xgb_long_path=str(xgb_long_path),
        xgb_short_path=str(xgb_short_path)
    )
    
    # Step 5: 執行回測
    logger.info(f"\nStep 5/5: Running backtest...")
    
    # 策略配置
    strategy_config = {
        'initial_capital': args.initial_capital,
        'target_multiplier': args.target_multiplier,
        'target_days': args.days,
        'base_position_pct': args.base_position,
        'max_position_pct': args.max_position,
        'use_martingale': args.use_martingale,
        'max_martingale_level': 3
    }
    
    # 只使用最近 N 天的資料
    cutoff_idx = -int(args.days * 24) if args.timeframe == '1h' else -int(args.days * 96)
    df_backtest = df.iloc[cutoff_idx:].reset_index(drop=True)
    df_chronos_backtest = df_chronos.iloc[cutoff_idx:].reset_index(drop=True)
    df_features_backtest = df_features.iloc[cutoff_idx:].reset_index(drop=True)
    
    result = backtester.run_backtest(
        df=df_backtest,
        df_chronos=df_chronos_backtest,
        df_features=df_features_backtest,
        strategy_config=strategy_config
    )
    
    # 顯示結果
    if not result['success']:
        logger.error(f"[ERROR] Backtest failed: {result.get('error')}")
        sys.exit(1)
    
    stats = result['stats']
    trades_df = result['trades']
    
    # 詳細統計
    logger.info(f"\n" + "="*80)
    logger.info("[STATS] DETAILED STATISTICS")
    logger.info("="*80)
    
    # 交易統計
    tp_trades = len(trades_df[trades_df['exit_reason'] == 'TP'])
    sl_trades = len(trades_df[trades_df['exit_reason'] == 'SL'])
    
    logger.info(f"\n[TRADES] Trade Breakdown:")
    logger.info(f"  TP Trades: {tp_trades} ({tp_trades/stats['total_trades']*100:.1f}%)")
    logger.info(f"  SL Trades: {sl_trades} ({sl_trades/stats['total_trades']*100:.1f}%)")
    logger.info(f"  Avg Win: +{stats['avg_win']:.2f}%")
    logger.info(f"  Avg Loss: {stats['avg_loss']:.2f}%")
    
    # 每日統計
    daily_trades = stats['total_trades'] / args.days
    daily_return = stats['total_return'] / args.days
    
    logger.info(f"\n[DAILY] Daily Statistics:")
    logger.info(f"  Trades/Day: {daily_trades:.1f}")
    logger.info(f"  Return/Day: {daily_return:+.2f}%")
    
    # 達成狀況
    target_return = (args.target_multiplier - 1) * 100
    achievement = (stats['total_return'] / target_return) * 100
    
    logger.info(f"\n[TARGET] Target Achievement:")
    logger.info(f"  Target: +{target_return:.0f}%")
    logger.info(f"  Actual: +{stats['total_return']:.2f}%")
    logger.info(f"  Achievement: {achievement:.1f}%")
    
    if achievement >= 100:
        logger.info(f"\n[SUCCESS] TARGET ACHIEVED!")
    elif achievement >= 70:
        logger.info(f"\n[GOOD] Close! Adjust parameters to reach target.")
    else:
        logger.info(f"\n[WARN] Far from target. Consider:")
        logger.info(f"  1. Increase leverage ({args.leverage}x -> {args.leverage * 1.5}x)")
        logger.info(f"  2. Increase position size ({args.base_position}% -> {args.base_position * 1.2:.0f}%)")
        logger.info(f"  3. Lower signal thresholds")
        logger.info(f"  4. Use martingale strategy")
    
    # 儲存結果
    output_dir = Path('backtest_results')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    trades_file = output_dir / f'hybrid_trades_{args.symbol}_{args.timeframe}_{timestamp}.csv'
    
    trades_df.to_csv(trades_file, index=False)
    logger.info(f"\n[SAVE] Results saved: {trades_file}")
    
    logger.info(f"\n" + "="*80)
    logger.info("[DONE] BACKTEST COMPLETE")
    logger.info("="*80)
    
    return result


if __name__ == "__main__":
    Path('logs').mkdir(exist_ok=True)
    Path('backtest_results').mkdir(exist_ok=True)
    
    try:
        result = main()
    except KeyboardInterrupt:
        logger.warning("\n[STOP] Backtest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n[ERROR] Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
