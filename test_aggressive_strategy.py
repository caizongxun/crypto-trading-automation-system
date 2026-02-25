#!/usr/bin/env python3
"""
激進複利策略測試

目標: 30天翻倉 (2x)
策略: Chronos + XGBoost 混合
使用: python test_aggressive_strategy.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.hybrid_predictor import HybridPredictor, print_strategy_comparison
from utils.aggressive_backtester import AggressiveBacktester
from utils.hf_data_loader import load_klines

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_aggressive_backtest(
    symbol: str = 'BTCUSDT',
    timeframe: str = '1h',
    days: int = 30,
    strategy: str = 'aggressive',
    initial_capital: float = 1000.0
):
    """
    執行激進複利回測
    """
    logger.info("="*80)
    logger.info("激進複利策略回測")
    logger.info("="*80)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Backtest days: {days}")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Initial capital: ${initial_capital}")
    
    # Step 1: 載入數據
    logger.info("\nStep 1/4: Loading data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 7)  # +7 for lookback
    
    df = load_klines(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    if len(df) == 0:
        logger.error("No data loaded!")
        return None
    
    logger.info(f"Loaded {len(df)} K-lines")
    
    # Step 2: 初始化混合預測器
    logger.info("\nStep 2/4: Initializing hybrid predictor...")
    logger.info("⚠️  首次使用會下載 Chronos 模型...")
    
    try:
        predictor = HybridPredictor(strategy=strategy)
        params = predictor.params
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        logger.error("⚠️  如果沒有 XGBoost 模型,只會使用 Chronos")
        predictor = HybridPredictor(strategy=strategy)
        params = predictor.params
    
    logger.info("✅ Predictor initialized")
    
    # Step 3: 初始化回測引擎
    logger.info("\nStep 3/4: Initializing backtester...")
    
    backtester = AggressiveBacktester(
        initial_capital=initial_capital,
        position_size=params['position_size'],
        tp_pct=params['tp_pct'],
        sl_pct=params['sl_pct'],
        max_trades_per_day=params['max_trades_per_day'],
        enable_compound=True
    )
    
    logger.info("✅ Backtester initialized")
    
    # Step 4: 執行回測
    logger.info("\nStep 4/4: Running backtest...")
    logger.info(f"Processing {len(df)} K-lines...")
    
    lookback = 168  # 7 days for Chronos
    
    for i in range(lookback, len(df)):
        if i % 100 == 0:
            progress = (i - lookback) / (len(df) - lookback) * 100
            logger.info(f"Progress: {progress:.1f}% ({i}/{len(df)})")
        
        current_row = df.iloc[i]
        
        # 檢查出倉
        if backtester.position is not None:
            backtester.check_exit(
                current_high=current_row['high'],
                current_low=current_row['low'],
                current_time=current_row['open_time']
            )
        
        # 檢查開倉 (每 4 根 K 線預測一次)
        if backtester.position is None and i % 4 == 0:
            # 準備 Chronos 輸入
            window = df.iloc[i-lookback:i]
            
            # 預測
            try:
                signal, confidence, details = predictor.predict(
                    df=window,
                    features_df=None  # 如果有 XGBoost 特徵,這裡傳入
                )
                
                # 根據信號開倉
                if signal in ['LONG', 'SHORT']:
                    backtester.open_position(
                        side=signal,
                        price=current_row['close'],
                        timestamp=current_row['open_time'],
                        confidence=confidence
                    )
            
            except Exception as e:
                logger.debug(f"Prediction failed at {i}: {e}")
                continue
    
    # Step 5: 顯示結果
    logger.info("\n✅ Backtest completed!")
    logger.info("="*80)
    
    backtester.print_summary()
    
    # 返回統計
    return backtester.get_stats(), backtester.trades


def compare_strategies():
    """
    比較不同策略
    """
    print("\n" + "="*80)
    print("策略比較")
    print("="*80)
    
    strategies = ['aggressive', 'moderate', 'conservative']
    results = {}
    
    for strategy in strategies:
        print(f"\n>>> Testing {strategy} strategy...")
        
        try:
            stats, trades = run_aggressive_backtest(
                symbol='BTCUSDT',
                timeframe='1h',
                days=30,
                strategy=strategy,
                initial_capital=1000.0
            )
            results[strategy] = stats
        
        except Exception as e:
            print(f"\u274c {strategy} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 比較結果
    print("\n" + "="*80)
    print("策略比較結果")
    print("="*80)
    
    print(f"\n{'Metric':<20} {'Aggressive':<15} {'Moderate':<15} {'Conservative':<15}")
    print("-"*65)
    
    metrics = [
        ('total_return', '%'),
        ('total_trades', ''),
        ('win_rate', '%'),
        ('profit_factor', ''),
        ('max_drawdown', '%')
    ]
    
    for metric, unit in metrics:
        row = [metric.replace('_', ' ').title()]
        for strategy in strategies:
            if strategy in results:
                value = results[strategy].get(metric, 0)
                if unit == '%':
                    row.append(f"{value:+.2f}%")
                else:
                    row.append(f"{value:.2f}")
            else:
                row.append("N/A")
        
        print(f"{row[0]:<20} {row[1]:<15} {row[2]:<15} {row[3]:<15}")
    
    print("\n" + "="*80)
    
    # 找出最佳策略
    best_strategy = max(results.keys(), key=lambda k: results[k]['total_return'])
    best_return = results[best_strategy]['total_return']
    
    print(f"\n🏆 最佳策略: {best_strategy.upper()}")
    print(f"   總報酬: {best_return:+.2f}% ({(100+best_return)/100:.2f}x)")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test aggressive compound strategy')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='交易對')
    parser.add_argument('--timeframe', type=str, default='1h', help='時間週期')
    parser.add_argument('--days', type=int, default=30, help='回測天數')
    parser.add_argument('--strategy', type=str, default='aggressive',
                        choices=['aggressive', 'moderate', 'conservative'],
                        help='策略')
    parser.add_argument('--capital', type=float, default=1000.0, help='初始資金')
    parser.add_argument('--compare', action='store_true', help='比較所有策略')
    
    args = parser.parse_args()
    
    # 顯示策略參數
    print_strategy_comparison()
    
    try:
        if args.compare:
            # 比較模式
            compare_strategies()
        else:
            # 單策略測試
            stats, trades = run_aggressive_backtest(
                symbol=args.symbol,
                timeframe=args.timeframe,
                days=args.days,
                strategy=args.strategy,
                initial_capital=args.capital
            )
            
            if stats:
                # 儲存交易記錄
                if trades:
                    trades_df = pd.DataFrame(trades)
                    output_file = f"aggressive_trades_{args.strategy}_{args.days}d.csv"
                    trades_df.to_csv(output_file, index=False)
                    print(f"\n✅ Trades saved to: {output_file}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n\u274c Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
