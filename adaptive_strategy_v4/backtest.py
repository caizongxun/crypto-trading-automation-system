"""
V4 Backtest Script
Kelly倉位管理回測腥本
"""
import sys
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from adaptive_strategy_v4.data.hf_loader import HFDataLoader
from adaptive_strategy_v4.core.feature_engineer import FeatureEngineer
from adaptive_strategy_v4.core.label_generator import LabelGenerator
from adaptive_strategy_v4.core.neural_predictor import NeuralPredictor
from adaptive_strategy_v4.backtest.engine import BacktestEngine

def run_v4_backtest(model_name: str, config: dict):
    """
    執行V4回測
    
    Args:
        model_name: 模型名稱
        config: 回測配置
    """
    print("="*60)
    print(f"V4 Backtest - {model_name}")
    print("="*60)
    
    # 1. 加載模型配置
    print("\n[1/6] 加載模型...")
    model_dir = project_root / 'models' / model_name
    
    if not model_dir.exists():
        print(f"[Error] 模型不存在: {model_dir}")
        return
    
    with open(model_dir / 'model_config.json', 'r') as f:
        model_config = json.load(f)
    
    symbol = model_config['symbol']
    timeframe = model_config['timeframe']
    feature_names = model_config['feature_names']
    
    print(f"[OK] 模型: {symbol} {timeframe}")
    print(f"[OK] 特徵數: {len(feature_names)}")
    
    # 2. 加載數據
    print("\n[2/6] 加載數據...")
    loader = HFDataLoader()
    df = loader.load_klines(symbol, timeframe)
    print(f"[OK] 數據: {len(df)} 筆")
    
    # 3. 特徵工程
    print("\n[3/6] 特徵工程...")
    feature_engineer = FeatureEngineer({})
    df = feature_engineer.create_features(df)
    
    # 輔助特徵
    label_generator = LabelGenerator(model_config.get('label_config', {}))
    df = label_generator._calculate_helper_features(df)
    print(f"[OK] 特徵完成")
    
    # 4. 加載預測模型
    print("\n[4/6] 加載預測模型...")
    predictor = NeuralPredictor(model_config['predictor_config'])
    predictor.load(model_dir)
    print(f"[OK] 模型加載完成")
    
    # 5. 生成預測
    print("\n[5/6] 生成預測...")
    
    # 檢查特徵
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        print(f"[Warning] 缺少 {len(missing_features)} 個特徵")
        feature_names = [f for f in feature_names if f in df.columns]
    
    X = df[feature_names].values
    
    # LSTM預測：方向、勝率、賠率、信心度
    directions, win_rates, payoffs, confidences = predictor.predict(X)
    
    print(f"[OK] 預測完成")
    print(f"  - 信號數: {(directions != 0).sum()}")
    print(f"  - 做多: {(directions == 1).sum()}")
    print(f"  - 做空: {(directions == -1).sum()}")
    print(f"  - 平均勝率: {win_rates.mean():.3f}")
    print(f"  - 平均賠率: {payoffs.mean():.3f}")
    print(f"  - 平均信心度: {confidences.mean():.3f}")
    
    # 6. 執行回測
    print("\n[6/6] 執行回測...")
    backtest_config = {
        'initial_capital': config.get('initial_capital', 10000),
        'commission': config.get('commission', 0.001),
        'slippage': config.get('slippage', 0.0005),
        'kelly_fraction': config.get('kelly_fraction', 0.25),
        'max_position': config.get('max_position', 0.20),
        'min_kelly': config.get('min_kelly', 0.10),
        'max_leverage': config.get('max_leverage', 3),
        'atr_tp_multiplier': config.get('atr_tp_multiplier', 2.0),
        'atr_sl_multiplier': config.get('atr_sl_multiplier', 1.0)
    }
    
    engine = BacktestEngine(backtest_config)
    results = engine.run(df, directions, win_rates, payoffs, confidences)
    
    # 顯示結果
    print("\n" + "="*60)
    print("V4 Backtest Results")
    print("="*60)
    
    metrics = results['metrics']
    
    if 'error' in metrics:
        print(f"[Error] {metrics['error']}")
        return
    
    print(f"\n[Trading Performance]")
    print(f"  Total Trades: {metrics['total_trades']}")
    print(f"  Winning Trades: {metrics['winning_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  Total Return: {metrics['total_return']:.2%}")
    print(f"  Total PnL: ${metrics['total_pnl']:.2f}")
    
    print(f"\n[Risk Metrics]")
    print(f"  Avg Win: ${metrics['avg_win']:.2f}")
    print(f"  Avg Loss: ${metrics['avg_loss']:.2f}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    print(f"\n[Kelly Metrics]")
    print(f"  Avg Kelly: {metrics['avg_kelly']:.2%}")
    print(f"  Avg Leverage: {metrics['avg_leverage']:.2f}x")
    
    # 評分
    print(f"\n[Assessment]")
    if metrics['win_rate'] >= 0.60 and metrics['profit_factor'] >= 2.0:
        print("  ✓ [Excellent] Win rate ≥60% and PF ≥2.0")
    elif metrics['win_rate'] >= 0.55 and metrics['profit_factor'] >= 1.5:
        print("  ✓ [Good] Win rate ≥55% and PF ≥1.5")
    elif metrics['profit_factor'] >= 1.2:
        print("  ~ [Acceptable] PF ≥1.2")
    else:
        print("  ✗ [Poor] Needs improvement")
    
    if metrics['sharpe_ratio'] >= 2.0:
        print("  ✓ [Excellent] Sharpe ≥2.0")
    elif metrics['sharpe_ratio'] >= 1.5:
        print("  ✓ [Good] Sharpe ≥1.5")
    
    if metrics['max_drawdown'] <= -0.20:
        print("  ✓ [Good] Max DD ≤20%")
    elif metrics['max_drawdown'] <= -0.30:
        print("  ~ [Acceptable] Max DD ≤30%")
    else:
        print("  ✗ [Warning] Max DD >30%")
    
    print("\n" + "="*60)
    
    # 保存結果
    results_dir = model_dir / 'backtest_results'
    results_dir.mkdir(exist_ok=True)
    
    trades_df = pd.DataFrame(results['trades'])
    if len(trades_df) > 0:
        trades_df.to_csv(results_dir / 'trades.csv', index=False)
    
    equity_df = pd.DataFrame(results['equity_curve'])
    equity_df.to_csv(results_dir / 'equity_curve.csv', index=False)
    
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[Saved] Results saved to {results_dir}")

def main():
    parser = argparse.ArgumentParser(description='V4 Backtest')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (e.g., BTCUSDT_15m_v4_20260227_141234)')
    parser.add_argument('--capital', type=int, default=10000,
                       help='Initial capital (default: 10000)')
    parser.add_argument('--kelly-fraction', type=float, default=0.25,
                       help='Kelly fraction (default: 0.25)')
    parser.add_argument('--max-leverage', type=int, default=3,
                       help='Max leverage (default: 3)')
    
    args = parser.parse_args()
    
    config = {
        'initial_capital': args.capital,
        'kelly_fraction': args.kelly_fraction,
        'max_leverage': args.max_leverage,
        'commission': 0.001,
        'slippage': 0.0005
    }
    
    run_v4_backtest(args.model, config)

if __name__ == '__main__':
    main()
