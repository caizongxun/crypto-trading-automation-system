"""
完整的回測流程腦本
"""
import sys
from pathlib import Path
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.signal_detector import SignalDetector
from core.feature_engineer import FeatureEngineer
from core.ml_predictor import MLPredictor
from core.risk_manager import RiskManager
from backtest.engine import BacktestEngine
from data.hf_loader import HFDataLoader

def run_backtest(model_name, data_source='binance', backtest_days=30, 
                initial_capital=10, leverage=3, min_signal_strength=2, min_confidence=0.6):
    """執行完整回測流程"""
    
    print(f"\n{'='*60}")
    print(f"開始回測: {model_name}")
    print(f"{'='*60}\n")
    
    # 1. 加載模型
    print("[1/6] 加載訓練模型...")
    model_dir = Path('models') / model_name
    
    if not model_dir.exists():
        print(f"錯誤: 模型不存在 {model_dir}")
        return None
    
    with open(model_dir / 'model_config.json', 'r') as f:
        model_config = json.load(f)
    
    symbol = model_config['symbol']
    timeframe = model_config['timeframe']
    config = model_config['config']
    
    print(f"模型資訊: {symbol} {timeframe}")
    
    # 2. 加載數據
    print("\n[2/6] 加載歷史數據...")
    if data_source == 'binance':
        backtest_engine = BacktestEngine({
            'initial_capital': initial_capital,
            'leverage': leverage,
            'maker_fee': config['backtest']['maker_fee'],
            'taker_fee': config['backtest']['taker_fee'],
            'slippage': config['backtest']['slippage']
        })
        df = backtest_engine.fetch_latest_data(symbol, timeframe, days=backtest_days)
    else:
        loader = HFDataLoader()
        df = loader.load_klines(symbol, timeframe)
        df = df.tail(backtest_days * 96)  # 15m: 96 bars per day
    
    if df.empty:
        print("錯誤: 無法加載數據")
        return None
    
    print(f"加載完成: {len(df)} 筆數據")
    
    # 3. 信號檢測
    print("\n[3/6] 檢測交易信號...")
    signal_detector = SignalDetector(config['signal_detection'])
    df = signal_detector.detect_signals(df)
    
    # 4. 特徵工程
    print("\n[4/6] 生成ML特徵...")
    feature_engineer = FeatureEngineer(config['feature_engineering'])
    df = feature_engineer.create_features(df)
    
    # 5. ML預測
    print("\n[5/6] ML信號驗證...")
    ml_predictor = MLPredictor(config['ml_model'])
    ml_predictor.load(model_dir)
    df = ml_predictor.predict(df)
    
    valid_long = ((df['signal_long']==1) & (df['pred_long_valid']==1)).sum()
    valid_short = ((df['signal_short']==1) & (df['pred_short_valid']==1)).sum()
    print(f"有效做多信號: {valid_long} | 有效做空信號: {valid_short}")
    
    # 6. 執行回測
    print("\n[6/6] 執行回測...")
    risk_manager = RiskManager(config['risk_management'])
    
    # 計算止損止盈
    for i in range(len(df)):
        if df.iloc[i]['signal_long'] == 1:
            sltp = risk_manager.calculate_stop_loss_take_profit(df.iloc[:i+1], 'LONG')
            df.loc[df.index[i], 'stop_loss'] = sltp['stop_loss']
            df.loc[df.index[i], 'take_profit'] = sltp['take_profit']
        elif df.iloc[i]['signal_short'] == 1:
            sltp = risk_manager.calculate_stop_loss_take_profit(df.iloc[:i+1], 'SHORT')
            df.loc[df.index[i], 'stop_loss'] = sltp['stop_loss']
            df.loc[df.index[i], 'take_profit'] = sltp['take_profit']
    
    backtest_engine = BacktestEngine({
        'initial_capital': initial_capital,
        'leverage': leverage,
        'maker_fee': config['backtest']['maker_fee'],
        'taker_fee': config['backtest']['taker_fee'],
        'slippage': config['backtest']['slippage']
    })
    
    results = backtest_engine.run_backtest(df, min_signal_strength, min_confidence)
    
    if 'error' in results:
        print(f"\n錯誤: {results['error']}")
        return None
    
    # 7. 顯示結果
    print(f"\n{'='*60}")
    print("回測結果")
    print(f"{'='*60}")
    print(f"初始資金: {results['initial_capital']:.2f} USDT")
    print(f"最終資金: {results['final_capital']:.2f} USDT")
    print(f"總收益率: {results['total_return']:.2f}%")
    print(f"總盈虧: {results['total_pnl']:.2f} USDT")
    print(f"交易次數: {results['total_trades']}")
    print(f"勝率: {results['win_rate']:.2f}%")
    print(f"平均盈利: {results['avg_win']:.2f} USDT")
    print(f"平均號損: {results['avg_loss']:.2f} USDT")
    print(f"盈虧因子: {results['profit_factor']:.2f}")
    print(f"最大回撤: {results['max_drawdown']:.2f}%")
    print(f"Sharpe比率: {results['sharpe_ratio']:.2f}")
    print(f"{'='*60}\n")
    
    # 8. 生成圖表
    print("生成圖表...")
    save_charts(results, model_name)
    
    # 9. 保存結果
    output_dir = Path('backtest_results') / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'results.json', 'w') as f:
        results_to_save = {k: v for k, v in results.items() if k not in ['trades', 'equity_curve']}
        json.dump(results_to_save, f, indent=2)
    
    trades_df = pd.DataFrame(results['trades'])
    trades_df.to_csv(output_dir / 'trades.csv', index=False)
    
    equity_df = pd.DataFrame(results['equity_curve'])
    equity_df.to_csv(output_dir / 'equity_curve.csv', index=False)
    
    print(f"結果已保存至: {output_dir}")
    
    return results

def save_charts(results, model_name):
    """生成並保存圖表"""
    
    equity_df = pd.DataFrame(results['equity_curve'])
    trades_df = pd.DataFrame(results['trades'])
    
    # 權益曲線圖
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=('權益曲線', '水下回撤'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=equity_df['time'], y=equity_df['equity'], 
                  name='權益', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    equity_df['cummax'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
    
    fig.add_trace(
        go.Scatter(x=equity_df['time'], y=equity_df['drawdown'],
                  name='回撤', fill='tozeroy', line=dict(color='red', width=1)),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="時間", row=2, col=1)
    fig.update_yaxes(title_text="權益 (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="回撤 (%)", row=2, col=1)
    
    fig.update_layout(
        title=f'回測結果 - {model_name}',
        height=800,
        showlegend=True
    )
    
    output_dir = Path('backtest_results') / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_dir / 'equity_curve.html')
    
    print(f"圖表已保存: {output_dir / 'equity_curve.html'}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='回測反轉策略')
    parser.add_argument('--model', type=str, required=True, help='模型名稱')
    parser.add_argument('--source', type=str, default='binance', choices=['binance', 'hf'], help='數據源')
    parser.add_argument('--days', type=int, default=30, help='回測天數')
    parser.add_argument('--capital', type=float, default=10, help='初始資金')
    parser.add_argument('--leverage', type=int, default=3, help='槓桿倍數')
    parser.add_argument('--strength', type=int, default=2, help='最小信號強度')
    parser.add_argument('--confidence', type=float, default=0.6, help='最小置信度')
    
    args = parser.parse_args()
    
    run_backtest(
        args.model,
        data_source=args.source,
        backtest_days=args.days,
        initial_capital=args.capital,
        leverage=args.leverage,
        min_signal_strength=args.strength,
        min_confidence=args.confidence
    )
