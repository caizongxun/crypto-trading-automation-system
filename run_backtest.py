#!/usr/bin/env python3
"""
雙向智能體回測執行腳本

使用方法:
python run_backtest.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

from config import Config
from utils.logger import setup_logger
from utils.agent_backtester import BidirectionalAgentBacktester
from utils.feature_engineering import FeatureEngineer

logger = setup_logger('run_backtest', 'logs/run_backtest.log')

def load_klines(symbol: str, timeframe: str) -> pd.DataFrame:
    """從 HuggingFace 載入 K 線數據"""
    try:
        repo_id = Config.HF_REPO_ID
        base = symbol.replace("USDT", "")
        filename = f"{base}_{timeframe}.parquet"
        path_in_repo = f"klines/{symbol}/{filename}"
        
        logger.info(f"Loading {symbol} {timeframe} from HuggingFace")
        
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            token=Config.HF_TOKEN
        )
        df = pd.read_parquet(local_path)
        logger.info(f"Loaded {len(df):,} records for {symbol} {timeframe}")
        return df
    
    except Exception as e:
        logger.error(f"Failed to load {symbol} {timeframe}: {str(e)}")
        raise

def main():
    logger.info("="*80)
    logger.info("BIDIRECTIONAL AGENT BACKTEST")
    logger.info("="*80)
    
    # ============================
    # Step 1: 載入 1m K 線
    # ============================
    logger.info("Step 1: Loading 1m K-lines...")
    df_1m = load_klines("BTCUSDT", "1m")
    
    # 設定 index
    if 'open_time' in df_1m.columns:
        df_1m['open_time'] = pd.to_datetime(df_1m['open_time'])
        df_1m.set_index('open_time', inplace=True)
    
    # ============================
    # Step 2: 生成特徵 (滾動視窗)
    # ============================
    logger.info("Step 2: Generating features (rolling window architecture)...")
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.create_features_from_1m(
        df_1m,
        use_micro_structure=True,
        label_type='both'
    )
    
    logger.info(f"Features generated: {len(df_features):,} samples")
    
    # ============================
    # Step 3: Train/Test Split
    # ============================
    logger.info("Step 3: Splitting train/test...")
    split_idx = int(len(df_features) * 0.8)
    df_train = df_features.iloc[:split_idx]
    df_test = df_features.iloc[split_idx:]
    
    logger.info(f"Train: {len(df_train):,} samples ({df_train.index[0]} to {df_train.index[-1]})")
    logger.info(f"Test:  {len(df_test):,} samples ({df_test.index[0]} to {df_test.index[-1]})")
    
    # ============================
    # Step 4: 初始化回測器
    # ============================
    logger.info("Step 4: Initializing backtester...")
    
    # 尋找最新的模型檔案
    models_dir = Path('models_output')
    model_files = list(models_dir.glob('catboost_long_*.pkl'))
    
    if not model_files:
        logger.error("No trained models found in models_output/")
        logger.error("Please train models first using the Model Training tab")
        return
    
    # 使用最新的模型
    latest_long_model = max(model_files, key=lambda x: x.stat().st_mtime)
    model_prefix = latest_long_model.stem.replace('_long_', '_short_')
    latest_short_model = models_dir / f"{model_prefix}.pkl"
    
    if not latest_short_model.exists():
        logger.error(f"Short model not found: {latest_short_model}")
        return
    
    logger.info(f"Using models:")
    logger.info(f"  Long:  {latest_long_model.name}")
    logger.info(f"  Short: {latest_short_model.name}")
    
    # 初始化回測器
    backtester = BidirectionalAgentBacktester(
        model_long_path=str(latest_long_model),
        model_short_path=str(latest_short_model),
        initial_capital=10000.0,
        position_size_pct=0.95,
        prob_threshold_long=0.65,
        prob_threshold_short=0.65,
        tp_pct=0.02,
        sl_pct=0.01,
        hunting_expire_bars=15,
        trading_hours=[(0, 24)],  # 24/7 交易
        maker_fee=0.0001,
        taker_fee=0.0004,
        slippage=0.0002
    )
    
    # ============================
    # Step 5: 執行回測
    # ============================
    logger.info("Step 5: Running backtest...")
    
    feature_cols = ['efficiency_ratio', 'extreme_time_diff', 'vol_imbalance_ratio',
                   'z_score', 'bb_width_pct', 'rsi', 'atr_pct', 'z_score_1h', 'atr_pct_1d']
    available_features = [col for col in feature_cols if col in df_test.columns]
    
    results = backtester.run(df_test, available_features)
    
    # ============================
    # Step 6: 儲存結果
    # ============================
    logger.info("Step 6: Saving results...")
    
    # 創建輸出目錄
    output_dir = Path('backtest_results')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 儲存 JSON 報告
    report_path = output_dir / f'backtest_report_{timestamp}.json'
    with open(report_path, 'w') as f:
        # 轉換 numpy 類型
        results_serializable = {}
        for k, v in results.items():
            if isinstance(v, (np.integer, np.floating)):
                results_serializable[k] = float(v)
            else:
                results_serializable[k] = v
        json.dump(results_serializable, f, indent=4)
    
    logger.info(f"Report saved: {report_path}")
    
    # 儲存交易記錄
    trades_df = backtester.get_trades_df()
    if not trades_df.empty:
        trades_path = output_dir / f'trades_{timestamp}.csv'
        trades_df.to_csv(trades_path, index=False)
        logger.info(f"Trades saved: {trades_path}")
    
    # 儲存權益曲線
    equity_df = backtester.get_equity_curve()
    equity_path = output_dir / f'equity_curve_{timestamp}.csv'
    equity_df.to_csv(equity_path, index=False)
    logger.info(f"Equity curve saved: {equity_path}")
    
    # ============================
    # Step 7: 可視化
    # ============================
    logger.info("Step 7: Generating visualizations...")
    
    # 權益曲線圖
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 子圖 1: 資金曲線
    ax1 = axes[0]
    ax1.plot(equity_df['timestamp'], equity_df['capital'], linewidth=1.5, color='#2E86AB')
    ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.set_title('Capital Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Capital ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 子圖 2: 狀態分布
    ax2 = axes[1]
    state_mapping = {
        'IDLE': 0,
        'HUNTING_LONG': 1,
        'HUNTING_SHORT': 2,
        'LONG_POSITION': 3,
        'SHORT_POSITION': 4
    }
    equity_df['state_num'] = equity_df['state'].map(state_mapping)
    ax2.scatter(equity_df['timestamp'], equity_df['state_num'], s=1, alpha=0.3, color='#A23B72')
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_yticklabels(['IDLE', 'HUNTING_LONG', 'HUNTING_SHORT', 'LONG_POS', 'SHORT_POS'])
    ax2.set_title('Agent State Timeline', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('State')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    chart_path = output_dir / f'backtest_chart_{timestamp}.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    logger.info(f"Chart saved: {chart_path}")
    
    # ============================
    # Step 8: 顯示結果摘要
    # ============================
    logger.info("="*80)
    logger.info("BACKTEST SUMMARY")
    logger.info("="*80)
    logger.info(f"Test Period: {df_test.index[0]} to {df_test.index[-1]}")
    logger.info(f"Total Bars: {len(df_test):,}")
    logger.info(f"Models Used:")
    logger.info(f"  Long:  {latest_long_model.name}")
    logger.info(f"  Short: {latest_short_model.name}")
    logger.info("="*80)
    logger.info("PERFORMANCE METRICS")
    logger.info("="*80)
    if results:
        logger.info(f"Total Trades: {results.get('total_trades', 0)}")
        logger.info(f"  Long: {results.get('long_trades', 0)} | Short: {results.get('short_trades', 0)}")
        logger.info(f"Win Rate: {results.get('win_rate', 0)*100:.2f}%")
        logger.info(f"Total Return: {results.get('total_return_pct', 0)*100:+.2f}%")
        logger.info(f"Final Capital: ${results.get('final_capital', 0):,.2f}")
        logger.info(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
        logger.info(f"TP Rate: {results.get('tp_rate', 0)*100:.1f}%")
        logger.info(f"SL Rate: {results.get('sl_rate', 0)*100:.1f}%")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*80)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}", exc_info=True)
        raise