#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v10 策略參數自動優化器

功能:
1. 網格搜索 (Grid Search)
2. 貪婪搜索 (Greedy Search)
3. 随機搜索 (Random Search)
4. 多目標優化 (Pareto Frontier)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
import itertools
from typing import Dict, List, Tuple
import logging
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/optimize_v10.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class V10ParameterOptimizer:
    def __init__(
        self,
        long_model_path: str,
        short_model_path: str,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        initial_capital: float = 10000,
        leverage: int = 10
    ):
        self.long_model_path = long_model_path
        self.short_model_path = short_model_path
        self.df = df
        self.train_ratio = train_ratio
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.oos_start = int(len(df) * train_ratio)
        
        logger.info(f"數據範圍: {len(df)} 根K線")
        logger.info(f"訓練集: 0-{self.oos_start}")
        logger.info(f"測試集: {self.oos_start}-{len(df)}")
    
    def run_single_backtest(
        self,
        params: Dict,
        verbose: bool = False
    ) -> Dict:
        """執行單次回測"""
        from backtest_v10_scalping_advanced import AdvancedScalpingBacktester
        
        try:
            backtester = AdvancedScalpingBacktester(
                long_model_path=self.long_model_path,
                short_model_path=self.short_model_path,
                initial_capital=self.initial_capital,
                leverage=self.leverage,
                **params
            )
            
            report = backtester.run_backtest(
                self.df,
                start_idx=self.oos_start,
                long_enabled=True,
                short_enabled=True
            )
            
            if not report:
                return {'valid': False}
            
            summary = report['summary']
            
            result = {
                'valid': True,
                'total_trades': summary['total_trades'],
                'win_rate': summary['win_rate'],
                'total_return_pct': summary['total_return_pct'],
                'sharpe_ratio': summary['sharpe_ratio'],
                'profit_factor': summary['profit_factor'],
                'max_drawdown': summary['max_drawdown'],
                'avg_win': summary['avg_win'],
                'avg_loss': summary['avg_loss']
            }
            
            if verbose:
                logger.info(f"  Return: {result['total_return_pct']*100:.2f}%, "
                          f"Sharpe: {result['sharpe_ratio']:.2f}, "
                          f"Trades: {result['total_trades']}")
            
            return result
            
        except Exception as e:
            logger.error(f"回測失敗: {e}")
            return {'valid': False}
    
    def grid_search(
        self,
        param_grid: Dict[str, List],
        max_combinations: int = 1000,
        sort_by: str = 'sharpe_ratio'
    ) -> pd.DataFrame:
        """
        網格搜索 - 遍歷所有參數組合
        
        Args:
            param_grid: 參數空間
            max_combinations: 最大組合數 (防止爆破)
            sort_by: 排序指標
        """
        logger.info("="*80)
        logger.info("[START] Grid Search Optimization")
        logger.info("="*80)
        
        # 生成所有組合
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        
        total_combinations = len(combinations)
        logger.info(f"總組合數: {total_combinations}")
        
        if total_combinations > max_combinations:
            logger.warning(f"組合數過多,隨機抽樣 {max_combinations} 個")
            import random
            random.seed(42)
            combinations = random.sample(combinations, max_combinations)
        
        results = []
        
        for i, combo in enumerate(combinations, 1):
            params = dict(zip(keys, combo))
            
            if i % 10 == 0:
                logger.info(f"\n[{i}/{len(combinations)}] 測試配置:")
                for k, v in params.items():
                    if isinstance(v, float):
                        logger.info(f"  {k}: {v:.4f}")
                    else:
                        logger.info(f"  {k}: {v}")
            
            result = self.run_single_backtest(params, verbose=(i % 10 == 0))
            
            if result['valid']:
                results.append({**params, **result})
        
        if not results:
            logger.error("所有配置皆失敗")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(sort_by, ascending=False)
        
        logger.info("\n" + "="*80)
        logger.info("[TOP 5 CONFIGS]")
        logger.info("="*80)
        
        for idx, row in results_df.head(5).iterrows():
            logger.info(f"\nRank #{results_df.index.get_loc(idx) + 1}:")
            logger.info(f"  Return: {row['total_return_pct']*100:.2f}%")
            logger.info(f"  Sharpe: {row['sharpe_ratio']:.2f}")
            logger.info(f"  Win Rate: {row['win_rate']*100:.2f}%")
            logger.info(f"  Trades: {int(row['total_trades'])}")
            logger.info(f"  Max DD: {row['max_drawdown']*100:.2f}%")
            logger.info(f"  Config:")
            for k in keys:
                v = row[k]
                if isinstance(v, float):
                    logger.info(f"    {k}: {v:.4f}")
                else:
                    logger.info(f"    {k}: {v}")
        
        return results_df
    
    def greedy_search(
        self,
        base_params: Dict,
        param_ranges: Dict[str, List],
        max_iterations: int = 50
    ) -> Tuple[Dict, Dict]:
        """
        貪婪搜索 - 每次只改一個參數
        
        Args:
            base_params: 起始參數
            param_ranges: 每個參數的可選值
            max_iterations: 最大迭代次數
        """
        logger.info("="*80)
        logger.info("[START] Greedy Search Optimization")
        logger.info("="*80)
        
        current_params = base_params.copy()
        current_result = self.run_single_backtest(current_params, verbose=True)
        
        if not current_result['valid']:
            logger.error("起始配置無效")
            return {}, {}
        
        current_score = current_result['sharpe_ratio']
        logger.info(f"\n起始 Sharpe: {current_score:.2f}")
        
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            iteration += 1
            improved = False
            
            logger.info(f"\n[Iteration {iteration}]")
            
            # 嘗試改變每個參數
            for param_name, values in param_ranges.items():
                original_value = current_params[param_name]
                
                for new_value in values:
                    if new_value == original_value:
                        continue
                    
                    # 測試新值
                    test_params = current_params.copy()
                    test_params[param_name] = new_value
                    
                    result = self.run_single_backtest(test_params)
                    
                    if not result['valid']:
                        continue
                    
                    new_score = result['sharpe_ratio']
                    
                    # 如果更好則更新
                    if new_score > current_score:
                        logger.info(f"  改進! {param_name}: {original_value} -> {new_value}")
                        logger.info(f"    Sharpe: {current_score:.2f} -> {new_score:.2f}")
                        
                        current_params[param_name] = new_value
                        current_result = result
                        current_score = new_score
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                logger.info("  無法繼續改進")
        
        logger.info("\n" + "="*80)
        logger.info("[BEST CONFIG]")
        logger.info("="*80)
        logger.info(f"Sharpe: {current_score:.2f}")
        logger.info(f"Return: {current_result['total_return_pct']*100:.2f}%")
        logger.info(f"Config: {current_params}")
        
        return current_params, current_result
    
    def random_search(
        self,
        param_distributions: Dict[str, Tuple],
        n_iterations: int = 100,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        随機搜索 - 在範圍內隨機抽樣
        
        Args:
            param_distributions: {param: (min, max, type)}
            n_iterations: 抽樣次數
        """
        logger.info("="*80)
        logger.info("[START] Random Search Optimization")
        logger.info("="*80)
        
        np.random.seed(seed)
        results = []
        
        for i in range(1, n_iterations + 1):
            params = {}
            
            for param_name, (min_val, max_val, param_type) in param_distributions.items():
                if param_type == 'float':
                    params[param_name] = np.random.uniform(min_val, max_val)
                elif param_type == 'int':
                    params[param_name] = np.random.randint(min_val, max_val + 1)
                elif param_type == 'bool':
                    params[param_name] = bool(np.random.randint(0, 2))
            
            if i % 20 == 0:
                logger.info(f"\n[{i}/{n_iterations}] 測試配置: {params}")
            
            result = self.run_single_backtest(params, verbose=(i % 20 == 0))
            
            if result['valid']:
                results.append({**params, **result})
        
        if not results:
            logger.error("所有配置皆失敗")
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)
        
        logger.info("\n" + "="*80)
        logger.info("[TOP 3 CONFIGS]")
        logger.info("="*80)
        
        for idx, row in results_df.head(3).iterrows():
            logger.info(f"\nRank #{results_df.index.get_loc(idx) + 1}:")
            logger.info(f"  Return: {row['total_return_pct']*100:.2f}%")
            logger.info(f"  Sharpe: {row['sharpe_ratio']:.2f}")
            logger.info(f"  Config: {row.to_dict()}")
        
        return results_df
    
    def pareto_frontier(
        self,
        results_df: pd.DataFrame,
        objectives: List[str] = ['total_return_pct', 'sharpe_ratio'],
        maximize: List[bool] = [True, True]
    ) -> pd.DataFrame:
        """
        尋找 Pareto 最優解 - 多目標優化
        
        Args:
            results_df: 結果 DataFrame
            objectives: 目標指標列表
            maximize: 是否最大化
        """
        logger.info("\n" + "="*80)
        logger.info("[Pareto Frontier Analysis]")
        logger.info("="*80)
        
        pareto_front = []
        
        for i, row_i in results_df.iterrows():
            is_dominated = False
            
            for j, row_j in results_df.iterrows():
                if i == j:
                    continue
                
                # 檢查 row_j 是否支配 row_i
                dominates = True
                strictly_better = False
                
                for obj, max_flag in zip(objectives, maximize):
                    val_i = row_i[obj]
                    val_j = row_j[obj]
                    
                    if max_flag:
                        if val_j < val_i:
                            dominates = False
                            break
                        if val_j > val_i:
                            strictly_better = True
                    else:
                        if val_j > val_i:
                            dominates = False
                            break
                        if val_j < val_i:
                            strictly_better = True
                
                if dominates and strictly_better:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(i)
        
        pareto_df = results_df.loc[pareto_front].copy()
        
        logger.info(f"找到 {len(pareto_df)} 個 Pareto 最優解")
        
        for idx, row in pareto_df.iterrows():
            logger.info(f"\nConfig #{results_df.index.get_loc(idx) + 1}:")
            for obj in objectives:
                logger.info(f"  {obj}: {row[obj]:.4f}")
        
        return pareto_df


def run_quick_optimization(
    symbol: str = 'BTCUSDT',
    timeframe: str = '15m',
    days: int = 90
):
    """快速優化 - 針對 TP/SL 和 Threshold"""
    logger.info("="*80)
    logger.info("[QUICK OPTIMIZATION] TP/SL + Threshold")
    logger.info("="*80)
    
    # 載入數據
    from utils.hf_data_loader import load_klines
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = load_klines(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    if df is None or len(df) == 0:
        logger.error("數據載入失敗")
        return None
    
    logger.info(f"載入 {len(df)} 根K線")
    
    # 找模型
    models_dir = Path('models_output')
    long_models = sorted(models_dir.glob('scalping_long_*_v10_*.pkl'))
    short_models = sorted(models_dir.glob('scalping_short_*_v10_*.pkl'))
    
    if not long_models or not short_models:
        logger.error("未找到 v10 模型")
        return None
    
    optimizer = V10ParameterOptimizer(
        long_model_path=str(long_models[-1]),
        short_model_path=str(short_models[-1]),
        df=df,
        train_ratio=0.8
    )
    
    # 參數空間
    param_grid = {
        'position_size': [0.01, 0.015, 0.02, 0.025, 0.03],
        'threshold': [0.50, 0.55, 0.60, 0.65, 0.70],
        'tp_pct': [0.003, 0.004, 0.005, 0.006, 0.007, 0.008],
        'sl_pct': [0.002, 0.0025, 0.003, 0.0035, 0.004],
        'enable_dynamic_tpsl': [False],
        'enable_quality_sizing': [False],
        'enable_trailing_stop': [False],
        'enable_time_filter': [False],
        'enable_strict_filter': [False]
    }
    
    # 網格搜索
    results_df = optimizer.grid_search(
        param_grid,
        max_combinations=500,
        sort_by='sharpe_ratio'
    )
    
    if results_df.empty:
        logger.error("優化失敗")
        return None
    
    # 保存結果
    output_dir = Path('backtest_results/v10_optimization')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = output_dir / f'optimization_results_{timestamp}.csv'
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    
    logger.info(f"\n結果已保存: {results_path}")
    
    # 保存最佳配置
    best = results_df.iloc[0]
    best_config = {
        'timestamp': timestamp,
        'symbol': symbol,
        'timeframe': timeframe,
        'days': days,
        'best_params': {
            'position_size': float(best['position_size']),
            'threshold': float(best['threshold']),
            'tp_pct': float(best['tp_pct']),
            'sl_pct': float(best['sl_pct'])
        },
        'performance': {
            'total_return_pct': float(best['total_return_pct']),
            'sharpe_ratio': float(best['sharpe_ratio']),
            'win_rate': float(best['win_rate']),
            'total_trades': int(best['total_trades']),
            'profit_factor': float(best['profit_factor']),
            'max_drawdown': float(best['max_drawdown'])
        }
    }
    
    config_path = output_dir / f'best_config_{timestamp}.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"最佳配置已保存: {config_path}")
    
    return results_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize v10 strategy parameters')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='15m')
    parser.add_argument('--days', type=int, default=90)
    parser.add_argument('--mode', type=str, default='quick', choices=['quick', 'full', 'random'])
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        results = run_quick_optimization(
            symbol=args.symbol,
            timeframe=args.timeframe,
            days=args.days
        )
    
    logger.info("\n優化完成!")
