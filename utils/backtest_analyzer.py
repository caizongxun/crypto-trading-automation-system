import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from typing import Dict, List
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger('backtest_analyzer', 'logs/backtest_analyzer.log')

class BacktestAnalyzer:
    """
    完整的回測診斷分析工具
    
    核心功能:
    - 特徵重要性分析
    - 時段績效分解
    - 機率分層測試
    - Long vs Short 對比
    - 失敗案例診斷
    - 波動率環境分類
    """
    
    def __init__(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame,
                 model_long_path: str, model_short_path: str):
        self.trades_df = trades_df
        self.equity_df = equity_df
        
        # 載入模型以獲取特徵重要性
        try:
            self.model_long = joblib.load(model_long_path)
            self.model_short = joblib.load(model_short_path)
            logger.info("Models loaded for feature importance analysis")
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
            self.model_long = None
            self.model_short = None
    
    def analyze_all(self) -> Dict:
        """
        執行完整診斷分析
        """
        logger.info("="*80)
        logger.info("COMPREHENSIVE BACKTEST ANALYSIS")
        logger.info("="*80)
        
        results = {}
        
        # 1. 基礎統計
        results['basic_stats'] = self.analyze_basic_stats()
        
        # 2. 時段分析
        results['hourly_performance'] = self.analyze_hourly_performance()
        
        # 3. 機率分層分析
        results['probability_layers'] = self.analyze_probability_layers()
        
        # 4. Long vs Short 對比
        results['direction_comparison'] = self.analyze_direction_comparison()
        
        # 5. 失敗案例分析
        results['failure_analysis'] = self.analyze_failures()
        
        # 6. 特徵重要性
        if self.model_long and self.model_short:
            results['feature_importance'] = self.analyze_feature_importance()
        
        # 7. 關鍵建議
        results['recommendations'] = self.generate_recommendations(results)
        
        logger.info("="*80)
        logger.info("ANALYSIS COMPLETED")
        logger.info("="*80)
        
        return results
    
    def analyze_basic_stats(self) -> Dict:
        """
        基礎統計分析
        """
        if self.trades_df.empty:
            return {'total_trades': 0}
        
        total_trades = len(self.trades_df)
        winning = self.trades_df[self.trades_df['pnl_net'] > 0]
        losing = self.trades_df[self.trades_df['pnl_net'] <= 0]
        
        stats = {
            'total_trades': total_trades,
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': len(winning) / total_trades if total_trades > 0 else 0,
            'avg_win': winning['pnl_net'].mean() if len(winning) > 0 else 0,
            'avg_loss': losing['pnl_net'].mean() if len(losing) > 0 else 0,
            'best_trade': self.trades_df['pnl_net'].max(),
            'worst_trade': self.trades_df['pnl_net'].min(),
            'avg_probability': self.trades_df['probability'].mean(),
            'statistical_significance': 'YES' if total_trades >= 30 else 'NO'
        }
        
        logger.info(f"Basic Stats: {total_trades} trades, {stats['win_rate']*100:.2f}% win rate")
        
        return stats
    
    def analyze_hourly_performance(self) -> pd.DataFrame:
        """
        逐小時績效分解
        """
        if self.trades_df.empty:
            return pd.DataFrame()
        
        self.trades_df['entry_hour'] = pd.to_datetime(self.trades_df['entry_time']).dt.hour
        
        hourly = self.trades_df.groupby('entry_hour').agg({
            'pnl_net': ['count', 'sum', 'mean'],
            'exit_reason': lambda x: (x == 'TP').sum() / len(x) if len(x) > 0 else 0
        }).round(4)
        
        hourly.columns = ['trade_count', 'total_pnl', 'avg_pnl', 'tp_rate']
        hourly['profit_factor'] = hourly.apply(
            lambda row: self._calc_hourly_pf(row.name), axis=1
        )
        
        logger.info("Hourly Performance:")
        logger.info(f"Best hour: {hourly['total_pnl'].idxmax()} UTC (PnL: ${hourly['total_pnl'].max():.2f})")
        logger.info(f"Worst hour: {hourly['total_pnl'].idxmin()} UTC (PnL: ${hourly['total_pnl'].min():.2f})")
        
        return hourly
    
    def _calc_hourly_pf(self, hour: int) -> float:
        """
        計算特定小時的 Profit Factor
        """
        hour_trades = self.trades_df[self.trades_df['entry_hour'] == hour]
        wins = hour_trades[hour_trades['pnl_net'] > 0]['pnl_net'].sum()
        losses = abs(hour_trades[hour_trades['pnl_net'] <= 0]['pnl_net'].sum())
        return wins / losses if losses > 0 else (np.inf if wins > 0 else 1.0)
    
    def analyze_probability_layers(self) -> pd.DataFrame:
        """
        機率分層測試
        """
        if self.trades_df.empty:
            return pd.DataFrame()
        
        bins = [0, 0.15, 0.18, 0.22, 1.0]
        labels = ['<0.15', '0.15-0.18', '0.18-0.22', '0.22+']
        
        self.trades_df['prob_layer'] = pd.cut(
            self.trades_df['probability'],
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        
        layers = self.trades_df.groupby('prob_layer').agg({
            'pnl_net': ['count', 'sum', 'mean'],
            'exit_reason': lambda x: (x == 'TP').sum() / len(x) if len(x) > 0 else 0
        }).round(4)
        
        layers.columns = ['trade_count', 'total_pnl', 'avg_pnl', 'win_rate']
        
        logger.info("Probability Layer Analysis:")
        for layer in labels:
            if layer in layers.index:
                row = layers.loc[layer]
                logger.info(f"  {layer}: {row['trade_count']:.0f} trades, "
                           f"Win rate: {row['win_rate']*100:.1f}%, "
                           f"PnL: ${row['total_pnl']:.2f}")
        
        return layers
    
    def analyze_direction_comparison(self) -> Dict:
        """
        Long vs Short 對比分析
        """
        if self.trades_df.empty:
            return {}
        
        long_trades = self.trades_df[self.trades_df['direction'] == 'LONG']
        short_trades = self.trades_df[self.trades_df['direction'] == 'SHORT']
        
        def calc_metrics(df):
            if len(df) == 0:
                return {
                    'count': 0,
                    'win_rate': 0.0,
                    'total_pnl': 0.0,
                    'avg_pnl': 0.0,
                    'profit_factor': 0.0
                }
            
            wins = df[df['pnl_net'] > 0]
            losses = df[df['pnl_net'] <= 0]
            
            total_wins = wins['pnl_net'].sum() if len(wins) > 0 else 0
            total_losses = abs(losses['pnl_net'].sum()) if len(losses) > 0 else 0
            
            # FIXED: Handle case when no losses
            if total_losses == 0:
                pf = np.inf if total_wins > 0 else 0.0
            else:
                pf = total_wins / total_losses
            
            return {
                'count': len(df),
                'win_rate': len(wins) / len(df),
                'total_pnl': df['pnl_net'].sum(),
                'avg_pnl': df['pnl_net'].mean(),
                'profit_factor': pf
            }
        
        comparison = {
            'long': calc_metrics(long_trades),
            'short': calc_metrics(short_trades)
        }
        
        logger.info("Direction Comparison:")
        logger.info(f"  LONG:  {comparison['long']['count']} trades, "
                   f"WR: {comparison['long']['win_rate']*100:.1f}%, "
                   f"PF: {comparison['long']['profit_factor']:.2f}")
        logger.info(f"  SHORT: {comparison['short']['count']} trades, "
                   f"WR: {comparison['short']['win_rate']*100:.1f}%, "
                   f"PF: {comparison['short']['profit_factor']:.2f}")
        
        return comparison
    
    def analyze_failures(self) -> Dict:
        """
        失敗案例深度分析
        """
        if self.trades_df.empty:
            return {}
        
        losing_trades = self.trades_df[self.trades_df['pnl_net'] <= 0].copy()
        
        if len(losing_trades) == 0:
            return {'total_losses': 0}
        
        # 分析失敗原因
        losing_trades['entry_hour'] = pd.to_datetime(losing_trades['entry_time']).dt.hour
        
        failure_analysis = {
            'total_losses': len(losing_trades),
            'worst_hour': losing_trades.groupby('entry_hour')['pnl_net'].sum().idxmin(),
            'worst_direction': losing_trades.groupby('direction')['pnl_net'].sum().idxmin(),
            'avg_loss_prob': losing_trades['probability'].mean(),
            'quick_losses': len(losing_trades[
                (pd.to_datetime(losing_trades['exit_time']) - 
                 pd.to_datetime(losing_trades['entry_time'])).dt.total_seconds() < 300
            ])
        }
        
        logger.info("Failure Analysis:")
        logger.info(f"  Total losses: {failure_analysis['total_losses']}")
        logger.info(f"  Worst hour: {failure_analysis['worst_hour']} UTC")
        logger.info(f"  Worst direction: {failure_analysis['worst_direction']}")
        logger.info(f"  Quick losses (<5min): {failure_analysis['quick_losses']}")
        
        return failure_analysis
    
    def analyze_feature_importance(self) -> Dict:
        """
        特徵重要性分析
        """
        importance = {}
        
        try:
            # 從 CatBoost 模型提取特徵重要性
            if hasattr(self.model_long, 'estimators_'):
                base_long = self.model_long.estimators_[0].estimator
                base_short = self.model_short.estimators_[0].estimator
            else:
                base_long = self.model_long
                base_short = self.model_short
            
            long_importance = base_long.get_feature_importance()
            short_importance = base_short.get_feature_importance()
            long_features = base_long.feature_names_
            short_features = base_short.feature_names_
            
            importance['long'] = dict(zip(long_features, long_importance))
            importance['short'] = dict(zip(short_features, short_importance))
            
            # 排序並顯示 Top 5
            logger.info("Feature Importance (Top 5):")
            
            logger.info("  Long Oracle:")
            for feat, imp in sorted(importance['long'].items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"    {feat}: {imp:.2f}")
            
            logger.info("  Short Oracle:")
            for feat, imp in sorted(importance['short'].items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"    {feat}: {imp:.2f}")
        
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
        
        return importance
    
    def generate_recommendations(self, analysis_results: Dict) -> List[str]:
        """
        根據分析結果生成優化建議
        """
        recommendations = []
        
        basic = analysis_results.get('basic_stats', {})
        hourly = analysis_results.get('hourly_performance', pd.DataFrame())
        layers = analysis_results.get('probability_layers', pd.DataFrame())
        direction = analysis_results.get('direction_comparison', {})
        
        # 1. 統計顯著性檢查
        if basic.get('total_trades', 0) < 30:
            recommendations.append(
                "CRITICAL: 樣本數不足 (<30). 降低閾值到 0.12-0.14 以增加交易數量"
            )
        
        # 2. 時段優化
        if not hourly.empty:
            best_hours = hourly.nlargest(3, 'profit_factor').index.tolist()
            worst_hours = hourly.nsmallest(3, 'profit_factor').index.tolist()
            recommendations.append(
                f"專注最佳時段: {best_hours} UTC, 避開: {worst_hours} UTC"
            )
        
        # 3. 機率分層建議
        if not layers.empty and '0.18-0.22' in layers.index:
            sweet_spot = layers.loc['0.18-0.22']
            if sweet_spot['win_rate'] > 0.40:
                recommendations.append(
                    f"甜蜜點: 0.18-0.22 區間表現優異 (勝率 {sweet_spot['win_rate']*100:.1f}%). 考慮提高閾值到 0.18"
                )
        
        # 4. 方向偏差
        if direction:
            long_pf = direction.get('long', {}).get('profit_factor', 0)
            short_pf = direction.get('short', {}).get('profit_factor', 0)
            
            # Handle inf values
            if np.isinf(long_pf) or np.isinf(short_pf):
                if np.isinf(long_pf) and not np.isinf(short_pf):
                    recommendations.append(
                        "LONG 表現完美 (無虧損交易). SHORT 可能需要調整閾值或暫停"
                    )
                elif np.isinf(short_pf) and not np.isinf(long_pf):
                    recommendations.append(
                        "SHORT 表現完美 (無虧損交易). LONG 可能需要調整閾值或暫停"
                    )
            elif abs(long_pf - short_pf) > 0.5:
                better = 'LONG' if long_pf > short_pf else 'SHORT'
                recommendations.append(
                    f"方向偏差明顯: {better} 表現更好 (PF差距 {abs(long_pf - short_pf):.2f}). 考慮單向策略或調整閾值"
                )
        
        # 5. 勝率建議
        win_rate = basic.get('win_rate', 0)
        if win_rate < 0.33:
            recommendations.append(
                "勝率偏低 (<33%). 建議: 1) 提高閾值到 0.16+, 或 2) 壓縮 TP 到 1.5%"
            )
        elif win_rate > 0.40:
            recommendations.append(
                "勝率優異 (>40%). 可嘗試拉高 TP 到 2.0-2.5% 以提升盈虧比"
            )
        
        logger.info("="*80)
        logger.info("OPTIMIZATION RECOMMENDATIONS")
        logger.info("="*80)
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"{i}. {rec}")
        logger.info("="*80)
        
        return recommendations
    
    def generate_html_report(self, output_path: Path):
        """
        生成完整 HTML 報告
        """
        logger.info(f"Generating HTML report: {output_path}")
        
        # 執行完整分析
        results = self.analyze_all()
        
        # 創建圖表
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Hourly Performance', 'Probability Layers',
                'Long vs Short', 'Equity Curve',
                'Trade Distribution', 'Win Rate by Hour'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'histogram'}, {'type': 'bar'}]
            ]
        )
        
        # 1. 逐小時績效
        hourly = results['hourly_performance']
        if not hourly.empty:
            fig.add_trace(
                go.Bar(x=hourly.index, y=hourly['total_pnl'], name='Hourly PnL'),
                row=1, col=1
            )
        
        # 2. 機率分層
        layers = results['probability_layers']
        if not layers.empty:
            fig.add_trace(
                go.Bar(x=layers.index, y=layers['win_rate']*100, name='Win Rate %'),
                row=1, col=2
            )
        
        # 3. Long vs Short
        direction = results['direction_comparison']
        if direction:
            long_pf = direction['long']['profit_factor']
            short_pf = direction['short']['profit_factor']
            
            # Cap inf values for display
            if np.isinf(long_pf):
                long_pf = 10.0
            if np.isinf(short_pf):
                short_pf = 10.0
            
            fig.add_trace(
                go.Bar(
                    x=['Long', 'Short'],
                    y=[long_pf, short_pf],
                    name='Profit Factor'
                ),
                row=2, col=1
            )
        
        # 4. 資金曲線
        if not self.equity_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=self.equity_df['timestamp'],
                    y=self.equity_df['capital'],
                    mode='lines',
                    name='Equity'
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=1200, showlegend=False, title_text="Backtest Diagnostic Report")
        fig.write_html(output_path)
        
        logger.info(f"HTML report saved: {output_path}")