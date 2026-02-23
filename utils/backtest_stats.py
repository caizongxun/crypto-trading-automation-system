import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Tuple
from datetime import timedelta

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger('backtest_stats', 'logs/backtest_stats.log')

class BacktestStatistics:
    """
    回測統計分析工具
    
    **核心功能**:
    - 計算回測天數與時段分布
    - 日均交易頻率分析
    - 報酬率組成分析
    - 年化指標計算
    - 提供優化建議
    """
    
    def __init__(self, trades_df: pd.DataFrame, equity_df: pd.DataFrame, 
                 initial_capital: float):
        self.trades_df = trades_df
        self.equity_df = equity_df
        self.initial_capital = initial_capital
        
        if not trades_df.empty:
            self.trades_df['entry_time'] = pd.to_datetime(self.trades_df['entry_time'])
            self.trades_df['exit_time'] = pd.to_datetime(self.trades_df['exit_time'])
    
    def calculate_all_stats(self) -> Dict:
        """
        計算完整統計數據
        """
        logger.info("="*80)
        logger.info("COMPREHENSIVE BACKTEST STATISTICS")
        logger.info("="*80)
        
        stats = {}
        
        # 1. 時間統計
        stats['time_analysis'] = self.analyze_time_period()
        
        # 2. 交易頻率
        stats['frequency_analysis'] = self.analyze_trade_frequency()
        
        # 3. 報酬分析
        stats['return_analysis'] = self.analyze_returns()
        
        # 4. 年化指標
        stats['annualized_metrics'] = self.calculate_annualized_metrics()
        
        # 5. 交易質量
        stats['quality_analysis'] = self.analyze_trade_quality()
        
        # 6. 優化建議
        stats['recommendations'] = self.generate_recommendations(stats)
        
        self.print_summary(stats)
        
        return stats
    
    def analyze_time_period(self) -> Dict:
        """
        分析回測時間範圍
        """
        if self.trades_df.empty:
            return {}
        
        start_time = self.trades_df['entry_time'].min()
        end_time = self.trades_df['exit_time'].max()
        
        total_days = (end_time - start_time).days + 1
        total_hours = (end_time - start_time).total_seconds() / 3600
        total_weeks = total_days / 7
        total_months = total_days / 30
        
        # 計算實際有交易的天數
        trading_days = self.trades_df['entry_time'].dt.date.nunique()
        
        # 計算時段分布
        self.trades_df['entry_hour'] = self.trades_df['entry_time'].dt.hour
        hour_distribution = self.trades_df['entry_hour'].value_counts().sort_index()
        
        analysis = {
            'start_date': start_time,
            'end_date': end_time,
            'total_days': total_days,
            'total_hours': total_hours,
            'total_weeks': total_weeks,
            'total_months': total_months,
            'trading_days': trading_days,
            'trading_days_pct': trading_days / total_days * 100,
            'hour_distribution': hour_distribution.to_dict()
        }
        
        logger.info("Time Period Analysis:")
        logger.info(f"  Period: {start_time.date()} to {end_time.date()}")
        logger.info(f"  Total Days: {total_days} days ({total_weeks:.1f} weeks, {total_months:.1f} months)")
        logger.info(f"  Trading Days: {trading_days} days ({trading_days/total_days*100:.1f}%)")
        logger.info(f"  Most Active Hours: {hour_distribution.nlargest(3).index.tolist()} UTC")
        
        return analysis
    
    def analyze_trade_frequency(self) -> Dict:
        """
        分析交易頻率
        """
        if self.trades_df.empty:
            return {}
        
        time_analysis = self.analyze_time_period()
        total_trades = len(self.trades_df)
        
        # 日均交易數
        trades_per_day = total_trades / time_analysis['total_days']
        trades_per_week = total_trades / time_analysis['total_weeks']
        trades_per_month = total_trades / time_analysis['total_months']
        
        # 交易時長統計
        self.trades_df['duration_minutes'] = (
            (self.trades_df['exit_time'] - self.trades_df['entry_time']).dt.total_seconds() / 60
        )
        avg_duration = self.trades_df['duration_minutes'].mean()
        median_duration = self.trades_df['duration_minutes'].median()
        
        # 日內分布
        daily_trades = self.trades_df.groupby(
            self.trades_df['entry_time'].dt.date
        ).size()
        
        analysis = {
            'total_trades': total_trades,
            'trades_per_day': trades_per_day,
            'trades_per_week': trades_per_week,
            'trades_per_month': trades_per_month,
            'avg_duration_minutes': avg_duration,
            'median_duration_minutes': median_duration,
            'max_daily_trades': daily_trades.max(),
            'min_daily_trades': daily_trades[daily_trades > 0].min(),
            'days_with_trades': len(daily_trades)
        }
        
        logger.info("Trade Frequency Analysis:")
        logger.info(f"  Total Trades: {total_trades}")
        logger.info(f"  Frequency: {trades_per_day:.2f}/day, {trades_per_week:.1f}/week, {trades_per_month:.1f}/month")
        logger.info(f"  Avg Duration: {avg_duration:.1f} minutes ({avg_duration/60:.1f} hours)")
        logger.info(f"  Daily Range: {daily_trades[daily_trades > 0].min():.0f} - {daily_trades.max():.0f} trades")
        
        return analysis
    
    def analyze_returns(self) -> Dict:
        """
        分析報酬組成
        """
        if self.trades_df.empty:
            return {}
        
        total_pnl = self.trades_df['pnl_net'].sum()
        total_fees = self.trades_df['fees'].sum()
        gross_pnl = total_pnl + total_fees
        
        winning_trades = self.trades_df[self.trades_df['pnl_net'] > 0]
        losing_trades = self.trades_df[self.trades_df['pnl_net'] <= 0]
        
        total_wins = winning_trades['pnl_net'].sum()
        total_losses = abs(losing_trades['pnl_net'].sum())
        
        # 報酬率計算
        total_return_pct = (total_pnl / self.initial_capital) * 100
        
        # 平均報酬
        avg_trade_return = total_pnl / len(self.trades_df)
        avg_trade_return_pct = (avg_trade_return / self.initial_capital) * 100
        
        analysis = {
            'gross_pnl': gross_pnl,
            'total_fees': total_fees,
            'net_pnl': total_pnl,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'total_return_pct': total_return_pct,
            'avg_trade_return': avg_trade_return,
            'avg_trade_return_pct': avg_trade_return_pct,
            'fees_impact_pct': (total_fees / gross_pnl * 100) if gross_pnl != 0 else 0,
            'win_amount_pct': (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) != 0 else 0
        }
        
        logger.info("Return Analysis:")
        logger.info(f"  Gross PnL: ${gross_pnl:+.2f}")
        logger.info(f"  Total Fees: ${total_fees:.2f} ({analysis['fees_impact_pct']:.1f}% of gross)")
        logger.info(f"  Net PnL: ${total_pnl:+.2f} ({total_return_pct:+.2f}%)")
        logger.info(f"  Avg Per Trade: ${avg_trade_return:+.2f} ({avg_trade_return_pct:+.3f}%)")
        logger.info(f"  Win/Loss Amount: ${total_wins:.2f} / ${total_losses:.2f}")
        
        return analysis
    
    def calculate_annualized_metrics(self) -> Dict:
        """
        計算年化指標
        """
        if self.trades_df.empty:
            return {}
        
        time_analysis = self.analyze_time_period()
        return_analysis = self.analyze_returns()
        
        years = time_analysis['total_days'] / 365.25
        
        # 年化報酬
        annualized_return = (return_analysis['total_return_pct'] / years) if years > 0 else 0
        
        # 年化交易數
        annualized_trades = len(self.trades_df) / years if years > 0 else 0
        
        # 月化報酬
        months = time_analysis['total_months']
        monthly_return = return_analysis['total_return_pct'] / months if months > 0 else 0
        
        analysis = {
            'years': years,
            'annualized_return_pct': annualized_return,
            'annualized_trades': annualized_trades,
            'monthly_return_pct': monthly_return,
            'monthly_trades': len(self.trades_df) / months if months > 0 else 0
        }
        
        logger.info("Annualized Metrics:")
        logger.info(f"  Test Period: {years:.2f} years ({months:.1f} months)")
        logger.info(f"  Annualized Return: {annualized_return:+.2f}% per year")
        logger.info(f"  Monthly Return: {monthly_return:+.2f}% per month")
        logger.info(f"  Annualized Trades: {annualized_trades:.0f} trades/year")
        
        return analysis
    
    def analyze_trade_quality(self) -> Dict:
        """
        分析交易質量
        """
        if self.trades_df.empty:
            return {}
        
        # 按方向分析
        long_trades = self.trades_df[self.trades_df['direction'] == 'LONG']
        short_trades = self.trades_df[self.trades_df['direction'] == 'SHORT']
        
        # 按結果分析
        tp_trades = self.trades_df[self.trades_df['exit_reason'] == 'TP']
        sl_trades = self.trades_df[self.trades_df['exit_reason'] == 'SL']
        
        # 按機率區間分析
        high_prob = self.trades_df[self.trades_df['probability'] >= 0.20]
        mid_prob = self.trades_df[
            (self.trades_df['probability'] >= 0.15) & 
            (self.trades_df['probability'] < 0.20)
        ]
        low_prob = self.trades_df[self.trades_df['probability'] < 0.15]
        
        analysis = {
            'long_count': len(long_trades),
            'short_count': len(short_trades),
            'tp_count': len(tp_trades),
            'sl_count': len(sl_trades),
            'tp_rate': len(tp_trades) / len(self.trades_df) * 100,
            'high_prob_count': len(high_prob),
            'mid_prob_count': len(mid_prob),
            'low_prob_count': len(low_prob),
            'high_prob_win_rate': (high_prob['pnl_net'] > 0).sum() / len(high_prob) * 100 if len(high_prob) > 0 else 0,
            'mid_prob_win_rate': (mid_prob['pnl_net'] > 0).sum() / len(mid_prob) * 100 if len(mid_prob) > 0 else 0,
            'low_prob_win_rate': (low_prob['pnl_net'] > 0).sum() / len(low_prob) * 100 if len(low_prob) > 0 else 0
        }
        
        logger.info("Trade Quality Analysis:")
        logger.info(f"  Direction: Long {analysis['long_count']}, Short {analysis['short_count']}")
        logger.info(f"  Exit: TP {analysis['tp_count']} ({analysis['tp_rate']:.1f}%), SL {analysis['sl_count']}")
        logger.info(f"  Probability Distribution:")
        logger.info(f"    High (>=0.20): {analysis['high_prob_count']} trades, WR {analysis['high_prob_win_rate']:.1f}%")
        logger.info(f"    Mid (0.15-0.20): {analysis['mid_prob_count']} trades, WR {analysis['mid_prob_win_rate']:.1f}%")
        logger.info(f"    Low (<0.15): {analysis['low_prob_count']} trades, WR {analysis['low_prob_win_rate']:.1f}%")
        
        return analysis
    
    def generate_recommendations(self, stats: Dict) -> list:
        """
        生成優化建議
        """
        recommendations = []
        
        freq = stats.get('frequency_analysis', {})
        ret = stats.get('return_analysis', {})
        ann = stats.get('annualized_metrics', {})
        qual = stats.get('quality_analysis', {})
        
        # 1. 交易頻率建議
        if freq.get('trades_per_day', 0) < 1:
            recommendations.append(
                f"⚠️ 交易頻率偏低 ({freq['trades_per_day']:.2f}/天). "
                f"建議: 降低閾值到 0.14-0.15 或啟用 24/7 交易以增加機會"
            )
        
        # 2. 報酬率建議
        if ret.get('total_return_pct', 0) < 1:
            recommendations.append(
                f"⚠️ 總報酬偏低 ({ret['total_return_pct']:.2f}%). "
                f"建議: 提高閾值到 0.18 以提升交易品質,或增加倉位到 15%"
            )
        
        # 3. 年化報酬建議
        if ann.get('annualized_return_pct', 0) < 10:
            recommendations.append(
                f"💡 年化報酬 {ann['annualized_return_pct']:.1f}% 有提升空間. "
                f"目標: 通過提升勝率或增加交易頻率達到 15-20%"
            )
        
        # 4. 交易質量建議
        if qual.get('tp_rate', 0) < 35:
            recommendations.append(
                f"⚠️ 止盈率偏低 ({qual['tp_rate']:.1f}%). "
                f"建議: 壓縮 TP 到 1.5% 或啟用波動率自適應"
            )
        
        # 5. 機率分布建議
        high_prob_wr = qual.get('high_prob_win_rate', 0)
        if high_prob_wr > 50:
            recommendations.append(
                f"🎯 高機率交易表現優異 (勝率 {high_prob_wr:.1f}%). "
                f"建議: 提高閾值到 0.18-0.20 專注高品質交易"
            )
        
        # 6. 手續費建議
        fees_impact = ret.get('fees_impact_pct', 0)
        if fees_impact > 30:
            recommendations.append(
                f"⚠️ 手續費占比過高 ({fees_impact:.1f}%). "
                f"建議: 延長持倉時間或減少交易頻率"
            )
        
        return recommendations
    
    def print_summary(self, stats: Dict):
        """
        打印統計摘要
        """
        logger.info("="*80)
        logger.info("OPTIMIZATION RECOMMENDATIONS")
        logger.info("="*80)
        
        recommendations = stats.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"{i}. {rec}")
        else:
            logger.info("✅ 策略表現良好,暫無明顯優化建議")
        
        logger.info("="*80)
    
    def generate_detailed_report(self) -> str:
        """
        生成詳細文字報告
        """
        stats = self.calculate_all_stats()
        
        time = stats['time_analysis']
        freq = stats['frequency_analysis']
        ret = stats['return_analysis']
        ann = stats['annualized_metrics']
        qual = stats['quality_analysis']
        recs = stats['recommendations']
        
        report = f"""
# 回測統計詳細報告

## 時間統計
- 回測期間: {time['start_date'].date()} 至 {time['end_date'].date()}
- 總天數: {time['total_days']} 天 ({time['total_weeks']:.1f} 週, {time['total_months']:.1f} 個月)
- 交易天數: {time['trading_days']} 天 ({time['trading_days_pct']:.1f}%)

## 交易頻率
- 總交易數: {freq['total_trades']} 筆
- 日均交易: {freq['trades_per_day']:.2f} 筆/天
- 週均交易: {freq['trades_per_week']:.1f} 筆/週
- 月均交易: {freq['trades_per_month']:.1f} 筆/月
- 平均持倉時長: {freq['avg_duration_minutes']:.1f} 分鐘 ({freq['avg_duration_minutes']/60:.1f} 小時)

## 報酬分析
- 毛利潤: ${ret['gross_pnl']:+.2f}
- 總手續費: ${ret['total_fees']:.2f} ({ret['fees_impact_pct']:.1f}% of gross)
- 淨利潤: ${ret['net_pnl']:+.2f}
- 總報酬率: {ret['total_return_pct']:+.2f}%
- 每筆平均: ${ret['avg_trade_return']:+.2f} ({ret['avg_trade_return_pct']:+.3f}%)

## 年化指標
- 測試時長: {ann['years']:.2f} 年
- 年化報酬: {ann['annualized_return_pct']:+.2f}%
- 月化報酬: {ann['monthly_return_pct']:+.2f}%
- 年化交易數: {ann['annualized_trades']:.0f} 筆/年

## 交易品質
- 方向分布: Long {qual['long_count']}, Short {qual['short_count']}
- 止盈率: {qual['tp_rate']:.1f}% ({qual['tp_count']} 筆)
- 機率分布:
  - 高機率 (>=0.20): {qual['high_prob_count']} 筆, 勝率 {qual['high_prob_win_rate']:.1f}%
  - 中機率 (0.15-0.20): {qual['mid_prob_count']} 筆, 勝率 {qual['mid_prob_win_rate']:.1f}%
  - 低機率 (<0.15): {qual['low_prob_count']} 筆, 勝率 {qual['low_prob_win_rate']:.1f}%

## 優化建議
"""
        
        for i, rec in enumerate(recs, 1):
            report += f"{i}. {rec}\n"
        
        return report