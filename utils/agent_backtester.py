import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger
from utils.execution_agent import ExecutionAgent

logger = setup_logger('agent_backtester', 'logs/agent_backtester.log')

class AgentBacktester:
    """
    Agent 專用回測引擎
    
    核心特性:
    1. 1m K 線運作，15m 整點同步大腦
    2. 悲觀成交規則: low < limit_price 才成交
    3. 非對稱手續費: Maker/Taker 分離計算
    4. 幽靈部位防護: 新訊號自動取消舊單
    """
    
    def __init__(self, agent: ExecutionAgent, model_path: str):
        """
        Args:
            agent: ExecutionAgent 實例
            model_path: 模型路徑 (.pkl 檔案)
        """
        logger.info("Initializing AgentBacktester")
        
        self.agent = agent
        
        # 載入模型
        logger.info(f"Loading model from {model_path}")
        self.model = joblib.load(model_path)
        
        # 記錄
        self.actions = []
        self.probabilities = []
        
        logger.info("AgentBacktester initialized")
    
    def run(self, df_1m: pd.DataFrame, df_15m: pd.DataFrame) -> dict:
        """
        執行 Agent 回測
        
        Args:
            df_1m: 1m K 線資料 (open_time, open, high, low, close, volume)
            df_15m: 15m 特徵資料 (包含所有特徵)
        
        Returns:
            results: 回測結果
        """
        logger.info("="*80)
        logger.info("AGENT BACKTESTING")
        logger.info("="*80)
        logger.info(f"1m bars: {len(df_1m)}")
        logger.info(f"15m bars: {len(df_15m)}")
        
        # 確保索引是 DatetimeIndex
        if not isinstance(df_1m.index, pd.DatetimeIndex):
            df_1m = df_1m.set_index('open_time')
        
        if not isinstance(df_15m.index, pd.DatetimeIndex):
            df_15m = df_15m.set_index('open_time')
        
        # 預測 15m 機率
        logger.info("Generating 15m probabilities...")
        probabilities_15m = self._generate_probabilities(df_15m)
        
        # 遍歷每根 1m K 線
        logger.info("Processing 1m bars...")
        
        for timestamp, row in df_1m.iterrows():
            # 準備 1m bar 資料
            bar_1m = {
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close'],
                'volume': row['volume']
            }
            
            # 檢查是否是 15m 整點 (xx:00, xx:15, xx:30, xx:45)
            minute = timestamp.minute
            is_15m_point = (minute % 15 == 0)
            
            # 在 15m 整點時，提供大腦機率
            if is_15m_point:
                # 對齊到 15m 時間
                aligned_15m_time = timestamp.floor('15min')
                
                # 幽靈部位防護: 新訊號前取消所有未成交訂單
                self.agent.cancel_all_pending_orders()
                
                # 獲取機率
                if aligned_15m_time in probabilities_15m.index:
                    probability = probabilities_15m.loc[aligned_15m_time]
                    self.probabilities.append({
                        'timestamp': timestamp,
                        'probability': probability
                    })
                else:
                    probability = None
            else:
                probability = None
            
            # Agent 處理當前 bar
            action = self.agent.process_bar(timestamp, bar_1m, probability)
            self.actions.append(action)
            
            # 每 1000 根輸出進度
            if len(self.actions) % 1000 == 0:
                logger.info(f"  Processed {len(self.actions)} bars, Capital: ${self.agent.capital:.2f}")
        
        # 結束回測
        logger.info("Backtesting completed")
        
        # 強制平倉未結清部位
        if self.agent.position is not None:
            logger.warning("Closing open position at end of backtest")
            final_bar = df_1m.iloc[-1]
            final_action = self.agent._close_position(
                timestamp=df_1m.index[-1],
                exit_price=final_bar['close'],
                exit_reason='backtest_end',
                exit_type='taker'
            )
            self.actions.append(final_action)
        
        # 獲取統計
        stats = self.agent.get_statistics()
        
        # 生成報告
        results = self._generate_report(stats)
        
        return results
    
    def _generate_probabilities(self, df_15m: pd.DataFrame) -> pd.Series:
        """
        生成 15m 機率
        
        Args:
            df_15m: 15m 特徵資料
        
        Returns:
            probabilities: 機率序列
        """
        # 獲取特徵欄位
        feature_cols = [
            '1m_bull_sweep', '1m_bear_sweep', '1m_bull_bos', '1m_bear_bos', '1m_dist_to_poc',
            '15m_z_score', '15m_bb_width_pct',
            '1h_z_score', '1d_atr_pct',
            'hour', 'day_of_week', 'session_asia', 'session_europe', 'session_us',
            'session_overlap', 'is_weekend', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'rvol', 'sweep_with_high_rvol',
            'volatility_squeeze_ratio',
            'is_bullish_divergence', 'is_macd_bullish_divergence'
        ]
        
        available_features = [col for col in feature_cols if col in df_15m.columns]
        
        X = df_15m[available_features].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 預測機率
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return pd.Series(probabilities, index=df_15m.index)
    
    def _generate_report(self, stats: dict) -> dict:
        """生成報告"""
        logger.info("="*80)
        logger.info("AGENT BACKTEST RESULTS")
        logger.info("="*80)
        logger.info(f"Total Trades: {stats['total_trades']}")
        logger.info(f"Winning Trades: {stats['winning_trades']}")
        logger.info(f"Losing Trades: {stats['losing_trades']}")
        logger.info(f"Win Rate: {stats['win_rate']*100:.2f}%")
        logger.info(f"Total PnL: ${stats['total_pnl']:.2f}")
        logger.info(f"Avg PnL: ${stats['avg_pnl']:.2f}")
        logger.info(f"Final Capital: ${stats['final_capital']:.2f}")
        logger.info(f"Return: {stats['return_pct']*100:.2f}%")
        logger.info("="*80)
        
        # 按出場原因分類
        if stats['total_trades'] > 0:
            df_trades = stats['trades_df']
            
            logger.info("\nExit Reason Breakdown:")
            exit_counts = df_trades['exit_reason'].value_counts()
            for reason, count in exit_counts.items():
                pct = count / len(df_trades) * 100
                avg_pnl = df_trades[df_trades['exit_reason'] == reason]['pnl'].mean()
                logger.info(f"  {reason}: {count} ({pct:.1f}%), Avg PnL: ${avg_pnl:.2f}")
            
            logger.info("\nExit Type Breakdown:")
            type_counts = df_trades['exit_type'].value_counts()
            for exit_type, count in type_counts.items():
                pct = count / len(df_trades) * 100
                logger.info(f"  {exit_type}: {count} ({pct:.1f}%)")
        
        return {
            'statistics': stats,
            'actions': self.actions,
            'probabilities': self.probabilities
        }
    
    def save_results(self, output_path: str):
        """保存結果"""
        stats = self.agent.get_statistics()
        
        if stats['total_trades'] > 0:
            df_trades = stats['trades_df']
            df_trades.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        else:
            logger.warning("No trades to save")