import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from typing import Dict, List, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger('backtester', 'logs/backtester.log')

class EventDrivenBacktester:
    """事件驅動回測引擎 - 機構級別驗證框架"""
    
    def __init__(self, initial_capital: float = 10000.0, 
                 risk_reward_ratio: float = 2.0,
                 stop_loss_pct: float = 0.01,
                 fee_rate: float = 0.0004,
                 slippage_pct: float = 0.0001):
        """
        Args:
            initial_capital: 初始資金
            risk_reward_ratio: 盈虧比 (TP/SL)
            stop_loss_pct: 停損百分比 (0.01 = 1%)
            fee_rate: 單邊手續費 (0.04%)
            slippage_pct: 滑價百分比 (0.01%)
        """
        logger.info("Initializing EventDrivenBacktester")
        
        self.initial_capital = initial_capital
        self.risk_reward_ratio = risk_reward_ratio
        self.stop_loss_pct = stop_loss_pct
        self.fee_rate = fee_rate
        self.slippage_pct = slippage_pct
        
        # 總摩擦成本 (Taker 手續費 + 滑價)
        self.total_friction = (fee_rate + slippage_pct) * 2  # 一進一出
        
        logger.info(f"Initial capital: ${initial_capital}")
        logger.info(f"Risk/Reward: 1:{risk_reward_ratio}")
        logger.info(f"Stop Loss: {stop_loss_pct*100}%")
        logger.info(f"Total friction per trade: {self.total_friction*100:.4f}%")
        
        # 狀態追蹤
        self.reset_state()
    
    def reset_state(self):
        """重置回測狀態"""
        self.capital = self.initial_capital
        self.position = None  # None = 空手, dict = 持有多單
        self.trades = []  # 所有交易記錄
        self.equity_curve = []  # 資金曲線
        self.current_time = None
    
    def calculate_position_size(self) -> float:
        """計算每次開單的位置大小 (固定 10% 資金)"""
        return self.capital * 0.1
    
    def generate_signal(self, features: pd.Series, model, threshold: float) -> bool:
        """模組 2: 推論引擎 - 生成交易訊號"""
        try:
            # 提取特徵
            feature_cols = [
                '1m_bull_sweep', '1m_bear_sweep', '1m_bull_bos', '1m_bear_bos', '1m_dist_to_poc',
                '15m_z_score', '15m_bb_width_pct', '1h_z_score', '1d_atr_pct',
                'hour', 'day_of_week', 'session_asia', 'session_europe', 'session_us',
                'session_overlap', 'is_weekend', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                'rvol', 'sweep_with_high_rvol', 'volatility_squeeze_ratio',
                'is_bullish_divergence', 'is_macd_bullish_divergence'
            ]
            
            available_features = [col for col in feature_cols if col in features.index]
            X = features[available_features].values.reshape(1, -1)
            
            # 預測
            proba = model.predict(X)[0]
            
            # 訊號生成
            return proba >= threshold
        
        except Exception as e:
            logger.error(f"Signal generation error: {str(e)}")
            return False
    
    def open_position(self, entry_price: float, timestamp: pd.Timestamp):
        """模組 3: 狀態機 - 開倉"""
        position_size = self.calculate_position_size()
        
        # 計算停損停利
        sl_price = entry_price * (1 - self.stop_loss_pct)
        tp_price = entry_price * (1 + self.stop_loss_pct * self.risk_reward_ratio)
        
        self.position = {
            'entry_time': timestamp,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'position_size': position_size,
            'quantity': position_size / entry_price
        }
        
        logger.info(f"[OPEN] Time={timestamp}, Entry={entry_price:.2f}, SL={sl_price:.2f}, TP={tp_price:.2f}")
    
    def check_exit(self, high: float, low: float, timestamp: pd.Timestamp) -> Tuple[bool, str, float]:
        """模組 3: 狀態機 - 檢查出場條件"""
        if self.position is None:
            return False, None, None
        
        sl_price = self.position['sl_price']
        tp_price = self.position['tp_price']
        
        # 防線二: 悉觀單根 K 線假設
        # 如果同時觸及 SL 與 TP，一律視為先觸及 SL
        sl_hit = low <= sl_price
        tp_hit = high >= tp_price
        
        if sl_hit and tp_hit:
            # 同時觸及，悉觀假設
            logger.warning(f"[PESSIMISTIC] Both SL and TP hit in same bar at {timestamp}, assuming SL first")
            return True, 'SL', sl_price
        elif sl_hit:
            return True, 'SL', sl_price
        elif tp_hit:
            return True, 'TP', tp_price
        
        return False, None, None
    
    def close_position(self, exit_price: float, exit_reason: str, timestamp: pd.Timestamp):
        """模組 3: 狀態機 - 平倉"""
        entry_price = self.position['entry_price']
        position_size = self.position['position_size']
        quantity = self.position['quantity']
        
        # 計算原始報酬
        gross_return = (exit_price - entry_price) / entry_price
        
        # 扣除摩擦成本 (防線三)
        net_return = gross_return - self.total_friction
        
        # 實際損益
        pnl = position_size * net_return
        self.capital += pnl
        
        # 記錄交易
        trade = {
            'entry_time': self.position['entry_time'],
            'exit_time': timestamp,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'position_size': position_size,
            'gross_return': gross_return,
            'net_return': net_return,
            'pnl': pnl,
            'capital': self.capital
        }
        self.trades.append(trade)
        
        logger.info(f"[CLOSE] Time={timestamp}, Exit={exit_price:.2f}, Reason={exit_reason}, PnL=${pnl:.2f}, Capital=${self.capital:.2f}")
        
        # 清空持倉
        self.position = None
    
    def run_backtest(self, test_df: pd.DataFrame, model_path: str, threshold: float = 0.5) -> Dict:
        """模組 1: 資料餼送器 - 運行完整回測"""
        logger.info("="*80)
        logger.info("Starting event-driven backtest")
        logger.info(f"Test set size: {len(test_df)} bars")
        logger.info(f"Model: {model_path}")
        logger.info(f"Threshold: {threshold}")
        logger.info("="*80)
        
        # 載入模型
        try:
            model = lgb.Booster(model_file=model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return None
        
        # 重置狀態
        self.reset_state()
        
        # 確保時間排序
        test_df = test_df.sort_values('open_time').reset_index(drop=True)
        
        # 逐根 K 線處理
        for i in range(len(test_df) - 1):  # -1 因為需要下一根 K 線的 Open
            current_bar = test_df.iloc[i]
            next_bar = test_df.iloc[i + 1]
            
            self.current_time = current_bar['open_time']
            
            # 如果持有多單，檢查出場
            if self.position is not None:
                should_exit, exit_reason, exit_price = self.check_exit(
                    current_bar['high'],
                    current_bar['low'],
                    self.current_time
                )
                
                if should_exit:
                    self.close_position(exit_price, exit_reason, self.current_time)
            
            # 如果空手，檢查進場訊號
            if self.position is None:
                signal = self.generate_signal(current_bar, model, threshold)
                
                if signal:
                    # 防線一: T+1 執行 (使用下一根 K 線的開盤價)
                    entry_price = next_bar['open']
                    self.open_position(entry_price, next_bar['open_time'])
            
            # 記錄資金曲線
            current_equity = self.capital
            if self.position is not None:
                # 未實現損益
                unrealized_pnl = self.position['quantity'] * (current_bar['close'] - self.position['entry_price'])
                current_equity += unrealized_pnl
            
            self.equity_curve.append({
                'time': self.current_time,
                'equity': current_equity
            })
        
        # 強制平倉最後一單
        if self.position is not None:
            last_bar = test_df.iloc[-1]
            self.close_position(last_bar['close'], 'FORCE_CLOSE', last_bar['open_time'])
        
        # 模組 4: 績效結算器
        results = self.calculate_performance()
        
        logger.info("="*80)
        logger.info("Backtest completed")
        logger.info(f"Total trades: {results['total_trades']}")
        logger.info(f"Final capital: ${results['final_capital']:.2f}")
        logger.info(f"Total return: {results['total_return']*100:.2f}%")
        logger.info(f"Win rate: {results['win_rate']*100:.2f}%")
        logger.info(f"Max drawdown: {results['max_drawdown']*100:.2f}%")
        logger.info("="*80)
        
        return results
    
    def calculate_performance(self) -> Dict:
        """模組 4: 績效結算器 - 計算所有關鍵指標"""
        logger.info("Calculating performance metrics")
        
        if not self.trades:
            logger.warning("No trades executed")
            return {
                'total_trades': 0,
                'final_capital': self.capital,
                'total_return': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'trades_df': pd.DataFrame(),
                'equity_df': pd.DataFrame()
            }
        
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        # 基本指標
        total_trades = len(self.trades)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        # 報酬指標
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        # 最大回撤 (Max Drawdown)
        equity_series = equity_df['equity']
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = abs(drawdown.min())
        
        # 夏普比率 (Sharpe Ratio)
        returns = trades_df['net_return']
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # 年化
        else:
            sharpe_ratio = 0
        
        # 盈虧比 (Profit Factor)
        total_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        total_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        results = {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_return': total_return,
            'final_capital': self.capital,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'trades_df': trades_df,
            'equity_df': equity_df
        }
        
        return results