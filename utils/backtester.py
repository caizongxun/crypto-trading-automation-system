import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import joblib
from typing import Dict, List, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger('backtester', 'logs/backtester.log')

class EventDrivenBacktester:
    """
    事件驅動回測引擎 - 三層物理濾網優化
    
    層級一: 拉高機率閉值 (0.65+) - 只做高品質訊號
    層級二: 黃金時段濾網 (9-13, 18-21 UTC) - 迴避洗盤時段
    層級三: Maker 限價進場 (0.01%) - 減少摩擦成本
    """
    
    def __init__(self, initial_capital: float = 10000.0, 
                 risk_reward_ratio: float = 2.0,
                 stop_loss_pct: float = 0.01,
                 maker_fee: float = 0.0001,
                 taker_fee: float = 0.0004,
                 slippage_pct: float = 0.0002,
                 use_time_filter: bool = True,
                 probability_threshold: float = 0.65):
        """
        Args:
            initial_capital: 初始資金
            risk_reward_ratio: 盈虧比 (TP/SL)
            stop_loss_pct: 停損百分比 (0.01 = 1%)
            maker_fee: Maker 手續費 (0.01%)
            taker_fee: Taker 手續費 (0.04%)
            slippage_pct: 滑價百分比 (0.02%)
            use_time_filter: 是否啟用時間濾網
            probability_threshold: 機率閉值 (0.65 = 65%)
        """
        logger.info("="*80)
        logger.info("Initializing EventDrivenBacktester with 3-layer filters")
        logger.info("="*80)
        
        self.initial_capital = initial_capital
        self.risk_reward_ratio = risk_reward_ratio
        self.stop_loss_pct = stop_loss_pct
        self.slippage_pct = slippage_pct
        self.use_time_filter = use_time_filter
        self.probability_threshold = probability_threshold
        
        # 非對稱手續費
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        
        # 黃金時段 (基於回測數據分析)
        self.golden_hours = [9, 10, 11, 12, 13, 18, 19, 20, 21]
        
        logger.info(f"Initial capital: ${initial_capital:,.2f}")
        logger.info(f"Risk/Reward: 1:{risk_reward_ratio}")
        logger.info(f"Stop Loss: {stop_loss_pct*100:.2f}%")
        logger.info("")
        logger.info("✅ Layer 1: Probability Threshold")
        logger.info(f"  - Threshold: {probability_threshold:.2f} ({probability_threshold*100:.0f}%)")
        logger.info(f"  - Effect: Filter low-quality signals (51%-64%)")
        logger.info("")
        logger.info("✅ Layer 2: Golden Hour Filter")
        logger.info(f"  - Status: {'ENABLED' if use_time_filter else 'DISABLED'}")
        if use_time_filter:
            logger.info(f"  - Allowed hours (UTC): {self.golden_hours}")
            logger.info(f"  - Effect: Avoid wash trading periods")
        logger.info("")
        logger.info("✅ Layer 3: Asymmetric Fees (Maker Entry)")
        logger.info(f"  - Entry (Maker): {maker_fee*100:.2f}%")
        logger.info(f"  - Exit SL (Taker): {taker_fee*100:.2f}% + Slippage {slippage_pct*100:.2f}% = {(taker_fee+slippage_pct)*100:.2f}%")
        logger.info(f"  - Exit TP (Maker): {maker_fee*100:.2f}%")
        logger.info(f"  - Effect: Reduce friction by 75% on entry")
        logger.info("="*80)
        
        self.reset_state()
        self.filtered_signals_prob = 0  # 被機率濾掉的訊號
        self.filtered_signals_time = 0  # 被時間濾掉的訊號
    
    def reset_state(self):
        """重置回測狀態"""
        self.capital = self.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []
        self.current_time = None
        self.filtered_signals_prob = 0
        self.filtered_signals_time = 0
    
    def calculate_position_size(self) -> float:
        """計算每次開單的位置大小"""
        return self.capital * 0.1
    
    def is_golden_hour(self, timestamp: pd.Timestamp) -> bool:
        """檢查是否在黃金時段"""
        if not self.use_time_filter:
            return True
        
        hour = timestamp.hour
        return hour in self.golden_hours
    
    def generate_signal(self, features: pd.Series, model, proba_only: bool = False):
        """
        模組 2: 推論引擎 - 生成交易訊號
        
        Args:
            features: 特徵序列
            model: 模型
            proba_only: 只返回機率，不判斷訊號
        
        Returns:
            signal: bool 或 probability: float
        """
        try:
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
            
            # 預測機率
            proba = model.predict_proba(X)[0, 1]  # CatBoost/LightGBM
            
            if proba_only:
                return proba
            
            # Layer 1: 機率閉值濾網
            return proba >= self.probability_threshold, proba
        
        except Exception as e:
            logger.error(f"Signal generation error: {str(e)}")
            if proba_only:
                return 0.0
            return False, 0.0
    
    def open_position(self, entry_price: float, timestamp: pd.Timestamp, probability: float):
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
            'quantity': position_size / entry_price,
            'probability': probability  # 記錄機率
        }
        
        logger.info(f"[OPEN] Time={timestamp}, Hour={timestamp.hour}, Prob={probability:.3f}, Entry=${entry_price:.2f}, SL=${sl_price:.2f}, TP=${tp_price:.2f}")
    
    def check_exit(self, high: float, low: float, timestamp: pd.Timestamp) -> Tuple[bool, str, float]:
        """模組 3: 狀態機 - 檢查出場條件"""
        if self.position is None:
            return False, None, None
        
        sl_price = self.position['sl_price']
        tp_price = self.position['tp_price']
        
        sl_hit = low <= sl_price
        tp_hit = high >= tp_price
        
        # 悲觀假設: 同時觸發則先執行停損
        if sl_hit and tp_hit:
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
        probability = self.position['probability']
        
        # 非對稱手續費計算
        entry_fee = entry_price * self.maker_fee  # 進場 Maker
        
        if exit_reason == 'SL':
            # 停損必須用 Taker + 滑價
            exit_fee = exit_price * (self.taker_fee + self.slippage_pct)
        elif exit_reason == 'TP':
            # 停利用 Maker (限價單掉好)
            exit_fee = exit_price * self.maker_fee
        else:
            # 強制平倉用 Taker
            exit_fee = exit_price * (self.taker_fee + self.slippage_pct)
        
        # 計算總手續費
        total_fee = (entry_fee + exit_fee) * quantity
        
        # 計算 PnL
        gross_pnl = quantity * (exit_price - entry_price)
        net_pnl = gross_pnl - total_fee
        
        # 更新資金
        self.capital += net_pnl
        
        # 記錄交易
        trade = {
            'entry_time': self.position['entry_time'],
            'exit_time': timestamp,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'position_size': position_size,
            'gross_return': (exit_price - entry_price) / entry_price,
            'pnl': net_pnl,
            'capital': self.capital,
            'entry_hour': self.position['entry_time'].hour,
            'probability': probability,
            'entry_fee': entry_fee * quantity,
            'exit_fee': exit_fee * quantity,
            'total_fee': total_fee
        }
        self.trades.append(trade)
        
        logger.info(f"[CLOSE] Time={timestamp}, Exit=${exit_price:.2f}, Reason={exit_reason}, PnL=${net_pnl:.2f}, Fee=${total_fee:.2f}, Capital=${self.capital:.2f}")
        
        self.position = None
    
    def run_backtest(self, test_df: pd.DataFrame, model_path: str) -> Dict:
        """模組 1: 資料餵送器 - 運行完整回測"""
        logger.info("="*80)
        logger.info("Starting 3-layer filtered backtest")
        logger.info(f"Test set size: {len(test_df)} bars")
        logger.info(f"Model: {model_path}")
        logger.info("="*80)
        
        # 載入模型
        try:
            if model_path.endswith('.pkl'):
                model = joblib.load(model_path)
                logger.info("CatBoost model loaded successfully")
            else:
                model = lgb.Booster(model_file=model_path)
                logger.info("LightGBM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return None
        
        self.reset_state()
        test_df = test_df.sort_values('open_time').reset_index(drop=True)
        
        for i in range(len(test_df) - 1):
            current_bar = test_df.iloc[i]
            next_bar = test_df.iloc[i + 1]
            
            self.current_time = current_bar['open_time']
            
            # 檢查是否需要出場
            if self.position is not None:
                should_exit, exit_reason, exit_price = self.check_exit(
                    current_bar['high'],
                    current_bar['low'],
                    self.current_time
                )
                
                if should_exit:
                    self.close_position(exit_price, exit_reason, self.current_time)
            
            # 檢查是否需要進場
            if self.position is None:
                signal, probability = self.generate_signal(current_bar, model)
                
                if not signal:
                    # Layer 1: 機率不足
                    self.filtered_signals_prob += 1
                    continue
                
                # Layer 2: 時間濾網
                if not self.is_golden_hour(next_bar['open_time']):
                    self.filtered_signals_time += 1
                    if self.filtered_signals_time <= 5:  # 只記錄前 5 個
                        logger.info(f"[FILTERED] Signal at {next_bar['open_time']} (Hour {next_bar['open_time'].hour}) filtered by time")
                    continue
                
                # Layer 3: Maker 進場
                entry_price = next_bar['open']
                self.open_position(entry_price, next_bar['open_time'], probability)
            
            # 記錄權益曲線
            current_equity = self.capital
            if self.position is not None:
                unrealized_pnl = self.position['quantity'] * (current_bar['close'] - self.position['entry_price'])
                current_equity += unrealized_pnl
            
            self.equity_curve.append({
                'time': self.current_time,
                'equity': current_equity
            })
        
        # 強制平倉未結清部位
        if self.position is not None:
            last_bar = test_df.iloc[-1]
            self.close_position(last_bar['close'], 'FORCE_CLOSE', last_bar['open_time'])
        
        # 計算績效
        results = self.calculate_performance()
        
        logger.info("="*80)
        logger.info("3-LAYER FILTER RESULTS")
        logger.info("="*80)
        logger.info(f"Layer 1 (Probability): {self.filtered_signals_prob} signals filtered (< {self.probability_threshold:.0%})")
        logger.info(f"Layer 2 (Time): {self.filtered_signals_time} signals filtered (outside golden hours)")
        logger.info(f"Layer 3 (Maker): Reduced entry fee by 75% (0.04% → 0.01%)")
        logger.info("")
        logger.info(f"Total trades executed: {results['total_trades']}")
        logger.info(f"Win rate: {results['win_rate']*100:.2f}%")
        logger.info(f"Final capital: ${results['final_capital']:.2f}")
        logger.info(f"Total return: {results['total_return']*100:.2f}%")
        logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
        logger.info(f"Max drawdown: {results['max_drawdown']*100:.2f}%")
        logger.info("="*80)
        
        results['filtered_signals_prob'] = self.filtered_signals_prob
        results['filtered_signals_time'] = self.filtered_signals_time
        return results
    
    def calculate_performance(self) -> Dict:
        """模組 4: 績效結算器"""
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
                'profit_factor': 0,
                'trades_df': pd.DataFrame(),
                'equity_df': pd.DataFrame()
            }
        
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        total_trades = len(self.trades)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        # 最大回撤
        equity_series = equity_df['equity']
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = abs(drawdown.min())
        
        # Sharpe Ratio
        returns = trades_df['gross_return']
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Profit Factor
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