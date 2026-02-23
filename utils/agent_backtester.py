import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger('agent_backtester', 'logs/agent_backtester.log')

class AgentState(Enum):
    """智能體狀態機"""
    IDLE = "IDLE"
    HUNTING_LONG = "HUNTING_LONG"
    HUNTING_SHORT = "HUNTING_SHORT"
    LONG_POSITION = "LONG_POSITION"
    SHORT_POSITION = "SHORT_POSITION"

@dataclass
class Trade:
    """交易記錄"""
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    direction: str
    exit_reason: str
    pnl_pct: float
    pnl_net: float
    fees: float
    probability: float

class BidirectionalAgentBacktester:
    """
    雙向事件驅動智能體 - 動態特徵抓取版
    
    **核心修復**:
    - 從模型直接讀取特徵名稱 (永不錯位)
    - 批次預測機率 (避免逐筆報錯)
    - 雙向互斥決策邏輯
    - 完整的非對稱手續費
    """
    
    def __init__(self,
                 model_long_path: str,
                 model_short_path: str,
                 initial_capital: float = 10000.0,
                 position_size_pct: float = 0.10,
                 prob_threshold_long: float = 0.65,
                 prob_threshold_short: float = 0.65,
                 tp_pct: float = 0.02,
                 sl_pct: float = 0.01,
                 hunting_expire_bars: int = 15,
                 trading_hours: Optional[List[tuple]] = None,
                 maker_fee: float = 0.0001,
                 taker_fee: float = 0.0004,
                 slippage: float = 0.0002):
        
        logger.info("="*80)
        logger.info("INITIALIZING BIDIRECTIONAL AGENT - DYNAMIC FEATURE VERSION")
        logger.info("="*80)
        
        # 載入模型
        logger.info("Loading Long and Short Oracles...")
        self.model_long = joblib.load(model_long_path)
        self.model_short = joblib.load(model_short_path)
        
        # [核心修復] 從模型直接讀取訓練時用的特徵名稱
        try:
            # CalibratedClassifierCV 需要訪問 base_estimator
            if hasattr(self.model_long, 'estimators_'):
                # CalibratedClassifierCV 的情況
                base_model_long = self.model_long.estimators_[0].estimator
                base_model_short = self.model_short.estimators_[0].estimator
            else:
                # 直接是 CatBoost 模型
                base_model_long = self.model_long
                base_model_short = self.model_short
            
            self.features_long = list(base_model_long.feature_names_)
            self.features_short = list(base_model_short.feature_names_)
            
            logger.info(f"Long Oracle features: {self.features_long}")
            logger.info(f"Short Oracle features: {self.features_short}")
            
        except Exception as e:
            logger.error(f"Failed to extract feature names: {str(e)}")
            # Fallback: 使用預設特徵
            self.features_long = ['efficiency_ratio', 'extreme_time_diff', 'vol_imbalance_ratio',
                                 'z_score', 'bb_width_pct', 'rsi', 'atr_pct', 'z_score_1h', 'atr_pct_1d']
            self.features_short = self.features_long
            logger.warning(f"Using fallback features: {self.features_long}")
        
        # 資金設定
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.capital = initial_capital
        
        # 機率閾值
        self.prob_threshold_long = prob_threshold_long
        self.prob_threshold_short = prob_threshold_short
        
        # 風控參數
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.hunting_expire_bars = hunting_expire_bars
        
        # 交易時段
        self.trading_hours = trading_hours or [(0, 24)]
        
        # 成本參數
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage
        
        # 狀態追蹤
        self.state = AgentState.IDLE
        self.hunting_entry_bar = None
        self.limit_order_price = None
        self.entry_time = None
        self.entry_price = None
        self.position_size = None
        self.tp_price = None
        self.sl_price = None
        self.entry_prob = None
        
        # 回測結果
        self.trades: List[Trade] = []
        self.equity_curve = []
        
        logger.info(f"Initial capital: ${initial_capital:,.2f}")
        logger.info(f"Position size: {position_size_pct*100:.1f}%")
        logger.info(f"Prob thresholds: Long={prob_threshold_long:.2f}, Short={prob_threshold_short:.2f}")
        logger.info(f"TP/SL: {tp_pct*100:.1f}% / {sl_pct*100:.1f}%")
        logger.info("="*80)
    
    def is_trading_hours(self, timestamp: pd.Timestamp) -> bool:
        """檢查是否在交易時段"""
        hour = timestamp.hour
        for start, end in self.trading_hours:
            if start <= hour < end:
                return True
        return False
    
    def calculate_fees(self, trade_value: float, is_maker: bool = True) -> float:
        """計算手續費"""
        fee_rate = self.maker_fee if is_maker else (self.taker_fee + self.slippage)
        return trade_value * fee_rate
    
    def execute_trade(self, direction: str, entry_price: float, exit_price: float,
                     exit_reason: str, entry_time: pd.Timestamp, exit_time: pd.Timestamp,
                     entry_prob: float):
        """執行交易並記錄"""
        # 計算比例獲利
        if direction == 'LONG':
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price
        
        # 計算手續費
        trade_value = self.position_size
        entry_fee = self.calculate_fees(trade_value, is_maker=True)
        
        if exit_reason == 'SL':
            exit_fee = self.calculate_fees(trade_value, is_maker=False)
        else:
            exit_fee = self.calculate_fees(trade_value, is_maker=True)
        
        total_fees = entry_fee + exit_fee
        
        # 淨利
        gross_pnl = self.position_size * pnl_pct
        net_pnl = gross_pnl - total_fees
        
        # 更新資金
        self.capital += net_pnl
        
        # 記錄交易
        trade = Trade(
            entry_time=entry_time,
            entry_price=entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            direction=direction,
            exit_reason=exit_reason,
            pnl_pct=pnl_pct,
            pnl_net=net_pnl,
            fees=total_fees,
            probability=entry_prob
        )
        self.trades.append(trade)
        
        logger.info(f"TRADE: {direction} | Entry={entry_price:.2f} | Exit={exit_price:.2f} | "
                   f"Reason={exit_reason} | PnL={net_pnl:+.2f} ({pnl_pct*100:+.2f}%) | "
                   f"Prob={entry_prob:.3f} | Capital={self.capital:.2f}")
    
    def process_bar(self, bar_idx: int, row: pd.Series, prob_long: float, prob_short: float):
        """處理單根 K 線"""
        timestamp = row.name
        open_price = row['open']
        high_price = row['high']
        low_price = row['low']
        close_price = row['close']
        
        # === 狀態 1: IDLE ===
        if self.state == AgentState.IDLE:
            if not self.is_trading_hours(timestamp):
                return
            
            # 雙向互斥決策 (避免同時做多做空)
            if prob_long >= self.prob_threshold_long and prob_short < 0.40:
                self.state = AgentState.HUNTING_LONG
                self.hunting_entry_bar = bar_idx
                self.limit_order_price = close_price * 0.9995  # 下方 0.05% 接刀
                self.entry_prob = prob_long
                logger.info(f"[{timestamp}] IDLE -> HUNTING_LONG | prob={prob_long:.4f} | limit={self.limit_order_price:.2f}")
                return
            
            if prob_short >= self.prob_threshold_short and prob_long < 0.40:
                self.state = AgentState.HUNTING_SHORT
                self.hunting_entry_bar = bar_idx
                self.limit_order_price = close_price * 1.0005  # 上方 0.05% 狙擊
                self.entry_prob = prob_short
                logger.info(f"[{timestamp}] IDLE -> HUNTING_SHORT | prob={prob_short:.4f} | limit={self.limit_order_price:.2f}")
                return
        
        # === 狀態 2: HUNTING_LONG ===
        elif self.state == AgentState.HUNTING_LONG:
            if bar_idx - self.hunting_entry_bar >= self.hunting_expire_bars:
                logger.info(f"[{timestamp}] HUNTING_LONG expired -> IDLE")
                self.state = AgentState.IDLE
                self.hunting_entry_bar = None
                self.limit_order_price = None
                return
            
            # 悲觀成交 (嚴格小於)
            if low_price < self.limit_order_price:
                self.state = AgentState.LONG_POSITION
                self.entry_time = timestamp
                self.entry_price = self.limit_order_price
                self.position_size = self.capital * self.position_size_pct
                self.tp_price = self.entry_price * (1 + self.tp_pct)
                self.sl_price = self.entry_price * (1 - self.sl_pct)
                logger.info(f"[{timestamp}] HUNTING_LONG -> LONG_POSITION | entry={self.entry_price:.2f} | "
                           f"TP={self.tp_price:.2f} | SL={self.sl_price:.2f}")
                return
        
        # === 狀態 3: HUNTING_SHORT ===
        elif self.state == AgentState.HUNTING_SHORT:
            if bar_idx - self.hunting_entry_bar >= self.hunting_expire_bars:
                logger.info(f"[{timestamp}] HUNTING_SHORT expired -> IDLE")
                self.state = AgentState.IDLE
                self.hunting_entry_bar = None
                self.limit_order_price = None
                return
            
            # 悲觀成交 (嚴格大於)
            if high_price > self.limit_order_price:
                self.state = AgentState.SHORT_POSITION
                self.entry_time = timestamp
                self.entry_price = self.limit_order_price
                self.position_size = self.capital * self.position_size_pct
                self.tp_price = self.entry_price * (1 - self.tp_pct)
                self.sl_price = self.entry_price * (1 + self.sl_pct)
                logger.info(f"[{timestamp}] HUNTING_SHORT -> SHORT_POSITION | entry={self.entry_price:.2f} | "
                           f"TP={self.tp_price:.2f} | SL={self.sl_price:.2f}")
                return
        
        # === 狀態 4: LONG_POSITION ===
        elif self.state == AgentState.LONG_POSITION:
            tp_hit = high_price >= self.tp_price
            sl_hit = low_price <= self.sl_price
            
            if tp_hit and sl_hit:
                # 薛丁格案例: 假設最壞情況
                self.execute_trade(
                    direction='LONG',
                    entry_price=self.entry_price,
                    exit_price=self.sl_price,
                    exit_reason='SL',
                    entry_time=self.entry_time,
                    exit_time=timestamp,
                    entry_prob=self.entry_prob
                )
                self.state = AgentState.IDLE
                return
            
            if tp_hit:
                self.execute_trade(
                    direction='LONG',
                    entry_price=self.entry_price,
                    exit_price=self.tp_price,
                    exit_reason='TP',
                    entry_time=self.entry_time,
                    exit_time=timestamp,
                    entry_prob=self.entry_prob
                )
                self.state = AgentState.IDLE
                return
            
            if sl_hit:
                self.execute_trade(
                    direction='LONG',
                    entry_price=self.entry_price,
                    exit_price=self.sl_price,
                    exit_reason='SL',
                    entry_time=self.entry_time,
                    exit_time=timestamp,
                    entry_prob=self.entry_prob
                )
                self.state = AgentState.IDLE
                return
        
        # === 狀態 5: SHORT_POSITION ===
        elif self.state == AgentState.SHORT_POSITION:
            tp_hit = low_price <= self.tp_price
            sl_hit = high_price >= self.sl_price
            
            if tp_hit and sl_hit:
                self.execute_trade(
                    direction='SHORT',
                    entry_price=self.entry_price,
                    exit_price=self.sl_price,
                    exit_reason='SL',
                    entry_time=self.entry_time,
                    exit_time=timestamp,
                    entry_prob=self.entry_prob
                )
                self.state = AgentState.IDLE
                return
            
            if tp_hit:
                self.execute_trade(
                    direction='SHORT',
                    entry_price=self.entry_price,
                    exit_price=self.tp_price,
                    exit_reason='TP',
                    entry_time=self.entry_time,
                    exit_time=timestamp,
                    entry_prob=self.entry_prob
                )
                self.state = AgentState.IDLE
                return
            
            if sl_hit:
                self.execute_trade(
                    direction='SHORT',
                    entry_price=self.entry_price,
                    exit_price=self.sl_price,
                    exit_reason='SL',
                    entry_time=self.entry_time,
                    exit_time=timestamp,
                    entry_prob=self.entry_prob
                )
                self.state = AgentState.IDLE
                return
    
    def run(self, df_test: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """
        執行回測 - 批次預測版
        
        Args:
            df_test: 測試集 (OHLCV + features)
            feature_cols: 特徵欄位名稱 (會被忽略,使用模型內建特徵)
        """
        logger.info("="*80)
        logger.info("STARTING BIDIRECTIONAL BACKTEST - BATCH PREDICTION MODE")
        logger.info("="*80)
        logger.info(f"Test period: {df_test.index[0]} to {df_test.index[-1]}")
        logger.info(f"Total bars: {len(df_test):,}")
        
        # [核心修復] 批次預測機率 (避免逐筆報錯)
        logger.info("Batch predicting probabilities...")
        
        try:
            # 提取正確的特徵
            X_long = df_test[self.features_long].fillna(0).values
            X_short = df_test[self.features_short].fillna(0).values
            
            # 批次預測
            df_test['prob_long'] = self.model_long.predict_proba(X_long)[:, 1]
            df_test['prob_short'] = self.model_short.predict_proba(X_short)[:, 1]
            
            logger.info(f"Batch prediction completed")
            logger.info(f"Long prob range: {df_test['prob_long'].min():.3f} - {df_test['prob_long'].max():.3f}")
            logger.info(f"Short prob range: {df_test['prob_short'].min():.3f} - {df_test['prob_short'].max():.3f}")
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)
            return {}
        
        logger.info("="*80)
        
        # 逐根 K 線處理
        for bar_idx, (timestamp, row) in enumerate(df_test.iterrows()):
            prob_long = row['prob_long']
            prob_short = row['prob_short']
            
            self.process_bar(bar_idx, row, prob_long, prob_short)
            
            # 記錄權益曲線
            self.equity_curve.append({
                'timestamp': timestamp,
                'capital': self.capital,
                'state': self.state.value
            })
            
            if (bar_idx + 1) % 10000 == 0:
                logger.info(f"Processed {bar_idx+1:,}/{len(df_test):,} bars ({(bar_idx+1)/len(df_test)*100:.1f}%)")
        
        results = self.calculate_metrics()
        
        logger.info("="*80)
        logger.info("BACKTEST COMPLETED")
        logger.info("="*80)
        
        return results
    
    def calculate_metrics(self) -> Dict:
        """計算回測統計指標"""
        if len(self.trades) == 0:
            logger.warning("No trades executed during backtest")
            return {'total_trades': 0}
        
        total_trades = len(self.trades)
        long_trades = [t for t in self.trades if t.direction == 'LONG']
        short_trades = [t for t in self.trades if t.direction == 'SHORT']
        
        winning_trades = [t for t in self.trades if t.pnl_net > 0]
        losing_trades = [t for t in self.trades if t.pnl_net <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t.pnl_net for t in self.trades)
        total_return_pct = (self.capital - self.initial_capital) / self.initial_capital
        
        avg_win = np.mean([t.pnl_net for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_net for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(sum(t.pnl_net for t in winning_trades) / sum(t.pnl_net for t in losing_trades)) if losing_trades else np.inf
        
        tp_trades = [t for t in self.trades if t.exit_reason == 'TP']
        sl_trades = [t for t in self.trades if t.exit_reason == 'SL']
        tp_rate = len(tp_trades) / total_trades if total_trades > 0 else 0
        sl_rate = len(sl_trades) / total_trades if total_trades > 0 else 0
        
        results = {
            'total_trades': total_trades,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'final_capital': self.capital,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'tp_rate': tp_rate,
            'sl_rate': sl_rate,
            'tp_count': len(tp_trades),
            'sl_count': len(sl_trades)
        }
        
        logger.info("="*80)
        logger.info("BACKTEST RESULTS")
        logger.info("="*80)
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"  Long: {len(long_trades)} | Short: {len(short_trades)}")
        logger.info(f"Win Rate: {win_rate*100:.2f}% ({len(winning_trades)}/{total_trades})")
        logger.info(f"Total PnL: ${total_pnl:+,.2f}")
        logger.info(f"Total Return: {total_return_pct*100:+.2f}%")
        logger.info(f"Final Capital: ${self.capital:,.2f}")
        logger.info(f"Avg Win: ${avg_win:+.2f} | Avg Loss: ${avg_loss:+.2f}")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info(f"TP Rate: {tp_rate*100:.1f}% ({len(tp_trades)}/{total_trades})")
        logger.info(f"SL Rate: {sl_rate*100:.1f}% ({len(sl_trades)}/{total_trades})")
        logger.info("="*80)
        
        return results
    
    def get_equity_curve(self) -> pd.DataFrame:
        """獲取權益曲線"""
        return pd.DataFrame(self.equity_curve)
    
    def get_trades_df(self) -> pd.DataFrame:
        """獲取交易記錄"""
        if not self.trades:
            return pd.DataFrame()
        
        trades_data = [{
            'entry_time': t.entry_time,
            'entry_price': t.entry_price,
            'exit_time': t.exit_time,
            'exit_price': t.exit_price,
            'direction': t.direction,
            'exit_reason': t.exit_reason,
            'pnl_pct': t.pnl_pct,
            'pnl_net': t.pnl_net,
            'fees': t.fees,
            'probability': t.probability
        } for t in self.trades]
        
        return pd.DataFrame(trades_data)