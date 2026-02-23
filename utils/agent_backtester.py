import pandas as pd
import numpy as np
from pathlib import Path
import sys
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List
import joblib

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger('agent_backtester', 'logs/agent_backtester.log')

class AgentState(Enum):
    """智能體狀態機"""
    IDLE = "IDLE"                          # 空手,等待機會
    HUNTING_LONG = "HUNTING_LONG"          # 做多狩獵中,限價單掉下方
    HUNTING_SHORT = "HUNTING_SHORT"        # 做空狩獵中,限價單挂上方
    LONG_POSITION = "LONG_POSITION"        # 持有多單
    SHORT_POSITION = "SHORT_POSITION"      # 持有空單

@dataclass
class Trade:
    """交易記錄"""
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    direction: str  # 'LONG' or 'SHORT'
    exit_reason: str  # 'TP', 'SL', 'TIMEOUT'
    pnl_pct: float
    pnl_net: float  # 扣除手續費後的淨利
    fees: float

class BidirectionalAgentBacktester:
    """
    雙向智能體回測框架 - 事件驅動狀態機
    
    **核心特性**:
    - 事件驅動: 逐根 1m K 線處理
    - 狀態機: 5 種狀態嚴格流轉
    - 悉觀成交: 限價單必須嚴格穿越
    - 薫丁格處理: 同時觸及 TP+SL 則 SL 優先
    - 不對稱成本: Maker/Taker 分離計算
    - 訂單過期: 15 分鐘未成交自動取消
    """
    
    def __init__(self,
                 model_long_path: str,
                 model_short_path: str,
                 initial_capital: float = 10000.0,
                 position_size_pct: float = 0.95,
                 prob_threshold_long: float = 0.65,
                 prob_threshold_short: float = 0.65,
                 tp_pct: float = 0.02,
                 sl_pct: float = 0.01,
                 hunting_expire_bars: int = 15,
                 trading_hours: Optional[List[tuple]] = None,
                 maker_fee: float = 0.0001,
                 taker_fee: float = 0.0004,
                 slippage: float = 0.0002):
        """
        Args:
            model_long_path: Long Oracle 模型路徑
            model_short_path: Short Oracle 模型路徑
            initial_capital: 初始資金
            position_size_pct: 每筆交易使用資金比例
            prob_threshold_long: Long 機率閾值
            prob_threshold_short: Short 機率閾值
            tp_pct: 停利百分比
            sl_pct: 停損百分比
            hunting_expire_bars: 狩獵狀態過期時間 (1m K 線數)
            trading_hours: 交易時段 [(9, 14), (18, 22)]
            maker_fee: Maker 手續費 (限價單)
            taker_fee: Taker 手續費 (市價單)
            slippage: 滑價
        """
        logger.info("="*80)
        logger.info("INITIALIZING BIDIRECTIONAL AGENT BACKTESTER")
        logger.info("="*80)
        
        # 載入模型
        self.model_long = joblib.load(model_long_path)
        self.model_short = joblib.load(model_short_path)
        logger.info(f"Loaded Long Oracle: {model_long_path}")
        logger.info(f"Loaded Short Oracle: {model_short_path}")
        
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
        self.trading_hours = trading_hours or [(9, 14), (18, 22)]
        
        # 成本參數
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage
        
        # 狀態追蹤
        self.state = AgentState.IDLE
        self.hunting_entry_bar = None  # 進入 HUNTING 的 bar index
        self.limit_order_price = None
        self.entry_time = None
        self.entry_price = None
        self.position_size = None
        self.tp_price = None
        self.sl_price = None
        
        # 回測結果
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.state_history = []
        
        logger.info(f"Initial capital: ${initial_capital:,.2f}")
        logger.info(f"Position size: {position_size_pct*100:.1f}% of capital")
        logger.info(f"Prob thresholds: Long={prob_threshold_long:.2f}, Short={prob_threshold_short:.2f}")
        logger.info(f"TP/SL: {tp_pct*100:.1f}% / {sl_pct*100:.1f}%")
        logger.info(f"Trading hours: {self.trading_hours}")
        logger.info(f"Fees: Maker={maker_fee*10000:.1f}bp, Taker={taker_fee*10000:.1f}bp, Slippage={slippage*10000:.1f}bp")
        logger.info("="*80)
    
    def is_trading_hours(self, timestamp: pd.Timestamp) -> bool:
        """檢查是否在交易時段"""
        hour = timestamp.hour
        for start, end in self.trading_hours:
            if start <= hour < end:
                return True
        return False
    
    def get_oracle_signals(self, features: np.ndarray) -> tuple:
        """
        獲取雙向大腦預測
        
        Args:
            features: 1D array of features for current bar
        
        Returns:
            (prob_long, prob_short)
        """
        features_2d = features.reshape(1, -1)
        prob_long = self.model_long.predict_proba(features_2d)[0, 1]
        prob_short = self.model_short.predict_proba(features_2d)[0, 1]
        return prob_long, prob_short
    
    def calculate_fees(self, trade_value: float, is_maker: bool = True) -> float:
        """計算手續費"""
        fee_rate = self.maker_fee if is_maker else (self.taker_fee + self.slippage)
        return trade_value * fee_rate
    
    def execute_trade(self, direction: str, entry_price: float, exit_price: float, 
                     exit_reason: str, entry_time: pd.Timestamp, exit_time: pd.Timestamp):
        """
        執行交易並記錄
        
        Args:
            direction: 'LONG' or 'SHORT'
            entry_price: 進場價
            exit_price: 出場價
            exit_reason: 'TP', 'SL', 'TIMEOUT'
            entry_time: 進場時間
            exit_time: 出場時間
        """
        # 計算比例獲利
        if direction == 'LONG':
            pnl_pct = (exit_price - entry_price) / entry_price
        else:  # SHORT
            pnl_pct = (entry_price - exit_price) / entry_price
        
        # 計算手續費
        trade_value = self.position_size
        entry_fee = self.calculate_fees(trade_value, is_maker=True)  # 限價進場
        
        if exit_reason == 'SL':
            # 停損用市價單 + 滑價
            exit_fee = self.calculate_fees(trade_value, is_maker=False)
        else:
            # TP 或 TIMEOUT 用限價單
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
            fees=total_fees
        )
        self.trades.append(trade)
        
        logger.info(f"TRADE CLOSED: {direction} | Entry={entry_price:.2f} | Exit={exit_price:.2f} | "
                   f"Reason={exit_reason} | PnL={net_pnl:+.2f} ({pnl_pct*100:+.2f}%) | "
                   f"Fees={total_fees:.2f} | Capital={self.capital:.2f}")
    
    def process_bar(self, bar_idx: int, row: pd.Series, features: np.ndarray):
        """
        處理單根 K 線 - 狀態機核心
        
        Args:
            bar_idx: K 線索引
            row: K 線數據 (open, high, low, close, volume)
            features: 當前 K 線的特徵
        """
        timestamp = row.name
        open_price = row['open']
        high_price = row['high']
        low_price = row['low']
        close_price = row['close']
        
        # === 狀態 1: IDLE ===
        if self.state == AgentState.IDLE:
            # 檢查交易時段
            if not self.is_trading_hours(timestamp):
                return
            
            # 獲取大腦信號
            prob_long, prob_short = self.get_oracle_signals(features)
            
            # 檢查 Long 信號
            if prob_long >= self.prob_threshold_long:
                # 進入 HUNTING_LONG,在下方挂限價單
                self.state = AgentState.HUNTING_LONG
                self.hunting_entry_bar = bar_idx
                # 限價單掉在 close 下方 0.5%
                self.limit_order_price = close_price * 0.995
                logger.info(f"[{timestamp}] IDLE -> HUNTING_LONG | prob={prob_long:.4f} | limit_order={self.limit_order_price:.2f}")
                return
            
            # 檢查 Short 信號
            if prob_short >= self.prob_threshold_short:
                # 進入 HUNTING_SHORT,在上方挂限價單
                self.state = AgentState.HUNTING_SHORT
                self.hunting_entry_bar = bar_idx
                # 限價單挂在 close 上方 0.5%
                self.limit_order_price = close_price * 1.005
                logger.info(f"[{timestamp}] IDLE -> HUNTING_SHORT | prob={prob_short:.4f} | limit_order={self.limit_order_price:.2f}")
                return
        
        # === 狀態 2: HUNTING_LONG ===
        elif self.state == AgentState.HUNTING_LONG:
            # 檢查訂單過期
            if bar_idx - self.hunting_entry_bar >= self.hunting_expire_bars:
                logger.info(f"[{timestamp}] HUNTING_LONG expired (15 min) -> IDLE")
                self.state = AgentState.IDLE
                self.hunting_entry_bar = None
                self.limit_order_price = None
                return
            
            # 檢查悉觀成交 (嚴格小於)
            if low_price < self.limit_order_price:
                # 成交
                self.state = AgentState.LONG_POSITION
                self.entry_time = timestamp
                self.entry_price = self.limit_order_price
                self.position_size = self.capital * self.position_size_pct
                self.tp_price = self.entry_price * (1 + self.tp_pct)
                self.sl_price = self.entry_price * (1 - self.sl_pct)
                logger.info(f"[{timestamp}] HUNTING_LONG -> LONG_POSITION | entry={self.entry_price:.2f} | "
                           f"TP={self.tp_price:.2f} | SL={self.sl_price:.2f} | size={self.position_size:.2f}")
                return
        
        # === 狀態 3: HUNTING_SHORT ===
        elif self.state == AgentState.HUNTING_SHORT:
            # 檢查訂單過期
            if bar_idx - self.hunting_entry_bar >= self.hunting_expire_bars:
                logger.info(f"[{timestamp}] HUNTING_SHORT expired (15 min) -> IDLE")
                self.state = AgentState.IDLE
                self.hunting_entry_bar = None
                self.limit_order_price = None
                return
            
            # 檢查悉觀成交 (嚴格大於)
            if high_price > self.limit_order_price:
                # 成交
                self.state = AgentState.SHORT_POSITION
                self.entry_time = timestamp
                self.entry_price = self.limit_order_price
                self.position_size = self.capital * self.position_size_pct
                self.tp_price = self.entry_price * (1 - self.tp_pct)
                self.sl_price = self.entry_price * (1 + self.sl_pct)
                logger.info(f"[{timestamp}] HUNTING_SHORT -> SHORT_POSITION | entry={self.entry_price:.2f} | "
                           f"TP={self.tp_price:.2f} | SL={self.sl_price:.2f} | size={self.position_size:.2f}")
                return
        
        # === 狀態 4: LONG_POSITION ===
        elif self.state == AgentState.LONG_POSITION:
            # 檢查薫丁格狀態 (同時觸及 TP 和 SL)
            tp_hit = high_price >= self.tp_price
            sl_hit = low_price <= self.sl_price
            
            if tp_hit and sl_hit:
                # 薫丁格案例: 假設最壞情況,先觸及 SL
                self.execute_trade(
                    direction='LONG',
                    entry_price=self.entry_price,
                    exit_price=self.sl_price,
                    exit_reason='SL',
                    entry_time=self.entry_time,
                    exit_time=timestamp
                )
                self.state = AgentState.IDLE
                logger.info(f"[{timestamp}] Schrodinger case: Both TP and SL hit -> SL executed first (worst case)")
                return
            
            if tp_hit:
                # 停利
                self.execute_trade(
                    direction='LONG',
                    entry_price=self.entry_price,
                    exit_price=self.tp_price,
                    exit_reason='TP',
                    entry_time=self.entry_time,
                    exit_time=timestamp
                )
                self.state = AgentState.IDLE
                return
            
            if sl_hit:
                # 停損
                self.execute_trade(
                    direction='LONG',
                    entry_price=self.entry_price,
                    exit_price=self.sl_price,
                    exit_reason='SL',
                    entry_time=self.entry_time,
                    exit_time=timestamp
                )
                self.state = AgentState.IDLE
                return
        
        # === 狀態 5: SHORT_POSITION ===
        elif self.state == AgentState.SHORT_POSITION:
            # 檢查薫丁格狀態
            tp_hit = low_price <= self.tp_price
            sl_hit = high_price >= self.sl_price
            
            if tp_hit and sl_hit:
                # 薫丁格案例: 假設最壞情況,先觸及 SL
                self.execute_trade(
                    direction='SHORT',
                    entry_price=self.entry_price,
                    exit_price=self.sl_price,
                    exit_reason='SL',
                    entry_time=self.entry_time,
                    exit_time=timestamp
                )
                self.state = AgentState.IDLE
                logger.info(f"[{timestamp}] Schrodinger case: Both TP and SL hit -> SL executed first (worst case)")
                return
            
            if tp_hit:
                # 停利
                self.execute_trade(
                    direction='SHORT',
                    entry_price=self.entry_price,
                    exit_price=self.tp_price,
                    exit_reason='TP',
                    entry_time=self.entry_time,
                    exit_time=timestamp
                )
                self.state = AgentState.IDLE
                return
            
            if sl_hit:
                # 停損
                self.execute_trade(
                    direction='SHORT',
                    entry_price=self.entry_price,
                    exit_price=self.sl_price,
                    exit_reason='SL',
                    entry_time=self.entry_time,
                    exit_time=timestamp
                )
                self.state = AgentState.IDLE
                return
    
    def run(self, df_test: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """
        執行回測
        
        Args:
            df_test: 測試集 (OHLCV + features)
            feature_cols: 特徵欄位名稱
        
        Returns:
            回測結果字典
        """
        logger.info("="*80)
        logger.info("STARTING BACKTEST")
        logger.info("="*80)
        logger.info(f"Test period: {df_test.index[0]} to {df_test.index[-1]}")
        logger.info(f"Total bars: {len(df_test):,}")
        logger.info(f"Features: {feature_cols}")
        logger.info("="*80)
        
        # 逐根 K 線處理
        for bar_idx, (timestamp, row) in enumerate(df_test.iterrows()):
            # 獲取特徵
            features = row[feature_cols].values
            
            # 處理當前 K 線
            self.process_bar(bar_idx, row, features)
            
            # 記錄權益曲線
            self.equity_curve.append({
                'timestamp': timestamp,
                'capital': self.capital,
                'state': self.state.value
            })
            
            # 進度顯示
            if (bar_idx + 1) % 10000 == 0:
                logger.info(f"Processed {bar_idx+1:,}/{len(df_test):,} bars ({(bar_idx+1)/len(df_test)*100:.1f}%)")
        
        # 計算統計指標
        results = self.calculate_metrics()
        
        logger.info("="*80)
        logger.info("BACKTEST COMPLETED")
        logger.info("="*80)
        
        return results
    
    def calculate_metrics(self) -> Dict:
        """計算回測統計指標"""
        if len(self.trades) == 0:
            logger.warning("No trades executed during backtest")
            return {}
        
        # 基礎指標
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
        
        # 停利/停損統計
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
        
        # 顯示結果
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
            'fees': t.fees
        } for t in self.trades]
        
        return pd.DataFrame(trades_data)