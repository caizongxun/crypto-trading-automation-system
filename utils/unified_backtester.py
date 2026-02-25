"""
統一回測器
支援 XGBoost (現有) 和 Chronos (Amazon 預訓練) 兩種模型
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Optional, List, Dict
from enum import Enum
from dataclasses import dataclass

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger
from utils.agent_backtester import BidirectionalAgentBacktester, AgentState, Trade

logger = setup_logger('unified_backtester', 'logs/unified_backtester.log')


class ChronosBacktester:
    """
    Chronos 模型回測器
    使用 Amazon Chronos 預訓練模型生成機率，執行與 XGBoost 版相同的交易邏輯
    """

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-small",
        device: str = "cpu",
        initial_capital: float = 10000.0,
        position_size_pct: float = 0.10,
        prob_threshold_long: float = 0.15,
        prob_threshold_short: float = 0.15,
        tp_pct: float = 0.02,
        sl_pct: float = 0.01,
        lookback: int = 168,
        horizon: int = 1,
        num_samples: int = 100,
        hunting_expire_bars: int = 15,
        trading_hours: Optional[List[tuple]] = None,
        maker_fee: float = 0.0001,
        taker_fee: float = 0.0004,
        slippage: float = 0.0002,
    ):
        self.model_name = model_name
        self.device = device
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.prob_threshold_long = prob_threshold_long
        self.prob_threshold_short = prob_threshold_short
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.lookback = lookback
        self.horizon = horizon
        self.num_samples = num_samples
        self.hunting_expire_bars = hunting_expire_bars
        self.trading_hours = trading_hours or [(0, 24)]
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage

        # 狀態
        self.capital = initial_capital
        self.state = AgentState.IDLE
        self.hunting_entry_bar = None
        self.limit_order_price = None
        self.entry_time = None
        self.entry_price = None
        self.position_size = None
        self.tp_price = None
        self.sl_price = None
        self.entry_prob = None

        self.trades: List[Trade] = []
        self.equity_curve = []

    def is_trading_hours(self, timestamp: pd.Timestamp) -> bool:
        hour = timestamp.hour
        for start, end in self.trading_hours:
            if start <= hour < end:
                return True
        return False

    def calculate_fees(self, trade_value: float, is_maker: bool = True) -> float:
        fee_rate = self.maker_fee if is_maker else (self.taker_fee + self.slippage)
        return trade_value * fee_rate

    def execute_trade(self, direction: str, entry_price: float, exit_price: float,
                      exit_reason: str, entry_time: pd.Timestamp, exit_time: pd.Timestamp,
                      entry_prob: float):
        if direction == 'LONG':
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        trade_value = self.position_size
        entry_fee = self.calculate_fees(trade_value, is_maker=True)
        exit_fee = self.calculate_fees(
            trade_value, is_maker=(exit_reason != 'SL')
        )
        total_fees = entry_fee + exit_fee
        net_pnl = self.position_size * pnl_pct - total_fees
        self.capital += net_pnl

        self.trades.append(Trade(
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
        ))

    def process_bar(self, bar_idx: int, row: pd.Series, prob_long: float, prob_short: float):
        timestamp = row.name
        open_price = row['open']
        high_price = row['high']
        low_price = row['low']
        close_price = row['close']

        if self.state == AgentState.IDLE:
            if not self.is_trading_hours(timestamp):
                return
            if prob_long >= self.prob_threshold_long and prob_short < 0.10:
                self.state = AgentState.HUNTING_LONG
                self.hunting_entry_bar = bar_idx
                self.limit_order_price = close_price * 0.9995
                self.entry_prob = prob_long
                return
            if prob_short >= self.prob_threshold_short and prob_long < 0.10:
                self.state = AgentState.HUNTING_SHORT
                self.hunting_entry_bar = bar_idx
                self.limit_order_price = close_price * 1.0005
                self.entry_prob = prob_short
                return

        elif self.state == AgentState.HUNTING_LONG:
            if bar_idx - self.hunting_entry_bar >= self.hunting_expire_bars:
                self.state = AgentState.IDLE
                self.hunting_entry_bar = None
                self.limit_order_price = None
                return
            if low_price < self.limit_order_price:
                self.state = AgentState.LONG_POSITION
                self.entry_time = timestamp
                self.entry_price = self.limit_order_price
                self.position_size = self.capital * self.position_size_pct
                self.tp_price = self.entry_price * (1 + self.tp_pct)
                self.sl_price = self.entry_price * (1 - self.sl_pct)
                return

        elif self.state == AgentState.HUNTING_SHORT:
            if bar_idx - self.hunting_entry_bar >= self.hunting_expire_bars:
                self.state = AgentState.IDLE
                self.hunting_entry_bar = None
                self.limit_order_price = None
                return
            if high_price > self.limit_order_price:
                self.state = AgentState.SHORT_POSITION
                self.entry_time = timestamp
                self.entry_price = self.limit_order_price
                self.position_size = self.capital * self.position_size_pct
                self.tp_price = self.entry_price * (1 - self.tp_pct)
                self.sl_price = self.entry_price * (1 + self.sl_pct)
                return

        elif self.state == AgentState.LONG_POSITION:
            tp_hit = high_price >= self.tp_price
            sl_hit = low_price <= self.sl_price
            exit_price = self.sl_price if (sl_hit and not tp_hit) or (tp_hit and sl_hit) else self.tp_price
            exit_reason = 'SL' if (sl_hit and not tp_hit) or (tp_hit and sl_hit) else 'TP'
            if tp_hit or sl_hit:
                self.execute_trade('LONG', self.entry_price, exit_price, exit_reason,
                                   self.entry_time, timestamp, self.entry_prob)
                self.state = AgentState.IDLE
                return

        elif self.state == AgentState.SHORT_POSITION:
            tp_hit = low_price <= self.tp_price
            sl_hit = high_price >= self.sl_price
            exit_price = self.sl_price if (sl_hit and not tp_hit) or (tp_hit and sl_hit) else self.tp_price
            exit_reason = 'SL' if (sl_hit and not tp_hit) or (tp_hit and sl_hit) else 'TP'
            if tp_hit or sl_hit:
                self.execute_trade('SHORT', self.entry_price, exit_price, exit_reason,
                                   self.entry_time, timestamp, self.entry_prob)
                self.state = AgentState.IDLE
                return

    def run(self, df_test: pd.DataFrame, feature_cols: List[str] = None) -> Dict:
        """
        執行 Chronos 回測

        df_test 必須包含 OHLCV 欄位 ('open', 'high', 'low', 'close', 'volume')
        feature_cols 參數保留以保持介面一致，Chronos 不使用
        """
        logger.info("Starting Chronos backtest")
        logger.info(f"Model: {self.model_name} | Device: {self.device}")
        logger.info(f"Test period: {df_test.index[0]} to {df_test.index[-1]}")
        logger.info(f"Total bars: {len(df_test):,}")

        # 延遲匯入以避免啟動時載入大型模型
        try:
            from models.chronos_predictor import ChronosPredictor
        except ImportError as e:
            logger.error(f"Failed to import ChronosPredictor: {e}")
            return {}

        predictor = ChronosPredictor(
            model_name=self.model_name,
            device=self.device
        )

        logger.info(f"Generating Chronos batch predictions (lookback={self.lookback})...")
        df_with_probs = predictor.predict_batch(
            df=df_test,
            lookback=self.lookback,
            horizon=self.horizon,
            num_samples=self.num_samples,
            tp_pct=self.tp_pct * 100,
            sl_pct=self.sl_pct * 100
        )

        # 逐根 K 線處理 (跳過沒有機率的初始 lookback 段)
        for bar_idx, (timestamp, row) in enumerate(df_with_probs.iterrows()):
            prob_long = row.get('prob_long', 0.0)
            prob_short = row.get('prob_short', 0.0)

            if pd.isna(prob_long) or pd.isna(prob_short):
                prob_long, prob_short = 0.0, 0.0

            self.process_bar(bar_idx, row, prob_long, prob_short)

            self.equity_curve.append({
                'timestamp': timestamp,
                'capital': self.capital,
                'state': self.state.value
            })

            if (bar_idx + 1) % 1000 == 0:
                logger.info(f"Processed {bar_idx+1:,}/{len(df_with_probs):,} bars")

        return self.calculate_metrics()

    def calculate_metrics(self) -> Dict:
        if len(self.trades) == 0:
            return {'total_trades': 0}

        total_trades = len(self.trades)
        long_trades = [t for t in self.trades if t.direction == 'LONG']
        short_trades = [t for t in self.trades if t.direction == 'SHORT']
        winning_trades = [t for t in self.trades if t.pnl_net > 0]
        losing_trades = [t for t in self.trades if t.pnl_net <= 0]

        win_rate = len(winning_trades) / total_trades
        total_return_pct = (self.capital - self.initial_capital) / self.initial_capital
        avg_win = np.mean([t.pnl_net for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_net for t in losing_trades]) if losing_trades else 0
        profit_factor = (
            abs(sum(t.pnl_net for t in winning_trades) / sum(t.pnl_net for t in losing_trades))
            if losing_trades else np.inf
        )

        tp_trades = [t for t in self.trades if t.exit_reason == 'TP']
        sl_trades = [t for t in self.trades if t.exit_reason == 'SL']

        return {
            'total_trades': total_trades,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': sum(t.pnl_net for t in self.trades),
            'total_return_pct': total_return_pct,
            'final_capital': self.capital,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'tp_rate': len(tp_trades) / total_trades,
            'sl_rate': len(sl_trades) / total_trades,
            'tp_count': len(tp_trades),
            'sl_count': len(sl_trades),
        }

    def get_equity_curve(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_curve)

    def get_trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([{
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
        } for t in self.trades])


class UnifiedBacktester:
    """
    統一回測器入口
    根據 model_type 自動選擇 XGBoost 或 Chronos 回測器
    """

    def __init__(
        self,
        model_type: str = 'xgboost',
        # XGBoost 參數
        model_long_path: Optional[str] = None,
        model_short_path: Optional[str] = None,
        # Chronos 參數
        chronos_model_name: str = "amazon/chronos-t5-small",
        chronos_device: str = "cpu",
        chronos_lookback: int = 168,
        chronos_horizon: int = 1,
        chronos_num_samples: int = 100,
        # 共用參數
        initial_capital: float = 10000.0,
        position_size_pct: float = 0.10,
        prob_threshold_long: float = 0.15,
        prob_threshold_short: float = 0.15,
        tp_pct: float = 0.02,
        sl_pct: float = 0.01,
        trading_hours: Optional[List[tuple]] = None,
        hunting_expire_bars: int = 15,
        maker_fee: float = 0.0001,
        taker_fee: float = 0.0004,
        slippage: float = 0.0002,
    ):
        self.model_type = model_type

        if model_type == 'xgboost':
            if not model_long_path or not model_short_path:
                raise ValueError("model_long_path and model_short_path are required for xgboost")
            self._backtester = BidirectionalAgentBacktester(
                model_long_path=model_long_path,
                model_short_path=model_short_path,
                initial_capital=initial_capital,
                position_size_pct=position_size_pct,
                prob_threshold_long=prob_threshold_long,
                prob_threshold_short=prob_threshold_short,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                hunting_expire_bars=hunting_expire_bars,
                trading_hours=trading_hours,
                maker_fee=maker_fee,
                taker_fee=taker_fee,
                slippage=slippage,
            )
        elif model_type == 'chronos':
            self._backtester = ChronosBacktester(
                model_name=chronos_model_name,
                device=chronos_device,
                initial_capital=initial_capital,
                position_size_pct=position_size_pct,
                prob_threshold_long=prob_threshold_long,
                prob_threshold_short=prob_threshold_short,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                lookback=chronos_lookback,
                horizon=chronos_horizon,
                num_samples=chronos_num_samples,
                hunting_expire_bars=hunting_expire_bars,
                trading_hours=trading_hours,
                maker_fee=maker_fee,
                taker_fee=taker_fee,
                slippage=slippage,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'xgboost' or 'chronos'")

    def run(self, df_test: pd.DataFrame, feature_cols: List[str] = None) -> Dict:
        return self._backtester.run(df_test, feature_cols)

    def get_trades_df(self) -> pd.DataFrame:
        return self._backtester.get_trades_df()

    def get_equity_curve(self) -> pd.DataFrame:
        return self._backtester.get_equity_curve()
