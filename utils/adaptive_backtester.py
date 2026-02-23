import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger

logger = setup_logger('adaptive_backtester', 'logs/adaptive_backtester.log')

class AgentState(Enum):
    IDLE = "IDLE"
    HUNTING_LONG = "HUNTING_LONG"
    HUNTING_SHORT = "HUNTING_SHORT"
    LONG_POSITION = "LONG_POSITION"
    SHORT_POSITION = "SHORT_POSITION"

class VolatilityRegime(Enum):
    LOW = "LOW"        # ATR < 2%
    MEDIUM = "MEDIUM"  # 2% <= ATR < 4%
    HIGH = "HIGH"      # ATR >= 4%

@dataclass
class Trade:
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
    volatility_regime: str
    entry_hour: int
    position_size_pct: float

class AdaptiveBacktester:
    """
    進階自適應回測器
    
    **核心特性**:
    - 波動率自適應 TP/SL
    - 機率分層倉位
    - 時段差異化
    - 風控強化
    """
    
    def __init__(self,
                 model_long_path: str,
                 model_short_path: str,
                 initial_capital: float = 10000.0,
                 base_position_size_pct: float = 0.10,
                 prob_threshold_long: float = 0.16,
                 prob_threshold_short: float = 0.16,
                 base_tp_pct: float = 0.02,
                 base_sl_pct: float = 0.01,
                 hunting_expire_bars: int = 15,
                 trading_hours: Optional[List[tuple]] = None,
                 maker_fee: float = 0.0001,
                 taker_fee: float = 0.0004,
                 slippage: float = 0.0002,
                 # 進階參數
                 enable_volatility_adaptation: bool = True,
                 enable_probability_layering: bool = True,
                 enable_time_based_strategy: bool = True,
                 enable_risk_controls: bool = True,
                 max_daily_loss_pct: float = 0.03,
                 max_consecutive_losses: int = 5):
        
        logger.info("="*80)
        logger.info("INITIALIZING ADAPTIVE BACKTESTER")
        logger.info("="*80)
        
        # 載入模型
        self.model_long = joblib.load(model_long_path)
        self.model_short = joblib.load(model_short_path)
        
        # 提取特徵
        try:
            if hasattr(self.model_long, 'estimators_'):
                base_model_long = self.model_long.estimators_[0].estimator
                base_model_short = self.model_short.estimators_[0].estimator
            else:
                base_model_long = self.model_long
                base_model_short = self.model_short
            
            self.features_long = list(base_model_long.feature_names_)
            self.features_short = list(base_model_short.feature_names_)
            
            logger.info(f"Features loaded: {len(self.features_long)} (Long), {len(self.features_short)} (Short)")
        except Exception as e:
            logger.error(f"Failed to extract features: {e}")
            self.features_long = ['efficiency_ratio', 'extreme_time_diff', 'vol_imbalance_ratio',
                                 'z_score', 'bb_width_pct', 'rsi', 'atr_pct', 'z_score_1h', 'atr_pct_1d']
            self.features_short = self.features_long
        
        # 基礎參數
        self.initial_capital = initial_capital
        self.base_position_size_pct = base_position_size_pct
        self.capital = initial_capital
        self.prob_threshold_long = prob_threshold_long
        self.prob_threshold_short = prob_threshold_short
        self.base_tp_pct = base_tp_pct
        self.base_sl_pct = base_sl_pct
        self.hunting_expire_bars = hunting_expire_bars
        self.trading_hours = trading_hours or [(0, 24)]
        
        # 成本參數
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage
        
        # 進階功能開關
        self.enable_volatility_adaptation = enable_volatility_adaptation
        self.enable_probability_layering = enable_probability_layering
        self.enable_time_based_strategy = enable_time_based_strategy
        self.enable_risk_controls = enable_risk_controls
        
        # 風控參數
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.consecutive_losses = 0
        self.daily_pnl = 0
        self.current_date = None
        
        # 狀態
        self.state = AgentState.IDLE
        self.hunting_entry_bar = None
        self.limit_order_price = None
        self.entry_time = None
        self.entry_price = None
        self.position_size = None
        self.tp_price = None
        self.sl_price = None
        self.entry_prob = None
        self.current_position_size_pct = base_position_size_pct
        
        # 記錄
        self.trades: List[Trade] = []
        self.equity_curve = []
        
        logger.info(f"Adaptive Features: Vol={enable_volatility_adaptation}, "
                   f"ProbLayer={enable_probability_layering}, "
                   f"TimeBased={enable_time_based_strategy}, "
                   f"RiskCtrl={enable_risk_controls}")
        logger.info("="*80)
    
    def determine_volatility_regime(self, row: pd.Series) -> VolatilityRegime:
        """
        根據 ATR 判斷波動率狀態
        """
        atr = row.get('atr_pct_1d', row.get('atr_pct', 0.02))
        
        if atr < 0.02:
            return VolatilityRegime.LOW
        elif atr < 0.04:
            return VolatilityRegime.MEDIUM
        else:
            return VolatilityRegime.HIGH
    
    def get_adaptive_tp_sl(self, volatility_regime: VolatilityRegime) -> Tuple[float, float]:
        """
        根據波動率調整 TP/SL
        """
        if not self.enable_volatility_adaptation:
            return self.base_tp_pct, self.base_sl_pct
        
        if volatility_regime == VolatilityRegime.LOW:
            return 0.015, 0.0075  # 1.5% / 0.75%
        elif volatility_regime == VolatilityRegime.HIGH:
            return 0.025, 0.0125  # 2.5% / 1.25%
        else:
            return self.base_tp_pct, self.base_sl_pct  # 2.0% / 1.0%
    
    def get_adaptive_position_size(self, probability: float) -> float:
        """
        根據機率調整倉位
        """
        if not self.enable_probability_layering:
            return self.base_position_size_pct
        
        # 連續停損保護
        if self.consecutive_losses >= self.max_consecutive_losses:
            return self.base_position_size_pct * 0.5
        
        # 機率分層
        if probability >= 0.25:
            return self.base_position_size_pct * 1.5  # 15%
        elif probability >= 0.18:
            return self.base_position_size_pct  # 10%
        else:
            return self.base_position_size_pct * 0.5  # 5%
    
    def get_adaptive_threshold(self, hour: int, direction: str) -> float:
        """
        根據時段調整閾值
        """
        if not self.enable_time_based_strategy:
            return self.prob_threshold_long if direction == 'LONG' else self.prob_threshold_short
        
        # 歐洲時段 (09-13 UTC): 趨勢性較弱
        if 9 <= hour <= 13:
            return (self.prob_threshold_long if direction == 'LONG' else self.prob_threshold_short) * 0.95
        
        # 美國時段 (18-21 UTC): 波動更大
        elif 18 <= hour <= 21:
            return (self.prob_threshold_long if direction == 'LONG' else self.prob_threshold_short) * 1.05
        
        else:
            return self.prob_threshold_long if direction == 'LONG' else self.prob_threshold_short
    
    def get_adaptive_expiry(self, volatility_regime: VolatilityRegime) -> int:
        """
        根據波動率調整訂單過期時間
        """
        if volatility_regime == VolatilityRegime.HIGH:
            return 5  # 快速市場 5 分鐘
        elif volatility_regime == VolatilityRegime.LOW:
            return 20  # 慢速市場 20 分鐘
        else:
            return self.hunting_expire_bars
    
    def is_trading_hours(self, timestamp: pd.Timestamp) -> bool:
        hour = timestamp.hour
        for start, end in self.trading_hours:
            if start <= hour < end:
                return True
        return False
    
    def check_risk_controls(self, timestamp: pd.Timestamp) -> bool:
        """
        檢查風控限制
        """
        if not self.enable_risk_controls:
            return True
        
        # 更新日期
        current_date = timestamp.date()
        if self.current_date != current_date:
            self.current_date = current_date
            self.daily_pnl = 0
        
        # 最大日內虧損
        if self.daily_pnl < -self.initial_capital * self.max_daily_loss_pct:
            logger.warning(f"[{timestamp}] Daily loss limit reached: ${self.daily_pnl:.2f}")
            return False
        
        return True
    
    def check_correlation(self, prob_long: float, prob_short: float) -> bool:
        """
        檢查雙向相關性
        """
        if abs(prob_long - prob_short) < 0.05:
            return False  # 方向不明確
        return True
    
    def calculate_fees(self, trade_value: float, is_maker: bool = True) -> float:
        fee_rate = self.maker_fee if is_maker else (self.taker_fee + self.slippage)
        return trade_value * fee_rate
    
    def execute_trade(self, direction: str, entry_price: float, exit_price: float,
                     exit_reason: str, entry_time: pd.Timestamp, exit_time: pd.Timestamp,
                     entry_prob: float, volatility_regime: str, position_size_pct: float):
        
        if direction == 'LONG':
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price
        
        trade_value = self.position_size
        entry_fee = self.calculate_fees(trade_value, is_maker=True)
        exit_fee = self.calculate_fees(trade_value, is_maker=(exit_reason=='TP'))
        
        total_fees = entry_fee + exit_fee
        gross_pnl = self.position_size * pnl_pct
        net_pnl = gross_pnl - total_fees
        
        self.capital += net_pnl
        self.daily_pnl += net_pnl
        
        # 更新連續停損計數器
        if net_pnl <= 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
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
            probability=entry_prob,
            volatility_regime=volatility_regime,
            entry_hour=entry_time.hour,
            position_size_pct=position_size_pct
        )
        self.trades.append(trade)
        
        logger.info(f"TRADE: {direction} | Entry={entry_price:.2f} @ {entry_time} | "
                   f"Exit={exit_price:.2f} @ {exit_time} | Reason={exit_reason} | "
                   f"PnL={net_pnl:+.2f} ({pnl_pct*100:+.2f}%) | Prob={entry_prob:.3f} | "
                   f"Vol={volatility_regime} | Size={position_size_pct*100:.0f}% | "
                   f"Capital={self.capital:.2f} | ConsecLoss={self.consecutive_losses}")
    
    def process_bar(self, bar_idx: int, row: pd.Series, prob_long: float, prob_short: float):
        timestamp = row.name
        open_price = row['open']
        high_price = row['high']
        low_price = row['low']
        close_price = row['close']
        
        volatility_regime = self.determine_volatility_regime(row)
        
        # === IDLE ===
        if self.state == AgentState.IDLE:
            if not self.is_trading_hours(timestamp):
                return
            
            if not self.check_risk_controls(timestamp):
                return
            
            if not self.check_correlation(prob_long, prob_short):
                return
            
            # 獲取自適應閾值
            adaptive_threshold_long = self.get_adaptive_threshold(timestamp.hour, 'LONG')
            adaptive_threshold_short = self.get_adaptive_threshold(timestamp.hour, 'SHORT')
            
            if prob_long >= adaptive_threshold_long and prob_short < 0.10:
                self.state = AgentState.HUNTING_LONG
                self.hunting_entry_bar = bar_idx
                self.limit_order_price = close_price * 0.9995
                self.entry_prob = prob_long
                self.current_volatility_regime = volatility_regime
                self.current_position_size_pct = self.get_adaptive_position_size(prob_long)
                logger.info(f"[{timestamp}] IDLE -> HUNTING_LONG | prob={prob_long:.4f} | "
                           f"threshold={adaptive_threshold_long:.4f} | vol={volatility_regime.value}")
                return
            
            if prob_short >= adaptive_threshold_short and prob_long < 0.10:
                self.state = AgentState.HUNTING_SHORT
                self.hunting_entry_bar = bar_idx
                self.limit_order_price = close_price * 1.0005
                self.entry_prob = prob_short
                self.current_volatility_regime = volatility_regime
                self.current_position_size_pct = self.get_adaptive_position_size(prob_short)
                logger.info(f"[{timestamp}] IDLE -> HUNTING_SHORT | prob={prob_short:.4f} | "
                           f"threshold={adaptive_threshold_short:.4f} | vol={volatility_regime.value}")
                return
        
        # === HUNTING_LONG ===
        elif self.state == AgentState.HUNTING_LONG:
            adaptive_expiry = self.get_adaptive_expiry(self.current_volatility_regime)
            
            if bar_idx - self.hunting_entry_bar >= adaptive_expiry:
                logger.info(f"[{timestamp}] HUNTING_LONG expired -> IDLE")
                self.state = AgentState.IDLE
                return
            
            if low_price < self.limit_order_price:
                self.state = AgentState.LONG_POSITION
                self.entry_time = timestamp
                self.entry_price = self.limit_order_price
                self.position_size = self.capital * self.current_position_size_pct
                
                tp_pct, sl_pct = self.get_adaptive_tp_sl(self.current_volatility_regime)
                self.tp_price = self.entry_price * (1 + tp_pct)
                self.sl_price = self.entry_price * (1 - sl_pct)
                
                logger.info(f"[{timestamp}] HUNTING_LONG -> LONG_POSITION | "
                           f"entry={self.entry_price:.2f} | TP={self.tp_price:.2f} | SL={self.sl_price:.2f} | "
                           f"size={self.current_position_size_pct*100:.0f}%")
                return
        
        # === HUNTING_SHORT ===
        elif self.state == AgentState.HUNTING_SHORT:
            adaptive_expiry = self.get_adaptive_expiry(self.current_volatility_regime)
            
            if bar_idx - self.hunting_entry_bar >= adaptive_expiry:
                logger.info(f"[{timestamp}] HUNTING_SHORT expired -> IDLE")
                self.state = AgentState.IDLE
                return
            
            if high_price > self.limit_order_price:
                self.state = AgentState.SHORT_POSITION
                self.entry_time = timestamp
                self.entry_price = self.limit_order_price
                self.position_size = self.capital * self.current_position_size_pct
                
                tp_pct, sl_pct = self.get_adaptive_tp_sl(self.current_volatility_regime)
                self.tp_price = self.entry_price * (1 - tp_pct)
                self.sl_price = self.entry_price * (1 + sl_pct)
                
                logger.info(f"[{timestamp}] HUNTING_SHORT -> SHORT_POSITION | "
                           f"entry={self.entry_price:.2f} | TP={self.tp_price:.2f} | SL={self.sl_price:.2f} | "
                           f"size={self.current_position_size_pct*100:.0f}%")
                return
        
        # === LONG_POSITION ===
        elif self.state == AgentState.LONG_POSITION:
            tp_hit = high_price >= self.tp_price
            sl_hit = low_price <= self.sl_price
            
            if tp_hit and sl_hit:
                self.execute_trade('LONG', self.entry_price, self.sl_price, 'SL',
                                  self.entry_time, timestamp, self.entry_prob,
                                  self.current_volatility_regime.value, self.current_position_size_pct)
                self.state = AgentState.IDLE
            elif tp_hit:
                self.execute_trade('LONG', self.entry_price, self.tp_price, 'TP',
                                  self.entry_time, timestamp, self.entry_prob,
                                  self.current_volatility_regime.value, self.current_position_size_pct)
                self.state = AgentState.IDLE
            elif sl_hit:
                self.execute_trade('LONG', self.entry_price, self.sl_price, 'SL',
                                  self.entry_time, timestamp, self.entry_prob,
                                  self.current_volatility_regime.value, self.current_position_size_pct)
                self.state = AgentState.IDLE
        
        # === SHORT_POSITION ===
        elif self.state == AgentState.SHORT_POSITION:
            tp_hit = low_price <= self.tp_price
            sl_hit = high_price >= self.sl_price
            
            if tp_hit and sl_hit:
                self.execute_trade('SHORT', self.entry_price, self.sl_price, 'SL',
                                  self.entry_time, timestamp, self.entry_prob,
                                  self.current_volatility_regime.value, self.current_position_size_pct)
                self.state = AgentState.IDLE
            elif tp_hit:
                self.execute_trade('SHORT', self.entry_price, self.tp_price, 'TP',
                                  self.entry_time, timestamp, self.entry_prob,
                                  self.current_volatility_regime.value, self.current_position_size_pct)
                self.state = AgentState.IDLE
            elif sl_hit:
                self.execute_trade('SHORT', self.entry_price, self.sl_price, 'SL',
                                  self.entry_time, timestamp, self.entry_prob,
                                  self.current_volatility_regime.value, self.current_position_size_pct)
                self.state = AgentState.IDLE
    
    def run(self, df_test: pd.DataFrame, feature_cols: List[str]) -> Dict:
        logger.info("="*80)
        logger.info("STARTING ADAPTIVE BACKTEST")
        logger.info("="*80)
        
        # 批次預測
        X_long = df_test[self.features_long].fillna(0).values
        X_short = df_test[self.features_short].fillna(0).values
        
        df_test['prob_long'] = self.model_long.predict_proba(X_long)[:, 1]
        df_test['prob_short'] = self.model_short.predict_proba(X_short)[:, 1]
        
        logger.info(f"Long prob: max={df_test['prob_long'].max():.4f}, mean={df_test['prob_long'].mean():.4f}")
        logger.info(f"Short prob: max={df_test['prob_short'].max():.4f}, mean={df_test['prob_short'].mean():.4f}")
        logger.info("="*80)
        
        # 事件迴圈
        for bar_idx, (timestamp, row) in enumerate(df_test.iterrows()):
            prob_long = row['prob_long']
            prob_short = row['prob_short']
            
            self.process_bar(bar_idx, row, prob_long, prob_short)
            
            self.equity_curve.append({
                'timestamp': timestamp,
                'capital': self.capital,
                'state': self.state.value
            })
            
            if (bar_idx + 1) % 10000 == 0:
                logger.info(f"Progress: {bar_idx+1:,}/{len(df_test):,} ({(bar_idx+1)/len(df_test)*100:.1f}%)")
        
        results = self.calculate_metrics()
        return results
    
    def calculate_metrics(self) -> Dict:
        if len(self.trades) == 0:
            return {'total_trades': 0}
        
        total_trades = len(self.trades)
        winning = [t for t in self.trades if t.pnl_net > 0]
        losing = [t for t in self.trades if t.pnl_net <= 0]
        
        win_rate = len(winning) / total_trades
        total_pnl = sum(t.pnl_net for t in self.trades)
        total_return_pct = (self.capital - self.initial_capital) / self.initial_capital
        
        avg_win = np.mean([t.pnl_net for t in winning]) if winning else 0
        avg_loss = np.mean([t.pnl_net for t in losing]) if losing else 0
        profit_factor = abs(sum(t.pnl_net for t in winning) / sum(t.pnl_net for t in losing)) if losing else np.inf
        
        long_trades = [t for t in self.trades if t.direction == 'LONG']
        short_trades = [t for t in self.trades if t.direction == 'SHORT']
        
        tp_trades = [t for t in self.trades if t.exit_reason == 'TP']
        sl_trades = [t for t in self.trades if t.exit_reason == 'SL']
        
        results = {
            'total_trades': total_trades,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'final_capital': self.capital,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'tp_rate': len(tp_trades) / total_trades,
            'sl_rate': len(sl_trades) / total_trades,
            'tp_count': len(tp_trades),
            'sl_count': len(sl_trades)
        }
        
        logger.info("="*80)
        logger.info("ADAPTIVE BACKTEST RESULTS")
        logger.info("="*80)
        logger.info(f"Total Trades: {total_trades} (Long: {len(long_trades)}, Short: {len(short_trades)})")
        logger.info(f"Win Rate: {win_rate*100:.2f}%")
        logger.info(f"Total Return: {total_return_pct*100:+.2f}%")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info("="*80)
        
        return results
    
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
            'probability': t.probability,
            'volatility_regime': t.volatility_regime,
            'entry_hour': t.entry_hour,
            'position_size_pct': t.position_size_pct
        } for t in self.trades])