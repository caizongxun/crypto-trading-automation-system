"""
Kelly Criterion Position Manager
Kelly準則倉位管理器

核心功能:
1. Kelly公式倉位計算
2. 分數Kelly風險控制
3. 動態參數調整
4. 槓桿優化
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

class KellyManager:
    """
    Kelly準則倉位管理器
    
    Kelly公式: Kelly% = (p * b - q) / b
    其中:
    - p: 勝率
    - q: 1 - p (敗率)
    - b: 平均盈利 / 平均虧損 (賠率)
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 配置字典
                - kelly_fraction: Kelly分數 (default: 0.25)
                - max_position: 最大兩位百分比 (default: 0.20)
                - min_kelly: 最小Kelly值門檻 (default: 0.10)
                - max_leverage: 最大槓桿 (default: 3)
                - win_rate_window: 勝率計算窗口 (default: 50)
        """
        self.kelly_fraction = config.get('kelly_fraction', 0.25)
        self.max_position = config.get('max_position', 0.20)
        self.min_kelly = config.get('min_kelly', 0.10)
        self.max_leverage = config.get('max_leverage', 3)
        self.win_rate_window = config.get('win_rate_window', 50)
        
        # 歷史記錄
        self.trade_history = []
        self.recent_win_rate = 0.5
        self.recent_payoff_ratio = 1.5
    
    def calculate_position_size(self, 
                               predicted_win_rate: float,
                               predicted_payoff: float,
                               confidence: float,
                               capital: float) -> Tuple[float, float, str]:
        """
        計算Kelly最優倉位
        
        Args:
            predicted_win_rate: 預測勝率
            predicted_payoff: 預測賠率 (平均盈利/平均虧損)
            confidence: 信心度 (0-1)
            capital: 當前資金
        
        Returns:
            (position_size, leverage, reason)
        """
        # 1. 基礎Kelly計算
        p = predicted_win_rate
        q = 1 - p
        b = predicted_payoff
        
        # Kelly百分比
        if b <= 0:
            return 0, 1, "invalid_payoff"
        
        kelly_pct = (p * b - q) / b
        
        # 2. 調整為分數Kelly
        adjusted_kelly = kelly_pct * self.kelly_fraction
        
        # 3. 根據信心度調整
        confidence_adjusted = adjusted_kelly * confidence
        
        # 4. 應用限制
        if confidence_adjusted < self.min_kelly:
            return 0, 1, f"kelly_too_low_{confidence_adjusted:.3f}"
        
        final_position = min(confidence_adjusted, self.max_position)
        
        # 5. 計算槓桿 (根據Kelly值和信心度)
        if kelly_pct > 0.4 and confidence > 0.7:
            leverage = min(3, self.max_leverage)
        elif kelly_pct > 0.3 and confidence > 0.6:
            leverage = min(2, self.max_leverage)
        else:
            leverage = 1
        
        position_value = capital * final_position * leverage
        
        return position_value, leverage, "success"
    
    def update_from_trade(self, pnl: float, is_win: bool, 
                         entry_price: float, exit_price: float):
        """
        更新交易歷史並重新計算參數
        
        Args:
            pnl: 盈虧
            is_win: 是否盈利
            entry_price: 進場價
            exit_price: 出場價
        """
        self.trade_history.append({
            'pnl': pnl,
            'is_win': is_win,
            'return_pct': abs(exit_price - entry_price) / entry_price
        })
        
        # 保持最近N筆記錄
        if len(self.trade_history) > self.win_rate_window:
            self.trade_history = self.trade_history[-self.win_rate_window:]
        
        # 重新計算歷史勝率和賠率
        if len(self.trade_history) >= 10:
            wins = [t for t in self.trade_history if t['is_win']]
            losses = [t for t in self.trade_history if not t['is_win']]
            
            self.recent_win_rate = len(wins) / len(self.trade_history)
            
            if len(wins) > 0 and len(losses) > 0:
                avg_win = np.mean([t['return_pct'] for t in wins])
                avg_loss = np.mean([t['return_pct'] for t in losses])
                self.recent_payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 1.5
    
    def get_dynamic_parameters(self) -> Dict:
        """
        獲取當前動態參數
        
        Returns:
            包含歷史勝率和賠率的字典
        """
        return {
            'historical_win_rate': self.recent_win_rate,
            'historical_payoff': self.recent_payoff_ratio,
            'trade_count': len(self.trade_history),
            'kelly_fraction': self.kelly_fraction,
            'max_position': self.max_position
        }
    
    def calculate_optimal_kelly_fraction(self) -> float:
        """
        根據最近表現動態調整Kelly分數
        
        Returns:
            優化後的Kelly分數
        """
        if len(self.trade_history) < 30:
            return self.kelly_fraction
        
        # 計算最近30筆的波動率
        recent_returns = [t['return_pct'] for t in self.trade_history[-30:]]
        volatility = np.std(recent_returns)
        
        # 波動率越高,Kelly分數越低
        if volatility > 0.05:  # 5%以上波動
            return min(0.2, self.kelly_fraction)
        elif volatility > 0.03:  # 3-5%波動
            return min(0.25, self.kelly_fraction)
        else:
            return min(0.3, self.kelly_fraction)
    
    def should_reduce_exposure(self) -> bool:
        """
        檢查是否應該減少曝險
        
        Returns:
            True 如果應該減少倉位
        """
        if len(self.trade_history) < 10:
            return False
        
        # 檢查連敗
        recent_10 = self.trade_history[-10:]
        losing_streak = 0
        for trade in reversed(recent_10):
            if not trade['is_win']:
                losing_streak += 1
            else:
                break
        
        # 3連敗或更多
        if losing_streak >= 3:
            return True
        
        # 檢查最近10筆勝率
        recent_win_rate = sum(1 for t in recent_10 if t['is_win']) / len(recent_10)
        if recent_win_rate < 0.4:
            return True
        
        return False
    
    def get_statistics(self) -> Dict:
        """
        獲取統計信息
        
        Returns:
            包含各種統計指標的字典
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'payoff_ratio': 0,
                'optimal_kelly': 0
            }
        
        wins = [t for t in self.trade_history if t['is_win']]
        losses = [t for t in self.trade_history if not t['is_win']]
        
        win_rate = len(wins) / len(self.trade_history)
        avg_win = np.mean([t['return_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['return_pct'] for t in losses]) if losses else 0
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # 計算理論最優Kelly
        if payoff_ratio > 0:
            theoretical_kelly = (win_rate * payoff_ratio - (1 - win_rate)) / payoff_ratio
        else:
            theoretical_kelly = 0
        
        return {
            'total_trades': len(self.trade_history),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'payoff_ratio': payoff_ratio,
            'optimal_kelly': theoretical_kelly,
            'current_kelly_fraction': self.kelly_fraction
        }
