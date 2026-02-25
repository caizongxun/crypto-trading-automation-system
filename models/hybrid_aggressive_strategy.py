#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合激進策略 - 30天翻倉目標

結合 Chronos + XGBoost v3 + 複利管理
目標: 30天內資金翻倍
策略: 高頻 + 動態倒金字塔 + 激進風控
"""

import sys
import os
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class HybridAggressiveStrategy:
    """
    混合激進策略
    
    特點:
    1. 雙模型信號融合 (Chronos + XGBoost)
    2. 動態位置管理 (複利加仓)
    3. 倒金字塔機制 (損失後加倍)
    4. 激進風控 (20-50% 每筆)
    5. 快速止盈/止損 (0.8-1.5%)
    """
    
    def __init__(
        self,
        initial_capital: float = 10000,
        target_multiplier: float = 2.0,  # 30天翻倍
        target_days: int = 30,
        base_position_pct: float = 20.0,  # 基礎倉位 20%
        max_position_pct: float = 50.0,   # 最大倉位 50%
        use_martingale: bool = True,      # 使用倒金字塔
        max_martingale_level: int = 3,    # 最大加倍次數
    ):
        self.initial_capital = initial_capital
        self.target_multiplier = target_multiplier
        self.target_days = target_days
        self.base_position_pct = base_position_pct
        self.max_position_pct = max_position_pct
        self.use_martingale = use_martingale
        self.max_martingale_level = max_martingale_level
        
        # 狀態追蹤
        self.current_capital = initial_capital
        self.consecutive_losses = 0
        self.total_trades = 0
        self.winning_trades = 0
        
        # 計算目標
        self.daily_target = self._calculate_daily_target()
        
        logger.info(f"Initialized HybridAggressiveStrategy")
        logger.info(f"Target: ${initial_capital:,.0f} -> ${initial_capital * target_multiplier:,.0f} in {target_days} days")
        logger.info(f"Daily target return: {self.daily_target:.2f}%")
    
    
    def _calculate_daily_target(self) -> float:
        """
        計算每日目標報酬率 (複利)
        
        公式: (1 + r)^n = target_multiplier
        r = target_multiplier^(1/n) - 1
        """
        daily_multiplier = self.target_multiplier ** (1 / self.target_days)
        daily_return = (daily_multiplier - 1) * 100
        return daily_return
    
    
    def calculate_position_size(self, is_winning_streak: bool = False) -> float:
        """
        計算位置大小 (使用倒金字塔)
        
        Args:
            is_winning_streak: 是否連勝
        
        Returns:
            倉位百分比 (%)
        """
        if not self.use_martingale:
            return self.base_position_pct
        
        # 連輸後加倍倉位
        if self.consecutive_losses > 0:
            martingale_level = min(self.consecutive_losses, self.max_martingale_level)
            position_pct = self.base_position_pct * (2 ** martingale_level)
            position_pct = min(position_pct, self.max_position_pct)
            logger.info(f"[MARTINGALE] Level {martingale_level}: {position_pct:.1f}% position")
            return position_pct
        
        # 連勝後加倉 (Anti-Martingale)
        elif is_winning_streak and self.winning_trades >= 3:
            position_pct = min(self.base_position_pct * 1.5, self.max_position_pct)
            logger.info(f"[WINNING] Winning streak: {position_pct:.1f}% position")
            return position_pct
        
        return self.base_position_pct
    
    
    def get_dynamic_tp_sl(self, volatility: float = 1.0) -> Tuple[float, float]:
        """
        動態 TP/SL (根據波動性調整)
        
        Args:
            volatility: 波動性係數 (ATR / 價格)
        
        Returns:
            (tp_pct, sl_pct)
        """
        # 基礎值: 激進型 TP/SL
        base_tp = 1.2  # 1.2% TP
        base_sl = 0.6  # 0.6% SL
        
        # 根據波動性調整
        if volatility > 1.5:
            # 高波動: 放寬 TP/SL
            tp_pct = base_tp * 1.3
            sl_pct = base_sl * 1.3
        elif volatility < 0.7:
            # 低波動: 縮緊 TP/SL
            tp_pct = base_tp * 0.8
            sl_pct = base_sl * 0.8
        else:
            tp_pct = base_tp
            sl_pct = base_sl
        
        return tp_pct, sl_pct
    
    
    def combine_signals(
        self,
        chronos_prob_long: float,
        chronos_prob_short: float,
        xgb_prob_long: float,
        xgb_prob_short: float,
        use_aggressive: bool = True
    ) -> Tuple[str, float, str]:
        """
        融合雙模型信號 (激進版)
        
        Args:
            chronos_prob_long: Chronos 看多機率
            chronos_prob_short: Chronos 看空機率
            xgb_prob_long: XGBoost 看多機率
            xgb_prob_short: XGBoost 看空機率
            use_aggressive: 使用激進模式
        
        Returns:
            (signal, confidence, reason)
            signal: 'LONG', 'SHORT', 'HOLD'
            confidence: 0.0-1.0
            reason: 信號來源
        """
        # 加權平均 (Chronos 權重較高)
        chronos_weight = 0.6
        xgb_weight = 0.4
        
        combined_long = chronos_prob_long * chronos_weight + xgb_prob_long * xgb_weight
        combined_short = chronos_prob_short * chronos_weight + xgb_prob_short * xgb_weight
        
        if use_aggressive:
            # 激進模式: 降低門檻,增加交易頻率
            long_threshold = 0.08   # 非常低
            short_threshold = 0.08
            strong_threshold = 0.15  # 強信號
        else:
            # 保守模式
            long_threshold = 0.12
            short_threshold = 0.12
            strong_threshold = 0.20
        
        # 判斷信號
        if combined_long > strong_threshold and chronos_prob_long > 0.12 and xgb_prob_long > 0.15:
            # 雙模型強一致 - 高信心
            confidence = min(combined_long, 0.95)
            reason = f"BOTH_STRONG (C:{chronos_prob_long:.2f} X:{xgb_prob_long:.2f})"
            return 'LONG', confidence, reason
        
        elif combined_short > strong_threshold and chronos_prob_short > 0.12 and xgb_prob_short > 0.15:
            confidence = min(combined_short, 0.95)
            reason = f"BOTH_STRONG (C:{chronos_prob_short:.2f} X:{xgb_prob_short:.2f})"
            return 'SHORT', confidence, reason
        
        elif combined_long > long_threshold:
            # 单模型或弱一致 - 中等信心
            if chronos_prob_long > xgb_prob_long:
                reason = f"CHRONOS_LEAD (C:{chronos_prob_long:.2f} X:{xgb_prob_long:.2f})"
            else:
                reason = f"XGB_LEAD (C:{chronos_prob_long:.2f} X:{xgb_prob_long:.2f})"
            confidence = min(combined_long, 0.75)
            return 'LONG', confidence, reason
        
        elif combined_short > short_threshold:
            if chronos_prob_short > xgb_prob_short:
                reason = f"CHRONOS_LEAD (C:{chronos_prob_short:.2f} X:{xgb_prob_short:.2f})"
            else:
                reason = f"XGB_LEAD (C:{chronos_prob_short:.2f} X:{xgb_prob_short:.2f})"
            confidence = min(combined_short, 0.75)
            return 'SHORT', confidence, reason
        
        # 沒有明確信號
        return 'HOLD', 0.0, f"NO_SIGNAL (L:{combined_long:.2f} S:{combined_short:.2f})"
    
    
    def update_after_trade(self, pnl: float, is_win: bool):
        """
        交易後更新狀態
        
        Args:
            pnl: 盈虧 (%)
            is_win: 是否獲利
        """
        self.total_trades += 1
        
        if is_win:
            self.winning_trades += 1
            self.consecutive_losses = 0
            logger.info(f"[WIN] Win #{self.winning_trades}/{self.total_trades} | PnL: +{pnl:.2f}%")
        else:
            self.consecutive_losses += 1
            logger.warning(f"[LOSS] Loss (streak: {self.consecutive_losses}) | PnL: {pnl:.2f}%")
        
        # 更新資金
        self.current_capital *= (1 + pnl / 100)
        
        # 進度查詢
        days_elapsed = self.total_trades / 10  # 假設每天 10 筆交易
        progress = (self.current_capital / self.initial_capital - 1) * 100
        target_progress = self.daily_target * days_elapsed
        
        if days_elapsed > 0:
            logger.info(f"[PROGRESS] Progress: {progress:.1f}% (Target: {target_progress:.1f}%)")
            logger.info(f"[CAPITAL] Capital: ${self.current_capital:,.2f} / ${self.initial_capital * self.target_multiplier:,.2f}")
    
    
    def get_stats(self) -> Dict[str, Any]:
        """獲取統計資訊"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        total_return = (self.current_capital / self.initial_capital - 1) * 100
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'current_capital': self.current_capital,
            'target_capital': self.initial_capital * self.target_multiplier,
            'consecutive_losses': self.consecutive_losses,
            'daily_target': self.daily_target
        }


def example_usage():
    """使用範例"""
    # 初始化策略
    strategy = HybridAggressiveStrategy(
        initial_capital=10000,
        target_multiplier=2.0,  # 30天翻倍
        target_days=30,
        base_position_pct=20.0,
        max_position_pct=50.0,
        use_martingale=True
    )
    
    # 模擬交易
    chronos_long = 0.18
    chronos_short = 0.08
    xgb_long = 0.22
    xgb_short = 0.12
    
    # 獲取信號
    signal, confidence, reason = strategy.combine_signals(
        chronos_long, chronos_short,
        xgb_long, xgb_short,
        use_aggressive=True
    )
    
    print(f"Signal: {signal}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Reason: {reason}")
    
    # 計算位置大小
    position_pct = strategy.calculate_position_size()
    print(f"Position: {position_pct:.1f}%")
    
    # 獲取動態 TP/SL
    tp, sl = strategy.get_dynamic_tp_sl(volatility=1.2)
    print(f"TP/SL: {tp:.2f}% / {sl:.2f}%")


if __name__ == "__main__":
    example_usage()
