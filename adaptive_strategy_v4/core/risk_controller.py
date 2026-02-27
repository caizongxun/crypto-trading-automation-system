"""
Multi-Layer Risk Controller
多層風險控制器

功能:
1. Kelly值門檻過濾
2. 倉位上限控制
3. 連敗保護
4. 波動率調整
5. 總曝險管理
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

class RiskController:
    """
    多層風險控制器
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 配置字典
                - max_single_position: 單筆最大倉位 (default: 0.20)
                - max_total_exposure: 總曝險上限 (default: 0.50)
                - min_kelly_threshold: Kelly最小門檻 (default: 0.10)
                - max_losing_streak: 最大允許連敗 (default: 3)
                - drawdown_limit: 回撤限制 (default: 0.30)
                - volatility_limit: 波動率限制 (default: 0.05)
        """
        self.max_single_position = config.get('max_single_position', 0.20)
        self.max_total_exposure = config.get('max_total_exposure', 0.50)
        self.min_kelly_threshold = config.get('min_kelly_threshold', 0.10)
        self.max_losing_streak = config.get('max_losing_streak', 3)
        self.drawdown_limit = config.get('drawdown_limit', 0.30)
        self.volatility_limit = config.get('volatility_limit', 0.05)
        
        # 狀態追蹤
        self.current_exposure = 0.0
        self.losing_streak = 0
        self.max_equity = 0
        self.current_equity = 0
        self.recent_returns = []
    
    def check_signal(self, 
                    kelly_pct: float,
                    position_size: float,
                    confidence: float,
                    capital: float) -> Tuple[bool, str, float]:
        """
        檢查信號是否通過風險控制
        
        Args:
            kelly_pct: Kelly百分比
            position_size: 建議倉位大小
            confidence: 信心度
            capital: 當前資金
        
        Returns:
            (is_approved, reason, adjusted_position_size)
        """
        adjusted_size = position_size
        
        # 1. Kelly門檻檢查
        if kelly_pct < self.min_kelly_threshold:
            return False, f"kelly_below_threshold_{kelly_pct:.3f}", 0
        
        # 2. 單筆倉位限制
        max_allowed = capital * self.max_single_position
        if adjusted_size > max_allowed:
            adjusted_size = max_allowed
        
        # 3. 總曝險檢查
        potential_exposure = (self.current_exposure + adjusted_size / capital)
        if potential_exposure > self.max_total_exposure:
            return False, f"total_exposure_exceeded_{potential_exposure:.2f}", 0
        
        # 4. 連敗保護
        if self.losing_streak >= self.max_losing_streak:
            # 減小倉位到原本的50%
            adjusted_size *= 0.5
            if self.losing_streak >= self.max_losing_streak + 2:
                return False, f"excessive_losing_streak_{self.losing_streak}", 0
        
        # 5. 回撤保護
        if self.max_equity > 0:
            current_drawdown = (self.max_equity - self.current_equity) / self.max_equity
            if current_drawdown > self.drawdown_limit:
                # 回撤太大,減小倉位
                adjusted_size *= 0.5
                if current_drawdown > self.drawdown_limit * 1.2:
                    return False, f"drawdown_limit_exceeded_{current_drawdown:.2f}", 0
        
        # 6. 波動率檢查
        if len(self.recent_returns) >= 20:
            volatility = np.std(self.recent_returns[-20:])
            if volatility > self.volatility_limit:
                # 高波動時減小倉位
                vol_factor = self.volatility_limit / volatility
                adjusted_size *= vol_factor
        
        # 7. 信心度調整
        if confidence < 0.6:
            adjusted_size *= 0.8
        elif confidence < 0.5:
            return False, f"low_confidence_{confidence:.3f}", 0
        
        return True, "approved", adjusted_size
    
    def update_position(self, position_value: float, capital: float):
        """
        更新當前倉位
        
        Args:
            position_value: 新建倉位價值
            capital: 當前資金
        """
        self.current_exposure += position_value / capital
    
    def close_position(self, position_value: float, capital: float, pnl: float):
        """
        關閉倉位
        
        Args:
            position_value: 倉位價值
            capital: 當前資金
            pnl: 盈虧
        """
        self.current_exposure -= position_value / capital
        self.current_exposure = max(0, self.current_exposure)  # 防止負值
        
        # 更新連敗/連勝
        if pnl > 0:
            self.losing_streak = 0
        else:
            self.losing_streak += 1
        
        # 記錄報酬率
        return_pct = pnl / capital
        self.recent_returns.append(return_pct)
        
        # 保持最近50筆記錄
        if len(self.recent_returns) > 50:
            self.recent_returns = self.recent_returns[-50:]
    
    def update_equity(self, equity: float):
        """
        更新權益
        
        Args:
            equity: 當前權益
        """
        self.current_equity = equity
        if equity > self.max_equity:
            self.max_equity = equity
    
    def get_current_drawdown(self) -> float:
        """
        獲取當前回撤
        
        Returns:
            回撤百分比
        """
        if self.max_equity == 0:
            return 0
        return (self.max_equity - self.current_equity) / self.max_equity
    
    def get_current_volatility(self) -> float:
        """
        獲取當前波動率
        
        Returns:
            波動率
        """
        if len(self.recent_returns) < 10:
            return 0
        return np.std(self.recent_returns)
    
    def should_pause_trading(self) -> Tuple[bool, str]:
        """
        檢查是否應該暂停交易
        
        Returns:
            (should_pause, reason)
        """
        # 1. 連敗太多
        if self.losing_streak >= self.max_losing_streak + 3:
            return True, f"excessive_losing_streak_{self.losing_streak}"
        
        # 2. 回撤過大
        drawdown = self.get_current_drawdown()
        if drawdown > self.drawdown_limit * 1.3:
            return True, f"excessive_drawdown_{drawdown:.2f}"
        
        # 3. 波動率過高
        volatility = self.get_current_volatility()
        if volatility > self.volatility_limit * 2:
            return True, f"excessive_volatility_{volatility:.3f}"
        
        return False, ""
    
    def get_risk_adjustment_factor(self) -> float:
        """
        獲取風險調整因子
        
        根據當前狀態調整倉位大小
        
        Returns:
            調整因子 (0-1)
        """
        factor = 1.0
        
        # 根據連敗調整
        if self.losing_streak > 0:
            factor *= (1 - 0.1 * self.losing_streak)
        
        # 根據回撤調整
        drawdown = self.get_current_drawdown()
        if drawdown > 0.1:
            factor *= (1 - drawdown * 0.5)
        
        # 根據波動率調整
        volatility = self.get_current_volatility()
        if volatility > self.volatility_limit * 0.5:
            factor *= (self.volatility_limit / volatility)
        
        return max(0.2, min(1.0, factor))  # 限制在 0.2-1.0
    
    def get_statistics(self) -> Dict:
        """
        獲取風險統計信息
        
        Returns:
            統計字典
        """
        return {
            'current_exposure': self.current_exposure,
            'losing_streak': self.losing_streak,
            'current_drawdown': self.get_current_drawdown(),
            'current_volatility': self.get_current_volatility(),
            'max_equity': self.max_equity,
            'current_equity': self.current_equity,
            'risk_adjustment_factor': self.get_risk_adjustment_factor(),
            'should_pause': self.should_pause_trading()[0]
        }
