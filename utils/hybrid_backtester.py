#!/usr/bin/env python3
"""
混合回測引擎

結合 Chronos + XGBoost v3 進行回測
支援激進策略、動態仓位、倒金字塔
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class HybridBacktester:
    """
    混合回測引擎
    """
    
    def __init__(
        self,
        initial_capital: float = 10000,
        leverage: float = 1.0,
        fee_rate: float = 0.0004,  # 0.04% 手續費
    ):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.fee_rate = fee_rate
        
        # 狀態
        self.capital = initial_capital
        self.position = None
        self.trades = []
    
    
    def load_models(self, xgb_long_path: str, xgb_short_path: str):
        """
        載入 XGBoost 模型
        
        Args:
            xgb_long_path: Long 模型路徑
            xgb_short_path: Short 模型路徑
        """
        logger.info(f"Loading XGBoost models...")
        
        # 載入 Long 模型
        with open(xgb_long_path, 'rb') as f:
            loaded = pickle.load(f)
            # 如果是 dict (v3 format),提取模型
            if isinstance(loaded, dict):
                self.xgb_long = loaded.get('model', loaded.get('calibrated_model'))
            else:
                self.xgb_long = loaded
        
        # 載入 Short 模型
        with open(xgb_short_path, 'rb') as f:
            loaded = pickle.load(f)
            # 如果是 dict (v3 format),提取模型
            if isinstance(loaded, dict):
                self.xgb_short = loaded.get('model', loaded.get('calibrated_model'))
            else:
                self.xgb_short = loaded
        
        logger.info("✅ XGBoost models loaded")
    
    
    def predict_xgboost(self, features: pd.DataFrame) -> tuple:
        """
        XGBoost 預測
        
        Args:
            features: 特徵 DataFrame
        
        Returns:
            (prob_long, prob_short)
        """
        try:
            prob_long = self.xgb_long.predict_proba(features)[0][1]
            prob_short = self.xgb_short.predict_proba(features)[0][1]
            return float(prob_long), float(prob_short)
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            return 0.05, 0.05
    
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        df_chronos: pd.DataFrame,
        df_features: pd.DataFrame,
        strategy_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        執行混合回測
        
        Args:
            df: 原始 K 線資料
            df_chronos: Chronos 預測結果 (prob_long, prob_short)
            df_features: XGBoost 特徵
            strategy_config: 策略配置
        
        Returns:
            回測結果
        """
        from models.hybrid_aggressive_strategy import HybridAggressiveStrategy
        
        # 初始化策略
        strategy = HybridAggressiveStrategy(**strategy_config)
        
        # 重置狀態
        self.capital = self.initial_capital
        self.position = None
        self.trades = []
        
        logger.info(f"\nStarting hybrid backtest...")
        logger.info(f"Data: {len(df)} bars")
        logger.info(f"Strategy: {strategy_config}")
        
        # 回測迴圈
        for i in range(len(df)):
            bar = df.iloc[i]
            
            # 獲取 Chronos 預測
            if i < len(df_chronos) and not pd.isna(df_chronos.iloc[i]['prob_long']):
                chronos_long = df_chronos.iloc[i]['prob_long']
                chronos_short = df_chronos.iloc[i]['prob_short']
            else:
                chronos_long = 0.05
                chronos_short = 0.05
            
            # 獲取 XGBoost 預測
            if i < len(df_features) and not df_features.iloc[i].isna().any():
                features = df_features.iloc[i:i+1]
                xgb_long, xgb_short = self.predict_xgboost(features)
            else:
                xgb_long = 0.05
                xgb_short = 0.05
            
            # 融合信號
            signal, confidence, reason = strategy.combine_signals(
                chronos_long, chronos_short,
                xgb_long, xgb_short,
                use_aggressive=True
            )
            
            # 無持倉,檢查開倉
            if self.position is None:
                if signal == 'LONG':
                    self._open_position(
                        bar, 'LONG', strategy, confidence, 
                        chronos_long, xgb_long, reason
                    )
                elif signal == 'SHORT':
                    self._open_position(
                        bar, 'SHORT', strategy, confidence,
                        chronos_short, xgb_short, reason
                    )
            
            # 有持倉,檢查出倉
            else:
                self._check_exit(bar, strategy)
        
        # 強制平倉
        if self.position is not None:
            self._force_close(df.iloc[-1])
        
        # 統計結果
        return self._calculate_stats(strategy)
    
    
    def _open_position(
        self,
        bar: pd.Series,
        side: str,
        strategy,
        confidence: float,
        chronos_prob: float,
        xgb_prob: float,
        reason: str
    ):
        """開倉"""
        # 計算位置大小
        position_pct = strategy.calculate_position_size()
        position_value = self.capital * (position_pct / 100) * self.leverage
        
        # 計算手續費
        fee = position_value * self.fee_rate
        self.capital -= fee
        
        # 動態 TP/SL
        tp_pct, sl_pct = strategy.get_dynamic_tp_sl()
        
        # 記錄持倉
        self.position = {
            'side': side,
            'entry_price': bar['close'],
            'entry_time': bar['open_time'],
            'entry_capital': self.capital,
            'position_value': position_value,
            'position_pct': position_pct,
            'confidence': confidence,
            'chronos_prob': chronos_prob,
            'xgb_prob': xgb_prob,
            'reason': reason,
            'tp': bar['close'] * (1 + tp_pct/100 if side == 'LONG' else 1 - tp_pct/100),
            'sl': bar['close'] * (1 - sl_pct/100 if side == 'LONG' else 1 + sl_pct/100),
            'tp_pct': tp_pct,
            'sl_pct': sl_pct,
            'fee_open': fee
        }
        
        if len(self.trades) % 50 == 0:
            logger.info(f"\n🟢 Trade #{len(self.trades)+1}: {side}")
            logger.info(f"   Price: ${bar['close']:.2f}")
            logger.info(f"   Position: {position_pct:.1f}% (${position_value:,.0f})")
            logger.info(f"   TP/SL: {tp_pct:.2f}% / {sl_pct:.2f}%")
            logger.info(f"   Confidence: {confidence:.2%}")
            logger.info(f"   Reason: {reason}")
    
    
    def _check_exit(self, bar: pd.Series, strategy):
        """檢查出倉條件"""
        if self.position is None:
            return
        
        exit_reason = None
        exit_price = None
        
        if self.position['side'] == 'LONG':
            # TP
            if bar['high'] >= self.position['tp']:
                exit_reason = 'TP'
                exit_price = self.position['tp']
            # SL
            elif bar['low'] <= self.position['sl']:
                exit_reason = 'SL'
                exit_price = self.position['sl']
        
        else:  # SHORT
            # TP
            if bar['low'] <= self.position['tp']:
                exit_reason = 'TP'
                exit_price = self.position['tp']
            # SL
            elif bar['high'] >= self.position['sl']:
                exit_reason = 'SL'
                exit_price = self.position['sl']
        
        if exit_reason:
            self._close_position(bar, exit_price, exit_reason, strategy)
    
    
    def _close_position(self, bar: pd.Series, exit_price: float, exit_reason: str, strategy):
        """平倉"""
        # 計算 PnL
        if self.position['side'] == 'LONG':
            price_change = (exit_price - self.position['entry_price']) / self.position['entry_price']
        else:
            price_change = (self.position['entry_price'] - exit_price) / self.position['entry_price']
        
        pnl_pct = price_change * 100 * self.leverage
        pnl_amount = self.position['position_value'] * price_change
        
        # 手續費
        fee_close = self.position['position_value'] * self.fee_rate
        pnl_amount -= (self.position['fee_open'] + fee_close)
        
        # 更新資金
        self.capital += pnl_amount
        is_win = pnl_amount > 0
        
        # 更新策略狀態
        strategy.update_after_trade(pnl_pct, is_win)
        
        # 記錄交易
        self.trades.append({
            'side': self.position['side'],
            'entry_time': self.position['entry_time'],
            'entry_price': self.position['entry_price'],
            'exit_time': bar['open_time'],
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'position_pct': self.position['position_pct'],
            'confidence': self.position['confidence'],
            'chronos_prob': self.position['chronos_prob'],
            'xgb_prob': self.position['xgb_prob'],
            'reason': self.position['reason'],
            'pnl_pct': pnl_pct,
            'pnl_amount': pnl_amount,
            'capital': self.capital,
            'is_win': is_win
        })
        
        # 清除持倉
        self.position = None
    
    
    def _force_close(self, bar: pd.Series):
        """強制平倉"""
        if self.position is None:
            return
        
        from models.hybrid_aggressive_strategy import HybridAggressiveStrategy
        strategy = HybridAggressiveStrategy(initial_capital=self.initial_capital)
        
        self._close_position(bar, bar['close'], 'FORCE_CLOSE', strategy)
        logger.warning("⚠️ Forced close at end of backtest")
    
    
    def _calculate_stats(self, strategy) -> Dict[str, Any]:
        """計算統計數據"""
        if len(self.trades) == 0:
            return {
                'success': False,
                'error': '沒有生成任何交易'
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        total_trades = len(trades_df)
        wins = len(trades_df[trades_df['is_win']])
        losses = total_trades - wins
        win_rate = wins / total_trades * 100
        
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        # Profit Factor
        winning_trades = trades_df[trades_df['pnl_amount'] > 0]
        losing_trades = trades_df[trades_df['pnl_amount'] <= 0]
        
        total_profit = winning_trades['pnl_amount'].sum() if len(winning_trades) > 0 else 0
        total_loss = abs(losing_trades['pnl_amount'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        # Sharpe Ratio
        returns = trades_df['pnl_pct'].values
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Max Drawdown
        cumulative = (trades_df['capital'] / self.initial_capital - 1) * 100
        running_max = cumulative.cummax()
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()
        
        logger.info(f"\n" + "="*60)
        logger.info(f"✅ Backtest Complete!")
        logger.info(f"="*60)
        logger.info(f"📊 Total Trades: {total_trades}")
        logger.info(f"✅ Wins: {wins} | ❌ Losses: {losses}")
        logger.info(f"🎯 Win Rate: {win_rate:.2f}%")
        logger.info(f"💰 Total Return: {total_return:+.2f}%")
        logger.info(f"📈 Final Capital: ${self.capital:,.2f}")
        logger.info(f"⭐ Profit Factor: {profit_factor:.2f}")
        logger.info(f"📊 Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"📉 Max Drawdown: {max_drawdown:.2f}%")
        logger.info(f"="*60)
        
        return {
            'success': True,
            'trades': trades_df,
            'stats': {
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_return': total_return,
                'final_capital': self.capital,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'avg_win': winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0,
                'avg_loss': losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0,
            },
            'strategy_stats': strategy.get_stats()
        }


if __name__ == "__main__":
    print("混合回測引擎")
    print("請使用 run_hybrid_backtest.py 執行回測")
