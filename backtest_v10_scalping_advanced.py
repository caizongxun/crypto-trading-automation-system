#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v10 進階回測引擎 - 整合所有優化方案

支援功能:
1. 動態 TP/SL
2. 信號質量分級倉位
3. 移動止損
4. 時段過濾
5. 嚴格篩選條件
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class AdvancedScalpingBacktester:
    def __init__(
        self,
        long_model_path: str,
        short_model_path: str,
        initial_capital: float = 10000,
        position_size: float = 0.02,
        leverage: int = 10,
        threshold: float = 0.6,
        tp_pct: float = 0.005,
        sl_pct: float = 0.003,
        # 優化方案開關
        enable_dynamic_tpsl: bool = False,
        enable_quality_sizing: bool = False,
        enable_trailing_stop: bool = False,
        enable_time_filter: bool = False,
        enable_strict_filter: bool = False,
        # 動態 TP/SL 參數
        low_vol_threshold: float = 0.005,
        high_vol_threshold: float = 0.015,
        low_vol_tp: float = 0.003,
        low_vol_sl: float = 0.002,
        mid_vol_tp: float = 0.005,
        mid_vol_sl: float = 0.0025,
        high_vol_tp: float = 0.008,
        high_vol_sl: float = 0.0035,
        # 質量分級參數
        high_conf_threshold: float = 0.75,
        mid_conf_threshold: float = 0.65,
        high_conf_size: float = 0.03,
        mid_conf_size: float = 0.02,
        low_conf_size: float = 0.01,
        # 移動止損參數
        trailing_activation: float = 0.5,  # TP * 50% 時啟動
        trailing_distance: float = 0.0015,  # 回撤 0.15% 出場
        # 嚴格篩選參數
        min_volume_ratio: float = 0.8,
        max_return_threshold: float = 0.02,
        high_volatility_threshold: float = 0.65
    ):
        # 載入模型
        with open(long_model_path, 'rb') as f:
            long_data = pickle.load(f)
            self.long_model = long_data['model']
            self.feature_names = long_data['features']
        
        with open(short_model_path, 'rb') as f:
            short_data = pickle.load(f)
            self.short_model = short_data['model']
        
        # 基礎參數
        self.initial_capital = initial_capital
        self.base_position_size = position_size
        self.leverage = leverage
        self.threshold = threshold
        self.base_tp_pct = tp_pct
        self.base_sl_pct = sl_pct
        
        # 優化方案開關
        self.enable_dynamic_tpsl = enable_dynamic_tpsl
        self.enable_quality_sizing = enable_quality_sizing
        self.enable_trailing_stop = enable_trailing_stop
        self.enable_time_filter = enable_time_filter
        self.enable_strict_filter = enable_strict_filter
        
        # 動態 TP/SL
        self.low_vol_threshold = low_vol_threshold
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_tp = low_vol_tp
        self.low_vol_sl = low_vol_sl
        self.mid_vol_tp = mid_vol_tp
        self.mid_vol_sl = mid_vol_sl
        self.high_vol_tp = high_vol_tp
        self.high_vol_sl = high_vol_sl
        
        # 質量分級
        self.high_conf_threshold = high_conf_threshold
        self.mid_conf_threshold = mid_conf_threshold
        self.high_conf_size = high_conf_size
        self.mid_conf_size = mid_conf_size
        self.low_conf_size = low_conf_size
        
        # 移動止損
        self.trailing_activation = trailing_activation
        self.trailing_distance = trailing_distance
        
        # 嚴格篩選
        self.min_volume_ratio = min_volume_ratio
        self.max_return_threshold = max_return_threshold
        self.high_volatility_threshold = high_volatility_threshold
    
    def get_dynamic_tpsl(self, volatility: float) -> Tuple[float, float]:
        """方案1: 動態 TP/SL"""
        if not self.enable_dynamic_tpsl:
            return self.base_tp_pct, self.base_sl_pct
        
        if volatility < self.low_vol_threshold:
            return self.low_vol_tp, self.low_vol_sl
        elif volatility < self.high_vol_threshold:
            return self.mid_vol_tp, self.mid_vol_sl
        else:
            return self.high_vol_tp, self.high_vol_sl
    
    def get_position_size_by_confidence(self, prob: float) -> float:
        """方案2: 信號質量分級倉位"""
        if not self.enable_quality_sizing:
            return self.base_position_size
        
        if prob >= self.high_conf_threshold:
            return self.high_conf_size
        elif prob >= self.mid_conf_threshold:
            return self.mid_conf_size
        else:
            return self.low_conf_size
    
    def is_high_quality_time(self, timestamp: pd.Timestamp) -> bool:
        """方案4: 時段過濾"""
        if not self.enable_time_filter:
            return True
        
        hour = timestamp.hour
        weekday = timestamp.weekday()
        
        # 週末
        if weekday >= 5:
            return False
        
        # 午休時段
        if 12 <= hour < 14:
            return False
        
        # 高流動性時段
        if (8 <= hour < 10) or (14 <= hour < 17) or (21 <= hour < 23):
            return True
        
        return False
    
    def should_enter(self, prob: float, features: pd.Series) -> bool:
        """方案6: 嚴格篩選"""
        if prob < self.threshold:
            return False
        
        if not self.enable_strict_filter:
            return True
        
        # 量能太低
        if 'volume_ratio_3' in features.index:
            if features['volume_ratio_3'] < self.min_volume_ratio:
                return False
        
        # 波動太大
        if 'return_1' in features.index:
            if abs(features['return_1']) > self.max_return_threshold:
                return False
        
        # 波動擴大中需要更高機率
        if 'volatility_expanding' in features.index:
            if features['volatility_expanding'] == 1:
                if prob < self.high_volatility_threshold:
                    return False
        
        return True
    
    def calculate_trailing_stop(
        self, 
        entry_price: float, 
        current_price: float,
        peak_price: float,
        side: str,
        tp_pct: float,
        sl_pct: float
    ) -> float:
        """方案5: 移動止損"""
        if not self.enable_trailing_stop:
            return entry_price * (1 - sl_pct) if side == 'long' else entry_price * (1 + sl_pct)
        
        if side == 'long':
            profit_pct = (current_price - entry_price) / entry_price
            
            # 未達啟動條件
            if profit_pct < tp_pct * self.trailing_activation:
                return entry_price * (1 - sl_pct)
            
            # 啟動移動止損
            trailing_sl = peak_price * (1 - self.trailing_distance)
            original_sl = entry_price * (1 - sl_pct)
            
            return max(trailing_sl, original_sl)
        
        else:  # short
            profit_pct = (entry_price - current_price) / entry_price
            
            if profit_pct < tp_pct * self.trailing_activation:
                return entry_price * (1 + sl_pct)
            
            trailing_sl = peak_price * (1 + self.trailing_distance)
            original_sl = entry_price * (1 + sl_pct)
            
            return min(trailing_sl, original_sl)
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算特徵 (複用 train_v10 的邏輯)"""
        from train_v10_high_frequency import calculate_microstructure_features
        features = calculate_microstructure_features(df)
        return features.fillna(0).replace([np.inf, -np.inf], 0)
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        start_idx: int = 0,
        long_enabled: bool = True,
        short_enabled: bool = True
    ) -> Optional[Dict]:
        """執行回測"""
        
        # 計算特徵
        features = self.calculate_features(df)
        
        # 確保時間索引
        if 'open_time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['open_time'])
        elif isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = df.index
        else:
            df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='15min')
        
        # 預測
        long_probs = self.long_model.predict_proba(features[self.feature_names])[:, 1] if long_enabled else np.zeros(len(df))
        short_probs = self.short_model.predict_proba(features[self.feature_names])[:, 1] if short_enabled else np.zeros(len(df))
        
        # 回測變數
        equity = self.initial_capital
        trades = []
        equity_curve = []
        position = None
        
        for i in range(start_idx, len(df)):
            current_bar = df.iloc[i]
            current_features = features.iloc[i]
            current_time = df['timestamp'].iloc[i]
            
            # 更新持倉
            if position:
                current_price = current_bar['close']
                
                # 更新峰值價格 (用於移動止損)
                if position['side'] == 'long':
                    position['peak_price'] = max(position['peak_price'], current_bar['high'])
                else:
                    position['peak_price'] = min(position['peak_price'], current_bar['low'])
                
                # 計算當前止損
                current_sl = self.calculate_trailing_stop(
                    position['entry_price'],
                    current_price,
                    position['peak_price'],
                    position['side'],
                    position['tp_pct'],
                    position['sl_pct']
                )
                
                # 檢查出場
                exit_reason = None
                exit_price = None
                
                if position['side'] == 'long':
                    # TP
                    if current_bar['high'] >= position['tp_price']:
                        exit_reason = 'tp'
                        exit_price = position['tp_price']
                    # SL (動態)
                    elif current_bar['low'] <= current_sl:
                        exit_reason = 'sl'
                        exit_price = current_sl
                
                else:  # short
                    if current_bar['low'] <= position['tp_price']:
                        exit_reason = 'tp'
                        exit_price = position['tp_price']
                    elif current_bar['high'] >= current_sl:
                        exit_reason = 'sl'
                        exit_price = current_sl
                
                # 持有時間限制
                position['bars_held'] += 1
                if position['bars_held'] >= 20 and not exit_reason:
                    exit_reason = 'timeout'
                    exit_price = current_price
                
                # 平倉
                if exit_reason:
                    if position['side'] == 'long':
                        pnl = (exit_price - position['entry_price']) / position['entry_price'] * position['notional']
                    else:
                        pnl = (position['entry_price'] - exit_price) / position['entry_price'] * position['notional']
                    
                    equity += pnl
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'side': position['side'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'return_pct': pnl / position['notional'],
                        'exit_reason': exit_reason,
                        'bars_held': position['bars_held'],
                        'win': pnl > 0,
                        'confidence': position['confidence']
                    })
                    
                    position = None
            
            # 開新倉
            if not position:
                # 時段過濾
                if not self.is_high_quality_time(current_time):
                    equity_curve.append({'time': current_time, 'equity': equity})
                    continue
                
                # 獲取當前波動性 (用於動態 TP/SL)
                volatility = current_features.get('volatility_3', 0.01)
                tp_pct, sl_pct = self.get_dynamic_tpsl(volatility)
                
                # Long 信號
                if long_enabled and long_probs[i] >= self.threshold:
                    if self.should_enter(long_probs[i], current_features):
                        position_size = self.get_position_size_by_confidence(long_probs[i])
                        
                        entry_price = current_bar['close']
                        notional = equity * position_size * self.leverage
                        
                        position = {
                            'side': 'long',
                            'entry_time': current_time,
                            'entry_price': entry_price,
                            'tp_price': entry_price * (1 + tp_pct),
                            'notional': notional,
                            'bars_held': 0,
                            'confidence': long_probs[i],
                            'tp_pct': tp_pct,
                            'sl_pct': sl_pct,
                            'peak_price': entry_price
                        }
                
                # Short 信號
                elif short_enabled and short_probs[i] >= self.threshold:
                    if self.should_enter(short_probs[i], current_features):
                        position_size = self.get_position_size_by_confidence(short_probs[i])
                        
                        entry_price = current_bar['close']
                        notional = equity * position_size * self.leverage
                        
                        position = {
                            'side': 'short',
                            'entry_time': current_time,
                            'entry_price': entry_price,
                            'tp_price': entry_price * (1 - tp_pct),
                            'notional': notional,
                            'bars_held': 0,
                            'confidence': short_probs[i],
                            'tp_pct': tp_pct,
                            'sl_pct': sl_pct,
                            'peak_price': entry_price
                        }
            
            equity_curve.append({'time': current_time, 'equity': equity})
        
        # 統計
        if not trades:
            return None
        
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        
        wins = trades_df['win'].sum()
        total_trades = len(trades_df)
        win_rate = wins / total_trades
        
        total_pnl = trades_df['pnl'].sum()
        total_return_pct = total_pnl / self.initial_capital
        
        winning_trades = trades_df[trades_df['win']]
        losing_trades = trades_df[~trades_df['win']]
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # 回撤
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()
        
        # Sharpe
        returns = trades_df['return_pct']
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        summary = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_equity': equity
        }
        
        return {
            'trades': trades_df,
            'equity': equity_df,
            'summary': summary
        }
