"""
Chronos 回測整合工具
為 tabs/backtesting_tab.py 提供 Chronos 支援
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def render_chronos_parameters():
    """
    渲染 Chronos 模型參數設定區
    
    Returns:
        dict: Chronos 參數
    """
    st.markdown("### Chronos 模型設定")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_size = st.selectbox(
            "模型大小",
            ["tiny", "small", "base"],
            index=1,
            help="tiny: 8M, 最快 | small: 20M, 平衡(推薦) | base: 200M, 最準"
        )
        
        lookback = st.number_input(
            "歷史窗口 (K線數)",
            min_value=24,
            max_value=720,
            value=168,
            step=24,
            help="用於預測的歷史 K 線數量 (168 = 7天1h K線)"
        )
    
    with col2:
        num_samples = st.number_input(
            "採樣數",
            min_value=10,
            max_value=200,
            value=100,
            step=10,
            help="蒙地卡羅採樣數，更多 = 更準但更慢"
        )
        
        horizon = st.number_input(
            "預測距離 (K線數)",
            min_value=1,
            max_value=10,
            value=1,
            help="預測未來幾根 K 線 (1 = 下一根)"
        )
    
    return {
        'model_size': model_size,
        'lookback': lookback,
        'num_samples': num_samples,
        'horizon': horizon
    }


def run_chronos_backtest(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    chronos_params: Dict[str, Any],
    tp_pct: float,
    sl_pct: float,
    prob_threshold: float = 0.15
) -> Dict[str, Any]:
    """
    執行 Chronos 模型回測
    
    Args:
        symbol: 交易對
        timeframe: 時間週期
        start_date: 開始日期
        end_date: 結束日期
        chronos_params: Chronos 參數
        tp_pct: 止盈百分比
        sl_pct: 止損百分比
        prob_threshold: 機率門檻
    
    Returns:
        回測結果字典
    """
    try:
        from models.chronos_predictor import ChronosPredictor
        from utils.hf_data_loader import load_klines
        
        # 載入資料
        logger.info(f"Loading {symbol} {timeframe} data...")
        df = load_klines(symbol, timeframe, start_date, end_date)
        
        if len(df) == 0:
            return {
                'success': False,
                'error': '無法載入資料'
            }
        
        # 初始化預測器
        logger.info(f"Initializing Chronos {chronos_params['model_size']} model...")
        predictor = ChronosPredictor(
            model_name=f"amazon/chronos-t5-{chronos_params['model_size']}",
            device="cpu"  # GUI 環境先用 CPU
        )
        
        # 批次預測
        logger.info("Running batch predictions...")
        df_pred = predictor.predict_batch(
            df=df,
            lookback=chronos_params['lookback'],
            horizon=chronos_params['horizon'],
            num_samples=chronos_params['num_samples'],
            tp_pct=tp_pct,
            sl_pct=sl_pct
        )
        
        # 簡易回測邏輯
        trades = []
        capital = 10000
        position = None
        
        for i in range(chronos_params['lookback'], len(df_pred)):
            row = df_pred.iloc[i]
            
            if pd.isna(row['prob_long']) or pd.isna(row['prob_short']):
                continue
            
            # 無持倉，檢查開倉機會
            if position is None:
                if row['prob_long'] > prob_threshold:
                    position = {
                        'side': 'LONG',
                        'entry_price': row['close'],
                        'entry_time': row['open_time'],
                        'tp': row['close'] * (1 + tp_pct / 100),
                        'sl': row['close'] * (1 - sl_pct / 100)
                    }
                elif row['prob_short'] > prob_threshold:
                    position = {
                        'side': 'SHORT',
                        'entry_price': row['close'],
                        'entry_time': row['open_time'],
                        'tp': row['close'] * (1 - tp_pct / 100),
                        'sl': row['close'] * (1 + sl_pct / 100)
                    }
            
            # 有持倉，檢查出倉條件
            else:
                exit_reason = None
                exit_price = None
                
                if position['side'] == 'LONG':
                    if row['high'] >= position['tp']:
                        exit_reason = 'TP'
                        exit_price = position['tp']
                    elif row['low'] <= position['sl']:
                        exit_reason = 'SL'
                        exit_price = position['sl']
                
                else:  # SHORT
                    if row['low'] <= position['tp']:
                        exit_reason = 'TP'
                        exit_price = position['tp']
                    elif row['high'] >= position['sl']:
                        exit_reason = 'SL'
                        exit_price = position['sl']
                
                if exit_reason:
                    # 計算 PnL
                    if position['side'] == 'LONG':
                        pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                    else:
                        pnl_pct = (position['entry_price'] - exit_price) / position['entry_price'] * 100
                    
                    pnl_amount = capital * (pnl_pct / 100)
                    capital += pnl_amount
                    
                    trades.append({
                        'side': position['side'],
                        'entry_time': position['entry_time'],
                        'entry_price': position['entry_price'],
                        'exit_time': row['open_time'],
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl_pct': pnl_pct,
                        'pnl_amount': pnl_amount,
                        'capital': capital
                    })
                    
                    position = None
        
        # 計算統計
        if len(trades) == 0:
            return {
                'success': False,
                'error': '沒有生成任何交易，請調低機率門檻'
            }
        
        trades_df = pd.DataFrame(trades)
        total_trades = len(trades_df)
        wins = len(trades_df[trades_df['pnl_pct'] > 0])
        losses = len(trades_df[trades_df['pnl_pct'] < 0])
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        
        total_return = (capital - 10000) / 10000 * 100
        
        return {
            'success': True,
            'trades': trades_df,
            'stats': {
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_return': total_return,
                'final_capital': capital
            },
            'df_pred': df_pred
        }
        
    except Exception as e:
        logger.error(f"Chronos backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


def display_chronos_results(result: Dict[str, Any]):
    """
    顯示 Chronos 回測結果
    
    Args:
        result: run_chronos_backtest 返回的結果
    """
    if not result['success']:
        st.error(f"回測失敗: {result['error']}")
        return
    
    stats = result['stats']
    trades_df = result['trades']
    
    # 顯示統計
    st.markdown("### 回測結果")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("總交易數", stats['total_trades'])
    
    with col2:
        st.metric("勝率", f"{stats['win_rate']:.2f}%")
    
    with col3:
        st.metric(
            "總報酬",
            f"{stats['total_return']:.2f}%",
            delta=f"${stats['final_capital'] - 10000:.2f}"
        )
    
    with col4:
        profit_factor = (
            trades_df[trades_df['pnl_amount'] > 0]['pnl_amount'].sum() /
            abs(trades_df[trades_df['pnl_amount'] < 0]['pnl_amount'].sum())
            if len(trades_df[trades_df['pnl_amount'] < 0]) > 0 else 0
        )
        st.metric("Profit Factor", f"{profit_factor:.2f}")
    
    # 顯示交易明細
    st.markdown("### 交易明細")
    st.dataframe(
        trades_df[[
            'side', 'entry_time', 'entry_price',
            'exit_time', 'exit_price', 'exit_reason',
            'pnl_pct', 'capital'
        ]],
        use_container_width=True
    )
