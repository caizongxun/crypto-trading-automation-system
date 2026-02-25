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
            index=0,  # 預設 tiny (最快)
            help="tiny: 8M, 最快(推薦) | small: 20M, 平衡 | base: 200M, 最準"
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
            value=50,  # 降低預設值
            step=10,
            help="蒙地卡羅採樣數，更多 = 更準但更慢"
        )
        
        stride = st.number_input(
            "預測間隔",
            min_value=1,
            max_value=24,
            value=4,
            help="每 N 根 K 線預測一次 (更大 = 更快但可能漏交易)"
        )
    
    return {
        'model_size': model_size,
        'lookback': lookback,
        'num_samples': num_samples,
        'stride': stride
    }


def run_chronos_backtest(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    chronos_params: Dict[str, Any],
    tp_pct: float,
    sl_pct: float,
    prob_threshold: float = 0.15,
    progress_callback=None
) -> Dict[str, Any]:
    """
    執行 Chronos 模型回測 (優化版)
    
    Args:
        symbol: 交易對
        timeframe: 時間週期
        start_date: 開始日期
        end_date: 結束日期
        chronos_params: Chronos 參數
        tp_pct: 止盈百分比
        sl_pct: 止損百分比
        prob_threshold: 機率門檻
        progress_callback: 進度回調函數
    
    Returns:
        回測結果字典
    """
    def update_progress(msg):
        if progress_callback:
            progress_callback(msg)
        logger.info(msg)
    
    try:
        # Step 1: 載入資料
        update_progress(f"步驟 1/4: 載入 {symbol} {timeframe} 資料...")
        from utils.hf_data_loader import load_klines
        
        df = load_klines(symbol, timeframe, start_date, end_date)
        
        if len(df) == 0:
            return {
                'success': False,
                'error': '無法載入資料'
            }
        
        update_progress(f"✅ 載入 {len(df)} 筆 K 線")
        
        # Step 2: 初始化預測器
        update_progress(f"\n步驟 2/4: 初始化 Chronos {chronos_params['model_size']} 模型...")
        
        from models.chronos_predictor import ChronosPredictor
        
        predictor = ChronosPredictor(
            model_name=f"amazon/chronos-t5-{chronos_params['model_size']}",
            device="cpu"
        )
        
        update_progress("✅ 模型載入完成")
        
        # Step 3: 批次預測 (優化版 - 使用 stride)
        stride = chronos_params.get('stride', 4)
        lookback = chronos_params['lookback']
        
        # 計算實際預測數量
        prediction_points = list(range(lookback, len(df), stride))
        total_predictions = len(prediction_points)
        
        update_progress(f"\n步驟 3/4: 執行批次預測...")
        update_progress(f"原始資料: {len(df)} 筆")
        update_progress(f"預測點 (stride={stride}): {total_predictions} 筆")
        update_progress(f"預估時間: ~{total_predictions * 0.3 / 60:.1f} 分鐘")
        
        # 儲存預測結果
        predictions = {}
        
        for idx, i in enumerate(prediction_points):
            if idx % 50 == 0:
                progress = idx / total_predictions * 100
                update_progress(f"進度: {progress:.1f}% ({idx}/{total_predictions})")
            
            window = df.iloc[i-lookback:i]
            prob_long, prob_short = predictor.predict_probabilities(
                window,
                lookback=lookback,
                horizon=1,
                num_samples=chronos_params['num_samples'],
                tp_pct=tp_pct,
                sl_pct=sl_pct
            )
            
            predictions[i] = {
                'prob_long': prob_long,
                'prob_short': prob_short
            }
        
        update_progress("✅ 預測完成")
        
        # Step 4: 回測模擬
        update_progress(f"\n步驟 4/4: 執行回測模擬...")
        
        trades = []
        capital = 10000
        position = None
        last_prediction_idx = lookback
        
        for i in range(lookback, len(df)):
            row = df.iloc[i]
            
            # 取得最近的預測
            if i in predictions:
                last_prediction_idx = i
                current_pred = predictions[i]
            elif last_prediction_idx in predictions:
                current_pred = predictions[last_prediction_idx]
            else:
                continue
            
            prob_long = current_pred['prob_long']
            prob_short = current_pred['prob_short']
            
            # 無持倉，檢查開倉機會
            if position is None:
                if prob_long > prob_threshold:
                    position = {
                        'side': 'LONG',
                        'entry_price': row['close'],
                        'entry_time': row['open_time'],
                        'entry_idx': i,
                        'tp': row['close'] * (1 + tp_pct / 100),
                        'sl': row['close'] * (1 - sl_pct / 100),
                        'prob': prob_long
                    }
                elif prob_short > prob_threshold:
                    position = {
                        'side': 'SHORT',
                        'entry_price': row['close'],
                        'entry_time': row['open_time'],
                        'entry_idx': i,
                        'tp': row['close'] * (1 - tp_pct / 100),
                        'sl': row['close'] * (1 + sl_pct / 100),
                        'prob': prob_short
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
                        'prob': position['prob'],
                        'pnl_pct': pnl_pct,
                        'pnl_amount': pnl_amount,
                        'capital': capital
                    })
                    
                    position = None
        
        # 計算統計
        if len(trades) == 0:
            return {
                'success': False,
                'error': f'沒有生成任何交易。建議: 1) 降低機率門檻到 0.10-0.12  2) 縮短回測天數到 30 天  3) 使用 1h 時間週期'
            }
        
        trades_df = pd.DataFrame(trades)
        total_trades = len(trades_df)
        wins = len(trades_df[trades_df['pnl_pct'] > 0])
        losses = len(trades_df[trades_df['pnl_pct'] < 0])
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        
        total_return = (capital - 10000) / 10000 * 100
        
        update_progress(f"\n✅ 回測完成!")
        update_progress(f"總交易: {total_trades}, 勝率: {win_rate:.1f}%, 報酬: {total_return:+.2f}%")
        
        return {
            'success': True,
            'trades': trades_df,
            'stats': {
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_return': total_return,
                'final_capital': capital,
                'avg_prob': trades_df['prob'].mean()
            },
            'predictions': predictions
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
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
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
    
    with col5:
        st.metric("平均機率", f"{stats.get('avg_prob', 0):.2%}")
    
    # 顯示交易明細
    st.markdown("### 交易明細")
    st.dataframe(
        trades_df[[
            'side', 'entry_time', 'entry_price', 'prob',
            'exit_time', 'exit_price', 'exit_reason',
            'pnl_pct', 'capital'
        ]],
        width='stretch'
    )
