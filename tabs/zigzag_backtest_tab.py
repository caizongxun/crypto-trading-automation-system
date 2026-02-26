#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZigZag 策略回測 Tab

使用已訓練的 ZigZag 模型進行歷史回測
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class ZigZagBacktestTab:
    def __init__(self):
        self.models_dir = Path('models_output/zigzag')
        self.results_dir = Path('backtest_results/zigzag')
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def render(self):
        st.header("📊 ZigZag 策略回測")
        
        # 說明
        st.info("""
        使用已訓練的 ZigZag 模型在歷史數據上模擬交易，評估策略表現。
        """)
        
        # 模型選擇
        st.subheader("🎯 模型選擇")
        
        col1, col2 = st.columns(2)
        
        # 掃描可用模型
        long_models = list(self.models_dir.glob('zigzag_long_*.pkl'))
        short_models = list(self.models_dir.glob('zigzag_short_*.pkl'))
        
        if not long_models and not short_models:
            st.error("未找到 ZigZag 模型，請先訓練模型")
            st.info("前往 'ZigZag 模型訓練' Tab 訓練模型")
            return
        
        with col1:
            if long_models:
                long_model_names = [m.name for m in long_models]
                selected_long = st.selectbox(
                    "Long 模型",
                    options=long_model_names,
                    help="選擇 Long 方向的模型"
                )
                long_model_path = self.models_dir / selected_long
                
                # 顯示模型信息
                long_info = self.load_model_info(long_model_path)
                if long_info:
                    st.caption(f"Symbol: {long_info.get('symbol', 'N/A')}")
                    st.caption(f"Timeframe: {long_info.get('timeframe', 'N/A')}")
            else:
                st.warning("無 Long 模型")
                long_model_path = None
        
        with col2:
            if short_models:
                short_model_names = [m.name for m in short_models]
                selected_short = st.selectbox(
                    "Short 模型",
                    options=short_model_names,
                    help="選擇 Short 方向的模型"
                )
                short_model_path = self.models_dir / selected_short
                
                # 顯示模型信息
                short_info = self.load_model_info(short_model_path)
                if short_info:
                    st.caption(f"Symbol: {short_info.get('symbol', 'N/A')}")
                    st.caption(f"Timeframe: {short_info.get('timeframe', 'N/A')}")
            else:
                st.warning("無 Short 模型")
                short_model_path = None
        
        st.markdown("---")
        
        # 回測參數
        st.subheader("⚙️ 回測參數")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**資料設定**")
            
            # 從模型讀取 symbol 和 timeframe
            if long_info:
                default_symbol = long_info.get('symbol', 'BTCUSDT')
                default_tf = long_info.get('timeframe', '1h')
            elif short_info:
                default_symbol = short_info.get('symbol', 'BTCUSDT')
                default_tf = short_info.get('timeframe', '1h')
            else:
                default_symbol = 'BTCUSDT'
                default_tf = '1h'
            
            symbol = st.selectbox(
                "交易對",
                options=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'],
                index=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'].index(default_symbol) if default_symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'] else 0
            )
            
            timeframe = st.selectbox(
                "時間框架",
                options=['15m', '1h', '4h', '1d'],
                index=['15m', '1h', '4h', '1d'].index(default_tf) if default_tf in ['15m', '1h', '4h', '1d'] else 1
            )
            
            backtest_days = st.number_input(
                "回測天數",
                min_value=30,
                max_value=365,
                value=90,
                step=30
            )
        
        with col2:
            st.markdown("**信號閾值**")
            
            threshold_long = st.slider(
                "Long 閾值",
                min_value=0.5,
                max_value=0.9,
                value=0.6,
                step=0.05,
                help="模型預測概率超過此值才開 Long"
            )
            
            threshold_short = st.slider(
                "Short 閾值",
                min_value=0.5,
                max_value=0.9,
                value=0.6,
                step=0.05,
                help="模型預測概率超過此值才開 Short"
            )
            
            st.markdown("**風險控制**")
            
            max_positions = st.number_input(
                "最大持倉數",
                min_value=1,
                max_value=5,
                value=1,
                help="同時最多持有幾個倉位"
            )
            
            max_daily_trades = st.number_input(
                "每日交易上限",
                min_value=1,
                max_value=20,
                value=10
            )
        
        with col3:
            st.markdown("**資金管理**")
            
            initial_capital = st.number_input(
                "初始本金 (USDT)",
                min_value=1000,
                max_value=1000000,
                value=10000,
                step=1000
            )
            
            leverage = st.slider(
                "槓桿倍數",
                min_value=1,
                max_value=10,
                value=5
            )
            
            position_size_pct = st.slider(
                "倉位比例 (%)",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                help="每次開倉佔本金的百分比"
            )
            
            st.markdown("**交易成本**")
            
            fee_rate = st.number_input(
                "手續費率 (%)",
                min_value=0.0,
                max_value=0.1,
                value=0.04,
                step=0.01,
                format="%.3f"
            ) / 100
            
            slippage = st.number_input(
                "滑價 (%)",
                min_value=0.0,
                max_value=0.1,
                value=0.02,
                step=0.01,
                format="%.3f"
            ) / 100
        
        st.markdown("---")
        
        # 執行按鈕
        col1, col2 = st.columns([3, 1])
        
        with col1:
            pass
        
        with col2:
            if st.button("🚀 開始回測", type="primary", use_container_width=True):
                if not long_model_path and not short_model_path:
                    st.error("請至少選擇一個模型")
                else:
                    config = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'backtest_days': backtest_days,
                        'long_model_path': long_model_path,
                        'short_model_path': short_model_path,
                        'threshold_long': threshold_long,
                        'threshold_short': threshold_short,
                        'initial_capital': initial_capital,
                        'leverage': leverage,
                        'position_size_pct': position_size_pct / 100,
                        'max_positions': max_positions,
                        'max_daily_trades': max_daily_trades,
                        'fee_rate': fee_rate,
                        'slippage': slippage
                    }
                    
                    self.run_backtest(config)
        
        # 歷史回測
        st.markdown("---")
        st.subheader("📚 回測歷史")
        self.display_backtest_history()
    
    def load_model_info(self, model_path):
        """載入模型基本信息"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            return {
                'symbol': model_data.get('symbol', 'N/A'),
                'timeframe': model_data.get('timeframe', 'N/A'),
                'feature_cols': model_data.get('feature_cols', [])
            }
        except:
            return None
    
    def run_backtest(self, config):
        """執行回測"""
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 步驟 1: 載入數據
            status_text.text("⏳ 載入歷史數據...")
            progress_bar.progress(10)
            
            from utils.hf_data_loader import load_klines
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=config['backtest_days'])
            
            df = load_klines(
                symbol=config['symbol'],
                timeframe=config['timeframe'],
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            if df is None or len(df) < 100:
                st.error("數據不足")
                return
            
            st.success(f"✅ 載入 {len(df)} 根 K 線")
            progress_bar.progress(25)
            
            # 步驟 2: 計算 ZigZag 和指標
            status_text.text("⏳ 計算技術指標...")
            
            from utils.v11_zigzag import calculate_zigzag_pivots
            from utils.v11_reversal_indicators import calculate_reversal_indicators
            from utils.v11_feature_engineering import create_v11_features
            
            df = calculate_zigzag_pivots(df, threshold_pct=3.0)
            df = calculate_reversal_indicators(df)
            
            feature_config = {
                'price': True,
                'volume': True,
                'trend': True,
                'reversal': True,
                'pattern': True
            }
            df = create_v11_features(df, feature_config)
            
            progress_bar.progress(40)
            
            # 步驟 3: 載入模型並預測
            status_text.text("⏳ 生成交易信號...")
            
            feature_cols = df.attrs.get('feature_columns', [])
            df_features = df[feature_cols].fillna(0)
            
            # Long 預測
            if config['long_model_path']:
                with open(config['long_model_path'], 'rb') as f:
                    long_model_data = pickle.load(f)
                long_model = long_model_data['model']
                df['prob_long'] = long_model.predict_proba(df_features)[:, 1]
                df['signal_long'] = (df['prob_long'] >= config['threshold_long']).astype(int)
            else:
                df['prob_long'] = 0
                df['signal_long'] = 0
            
            # Short 預測
            if config['short_model_path']:
                with open(config['short_model_path'], 'rb') as f:
                    short_model_data = pickle.load(f)
                short_model = short_model_data['model']
                df['prob_short'] = short_model.predict_proba(df_features)[:, 1]
                df['signal_short'] = (df['prob_short'] >= config['threshold_short']).astype(int)
            else:
                df['prob_short'] = 0
                df['signal_short'] = 0
            
            progress_bar.progress(60)
            
            # 步驟 4: 模擬交易
            status_text.text("⏳ 模擬交易...")
            
            trades = self.simulate_trading(df, config)
            
            progress_bar.progress(80)
            
            # 步驟 5: 計算績效
            status_text.text("⏳ 計算績效...")
            
            results = self.calculate_performance(df, trades, config)
            
            progress_bar.progress(100)
            status_text.text("✅ 回測完成!")
            
            # 顯示結果
            self.display_results(results, df, trades)
            
            # 保存結果
            self.save_backtest_result(results, config)
            
        except Exception as e:
            st.error(f"回測失敗: {str(e)}")
            import traceback
            with st.expander("錯誤詳情"):
                st.code(traceback.format_exc())
    
    def simulate_trading(self, df, config):
        """模擬交易邏輯"""
        
        trades = []
        positions = []  # 當前持倉
        daily_trades = {}
        
        capital = config['initial_capital']
        
        for i in range(len(df)):
            current_time = df.index[i]
            current_price = df['close'].iloc[i]
            current_date = current_time.date()
            
            # 檢查當日交易數
            if current_date not in daily_trades:
                daily_trades[current_date] = 0
            
            # 更新持倉 (檢查 TP/SL)
            for pos in positions[:]:
                # 簡易止盈止損 (實際應該用 target_tp/target_sl)
                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * (1 if pos['side'] == 'long' else -1)
                
                # 止盈 3% 或止損 1.5%
                if pnl_pct >= 0.03 or pnl_pct <= -0.015:
                    # 平倉
                    exit_price = current_price * (1 - config['slippage'] if pos['side'] == 'long' else 1 + config['slippage'])
                    
                    pnl = pos['size'] * (exit_price - pos['entry_price']) / pos['entry_price'] * config['leverage']
                    if pos['side'] == 'short':
                        pnl = -pnl
                    
                    # 扣除手續費
                    fee = pos['size'] * config['fee_rate'] * 2  # 開倉+平倉
                    pnl -= fee
                    
                    capital += pnl
                    
                    trades.append({
                        'entry_time': pos['entry_time'],
                        'exit_time': current_time,
                        'side': pos['side'],
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_pct': pnl / pos['size'] * 100,
                        'duration': (current_time - pos['entry_time']).total_seconds() / 3600
                    })
                    
                    positions.remove(pos)
            
            # 檢查新信號
            if len(positions) < config['max_positions'] and daily_trades[current_date] < config['max_daily_trades']:
                signal_long = df['signal_long'].iloc[i]
                signal_short = df['signal_short'].iloc[i]
                
                if signal_long and not any(p['side'] == 'long' for p in positions):
                    # 開 Long
                    size = capital * config['position_size_pct']
                    entry_price = current_price * (1 + config['slippage'])
                    
                    positions.append({
                        'entry_time': current_time,
                        'side': 'long',
                        'entry_price': entry_price,
                        'size': size
                    })
                    
                    daily_trades[current_date] += 1
                
                elif signal_short and not any(p['side'] == 'short' for p in positions):
                    # 開 Short
                    size = capital * config['position_size_pct']
                    entry_price = current_price * (1 - config['slippage'])
                    
                    positions.append({
                        'entry_time': current_time,
                        'side': 'short',
                        'entry_price': entry_price,
                        'size': size
                    })
                    
                    daily_trades[current_date] += 1
        
        return trades
    
    def calculate_performance(self, df, trades, config):
        """計算績效指標"""
        
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return': 0,
                'message': '無交易'
            }
        
        trades_df = pd.DataFrame(trades)
        
        total_trades = len(trades)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        total_return = total_pnl / config['initial_capital'] * 100
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].mean()) if losing_trades > 0 else 0
        
        profit_factor = (avg_win * winning_trades) / (avg_loss * losing_trades) if losing_trades > 0 and avg_loss > 0 else 0
        
        avg_duration = trades_df['duration'].mean()
        
        # 計算資金曲線
        equity_curve = [config['initial_capital']]
        for pnl in trades_df['pnl']:
            equity_curve.append(equity_curve[-1] + pnl)
        
        equity_series = pd.Series(equity_curve)
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak * 100
        max_drawdown = drawdown.min()
        
        # Long/Short 分別統計
        long_trades = trades_df[trades_df['side'] == 'long']
        short_trades = trades_df[trades_df['side'] == 'short']
        
        return {
            'total_trades': total_trades,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'avg_duration': avg_duration,
            'equity_curve': equity_curve,
            'trades_df': trades_df
        }
    
    def display_results(self, results, df, trades):
        """顯示回測結果"""
        
        if results['total_trades'] == 0:
            st.warning("回測期間無交易，請調整閾值或檢查模型")
            return
        
        st.success("🎉 回測完成!")
        
        # 整體統計
        st.subheader("📊 整體表現")
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("總交易數", f"{results['total_trades']}")
        col2.metric("勝率", f"{results['win_rate']*100:.1f}%")
        col3.metric("總報酬", f"{results['total_return']:.2f}%", 
                    delta=f"{results['total_pnl']:.2f} USDT")
        col4.metric("Profit Factor", f"{results['profit_factor']:.2f}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Long 交易", f"{results['long_trades']}")
        col2.metric("Short 交易", f"{results['short_trades']}")
        col3.metric("最大回撤", f"{results['max_drawdown']:.2f}%")
        col4.metric("平均持倉", f"{results['avg_duration']:.1f}h")
        
        # 資金曲線
        st.subheader("💰 資金曲線")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=results['equity_curve'],
            mode='lines',
            name='權益',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="資金曲線",
            xaxis_title="交易序號",
            yaxis_title="權益 (USDT)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 交易明細
        st.subheader("📋 交易明細")
        
        trades_df = results['trades_df']
        
        # 顯示表格
        display_df = trades_df[['entry_time', 'side', 'entry_price', 'exit_price', 'pnl', 'pnl_pct', 'duration']].copy()
        display_df['entry_time'] = display_df['entry_time'].dt.strftime('%Y-%m-%d %H:%M')
        display_df.columns = ['進場時間', '方向', '進場價', '出場價', 'PnL (USDT)', 'PnL (%)', '持倉(h)']
        
        st.dataframe(display_df, use_container_width=True, height=400)
    
    def save_backtest_result(self, results, config):
        """保存回測結果"""
        
        import json
        
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'symbol': config['symbol'],
                'timeframe': config['timeframe'],
                'backtest_days': config['backtest_days'],
                'threshold_long': config['threshold_long'],
                'threshold_short': config['threshold_short'],
                'initial_capital': config['initial_capital'],
                'leverage': config['leverage']
            },
            'results': {
                'total_trades': results['total_trades'],
                'win_rate': results['win_rate'],
                'total_return': results['total_return'],
                'profit_factor': results['profit_factor'],
                'max_drawdown': results['max_drawdown']
            }
        }
        
        report_file = self.results_dir / f'zigzag_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    def display_backtest_history(self):
        """顯示回測歷史"""
        
        import json
        
        reports = sorted(self.results_dir.glob('zigzag_backtest_*.json'), reverse=True)
        
        if not reports:
            st.info("尚無回測記錄")
            return
        
        for report_file in reports[:5]:
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                
                with st.expander(f"{report['timestamp']} - {report['config']['symbol']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("總交易", report['results']['total_trades'])
                        st.text(f"回測: {report['config']['backtest_days']}天")
                    
                    with col2:
                        st.metric("勝率", f"{report['results']['win_rate']*100:.1f}%")
                        st.text(f"槓桿: {report['config']['leverage']}x")
                    
                    with col3:
                        st.metric("總報酬", f"{report['results']['total_return']:.2f}%")
                        st.text(f"PF: {report['results']['profit_factor']:.2f}")
            except:
                continue


def render():
    tab = ZigZagBacktestTab()
    tab.render()
