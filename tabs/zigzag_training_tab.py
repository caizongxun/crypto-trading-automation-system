#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZigZag 模型訓練 Tab - 獨立版本

專注於 ZigZag 反轉點交易策略
不影響 V1/V2/V3/V10/Chronos 等其他版本
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class ZigZagTrainingTab:
    def __init__(self):
        self.models_dir = Path('models_output/zigzag')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.reports_dir = Path('training_reports/zigzag')
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def render(self):
        st.header("🔶 ZigZag 反轉策略 - 模型訓練")
        
        # 說明區
        with st.expander("📖 關於 ZigZag 反轉策略", expanded=False):
            st.markdown("""
            ### 什麼是 ZigZag?
            
            ZigZag 指標會過濾掉市場噪音，只保留**重要的高點和低點**。
            
            **核心概念:**
            - 當價格反轉超過設定門檻(如3%)，才標記為 swing high/low
            - 排除小幅震盪，只抓主要波段
            - 適合中線交易 (1h-4h timeframe)
            
            **V11 vs ZigZag 獨立版:**
            - V11 整合在 V3 版本中，與其他模型共用流程
            - **ZigZag 獨立版**：專注反轉交易，模型/報告獨立儲存
            - 不影響現有 V1/V2/V3/V10/Chronos 策略
            
            **適合市場:** 震盪市、趨勢轉折、波段交易
            """)
        
        st.markdown("---")
        
        # 配置區
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("⚙️ ZigZag 參數")
            
            zigzag_threshold = st.slider(
                "反轉門檻 (%)",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                help="價格反轉超過此百分比才標記為關鍵點。越大=越少信號但越重要"
            )
            
            min_swing_pct = st.slider(
                "最小振幅 (%)",
                min_value=0.3,
                max_value=2.0,
                value=0.5,
                step=0.1,
                help="過濾掉振幅太小的反轉點"
            )
            
            lookahead_bars = st.number_input(
                "提前預警 K 線數",
                min_value=0,
                max_value=5,
                value=2,
                help="在反轉發生前 N 根 K 線就發出信號"
            )
        
        with col2:
            st.subheader("🔍 確認條件")
            
            use_rsi_div = st.checkbox(
                "要求 RSI 背離",
                value=False,
                help="只標記有 RSI 背離的反轉點 (更嚴格)"
            )
            
            use_volume_div = st.checkbox(
                "要求量能背離",
                value=False,
                help="需要量能配合確認"
            )
            
            use_sr = st.checkbox(
                "要求支撐/阻力",
                value=False,
                help="只在支撐/阻力位附近的反轉"
            )
            
            st.markdown("**TP/SL 設定**")
            
            tp_multiplier = st.slider(
                "TP 倍數",
                min_value=0.5,
                max_value=3.0,
                value=1.5,
                step=0.1,
                help="根據振幅計算 TP"
            )
            
            sl_multiplier = st.slider(
                "SL 倍數",
                min_value=0.3,
                max_value=1.5,
                value=0.5,
                step=0.1,
                help="根據振幅計算 SL"
            )
            
            rr_ratio = tp_multiplier / sl_multiplier
            st.info(f"預期 RR 比: **{rr_ratio:.1f}:1**")
        
        with col3:
            st.subheader("📊 訓練資料")
            
            symbol = st.selectbox(
                "交易對",
                options=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'],
                index=0
            )
            
            timeframe = st.selectbox(
                "時間框架",
                options=['15m', '1h', '4h', '1d'],
                index=1,
                help="建議 1h 或 4h 用於波段交易"
            )
            
            train_days = st.number_input(
                "訓練天數",
                min_value=90,
                max_value=365,
                value=180,
                step=30
            )
            
            train_split = st.slider(
                "訓練集比例",
                min_value=0.6,
                max_value=0.9,
                value=0.8,
                step=0.05
            )
        
        # 模型配置
        st.markdown("---")
        
        with st.expander("🧠 模型進階設定"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                model_type = st.radio(
                    "模型類型",
                    options=['CatBoost', 'XGBoost', 'LightGBM'],
                    index=0
                )
                
                use_class_weights = st.checkbox(
                    "使用類別權重",
                    value=True,
                    help="平衡 Long/Short 樣本數量"
                )
            
            with col2:
                st.markdown("**特徵組**")
                use_price = st.checkbox("價格特徵", value=True)
                use_volume = st.checkbox("量能特徵", value=True)
                use_trend = st.checkbox("趨勢特徵", value=True)
            
            with col3:
                st.markdown("**進階特徵**")
                use_reversal = st.checkbox("反轉特徵", value=True)
                use_pattern = st.checkbox("型態特徵", value=True)
        
        # 訓練控制
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            train_long = st.checkbox("訓練 Long 模型", value=True)
        
        with col2:
            train_short = st.checkbox("訓練 Short 模型", value=True)
        
        with col3:
            preview_only = st.checkbox(
                "僅預覽標籤",
                value=False,
                help="只生成標籤和統計，不訓練模型"
            )
        
        with col4:
            if st.button("🚀 開始", type="primary", use_container_width=True):
                if not train_long and not train_short and not preview_only:
                    st.error("請至少選擇一個方向或勾選'僅預覽標籤'")
                else:
                    config = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'train_days': train_days,
                        'train_split': train_split,
                        'zigzag_threshold': zigzag_threshold,
                        'min_swing_pct': min_swing_pct,
                        'lookahead_bars': lookahead_bars,
                        'tp_multiplier': tp_multiplier,
                        'sl_multiplier': sl_multiplier,
                        'use_rsi_div': use_rsi_div,
                        'use_volume_div': use_volume_div,
                        'use_sr': use_sr,
                        'model_type': model_type,
                        'use_class_weights': use_class_weights,
                        'train_long': train_long,
                        'train_short': train_short,
                        'feature_config': {
                            'price': use_price,
                            'volume': use_volume,
                            'trend': use_trend,
                            'reversal': use_reversal,
                            'pattern': use_pattern
                        },
                        'preview_only': preview_only
                    }
                    
                    self.run_training(config)
        
        # 訓練歷史
        st.markdown("---")
        st.subheader("📚 訓練歷史")
        self.display_training_history()
    
    def run_training(self, config):
        """執行訓練流程"""
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 步驟 1: 載入數據
            status_text.text("⏳ 步驟 1/6: 載入數據...")
            progress_bar.progress(10)
            
            from utils.hf_data_loader import load_klines
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=config['train_days'])
            
            df = load_klines(
                symbol=config['symbol'],
                timeframe=config['timeframe'],
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            if df is None or len(df) < 500:
                st.error("數據不足或載入失敗")
                return
            
            st.success(f"✅ 載入 {len(df)} 根 K 線")
            progress_bar.progress(20)
            
            # 步驟 2: ZigZag 計算
            status_text.text("⏳ 步驟 2/6: 計算 ZigZag 反轉點...")
            
            from utils.v11_zigzag import calculate_zigzag_pivots
            
            df = calculate_zigzag_pivots(
                df,
                threshold_pct=config['zigzag_threshold']
            )
            
            pivot_count = df['pivot_type'].notna().sum()
            st.info(f"🔶 識別 {pivot_count} 個 ZigZag 反轉點")
            progress_bar.progress(35)
            
            # 步驟 3: 反轉指標
            status_text.text("⏳ 步驟 3/6: 計算反轉指標...")
            
            from utils.v11_reversal_indicators import calculate_reversal_indicators
            
            df = calculate_reversal_indicators(df)
            progress_bar.progress(50)
            
            # 步驟 4: 生成標籤
            status_text.text("⏳ 步驟 4/6: 生成訓練標籤...")
            
            from utils.v11_labeling import create_v11_labels
            
            df = create_v11_labels(
                df,
                lookahead_bars=config['lookahead_bars'],
                tp_multiplier=config['tp_multiplier'],
                sl_multiplier=config['sl_multiplier'],
                require_rsi_div=config['use_rsi_div'],
                require_volume=config['use_volume_div'],
                require_sr=config['use_sr']
            )
            
            long_signals = (df['label'] == 1).sum()
            short_signals = (df['label'] == -1).sum()
            total_signals = long_signals + short_signals
            signal_rate = total_signals / len(df) * 100
            
            st.success(f"""
            🎯 **標籤統計:**
            - Long: {long_signals} ({long_signals/len(df)*100:.2f}%)
            - Short: {short_signals} ({short_signals/len(df)*100:.2f}%)
            - 總信號率: {signal_rate:.2f}%
            """)
            
            progress_bar.progress(65)
            
            # 顯示 ZigZag 圖表
            self.plot_zigzag_preview(df.tail(200))
            
            # 如果只是預覽，到這裡就結束
            if config['preview_only']:
                status_text.text("✅ 標籤預覽完成")
                progress_bar.progress(100)
                st.info("💡 取消勾選'僅預覽標籤'以開始訓練模型")
                return
            
            # 步驟 5: 特徵工程
            status_text.text("⏳ 步驟 5/6: 特徵工程...")
            
            from utils.v11_feature_engineering import create_v11_features
            
            df = create_v11_features(df, config['feature_config'])
            progress_bar.progress(75)
            
            # 步驟 6: 訓練模型
            status_text.text("⏳ 步驟 6/6: 訓練模型...")
            
            from train_zigzag_model import train_zigzag_model
            
            results = train_zigzag_model(
                df=df,
                symbol=config['symbol'],
                timeframe=config['timeframe'],
                train_split=config['train_split'],
                model_type=config['model_type'],
                use_class_weights=config['use_class_weights'],
                train_long=config['train_long'],
                train_short=config['train_short']
            )
            
            progress_bar.progress(100)
            status_text.text("✅ 訓練完成!")
            
            # 顯示結果
            self.display_training_results(results)
            
        except Exception as e:
            st.error(f"訓練失敗: {str(e)}")
            import traceback
            with st.expander("錯誤詳情"):
                st.code(traceback.format_exc())
    
    def plot_zigzag_preview(self, df):
        """繪製 ZigZag 預覽圖"""
        
        st.subheader("📈 ZigZag 視覺化 (最近 200 根)")
        
        fig = go.Figure()
        
        # K 線
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K線'
        ))
        
        # ZigZag 線
        pivot_df = df[df['pivot_type'].notna()]
        if len(pivot_df) > 0:
            fig.add_trace(go.Scatter(
                x=pivot_df.index,
                y=pivot_df['pivot_price'],
                mode='lines+markers',
                name='ZigZag',
                line=dict(color='yellow', width=2),
                marker=dict(size=8)
            ))
        
        # 標記 Long/Short 信號
        long_df = df[df['label'] == 1]
        short_df = df[df['label'] == -1]
        
        if len(long_df) > 0:
            fig.add_trace(go.Scatter(
                x=long_df.index,
                y=long_df['close'],
                mode='markers',
                name='Long 信號',
                marker=dict(symbol='triangle-up', size=12, color='green')
            ))
        
        if len(short_df) > 0:
            fig.add_trace(go.Scatter(
                x=short_df.index,
                y=short_df['close'],
                mode='markers',
                name='Short 信號',
                marker=dict(symbol='triangle-down', size=12, color='red')
            ))
        
        fig.update_layout(
            title="ZigZag 反轉點與交易信號",
            xaxis_title="時間",
            yaxis_title="價格",
            height=500,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_training_results(self, results):
        """顯示訓練結果"""
        
        st.success("🎉 訓練完成!")
        
        col1, col2 = st.columns(2)
        
        if 'long' in results:
            with col1:
                st.subheader("🟢 Long 模型")
                metrics = results['long']['metrics']
                
                m1, m2, m3 = st.columns(3)
                m1.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")
                m2.metric("AUC-PR", f"{metrics['auc_pr']:.3f}")
                m3.metric("F1", f"{metrics['f1']:.3f}")
                
                st.metric("標籤率", f"{metrics['label_rate']*100:.2f}%")
                st.caption(f"模型: {results['long']['model_path']}")
        
        if 'short' in results:
            with col2:
                st.subheader("🔴 Short 模型")
                metrics = results['short']['metrics']
                
                m1, m2, m3 = st.columns(3)
                m1.metric("AUC-ROC", f"{metrics['auc_roc']:.3f}")
                m2.metric("AUC-PR", f"{metrics['auc_pr']:.3f}")
                m3.metric("F1", f"{metrics['f1']:.3f}")
                
                st.metric("標籤率", f"{metrics['label_rate']*100:.2f}%")
                st.caption(f"模型: {results['short']['model_path']}")
    
    def display_training_history(self):
        """顯示訓練歷史"""
        
        import json
        
        reports = sorted(self.reports_dir.glob('zigzag_training_*.json'), reverse=True)
        
        if not reports:
            st.info("尚無訓練記錄")
            return
        
        for report_file in reports[:5]:
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                
                with st.expander(f"{report['timestamp']} - {report['symbol']} {report['timeframe']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'long' in report['results']:
                            st.markdown("**Long 模型**")
                            m = report['results']['long']['metrics']
                            st.text(f"AUC: {m['auc_roc']:.3f}")
                            st.text(f"標籤率: {m['label_rate']*100:.2f}%")
                    
                    with col2:
                        if 'short' in report['results']:
                            st.markdown("**Short 模型**")
                            m = report['results']['short']['metrics']
                            st.text(f"AUC: {m['auc_roc']:.3f}")
                            st.text(f"標籤率: {m['label_rate']*100:.2f}%")
            except:
                continue


def render():
    tab = ZigZagTrainingTab()
    tab.render()
