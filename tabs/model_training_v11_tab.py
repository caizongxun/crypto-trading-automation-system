#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11 模型訓練 Tab - 使用 ZigZag 反轉點標籤

**V11 核心特色:**
1. ZigZag 反轉點標籤 (標記真實高低點)
2. 多種反轉指標結合
3. 提前 N根K線預測反轉
4. 更高的標籤率 (10-20%)
5. 適合波段交易
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


class ModelTrainingV11Tab:
    def __init__(self):
        self.models_dir = Path('models_output')
        self.models_dir.mkdir(exist_ok=True)
        
        self.reports_dir = Path('training_reports/v11')
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def render(self):  
        st.header("🔶 V11 模型訓練 - ZigZag 反轉點標籤")
        
        st.info("""
        **V11 核心改進:**
        
        🔶 **ZigZag 反轉點標籤**
        - 標記真實的高點和低點
        - 排除市場噪音
        - 捕捉重要的波段轉折
        
        📈 **多種反轉指標**
        - RSI 背離 (Regular + Hidden)
        - MACD 交叉 + 背離
        - 支撐阻力突破
        - 布林帶反彈
        - 量能發散
        
        ⏱️ **提前N根預測**
        - 在反轉發生前 1-3 根K線提示
        - 更實用的進場時機
        
        🎯 **預期效果**
        - 標籤率: 10-20% (比 v10 高)
        - 勝率: 50-60%
        - 持倉時間: 2-6 小時
        - 適合波段交易
        """)
        
        # 標籤配置
        st.subheader("⚙️ 標籤配置")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**ZigZag 參數**")
            zigzag_threshold = st.slider(
                "ZigZag 門檼 (%)",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                help="反轉門檼,越大越少反轉點但越重要"
            )
            
            lookahead_bars = st.number_input(
                "提前預警K線數",
                min_value=0,
                max_value=5,
                value=2,
                help="在反轉發生前 N 根K線就標記為信號"
            )
        
        with col2:
            st.markdown("**反轉確認**")
            
            require_rsi_div = st.checkbox(
                "RSI 背離確認",
                value=True,
                help="只標記有 RSI 背離的反轉點"
            )
            
            require_volume = st.checkbox(
                "量能確認",
                value=False,
                help="需要量能增加才標記"
            )
            
            require_sr = st.checkbox(
                "支撐/阻力確認",
                value=False,
                help="只標記在支撐/阻力位的反轉"
            )
        
        with col3:
            st.markdown("**目標設定**")
            
            tp_multiplier = st.slider(
                "TP 倍數",
                min_value=0.5,
                max_value=3.0,
                value=1.5,
                step=0.1,
                help="根據 ZigZag 振幅計算 TP"
            )
            
            sl_multiplier = st.slider(
                "SL 倍數",
                min_value=0.3,
                max_value=1.5,
                value=0.5,
                step=0.1,
                help="根據 ZigZag 振幅計算 SL"
            )
        
        st.markdown("---")
        
        # 訓練配置
        st.subheader("🏋️ 訓練配置")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            symbol = st.selectbox(
                "交易對",
                options=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'],
                index=0
            )
        
        with col2:
            timeframe = st.selectbox(
                "時間框架",
                options=['15m', '1h', '4h', '1d'],
                index=1  # 預設 1h,適合波段交易
            )
        
        with col3:
            train_days = st.number_input(
                "訓練天數",
                min_value=30,
                max_value=365,
                value=180,
                step=30
            )
        
        with col4:
            train_split = st.slider(
                "訓練集比例",
                min_value=0.6,
                max_value=0.9,
                value=0.8,
                step=0.05
            )
        
        # 模型配置
        with st.expander("🧠 模型進階配置"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**模型選擇**")
                model_type = st.radio(
                    "模型類型",
                    options=['CatBoost', 'XGBoost', 'LightGBM'],
                    index=0
                )
                
                use_class_weights = st.checkbox(
                    "使用類別權重",
                    value=True,
                    help="平衡long/short樣本數量"
                )
            
            with col2:
                st.markdown("**特徵選擇**")
                
                use_price_features = st.checkbox("價格特徵", value=True)
                use_volume_features = st.checkbox("量能特徵", value=True)
                use_trend_features = st.checkbox("趨勢特徵", value=True)
                use_reversal_features = st.checkbox("反轉特徵", value=True)
                use_pattern_features = st.checkbox("型態特徵", value=True)
        
        st.markdown("---")
        
        # 訓練控制
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            train_long = st.checkbox("訓練 Long 模型", value=True)
        
        with col2:
            train_short = st.checkbox("訓練 Short 模型", value=True)
        
        with col3:
            if st.button("🚀 開始訓練", type="primary", use_container_width=True):
                if not train_long and not train_short:
                    st.error("請至少選擇一個方向")
                else:
                    self.run_training(
                        symbol=symbol,
                        timeframe=timeframe,
                        train_days=train_days,
                        train_split=train_split,
                        zigzag_threshold=zigzag_threshold,
                        lookahead_bars=lookahead_bars,
                        tp_multiplier=tp_multiplier,
                        sl_multiplier=sl_multiplier,
                        require_rsi_div=require_rsi_div,
                        require_volume=require_volume,
                        require_sr=require_sr,
                        model_type=model_type,
                        use_class_weights=use_class_weights,
                        train_long=train_long,
                        train_short=train_short,
                        feature_config={
                            'price': use_price_features,
                            'volume': use_volume_features,
                            'trend': use_trend_features,
                            'reversal': use_reversal_features,
                            'pattern': use_pattern_features
                        }
                    )
        
        # 顯示最近訓練記錄
        st.markdown("---")
        st.subheader("📊 訓練歷史")
        
        self.display_training_history()
    
    def run_training(self, **config):
        """V11 訓練流程"""
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 步驟 1: 載入數據
            status_text.text("✅ 步驟 1/6: 載入數據...")
            progress_bar.progress(10)
            
            from utils.hf_data_loader import load_klines
            from datetime import datetime, timedelta
            
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
            
            st.success(f"✅ 載入 {len(df)} 根K線")
            progress_bar.progress(20)
            
            # 步驟 2: ZigZag 計算
            status_text.text("✅ 步驟 2/6: 計算 ZigZag 反轉點...")
            progress_bar.progress(30)
            
            from utils.v11_zigzag import calculate_zigzag_pivots
            
            df = calculate_zigzag_pivots(
                df,
                threshold_pct=config['zigzag_threshold']
            )
            
            pivot_count = df['pivot_type'].notna().sum()
            st.info(f"🔶 識別 {pivot_count} 個 ZigZag 反轉點")
            progress_bar.progress(40)
            
            # 步驟 3: 反轉指標計算
            status_text.text("✅ 步驟 3/6: 計算反轉指標...")
            progress_bar.progress(50)
            
            from utils.v11_reversal_indicators import calculate_reversal_indicators
            
            df = calculate_reversal_indicators(df)
            progress_bar.progress(60)
            
            # 步驟 4: 生成標籤
            status_text.text("✅ 步驟 4/6: 生成訓練標籤...")
            progress_bar.progress(70)
            
            from utils.v11_labeling import create_v11_labels
            
            df = create_v11_labels(
                df,
                lookahead_bars=config['lookahead_bars'],
                tp_multiplier=config['tp_multiplier'],
                sl_multiplier=config['sl_multiplier'],
                require_rsi_div=config['require_rsi_div'],
                require_volume=config['require_volume'],
                require_sr=config['require_sr']
            )
            
            long_signals = (df['label'] == 1).sum()
            short_signals = (df['label'] == -1).sum()
            signal_rate = (long_signals + short_signals) / len(df) * 100
            
            st.success(f"""
            🎯 標籤統計:
            - Long: {long_signals} ({long_signals/len(df)*100:.2f}%)
            - Short: {short_signals} ({short_signals/len(df)*100:.2f}%)
            - 總信號率: {signal_rate:.2f}%
            """)
            progress_bar.progress(75)
            
            # 步驟 5: 特徵工程
            status_text.text("✅ 步驟 5/6: 特徵工程...")
            
            from utils.v11_feature_engineering import create_v11_features
            
            df = create_v11_features(df, config['feature_config'])
            progress_bar.progress(80)
            
            # 步驟 6: 模型訓練
            status_text.text("✅ 步驟 6/6: 訓練模型...")
            
            from train_model_v11 import train_v11_model
            
            results = train_v11_model(
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
            st.code(traceback.format_exc())
    
    def display_training_results(self, results):
        """顯示訓練結果"""
        st.success("🎉 訓練完成!")
        
        col1, col2 = st.columns(2)
        
        if 'long' in results:
            with col1:
                st.subheader("🟢 Long 模型")
                long_metrics = results['long']['metrics']
                
                m1, m2, m3 = st.columns(3)
                m1.metric("AUC-ROC", f"{long_metrics['auc_roc']:.3f}")
                m2.metric("AUC-PR", f"{long_metrics['auc_pr']:.3f}")
                m3.metric("F1", f"{long_metrics['f1']:.3f}")
                
                st.metric("標籤率", f"{long_metrics['label_rate']*100:.2f}%")
                st.metric("樣本數", f"{long_metrics['n_samples']:,}")
        
        if 'short' in results:
            with col2:
                st.subheader("🔴 Short 模型")
                short_metrics = results['short']['metrics']
                
                m1, m2, m3 = st.columns(3)
                m1.metric("AUC-ROC", f"{short_metrics['auc_roc']:.3f}")
                m2.metric("AUC-PR", f"{short_metrics['auc_pr']:.3f}")
                m3.metric("F1", f"{short_metrics['f1']:.3f}")
                
                st.metric("標籤率", f"{short_metrics['label_rate']*100:.2f}%")
                st.metric("樣本數", f"{short_metrics['n_samples']:,}")
        
        st.info("✅ 模型已保存到 models_output/")
        st.info("📄 報告已保存到 training_reports/v11/")
    
    def display_training_history(self):
        """顯示訓練歷史"""
        import json
        
        reports = sorted(self.reports_dir.glob('v11_training_*.json'), reverse=True)
        
        if not reports:
            st.info("尚無訓練記錄")
            return
        
        # 顯示最近 5 筆
        for report_file in reports[:5]:
            with open(report_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            with st.expander(f"{report['timestamp']} - {report['symbol']} {report['timeframe']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**配置**")
                    st.text(f"ZigZag: {report['config']['zigzag_threshold']}%")
                    st.text(f"Lookahead: {report['config']['lookahead_bars']}")
                    st.text(f"TP/SL: {report['config']['tp_multiplier']:.1f}x / {report['config']['sl_multiplier']:.1f}x")
                
                with col2:
                    if 'long' in report['results']:
                        st.markdown("**Long 模型**")
                        m = report['results']['long']['metrics']
                        st.text(f"AUC: {m['auc_roc']:.3f}")
                        st.text(f"標籤率: {m['label_rate']*100:.2f}%")
                
                with col3:
                    if 'short' in report['results']:
                        st.markdown("**Short 模型**")
                        m = report['results']['short']['metrics']
                        st.text(f"AUC: {m['auc_roc']:.3f}")
                        st.text(f"標籤率: {m['label_rate']*100:.2f}%")


def render():
    tab = ModelTrainingV11Tab()
    tab.render()
