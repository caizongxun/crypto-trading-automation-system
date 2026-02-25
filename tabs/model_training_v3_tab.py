"""
V3 Model Training Tab - GUI Interface

Provides easy interface to train V3 models with optimized settings

Author: Zong
Version: 3.0.0
Date: 2026-02-25
"""

import streamlit as st
import subprocess
import sys
from pathlib import Path
import json
from datetime import datetime
import pandas as pd

class ModelTrainingV3Tab:
    """
    V3 Model Training GUI Tab
    """
    
    def __init__(self):
        self.version = "3.0.0"
        self.models_dir = Path("models_output")
        self.reports_dir = Path("training_reports")
    
    def render(self):
        st.header("⭐ V3 模型訓練 - 優化版")
        
        st.markdown("""
        V3 是完全重新設計的模型,修復了 V2 的所有問題。
        
        **核心改進**:
        - ✅ 更激進的標籤 (1.2% TP vs 2%)
        - ✅ 更高的信號率 (5-10% vs <2%)
        - ✅ 更好的機率校準 (Max 0.6-0.8 vs 0.21)
        - ✅ 30個精選特徵 (vs 54)
        """)
        
        st.markdown("---")
        
        # 主要區域
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🛠️ 訓練設定")
            
            # 基本設定
            with st.expander("🎯 基本設定", expanded=True):
                st.markdown("""
                **預設設定已優化,直接使用即可**
                """)
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("目標 TP", "1.2%", help="止盈目標 (從 2% 降低)")
                    st.metric("停損 SL", "0.8%", help="止損限制 (從 1% 降低)")
                
                with col_b:
                    st.metric("Lookahead", "240 bars (4h)", help="從 8h 縮短到 4h")
                    st.metric("特徵數", "30", help="從 54 精簡到 30")
                
                st.info("📚 [查看完整設定說明](V3_MODEL_GUIDE.md#訓練參數)")
            
            # 進階設定 (不建議修改)
            with st.expander("⚙️ 進階設定 (不建議修改)"):
                st.warning("⚠️ 修改這些設定可能導致模型效果下降")
                
                tp_target = st.slider(
                    "TP 目標 (%)",
                    min_value=0.8,
                    max_value=2.0,
                    value=1.2,
                    step=0.1,
                    help="太高會導致標籤太少"
                )
                
                sl_stop = st.slider(
                    "SL 限制 (%)",
                    min_value=0.5,
                    max_value=1.5,
                    value=0.8,
                    step=0.1,
                    help="太寬會導致勝率下降"
                )
                
                lookahead = st.slider(
                    "Lookahead (bars)",
                    min_value=120,
                    max_value=480,
                    value=240,
                    step=60,
                    help="120=2h, 240=4h, 480=8h"
                )
                
                st.session_state.v3_custom_params = {
                    'tp_target': tp_target / 100,
                    'sl_stop': sl_stop / 100,
                    'lookahead_bars': lookahead
                }
        
        with col2:
            st.subheader("📊 預期結果")
            
            st.markdown("""
            **訓練指標**:
            - Label Rate: 5-10%
            - AUC: 0.65-0.72
            - Max Prob: 0.60-0.80
            - Precision @ 0.15: 60-65%
            
            **回測指標 (90天)**:
            - 交易數: 150-300
            - 勝率: 45-55%
            - Profit Factor: 1.5-2.5
            - 總報酬: 5-15% (1x)
            """)
            
            st.info("📚 [查看詳細指標說明](V3_MODEL_GUIDE.md#預期績效)")
        
        st.markdown("---")
        
        # 訓練控制
        st.subheader("🚀 開始訓練")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ 開始 V3 訓練", type="primary", use_container_width=True):
                self._start_training()
        
        with col2:
            if st.button("📊 查看訓練 Log", use_container_width=True):
                self._show_training_log()
        
        with col3:
            if st.button("📝 查看最新報告", use_container_width=True):
                self._show_latest_report()
        
        st.markdown("---")
        
        # 現有模型
        st.subheader("📦 V3 模型列表")
        self._show_v3_models()
    
    def _start_training(self):
        """
        啟動 V3 訓練流程
        """
        st.info("🚀 正在啟動 V3 訓練...預計 30-60 分鐘")
        
        try:
            # 執行 train_v3.py
            process = subprocess.Popen(
                [sys.executable, "train_v3.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # 建立容器來顯示輸出
            output_container = st.empty()
            error_container = st.empty()
            
            output_lines = []
            
            # 即時顯示輸出
            with st.spinner("正在訓練..."):
                for line in process.stdout:
                    output_lines.append(line)
                    # 只顯示最後 20 行
                    output_container.code("\n".join(output_lines[-20:]), language="log")
                
                # 等待完成
                return_code = process.wait()
                
                if return_code == 0:
                    st.success("✅ V3 模型訓練完成!")
                    st.balloons()
                    
                    # 顯示下一步
                    st.info("""
                    **下一步**:
                    1. 點擊 "📝 查看最新報告" 查看訓練結果
                    2. 到 "📈 策略回測" 標籤進行回測
                    3. 選擇 V3 模型,設定閾值 0.15
                    """)
                else:
                    st.error(f"❌ 訓練失敗 (exit code: {return_code})")
                    stderr = process.stderr.read()
                    error_container.error(f"錯誤訊息:\n{stderr}")
        
        except Exception as e:
            st.error(f"❌ 訓練失敗: {str(e)}")
            st.info("""
            **手動訓練方式**:
            ```bash
            python train_v3.py
            ```
            """)
    
    def _show_training_log(self):
        """
        顯示訓練 log
        """
        log_file = Path("logs/train_v3.log")
        
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # 顯示最後 100 行
                st.code("".join(lines[-100:]), language="log")
        else:
            st.warning("⚠️ 尚未找到訓練 log")
            st.info("需要先執行訓練")
    
    def _show_latest_report(self):
        """
        顯示最新的訓練報告
        """
        if not self.reports_dir.exists():
            st.warning("⚠️ 尚未找到訓練報告")
            return
        
        # 找到最新的 V3 報告
        v3_reports = sorted(self.reports_dir.glob("v3_training_report_*.json"), reverse=True)
        
        if not v3_reports:
            st.warning("⚠️ 尚未找到 V3 訓練報告")
            st.info("需要先執行 V3 訓練")
            return
        
        latest_report = v3_reports[0]
        
        with open(latest_report, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        st.success(f"✅ 找到報告: {latest_report.name}")
        
        # 顯示關鍵指標
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👉 Long 模型")
            
            long_metrics = report['long_model']['metrics']
            prob_stats = report['long_model']['probability_stats']
            
            st.metric("AUC", f"{long_metrics['auc']:.4f}")
            st.metric("Precision", f"{long_metrics['precision']:.2%}")
            st.metric("Recall", f"{long_metrics['recall']:.2%}")
            st.metric("F1-Score", f"{long_metrics['f1']:.4f}")
            
            st.markdown("**機率分佈**:")
            st.metric("Max Prob", f"{prob_stats['max']:.4f}")
            st.metric("95th Percentile", f"{prob_stats['p95']:.4f}")
        
        with col2:
            st.subheader("👉 Short 模型")
            
            short_metrics = report['short_model']['metrics']
            prob_stats = report['short_model']['probability_stats']
            
            st.metric("AUC", f"{short_metrics['auc']:.4f}")
            st.metric("Precision", f"{short_metrics['precision']:.2%}")
            st.metric("Recall", f"{short_metrics['recall']:.2%}")
            st.metric("F1-Score", f"{short_metrics['f1']:.4f}")
            
            st.markdown("**機率分佈**:")
            st.metric("Max Prob", f"{prob_stats['max']:.4f}")
            st.metric("95th Percentile", f"{prob_stats['p95']:.4f}")
        
        # 驗證檢查
        st.markdown("---")
        st.subheader("✅ 驗證檢查")
        
        checks = []
        
        # 檢查 AUC
        if long_metrics['auc'] > 0.65 and short_metrics['auc'] > 0.65:
            checks.append("✅ AUC > 0.65 (通過)")
        else:
            checks.append("❌ AUC < 0.65 (建議重訓)")
        
        # 檢查 Max Prob
        long_max = report['long_model']['probability_stats']['max']
        short_max = report['short_model']['probability_stats']['max']
        
        if long_max > 0.60 and short_max > 0.60:
            checks.append("✅ Max Probability > 0.60 (通過)")
        elif long_max > 0.50 and short_max > 0.50:
            checks.append("⚠️ Max Probability 0.50-0.60 (可接受)")
        else:
            checks.append("❌ Max Probability < 0.50 (需要調整)")
        
        # 檢查 Precision
        if long_metrics['precision'] > 0.55 and short_metrics['precision'] > 0.55:
            checks.append("✅ Precision > 0.55 (通過)")
        else:
            checks.append("⚠️ Precision < 0.55 (注意)")
        
        for check in checks:
            st.markdown(check)
        
        # 完整 JSON
        with st.expander("📝 查看完整報告"):
            st.json(report)
    
    def _show_v3_models(self):
        """
        顯示現有 V3 模型
        """
        if not self.models_dir.exists():
            st.info("尚未找到模型目錄")
            return
        
        v3_long = list(self.models_dir.glob("catboost_long_v3_*.pkl"))
        v3_short = list(self.models_dir.glob("catboost_short_v3_*.pkl"))
        
        if not v3_long and not v3_short:
            st.info("📦 尚未訓練 V3 模型")
            st.markdown("點擊上方 **'▶️ 開始 V3 訓練'** 按鈕開始訓練")
            return
        
        # Long 模型
        if v3_long:
            st.markdown("**👉 Long 模型**:")
            for model_file in sorted(v3_long, reverse=True):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text(model_file.name)
                with col2:
                    size_mb = model_file.stat().st_size / 1024 / 1024
                    st.text(f"{size_mb:.2f} MB")
                with col3:
                    mod_time = datetime.fromtimestamp(model_file.stat().st_mtime)
                    st.text(mod_time.strftime("%m-%d %H:%M"))
        
        # Short 模型
        if v3_short:
            st.markdown("**👉 Short 模型**:")
            for model_file in sorted(v3_short, reverse=True):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text(model_file.name)
                with col2:
                    size_mb = model_file.stat().st_size / 1024 / 1024
                    st.text(f"{size_mb:.2f} MB")
                with col3:
                    mod_time = datetime.fromtimestamp(model_file.stat().st_mtime)
                    st.text(mod_time.strftime("%m-%d %H:%M"))