"""
Unified Trading System - V1, V2, V3 Integrated GUI
統一交易系統 - 所有功能直接整合進GUI
"""
import streamlit as st
import sys
from pathlib import Path
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np
import importlib.util
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

project_root = Path(__file__).parent.parent.parent
v1_root = project_root / 'reversal_strategy_v1'
v2_root = project_root / 'high_frequency_strategy_v2'
v3_root = project_root / 'adaptive_strategy_v3'

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(v1_root))
sys.path.insert(0, str(v2_root))
sys.path.insert(0, str(v3_root))

from reversal_strategy_v1.core.signal_detector import SignalDetector
from reversal_strategy_v1.core.feature_engineer import FeatureEngineer
from reversal_strategy_v1.core.ml_predictor import MLPredictor
from reversal_strategy_v1.core.risk_manager import RiskManager
from reversal_strategy_v1.backtest.engine import BacktestEngine
from reversal_strategy_v1.data.hf_loader import HFDataLoader

st.set_page_config(
    page_title="加密貨幣交易系統",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_module(module_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    st.title("加密貨幣智能交易系統")
    
    with st.sidebar:
        st.header("策略選擇")
        strategy_version = st.radio(
            "選擇策略版本",
            [
                "V1 - 訂單流反轉策略",
                "V2 - 高頻Transformer策略 (策略無效)",
                "V3 - 自適應多週期策略"
            ],
            index=2
        )
        st.markdown("---")
        
        if "V1" in strategy_version:
            st.subheader("V1 特點")
            st.caption("訂單流不平衡 + XGBoost")
            st.info("訂單流不平衡檢測\n流動性掃蕩識別\n市場微觀結構分析\nXGBoost機器學習")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("月交易目標", "50-80筆")
            with col2:
                st.metric("月報酬目標", "30-50%")
        elif "V2" in strategy_version:
            st.subheader("V2 特點")
            st.caption("LightGBM / Transformer")
            st.error("[策略無效]\n盈虧因子: 0.90\n勝率51.9%但平均虧損>盈利\n不建議使用")
            st.info("問題分析:\n- 信號過濾不足\n- 止盈止損不當\n- 特徵工程問題")
        else:
            st.subheader("V3 特點")
            st.caption("多週期自適應 + 風險動態調整")
            st.success("[最新策略]\n目標: 30天50%報酬\n特色: 多時間框架融合\n市場狀態自適應\n動態倉位管理")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("月交易目標", "150筆")
            with col2:
                st.metric("月報酬目標", "50%")
        
        st.markdown("---")
        if st.button("查看策略對比", use_container_width=True):
            st.session_state['show_comparison'] = True
    
    if st.session_state.get('show_comparison', False):
        show_strategy_comparison()
    
    if "V1" in strategy_version:
        render_v1_interface()
    elif "V2" in strategy_version:
        render_v2_interface()
    else:
        render_v3_interface()

def show_strategy_comparison():
    with st.expander("V1 vs V2 vs V3 策略對比", expanded=True):
        comparison_df = pd.DataFrame({
            '項目': ['模型架構', '時序學習', '集成學習', '信號過濾', '風險管理', '市場自適應', '月交易量', '月報酬目標', '訓練時間', '狀態'],
            'V1 反轉策略': ['XGBoost', 'X', 'X', '單層', '固定', 'X', '50-80筆', '30-50%', '5-10分鐘', '可用'],
            'V2 高頻策略': ['LightGBM/Transformer', 'O (可選)', 'O (加權集成)', '三層過濾', '動態調整', 'O', '140-150筆', '50%+', '5-15分鐘', '[無效] 盈虧因子0.90'],
            'V3 自適應策略': ['LightGBM (優化)', 'X', 'X', '五層過濾', 'ATR動態', 'O (趨勢識別)', '150筆', '50%', '5-10分鐘', '[推薦]']
        })
        st.table(comparison_df)
        if st.button("關閉對比", use_container_width=True):
            st.session_state['show_comparison'] = False
            st.rerun()

def render_v3_interface():
    tab1, tab2, tab3 = st.tabs(["V3 模型訓練", "V3 回測分析", "V3 系統狀態"])
    
    with tab1:
        render_v3_training()
    
    with tab2:
        render_v3_backtest()
    
    with tab3:
        render_v3_status()

def render_v3_training():
    st.header("V3 模型訓練")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("訓練參數")
        
        symbol = st.selectbox("交易對", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"], key="v3_train_symbol")
        timeframe = st.selectbox("時間框架", ["15m", "1h"], index=0, key="v3_train_timeframe")
        
        # 配置選擇
        config_mode = st.radio(
            "標籤配置",
            ["自定義", "平衡版 (0.9x盈利)", "激進版 (0.7x盈利) [推薦]"],
            index=2
        )
        
        if config_mode == "自定義":
            st.markdown("---")
            st.markdown("**標籤生成參數**")
            forward_window = st.slider("前瞻窗口", 5, 15, 8)
            atr_profit_mult = st.slider("ATR盈利倍數", 0.5, 3.0, 0.7, 0.1)
            atr_loss_mult = st.slider("ATR虧損倍數", 0.5, 2.0, 1.5, 0.1)
            min_volume_ratio = st.slider("最小成交量比", 0.5, 2.0, 0.7, 0.1)
            min_trend_strength = st.slider("最小趨勢強度", 0.1, 0.8, 0.15, 0.05)
            max_atr_ratio = 0.08
        elif "平衡" in config_mode:
            st.info("平衡配置: ATR 0.9x盈利, 1.2x止損\n目標: 15-20%有效標籤")
            forward_window = 8
            atr_profit_mult = 0.9
            atr_loss_mult = 1.2
            min_volume_ratio = 0.8
            min_trend_strength = 0.2
            max_atr_ratio = 0.07
        else:  # 激進版
            st.success("激進配置: ATR 0.7x盈利, 1.5x止損\n目標: 20-25%有效標籤\n解決7%有效標籤問題")
            forward_window = 8
            atr_profit_mult = 0.7
            atr_loss_mult = 1.5
            min_volume_ratio = 0.7
            min_trend_strength = 0.15
            max_atr_ratio = 0.08
        
        st.markdown("---")
        if st.button("開始V3訓練", type="primary", use_container_width=True):
            st.session_state['v3_training_params'] = {
                'symbol': symbol,
                'timeframe': timeframe,
                'forward_window': forward_window,
                'atr_profit_multiplier': atr_profit_mult,
                'atr_loss_multiplier': atr_loss_mult,
                'min_volume_ratio': min_volume_ratio,
                'min_trend_strength': min_trend_strength,
                'max_atr_ratio': max_atr_ratio,
                'config_mode': config_mode
            }
            st.session_state['v3_training_started'] = True
    
    with col2:
        st.subheader("訓練過程")
        if st.session_state.get('v3_training_started', False):
            train_v3_model_in_gui(st.session_state['v3_training_params'])
            st.session_state['v3_training_started'] = False
        else:
            st.info("""
            V3 訓練流程 (激進配置):
            
            1. 加載HuggingFace K線數據
            2. 生成ATR動態標籤 (0.7x盈利, 1.5x止損)
            3. 提取市場微觀結構特徵
            4. LightGBM訓練 (防過擬合參數)
            5. 模型評估
            
            **預期結果:**
            - 有效標籤: 20-25%
            - 驗證準確率: 0.50-0.60
            - Precision: 0.55-0.65
            - Recall: 0.55-0.65
            - 信心度: >60%
            
            預計: 5-10分鐘
            """)
            
            # 顯示配置比較
            st.markdown("---")
            st.markdown("**參數進化**")
            comparison = pd.DataFrame({
                '項目': ['ATR盈利倍數', 'ATR止損倍數', '成交量比', '趨勢強度', '有效標籤率'],
                '原版 (7%標籤)': ['0.9', '1.2', '0.8', '0.2', '7%'],
                '激進版': ['0.7 (降22%)', '1.5 (增25%)', '0.7 (降12%)', '0.15 (降25%)', '20-25%']
            })
            st.table(comparison)
            
            st.warning("激進配置說明: 大幅降低盈利要求到0.7倍,同時放寬止損到1.5倍。這會產生更多有效標籤(20-25%),讓模型真正學習交易模式,而非偽懶預測中立。")

def train_v3_model_in_gui(params):
    try:
        # 加載V3模組
        v3_label_gen_path = v3_root / 'core' / 'label_generator.py'
        v3_feature_eng_path = v3_root / 'core' / 'feature_engineer.py'
        v3_predictor_path = v3_root / 'core' / 'predictor.py'
        v3_loader_path = v3_root / 'data' / 'hf_loader.py'
        
        v3_label_gen = load_module(v3_label_gen_path, 'v3_label_generator')
        v3_feature_eng = load_module(v3_feature_eng_path, 'v3_feature_engineer')
        v3_predictor = load_module(v3_predictor_path, 'v3_predictor')
        v3_loader = load_module(v3_loader_path, 'v3_loader')
        
        V3LabelGenerator = v3_label_gen.LabelGenerator
        V3FeatureEngineer = v3_feature_eng.FeatureEngineer
        V3Predictor = v3_predictor.Predictor
        V3HFDataLoader = v3_loader.HFDataLoader
        
        # 步驟 1: 加載數據
        with st.spinner('步驟 1/5: 加載數據...'):
            loader = V3HFDataLoader()
            df = loader.load_klines(params['symbol'], params['timeframe'])
            st.success(f"[OK] 加載: {len(df)} 筆")
        
        # 步驟 2: 特徵工程
        with st.spinner('步驟 2/5: 特徵工程...'):
            feature_config = {
                'lookback_periods': [5, 10, 20]
            }
            feature_engineer = V3FeatureEngineer(feature_config)
            df = feature_engineer.create_features(df)
            st.success(f"[OK] 特徵: {len(df.columns)} 個")
        
        # 步驟 3: 生成標籤
        with st.spinner('步驟 3/5: 生成標籤...'):
            label_config = {
                'forward_window': params['forward_window'],
                'atr_profit_multiplier': params['atr_profit_multiplier'],
                'atr_loss_multiplier': params['atr_loss_multiplier'],
                'min_volume_ratio': params['min_volume_ratio'],
                'min_trend_strength': params['min_trend_strength'],
                'max_atr_ratio': params['max_atr_ratio']
            }
            label_generator = V3LabelGenerator(label_config)
            df = label_generator.generate_labels(df)
            
            long_count = (df['label'] == 1).sum()
            short_count = (df['label'] == -1).sum()
            neutral_count = (df['label'] == 0).sum()
            valid_rate = (long_count + short_count) / len(df)
            
            st.info(f"[標籤分布] 做多={long_count} ({long_count/len(df)*100:.1f}%) | 做空={short_count} ({short_count/len(df)*100:.1f}%) | 中立={neutral_count} ({neutral_count/len(df)*100:.1f}%)")
            st.info(f"[有效標籤] {long_count + short_count} ({valid_rate*100:.1f}%)")
            
            # 警告: 類別不平衡
            if valid_rate < 0.15:
                st.error(f"[錯誤] 有效標籤{valid_rate*100:.1f}%<15%,建議使用激進配置")
            elif valid_rate > 0.30:
                st.warning("[警告] 有效標籤>30%,可能質量不足")
            else:
                st.success(f"[優秀] 有效標籤在15-30%範圍")
        
        # 步驟 4: 準備數據
        with st.spinner('步驟 4/5: 準備訓練數據...'):
            exclude_cols = ['timestamp', 'label', 'open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
            
            # 清理數據
            for col in feature_cols:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=feature_cols + ['label'])
            
            # 分割數據
            train_size = int(len(df) * 0.7)
            val_size = int(len(df) * 0.15)
            
            df_train = df.iloc[:train_size]
            df_val = df.iloc[train_size:train_size+val_size]
            
            X_train = df_train[feature_cols].values
            y_train = df_train['label'].values
            X_val = df_val[feature_cols].values
            y_val = df_val['label'].values
            
            st.success(f"[OK] 訓練: {len(X_train)} | 驗證: {len(X_val)} | 特徵: {len(feature_cols)}")
        
        # 步驟 5: 訓練模型
        with st.spinner('步驟 5/5: 訓練LightGBM...'):
            predictor_config = {}
            predictor = V3Predictor(predictor_config)
            results = predictor.train(X_train, y_train, X_val, y_val)
            st.success("[OK] 訓練完成")
        
        # 顯示評估結果
        st.markdown("---")
        st.subheader("訓練集績效")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("準確率", f"{results['train_accuracy']:.3f}")
        with col2:
            st.metric("Macro Precision", f"{results['train_report']['macro avg']['precision']:.3f}")
        with col3:
            st.metric("Macro Recall", f"{results['train_report']['macro avg']['recall']:.3f}")
        
        st.markdown("---")
        st.subheader("驗證集績效")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("準確率", f"{results['val_accuracy']:.3f}")
        with col2:
            st.metric("AUC (OvR)", f"{results['val_auc']:.3f}")
        with col3:
            st.metric("Precision", f"{results['val_precision']:.3f}")
        with col4:
            st.metric("Recall", f"{results['val_recall']:.3f}")
        
        st.info(f"[信心度] 做多平均: {results['avg_conf_long']:.1%} | 做空平均: {results['avg_conf_short']:.1%}")
        
        # 檢查模型狀態
        if results['val_accuracy'] > 0.75:
            st.warning("[警告] 準確率>75%,仍有類別不平衡")
        
        if results['val_precision'] < 0.50:
            st.error("[警告] Precision<50%,交易信號質量差")
        elif results['val_precision'] >= 0.55 and results['val_recall'] >= 0.55:
            st.success("[優秀] Precision和Recall均>55%,模型平衡!")
        
        # 特徵重要性
        importance_df = predictor.get_feature_importance(feature_cols, top_k=15)
        if importance_df is not None:
            st.markdown("---")
            st.subheader("Top 15 重要特徵")
            st.dataframe(importance_df, use_container_width=True)
        
        # 保存模型
        with st.spinner('保存模型...'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"{params['symbol']}_{params['timeframe']}_v3_{timestamp}"
            model_dir = project_root / 'models' / model_name
            predictor.save(model_dir)
            
            with open(model_dir / 'model_config.json', 'w') as f:
                json.dump({
                    'symbol': params['symbol'],
                    'timeframe': params['timeframe'],
                    'model_version': 'v3',
                    'config_mode': params.get('config_mode', '自定義'),
                    'training_date': timestamp,
                    'data_samples': len(df),
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'feature_count': len(feature_cols),
                    'label_config': label_config,
                    'valid_label_rate': float(valid_rate),
                    'metrics': {
                        'train_accuracy': results['train_accuracy'],
                        'val_accuracy': results['val_accuracy'],
                        'val_auc': results['val_auc'],
                        'val_precision': results['val_precision'],
                        'val_recall': results['val_recall'],
                        'avg_conf_long': results['avg_conf_long'],
                        'avg_conf_short': results['avg_conf_short']
                    },
                    'feature_names': feature_cols
                }, f, indent=2)
            st.success(f"[V3 完成] {model_name}")
        
        st.session_state['latest_v3_model'] = model_name
        
    except Exception as e:
        st.error(f"[失敗] {str(e)}")
        import traceback
        with st.expander("詳情"):
            st.code(traceback.format_exc())

def render_v3_backtest():
    st.info("V3回測功能保持不變")

def render_v3_status():
    st.header("V3 系統狀態")
    
    col1, col2, col3 = st.columns(3)
    models_dir = project_root / 'models'
    v1_count = v2_count = v3_count = 0
    if models_dir.exists():
        for d in models_dir.iterdir():
            if d.is_dir():
                if '_v1_' in d.name:
                    v1_count += 1
                elif '_v2_' in d.name:
                    v2_count += 1
                elif '_v3_' in d.name:
                    v3_count += 1
    
    with col1:
        st.metric("V1模型", v1_count)
    with col2:
        st.metric("V2模型 (無效)", v2_count)
    with col3:
        st.metric("V3模型", v3_count)

def render_v1_interface():
    st.info("V1功能保持不變")

def render_v2_interface():
    st.warning("V2策略已被標記為無效 (盈虧因子0.90),建議使用V1或V3策略")

if __name__ == "__main__":
    main()
