#!/usr/bin/env python3
"""
Quick fix script to add V3 support to backtesting_tab.py

Usage:
    python fix_backtest_v3.py
"""

import re
from pathlib import Path

def fix_backtesting_tab():
    file_path = Path("tabs/backtesting_tab.py")
    
    if not file_path.exists():
        print(f"Error: {file_path} not found")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Applying V3 support patches...\n")
    
    # 1. Add V3 import
    if 'from utils.feature_engineering_v3 import FeatureEngineerV3' not in content:
        v2_block = '''try:
    from utils.feature_engineering_v2 import FeatureEngineerV2
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False'''
        
        v3_block = v2_block + '''\n\n# V3
try:
    from utils.feature_engineering_v3 import FeatureEngineerV3
    V3_AVAILABLE = True
except ImportError:
    V3_AVAILABLE = False'''
        
        content = content.replace(v2_block, v3_block)
        print("[1/7] Added V3 import")
    else:
        print("[1/7] V3 import already exists")
    
    # 2. Init V3
    if 'self.feature_engineer_v3' not in content:
        init_pattern = r'(if V2_AVAILABLE:\s+self\.feature_engineer_v2 = FeatureEngineerV2\([^)]+\))'
        init_replacement = r'\1\n        if V3_AVAILABLE:\n            self.feature_engineer_v3 = FeatureEngineerV3()'
        content = re.sub(init_pattern, init_replacement, content)
        print("[2/7] Added V3 initialization")
    else:
        print("[2/7] V3 initialization already exists")
    
    # 3. Update render status
    old_status = '''if V2_AVAILABLE:
            st.success("V2 系統已啟用 - 支持 V1 和 V2 模型")
        else:
            st.info("當前僅支持 V1 模型")'''
    
    new_status = '''if V3_AVAILABLE:
            st.success("V3 系統已啟用 - 支持 V1/V2/V3 模型 (推薦 V3)")
        elif V2_AVAILABLE:
            st.info("V2 系統已啟用 - 支持 V1 和 V2 模型")
        else:
            st.info("當前僅支持 V1 模型")'''
    
    content = content.replace(old_status, new_status)
    print("[3/7] Updated status display")
    
    # 4-6. Add V3 to model selector (replace entire function)
    selector_pattern = r'def render_model_selector\(self, prefix: str, direction: str\) -> tuple:.*?(?=\n    def )'
    
    new_selector = '''def render_model_selector(self, prefix: str, direction: str) -> tuple:
        """模型選擇器 - 支持 V1/V2/V3"""
        models_dir = Path("models_output")
        
        if direction == "long":
            v1_models = list(models_dir.glob("catboost_long_[0-9]*.pkl"))
            v2_models = list(models_dir.glob("catboost_long_v2_*.pkl"))
            v3_models = list(models_dir.glob("catboost_long_v3_*.pkl"))
        else:
            v1_models = list(models_dir.glob("catboost_short_[0-9]*.pkl"))
            v2_models = list(models_dir.glob("catboost_short_v2_*.pkl"))
            v3_models = list(models_dir.glob("catboost_short_v3_*.pkl"))
        
        v1_models = sorted(v1_models, key=lambda x: x.stat().st_mtime, reverse=True)
        v2_models = sorted(v2_models, key=lambda x: x.stat().st_mtime, reverse=True)
        v3_models = sorted(v3_models, key=lambda x: x.stat().st_mtime, reverse=True)
        
        version_options = ["V1"]
        if V2_AVAILABLE and v2_models:
            version_options.append("V2")
        if V3_AVAILABLE and v3_models:
            version_options.append("V3 (推薦)")
        
        default_idx = len(version_options) - 1
        
        version_choice = st.radio(
            f"{direction.upper()} 模型版本",
            options=version_options,
            index=default_idx,
            key=f"{prefix}_{direction}_version",
            horizontal=True
        )
        
        if "V3" in version_choice:
            if not v3_models:
                st.warning(f"沒有找到 {direction.upper()} V3 模型")
                return None, 'v3'
            selected_model = st.selectbox(
                f"{direction.upper()} Oracle (V3)",
                [f.name for f in v3_models],
                key=f"{prefix}_{direction}_model"
            )
            return selected_model, 'v3'
        elif "V2" in version_choice:
            if not v2_models:
                st.warning(f"沒有找到 {direction.upper()} V2 模型")
                return None, 'v2'
            selected_model = st.selectbox(
                f"{direction.upper()} Oracle (V2)",
                [f.name for f in v2_models],
                key=f"{prefix}_{direction}_model"
            )
            return selected_model, 'v2'
        else:
            if not v1_models:
                st.warning(f"沒有找到 {direction.upper()} V1 模型")
                return None, 'v1'
            selected_model = st.selectbox(
                f"{direction.upper()} Oracle (V1)",
                [f.name for f in v1_models],
                key=f"{prefix}_{direction}_model"
            )
            return selected_model, 'v1'
    
    '''
    
    content = re.sub(selector_pattern, new_selector, content, flags=re.DOTALL)
    print("[4/7] Replaced model selector")
    
    # 5. Add V3 to run_standard_backtest
    if 'model_version == \'v3\' and V3_AVAILABLE' not in content:
        v2_feature_gen = '''if model_version == 'v2' and V2_AVAILABLE:
            with st.spinner("生成 V2 特徵 (44-54個)..."):'''
        
        v3_feature_gen = '''if model_version == 'v3' and V3_AVAILABLE:
            with st.spinner("生成 V3 特徵 (30個)..."):
                df_features = self.feature_engineer_v3.create_features_from_1m(
                    df_1m,
                    tp_target=0.012,
                    sl_stop=0.008,
                    lookahead_bars=240,
                    label_type='both'
                )
                feature_cols = self.feature_engineer_v3.get_feature_list()
        elif model_version == 'v2' and V2_AVAILABLE:
            with st.spinner("生成 V2 特徵 (44-54個)..."):'''
        
        content = content.replace(v2_feature_gen, v3_feature_gen, 1)
        print("[5/7] Added V3 to run_standard_backtest")
    else:
        print("[5/7] V3 already in run_standard_backtest")
    
    # 6. Add V3 to run_adaptive_backtest  
    # Find the SECOND occurrence in run_adaptive_backtest
    parts = content.split('''if model_version == 'v2' and V2_AVAILABLE:
            with st.spinner("生成 V2 特徵..."''')
    
    if len(parts) >= 2:
        v3_adaptive = '''if model_version == 'v3' and V3_AVAILABLE:
            with st.spinner("生成 V3 特徵..."):
                df_features = self.feature_engineer_v3.create_features_from_1m(
                    df_1m,
                    tp_target=0.012,
                    sl_stop=0.008,
                    lookahead_bars=240,
                    label_type='both'
                )
                feature_cols = self.feature_engineer_v3.get_feature_list()
        elif model_version == 'v2' and V2_AVAILABLE:
            with st.spinner("生成 V2 特徵..."'''
        
        content = parts[0] + parts[1] + v3_adaptive + parts[2] if len(parts) > 2 else content
        print("[6/7] Added V3 to run_adaptive_backtest")
    
    # 7. Update display version info
    old_display = '''if model_version == 'v2':
            st.info("本次回測使用 V2 特徵 (44-54個)")
        else:
            st.info("本次回測使用 V1 特徵 (9個)")'''
    
    new_display = '''if model_version == 'v3':
            st.success("本次回測使用 V3 特徵 (30個)")
        elif model_version == 'v2':
            st.info("本次回測使用 V2 特徵 (44-54個)")
        else:
            st.info("本次回測使用 V1 特徵 (9個)")'''
    
    content = content.replace(old_display, new_display)
    print("[7/7] Updated version display")
    
    # Backup
    backup_path = file_path.with_suffix('.py.v2backup')
    with open(file_path, 'r', encoding='utf-8') as f:
        with open(backup_path, 'w', encoding='utf-8') as b:
            b.write(f.read())
    print(f"\nBackup saved: {backup_path}")
    
    # Write
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\nSuccess! {file_path} updated with V3 support.")
    return True

if __name__ == "__main__":
    print("Fixing backtesting_tab.py for V3...\n")
    if fix_backtesting_tab():
        print("\nDone! Now run: streamlit run main.py")
        print("\nV3 models will appear in the dropdown.")
        print("Recommended settings:")
        print("  - Threshold: 0.12-0.15")
        print("  - TP: 1.5-2.0%")
        print("  - SL: 0.8-1.0%")
        print("  - Backtest days: 90")
    else:
        print("\nFailed. Check errors above.")