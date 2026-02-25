#!/usr/bin/env python3
"""
Script to update tabs/backtesting_tab.py to support V3 models

Usage:
    python update_backtest_v3.py
"""

import re
from pathlib import Path

def update_backtesting_tab():
    """
    Update backtesting_tab.py to add V3 support
    """
    file_path = Path("tabs/backtesting_tab.py")
    
    if not file_path.exists():
        print(f"Error: {file_path} not found")
        return False
    
    # Read file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Starting updates...")
    
    # 1. Add V3 import after V2 import
    v2_import_block = '''# 試圖載入 V2
try:
    from utils.feature_engineering_v2 import FeatureEngineerV2
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False'''
    
    v3_import_block = '''# 試圖載入 V2
try:
    from utils.feature_engineering_v2 import FeatureEngineerV2
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False

# 試圖載入 V3
try:
    from utils.feature_engineering_v3 import FeatureEngineerV3
    V3_AVAILABLE = True
except ImportError:
    V3_AVAILABLE = False'''
    
    content = content.replace(v2_import_block, v3_import_block)
    print("[1/7] Added V3 import")
    
    # 2. Update __init__ to initialize V3
    init_old = '''    def __init__(self):
        logger.info("Initializing BacktestingTab")
        self.feature_engineer_v1 = FeatureEngineer()
        if V2_AVAILABLE:
            self.feature_engineer_v2 = FeatureEngineerV2(
                enable_advanced_features=True,
                enable_ml_features=True
            )'''
    
    init_new = '''    def __init__(self):
        logger.info("Initializing BacktestingTab")
        self.feature_engineer_v1 = FeatureEngineer()
        if V2_AVAILABLE:
            self.feature_engineer_v2 = FeatureEngineerV2(
                enable_advanced_features=True,
                enable_ml_features=True
            )
        if V3_AVAILABLE:
            self.feature_engineer_v3 = FeatureEngineerV3()'''
    
    content = content.replace(init_old, init_new)
    print("[2/7] Updated __init__")
    
    # 3. Update render() status message
    render_old = '''        # 檢查 V2 狀態
        if V2_AVAILABLE:
            st.success("V2 系統已啟用 - 支持 V1 和 V2 模型")
        else:
            st.info("當前僅支持 V1 模型")'''
    
    render_new = '''        # 檢查狀態
        if V3_AVAILABLE:
            st.success("V3 系統已啟用 - 支持 V1/V2/V3 模型 (推薦 V3)")
        elif V2_AVAILABLE:
            st.info("V2 系統已啟用 - 支持 V1 和 V2 模型")
        else:
            st.info("當前僅支持 V1 模型")'''
    
    content = content.replace(render_old, render_new)
    print("[3/7] Updated render() status")
    
    # 4. Replace render_model_selector entirely
    # Find start and end of the method
    selector_start = content.find('    def render_model_selector(self, prefix: str, direction: str) -> tuple:')
    selector_end = content.find('\n    def render_standard_backtest(self):', selector_start)
    
    new_selector = '''    def render_model_selector(self, prefix: str, direction: str) -> tuple:
        """模型選擇器 - 支持 V1/V2/V3"""
        models_dir = Path("models_output")
        
        # 搜尋所有版本的模型
        if direction == "long":
            v1_models = list(models_dir.glob("catboost_long_[0-9]*.pkl"))
            v2_models = list(models_dir.glob("catboost_long_v2_*.pkl"))
            v3_models = list(models_dir.glob("catboost_long_v3_*.pkl"))
        else:
            v1_models = list(models_dir.glob("catboost_short_[0-9]*.pkl"))
            v2_models = list(models_dir.glob("catboost_short_v2_*.pkl"))
            v3_models = list(models_dir.glob("catboost_short_v3_*.pkl"))
        
        # 排序
        v1_models = sorted(v1_models, key=lambda x: x.stat().st_mtime, reverse=True)
        v2_models = sorted(v2_models, key=lambda x: x.stat().st_mtime, reverse=True)
        v3_models = sorted(v3_models, key=lambda x: x.stat().st_mtime, reverse=True)
        
        # 建立版本選項
        version_options = ["V1"]
        if V2_AVAILABLE and v2_models:
            version_options.append("V2")
        if V3_AVAILABLE and v3_models:
            version_options.append("V3 (推薦)")
        
        # 預設選擇最新版本
        default_idx = len(version_options) - 1
        
        # 版本選擇
        version_choice = st.radio(
            f"{direction.upper()} 模型版本",
            options=version_options,
            index=default_idx,
            key=f"{prefix}_{direction}_version",
            horizontal=True
        )
        
        # 根據選擇顯示模型
        if "V3" in version_choice:
            if not v3_models:
                st.warning(f"沒有找到 {direction.upper()} V3 模型")
                return None, 'v3'
            
            selected_model = st.selectbox(
                f"{direction.UPPER()} Oracle (V3)",
                [f.name for f in v3_models],
                key=f"{prefix}_{direction}_model"
            )
            return selected_model, 'v3'
        
        elif "V2" in version_choice:
            if not v2_models:
                st.warning(f"沒有找到 {direction.UPPER()} V2 模型")
                return None, 'v2'
            
            selected_model = st.selectbox(
                f"{direction.UPPER()} Oracle (V2)",
                [f.name for f in v2_models],
                key=f"{prefix}_{direction}_model"
            )
            return selected_model, 'v2'
        
        else:  # V1
            if not v1_models:
                st.warning(f"沒有找到 {direction.UPPER()} V1 模型")
                return None, 'v1'
            
            selected_model = st.selectbox(
                f"{direction.UPPER()} Oracle (V1)",
                [f.name for f in v1_models],
                key=f"{prefix}_{direction}_model"
            )
            return selected_model, 'v1'
'''
    
    content = content[:selector_start] + new_selector + content[selector_end:]
    print("[4/7] Replaced render_model_selector")
    
    # 5. Update run_standard_backtest feature generation
    standard_old = '''        # 根據版本選擇 feature engineer
        if model_version == 'v2' and V2_AVAILABLE:'''
    
    standard_new = '''        # 根據版本選擇 feature engineer
        if model_version == 'v3' and V3_AVAILABLE:
            with st.spinner("生成 V3 特徵 (30個)..."):
                df_features = self.feature_engineer_v3.create_features_from_1m(
                    df_1m,
                    tp_target=0.012,
                    sl_stop=0.008,
                    lookahead_bars=240,
                    label_type='both'
                )
                feature_cols = self.feature_engineer_v3.get_feature_list()
        elif model_version == 'v2' and V2_AVAILABLE:'''
    
    content = content.replace(standard_old, standard_new)
    print("[5/7] Updated run_standard_backtest")
    
    # 6. Update run_adaptive_backtest feature generation
    adaptive_old = '''        # 根據版本選擇 feature engineer
        if model_version == 'v2' and V2_AVAILABLE:'''
    
    adaptive_new = '''        # 根據版本選擇 feature engineer
        if model_version == 'v3' and V3_AVAILABLE:
            with st.spinner("生成 V3 特徵..."):
                df_features = self.feature_engineer_v3.create_features_from_1m(
                    df_1m,
                    tp_target=0.012,
                    sl_stop=0.008,
                    lookahead_bars=240,
                    label_type='both'
                )
                feature_cols = self.feature_engineer_v3.get_feature_list()
        elif model_version == 'v2' and V2_AVAILABLE:'''
    
    # Find the second occurrence (in run_adaptive_backtest)
    parts = content.split(adaptive_old)
    if len(parts) >= 3:
        content = parts[0] + standard_new + parts[1] + adaptive_new + parts[2]
    print("[6/7] Updated run_adaptive_backtest")
    
    # 7. Update display_results_with_analysis version display
    display_old = '''        # 顯示版本
        if model_version == 'v2':
            st.info("本次回測使用 V2 特徵 (44-54個)")
        else:
            st.info("本次回測使用 V1 特徵 (9個)")'''
    
    display_new = '''        # 顯示版本
        if model_version == 'v3':
            st.success("本次回測使用 V3 特徵 (30個)")
        elif model_version == 'v2':
            st.info("本次回測使用 V2 特徵 (44-54個)")
        else:
            st.info("本次回測使用 V1 特徵 (9個)")'''
    
    content = content.replace(display_old, display_new)
    print("[7/7] Updated display_results_with_analysis")
    
    # Backup original file
    backup_path = file_path.with_suffix('.py.backup')
    with open(backup_path, 'w', encoding='utf-8') as f:
        with open(file_path, 'r', encoding='utf-8') as orig:
            f.write(orig.read())
    print(f"\nBackup created: {backup_path}")
    
    # Write updated content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\nSuccess! Updated {file_path}")
    print("\nV3 support has been added to the backtesting tab.")
    print("You can now select V3 models in the GUI.")
    
    return True

if __name__ == "__main__":
    print("Updating backtesting_tab.py for V3 support...\n")
    success = update_backtesting_tab()
    if success:
        print("\nNext steps:")
        print("1. Run: streamlit run main.py")
        print("2. Go to '策略回測' tab")
        print("3. Select V3 models for Long and Short")
        print("4. Set threshold to 0.12-0.15")
        print("5. Run backtest!")
    else:
        print("\nUpdate failed. Please check the error messages above.")