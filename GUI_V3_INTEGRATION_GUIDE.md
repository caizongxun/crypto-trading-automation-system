# V3 GUI Integration Guide

## 🎯 Overview

V3 已經整合到 GUI 中,包括:

1. ✅ 版本選擇器 (Sidebar)
2. ✅ V3 訓練標籤
3. ⚠️ V3 回測支持 (需要手動更新)

---

## ✅ 已完成的整合

### 1. Sidebar 版本選擇器

在 `main.py` 中已增加:

```python
# 版本選項包括 V3
version_options = [
    "V1 - 基礎版 (9特徵)",
    "V2 - 進階版 (44-54特徵)",
    "V3 - 優化版 (30特徵) ⭐"  # 新增!
]
```

### 2. V3 訓練標籤

在 `tabs/model_training_v3_tab.py`:

**功能**:
- ⚡ 一鍵訓練 V3 模型
- 📊 即時顯示訓練 log
- 📝 查看訓練報告
- ✅ 驗證檢查
- 📦 模型列表

### 3. 模型管理

V3 模型會顯示在模型管理標籤中

---

## ⚠️ 需要手動更新的部分

### Backtesting Tab 支持 V3

需要更新 `tabs/backtesting_tab.py` 加入 V3 特徵工程支持:

```python
# 1. 在文件開頭增加 import
try:
    from utils.feature_engineering_v3 import FeatureEngineerV3
    V3_AVAILABLE = True
except ImportError:
    V3_AVAILABLE = False

# 2. 在 __init__ 中初始化
class BacktestingTab:
    def __init__(self):
        self.feature_engineer_v1 = FeatureEngineer()
        if V2_AVAILABLE:
            self.feature_engineer_v2 = FeatureEngineerV2(...)
        if V3_AVAILABLE:
            self.feature_engineer_v3 = FeatureEngineerV3()

# 3. 在 render_model_selector 中加入 V3
def render_model_selector(self, prefix: str, direction: str) -> tuple:
    models_dir = Path("models_output")
    
    if direction == "long":
        v1_models = list(models_dir.glob("catboost_long_[0-9]*.pkl"))
        v2_models = list(models_dir.glob("catboost_long_v2_*.pkl"))
        v3_models = list(models_dir.glob("catboost_long_v3_*.pkl"))  # 新增
    else:
        v1_models = list(models_dir.glob("catboost_short_[0-9]*.pkl"))
        v2_models = list(models_dir.glob("catboost_short_v2_*.pkl"))
        v3_models = list(models_dir.glob("catboost_short_v3_*.pkl"))  # 新增
    
    # 版本選擇
    version_options = ["V1"]
    if V2_AVAILABLE and v2_models:
        version_options.append("V2")
    if V3_AVAILABLE and v3_models:
        version_options.append("V3 (推薦)")  # 新增
    
    version_choice = st.radio(
        f"{direction.upper()} 模型版本",
        options=version_options,
        index=len(version_options) - 1,  # 預設選最新版
        key=f"{prefix}_{direction}_version",
        horizontal=True
    )
    
    # 處理 V3 選擇
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
    # ... V2/V1 處理 ...

# 4. 在 run_standard_backtest 中加入 V3 特徵生成
def run_standard_backtest(self, model_version='v1', backtest_days=180, leverage=1, **params):
    # ... 載入數據 ...
    
    # 根據版本選擇 feature engineer
    if model_version == 'v3' and V3_AVAILABLE:
        with st.spinner("生成 V3 特徵 (30個)..."):
            df_features = self.feature_engineer_v3.create_features_from_1m(
                df_1m,
                tp_target=0.012,
                sl_stop=0.008,
                lookahead_bars=240,
                label_type='both'
            )
            feature_cols = [
                # 30 個 V3 特徵
                'returns_5m', 'returns_15m', 'returns_30m', 'returns_1h',
                'price_position_1h', 'price_position_4h',
                'atr_pct_14', 'atr_pct_60', 'vol_ratio', 'vol_expanding',
                'trend_9_21', 'trend_21_50', 'above_ema9', 'above_ema21', 'above_ema50',
                'volume_ratio', 'volume_trend', 'high_volume',
                'body_pct', 'bullish_candle', 'bearish_candle',
                'pressure_ratio_30m', 'green_streak',
                'rsi_14', 'rsi_oversold', 'rsi_overbought',
                'is_asian', 'is_london', 'is_nyc', 'hour_cos'
            ]
    elif model_version == 'v2' and V2_AVAILABLE:
        # ... V2 處理 ...
    else:
        # ... V1 處理 ...
    
    # ... 剩餘回測代碼 ...
```

---

## 🚀 快速更新指南

### Option 1: 使用現有功能 (暫時解決)

雖然回測標籤還未完全支持 V3,但你可以:

1. **訓練 V3 模型**:
   ```bash
   python train_v3.py
   ```

2. **手動回測**:
   ```python
   # 在 Python 中
   from utils.agent_backtester import BidirectionalAgentBacktester
   from utils.feature_engineering_v3 import FeatureEngineerV3
   
   # 載入數據和生成特徵
   # 執行回測
   ```

3. **使用 V1/V2 回測標籤測試**:
   - V3 模型的 30 個特徵中
   - 大部分在 V2 也有 (只是名稱不同)
   - 所以可以先用 V2 回測標籤

### Option 2: 完整更新 (建議)

執行以下步驟完全支持 V3:

```bash
# 1. 備份現有文件
cp tabs/backtesting_tab.py tabs/backtesting_tab.py.backup

# 2. 手動編輯 tabs/backtesting_tab.py
# 按照上面的代碼片段增加 V3 支持

# 3. 測試 GUI
streamlit run main.py
```

---

## 📝 V3 特徵列表 (供回測使用)

```python
V3_FEATURE_COLS = [
    # Price Momentum (6)
    'returns_5m', 'returns_15m', 'returns_30m', 'returns_1h',
    'price_position_1h', 'price_position_4h',
    
    # Volatility (4)
    'atr_pct_14', 'atr_pct_60', 'vol_ratio', 'vol_expanding',
    
    # Trend (5)
    'trend_9_21', 'trend_21_50', 'above_ema9', 'above_ema21', 'above_ema50',
    
    # Volume (3)
    'volume_ratio', 'volume_trend', 'high_volume',
    
    # Microstructure (3)
    'body_pct', 'bullish_candle', 'bearish_candle',
    
    # Directional Pressure (2)
    'pressure_ratio_30m', 'green_streak',
    
    # Oscillators (3)
    'rsi_14', 'rsi_oversold', 'rsi_overbought',
    
    # Market Regime (4) - 注意比文檔多 1 個 hour_cos
    'is_asian', 'is_london', 'is_nyc', 'hour_cos'
]

# 總計: 30 個特徵
```

---

## ✅ 驗證清單

### GUI 功能

```
☑️ Sidebar 顯示 V3 選項
☑️ V3 訓練標籤可用
☑️ V3 模型顯示在模型管理
☐ V3 回測支持 (待更新)
☐ V3 自動交易支持 (待更新)
```

### 基本流程

```
☑️ 可以在 GUI 中選擇 V3
☑️ 可以在 GUI 中訓練 V3
☑️ 可以查看 V3 訓練結果
☐ 可以在 GUI 中回測 V3 (需要更新)
```

---

## 💡 臨時解決方案

在回測功能完成更新之前:

### 1. 使用指令行

```bash
# 訓練
python train_v3.py

# 回測 (需要寫一個簡單的 backtest_v3.py)
python backtest_v3.py --long-model models_output/catboost_long_v3_TIMESTAMP.pkl \
                       --short-model models_output/catboost_short_v3_TIMESTAMP.pkl \
                       --threshold 0.15 \
                       --days 90
```

### 2. 使用 Jupyter Notebook

建立 `notebooks/test_v3.ipynb`:

```python
import pandas as pd
from utils.feature_engineering_v3 import FeatureEngineerV3
from utils.agent_backtester import BidirectionalAgentBacktester

# 1. 載入數據
# 2. 生成 V3 特徵
# 3. 執行回測
# 4. 分析結果
```

### 3. 暫時用 V2 回測標籤

由於 V3 的 30 個特徵大多在 V2 也有:
- 可以在 V2 回測標籤中選擇 V3 模型
- 但需要確保特徵名稱匹配

---

## 🔧 完整更新 Checklist

如果要完全支持 V3,需要更新:

1. ☑️ `main.py` - 已完成
2. ☑️ `tabs/model_training_v3_tab.py` - 已完成
3. ☐ `tabs/backtesting_tab.py` - 需要更新
4. ☐ `tabs/auto_trading_tab.py` - 需要更新
5. ☑️ `tabs/model_management_tab.py` - 應該已支持

---

## 📚 相關文檔

- [V3 Model Guide](V3_MODEL_GUIDE.md)
- [V3 README](V3_README.md)
- [V1/V2/V3 Comparison](V1_V2_V3_COMPARISON.md)
- [V3 Changelog](CHANGELOG_V3.md)

---

**狀態**: V3 GUI 部分支持 (訓練完成,回測待更新)

**下一步**: 更新 `tabs/backtesting_tab.py` 以完全支持 V3 模型回測