# Backtesting Tab V3 Update Guide

## 概要

需要更新 `tabs/backtesting_tab.py` 以支持 V3 模型回測。

## 更新步驟

### 1. 在文件開頭添加 V3 Import

在第 17 行附近增加:

```python
# 現有的 V2 import
try:
    from utils.feature_engineering_v2 import FeatureEngineerV2
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False

# 新增: V3 import
try:
    from utils.feature_engineering_v3 import FeatureEngineerV3
    V3_AVAILABLE = True
except ImportError:
    V3_AVAILABLE = False
```

---

### 2. 在 `__init__` 方法中初始化 V3

找到 `def __init__(self):` 方法 (約第 25 行),修改為:

```python
def __init__(self):
    logger.info("Initializing BacktestingTab")
    self.feature_engineer_v1 = FeatureEngineer()
    
    if V2_AVAILABLE:
        self.feature_engineer_v2 = FeatureEngineerV2(
            enable_advanced_features=True,
            enable_ml_features=True
        )
    
    # 新增: V3 初始化
    if V3_AVAILABLE:
        self.feature_engineer_v3 = FeatureEngineerV3()
```

---

### 3. 更新 `render_model_selector` 支持 V3

找到 `def render_model_selector(self, prefix: str, direction: str)` 方法 (約第 90 行),完全替換為:

```python
def render_model_selector(self, prefix: str, direction: str) -> tuple:
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
    
    else:  # V1
        if not v1_models:
            st.warning(f"沒有找到 {direction.upper()} V1 模型")
            return None, 'v1'
        
        selected_model = st.selectbox(
            f"{direction.upper()} Oracle (V1)",
            [f.name for f in v1_models],
            key=f"{prefix}_{direction}_model"
        )
        return selected_model, 'v1'
```

---

### 4. 更新 `run_standard_backtest` 支持 V3

找到 `def run_standard_backtest(...)` 方法 (約第 378 行)。

在特徵生成部分 ("根據版本選擇 feature engineer"),在 `if model_version == 'v2'` 之前增加:

```python
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
        feature_cols = self.feature_engineer_v3.get_feature_list()

elif model_version == 'v2' and V2_AVAILABLE:
    # ... 現有的 V2 代碼 ...
    
else:
    # ... 現有的 V1 代碼 ...
```

---

### 5. 更新 `run_adaptive_backtest` 支持 V3

找到 `def run_adaptive_backtest(...)` 方法 (約第 453 行)。

同樣在特徵生成部分加入 V3 支持:

```python
# 根據版本選擇 feature engineer
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

elif model_version == 'v2' and V2_AVAILABLE:
    # ... 現有的 V2 代碼 ...

else:
    # ... 現有的 V1 代碼 ...
```

---

### 6. 更新 `display_results_with_analysis` 支持 V3

找到 `def display_results_with_analysis(...)` 方法 (約第 521 行)。

在顯示版本的部分修改:

```python
# 顯示版本
if model_version == 'v3':
    st.success("本次回測使用 V3 特徵 (30個)")
elif model_version == 'v2':
    st.info("本次回測使用 V2 特徵 (44-54個)")
else:
    st.info("本次回測使用 V1 特徵 (9個)")
```

---

### 7. 更新 `render` 方法的狀態顯示

找到 `def render(self):` 方法 (約第 38 行)。

在檢查 V2 狀態之後增加:

```python
# 檢查 V2 狀態
if V2_AVAILABLE:
    st.success("V2 系統已啟用 - 支持 V1 和 V2 模型")
else:
    st.info("當前僅支持 V1 模型")

# 新增: 檢查 V3 狀態
if V3_AVAILABLE:
    st.success("V3 系統已啟用 - 支持 V1/V2/V3 模型")
```

---

## V3 特徵列表

V3 使用 30 個特徵:

```python
V3_FEATURES = [
    # Price Momentum (6)
    'returns_5m', 'returns_15m', 'returns_30m', 'returns_1h',
    'price_position_1h', 'price_position_4h',
    
    # Volatility (4)
    'atr_pct_14', 'atr_pct_60', 'vol_ratio', 'vol_expanding',
    
    # Trend (5)
    'trend_9_21', 'trend_21_50',
    'above_ema9', 'above_ema21', 'above_ema50',
    
    # Volume (3)
    'volume_ratio', 'volume_trend', 'high_volume',
    
    # Microstructure (3)
    'body_pct', 'bullish_candle', 'bearish_candle',
    
    # Directional Pressure (2)
    'pressure_ratio_30m', 'green_streak',
    
    # Oscillators (3)
    'rsi_14', 'rsi_oversold', 'rsi_overbought',
    
    # Market Regime (4)
    'is_asian', 'is_london', 'is_nyc'
]
```

注意: `get_feature_list()` 會自動返回這 29 個特徵 (沒有 hour_cos)。

---

## 建議的 V3 回測參數

基於你的訓練結果:

```python
推薦設定:
- Probability Threshold: 0.12-0.15 (不要用 0.16)
- Take Profit: 1.5-2.0%
- Stop Loss: 0.8-1.0%
- Position Size: 10-20%
- Leverage: 1-2x
- Trading Hours: 09:00-22:00 UTC 或 24/7
- Backtest Days: 90-180

預期結果 (90天):
- 交易數: 100-200 筆
- 勝率: 40-48%
- Profit Factor: 1.3-1.8
- 總報酬: 5-15% (1x), 10-30% (2x)
```

---

## 測試步驟

更新完成後:

1. **啟動 GUI**:
   ```bash
   streamlit run main.py
   ```

2. **進入策略回測標籤**

3. **選擇 V3 模型**:
   - Long: `catboost_long_v3_20260225_083414.pkl`
   - Short: `catboost_short_v3_20260225_083414.pkl`

4. **設定參數**:
   - 閾值: 0.12-0.15
   - TP: 1.5%
   - SL: 0.8%
   - 回測天數: 90
   - 桓桿: 1x

5. **執行回測**

6. **檢查結果**:
   - 交易數 > 100
   - 勝率 > 40%
   - Profit Factor > 1.3
   - 總報酬 > 5%

---

## 常見問題

### Q1: V3 模型不顯示?

檢查:
1. `models_output/` 目錄中是否有 `catboost_*_v3_*.pkl` 檔案
2. 是否正確 import `FeatureEngineerV3`
3. 查看 `logs/backtesting_tab.log`

### Q2: 特徵數量不匹配?

V3 模型必須使用 V3 特徵工程生成的 30 個特徵。不能用 V1 或 V2 的特徵。

### Q3: 沒有交易?

降低閾值:
- 從 0.16 降到 0.12
- 或使用 0.10

根據你的訓練報告:
- @ 0.15: 覆蓋率 57%
- @ 0.10: 覆蓋率 77%

---

## 完整更新檔案

如果手動更新太複雜,可以:

1. 備份現有檔案:
   ```bash
   cp tabs/backtesting_tab.py tabs/backtesting_tab.py.backup
   ```

2. 使用 Git 合併或手動替換

3. 或者等待我幫你生成完整的更新檔案

---

**狀態**: 待更新

**工作量**: 中等 (7 處修改)

**預計時間**: 10-15 分鐘