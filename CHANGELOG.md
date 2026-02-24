# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2026-02-24

### 🐛 Fixed
- **Critical**: 修復 V2 模型回測時 fallback 到 V1 特徵的問題
- 模型現在正確儲存和讀取特徵名稱
- 回測程式現在可以識別模型版本 (v1/v2)

### ✨ Added
- 新增 `save_model_with_metadata()` 函數在 `train_v2.py`
- 新增 `load_model_with_metadata()` 函數在 `utils/agent_backtester.py`
- 新增 `verify_model_metadata.py` 驗證工具
- 新增 `MODEL_METADATA_FIX.md` 完整文檔

### 📝 Changed
- 模型檔案格式從單純的 model object 改為 dict 包裝
- 模型現在包含: model, feature_names, version, timestamp, metadata

### 📊 Impact
- ✅ V2 特徵現在可以正確在回測中使用
- ✅ 不再會意外使用錯誤的特徵集
- ✅ 向後兼容舊模型 (會顯示警告)

### 🛠 Files Modified
- `train_v2.py` - 在儲存模型時加入 metadata
- `utils/agent_backtester.py` - 在載入模型時讀取 metadata
- `verify_model_metadata.py` - 新增驗證工具
- `MODEL_METADATA_FIX.md` - 新增使用說明

### 📝 Usage

```bash
# 1. 重新訓練模型 (包含 metadata)
python train_v2.py

# 2. 驗證模型
python verify_model_metadata.py \
  --long models_output/catboost_long_v2_xxx.pkl \
  --short models_output/catboost_short_v2_xxx.pkl

# 3. 執行回測
python run_backtest.py \
  --long models_output/catboost_long_v2_xxx.pkl \
  --short models_output/catboost_short_v2_xxx.pkl \
  --prob_long 0.10 \
  --prob_short 0.10
```

### ⚠️ Breaking Changes

**無** - 完全向後兼容

舊模型仍可使用,但會:
- 顯示警告訊息
- 自動 fallback 到 V1 特徵
- 建議重新訓練以獲取正確結果

---

## [1.0.0] - 2026-02-23

### Initial Release
- 基礎雙向交易系統
- V1/V2 特徵工程
- CatBoost + XGBoost Ensemble
- 機率校準
- 事件驅動回測器

---

**格式說明**:
- [version] - date
- ✨ Added: 新增功能
- 🐛 Fixed: Bug 修復
- 📝 Changed: 變更
- ⚠️ Breaking Changes: 不兼容變更