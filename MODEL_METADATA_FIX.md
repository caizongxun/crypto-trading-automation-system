# Model Metadata Fix - 解決 V2 特徵讀取問題

## 問題描述

在之前的實現中,**模型儲存時沒有包含特徵名稱**,導致回測程式無法讀取 V2 特徵,自動 fallback 到 V1 特徵。

### 問題症狀

```
2026-02-23 182304 - ERROR - Failed to extract feature names: 
'CalibratedClassifierCV' object has no attribute 'feature_names_'

2026-02-23 182304 - WARNING - Using fallback features: 
efficiency_ratio, extreme_time_diff, vol_imbalance_ratio, zscore, 
bb_width_pct, rsi, atr_pct, zscore_1h, atr_pct_1d
```

結果:
- 網頁選擇 V2 模型
- 實際使用 V1 特徵
- 回測結果不準確

---

## 解決方案

### 1. 修改檔案

✅ **train_v2.py** - 儲存時包含 metadata
```python
def save_model_with_metadata(model, feature_names, model_type, version, output_dir):
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'version': version,
        'timestamp': timestamp,
        'metadata': {...}
    }
    joblib.dump(model_data, filename)
```

✅ **utils/agent_backtester.py** - 讀取 metadata
```python
def load_model_with_metadata(model_path):
    data = joblib.load(model_path)
    
    if isinstance(data, dict):
        # 新格式: 包含 metadata
        model = data['model']
        feature_names = data['feature_names']
        version = data['version']
    else:
        # 舊格式: fallback
        model = data
        feature_names = V1_FEATURES
        version = 'v1'
    
    return model, feature_names, version
```

### 2. 新增驗證工具

✅ **verify_model_metadata.py** - 檢查模型 metadata

---

## 使用步驟

### Step 1: 重新訓練模型

```bash
# 訓練新的 V2 模型 (包含 metadata)
python train_v2.py
```

輸出示例:
```
✅ Model saved with metadata: models_output/catboost_long_v2_20260224_104530.pkl
   Version: v2
   Features (15): ['efficiency_ratio', 'extreme_time_diff', ...]
```

### Step 2: 驗證模型

```bash
# 驗證單個模型
python verify_model_metadata.py --single models_output/catboost_long_v2_xxx.pkl

# 驗證模型對
python verify_model_metadata.py \
  --long models_output/catboost_long_v2_xxx.pkl \
  --short models_output/catboost_short_v2_xxx.pkl
```

正常輸出:
```
================================================================================
Verifying: models_output/catboost_long_v2_20260224_104530.pkl
================================================================================
✅ Model loaded successfully

📊 Model Information:
  Version:        v2
  Feature Count:  15

📋 Feature Names:
   1. efficiency_ratio
   2. extreme_time_diff
   ...
  15. momentum_5m

================================================================================
FEATURE CONSISTENCY CHECK
================================================================================
✅ Features match perfectly!
   Both models use 15 features
```

### Step 3: 執行回測

```bash
python run_backtest.py \
  --long models_output/catboost_long_v2_xxx.pkl \
  --short models_output/catboost_short_v2_xxx.pkl \
  --prob_long 0.10 \
  --prob_short 0.10
```

LOG 檢查點:
```
✅ Loaded v2 model from ...
   Features (15): ['efficiency_ratio', 'extreme_time_diff', ...]

🚀 Backtester initialized
   Long:  v2 with 15 features
   Short: v2 with 15 features
```

❗ **如果看到這個警告就是有問題**:
```
⚠️ Model has no feature metadata, using V1 fallback
   Features (9): ['efficiency_ratio', ...]
```

---

## 兼容性

### 新模型 (V2 with metadata)
- ✅ 完整保存特徵名稱
- ✅ 版本追蹤
- ✅ 回測時正確讀取 V2 特徵

### 舊模型 (Legacy)
- ⚠️ 自動 fallback 到 V1 特徵
- ⚠️ 顯示警告訊息
- ✅ 不會崩潰,但結果不準

---

## 故障排除

### Q1: 回測時還是使用 V1 特徵

**原因**: 使用了舊的模型檔案

**解決**:
```bash
# 1. 檢查模型
python verify_model_metadata.py --single your_model.pkl

# 2. 如果顯示 v1,重新訓練
python train_v2.py

# 3. 使用新模型回測
```

### Q2: Long/Short 特徵不一致

**原因**: 使用了不同版本的模型

**解決**:
```bash
# 重新訓練並使用同一批次的模型
python train_v2.py

# 會生成
# - catboost_long_v2_20260224_104530.pkl
# - catboost_short_v2_20260224_104530.pkl  (相同 timestamp)
```

### Q3: 訓練時失敗

**錯誤**: `TypeError: save_model_with_metadata() missing required argument`

**解決**: 確保 pull 了最新的 `train_v2.py`
```bash
git pull origin main
```

---

## 檔案結構

### 新格式模型檔案

```python
{
    'model': CalibratedClassifierCV(...),  # 訓練好的模型
    'feature_names': [
        'efficiency_ratio',
        'extreme_time_diff',
        ...
    ],
    'version': 'v2',
    'timestamp': '20260224_104530',
    'metadata': {
        'n_features': 15,
        'model_type': 'long',
        'training_date': '2026-02-24T10:45:30'
    }
}
```

### 舊格式模型檔案

```python
CalibratedClassifierCV(...)  # 只有模型本體,沒有 metadata
```

---

## 測試確認

### 測試 Checklist

- [ ] 訓練模型並看到 `✅ Model saved with metadata`
- [ ] 驗證模型顯示 `Version: v2`
- [ ] Long/Short 特徵一致
- [ ] 回測 LOG 顯示 `✅ Using features from model metadata`
- [ ] 回測 LOG 不顯示 `⚠️ Using fallback features`
- [ ] 機率分布正常 (Max > 0.40)

### 執行完整測試

```bash
#!/bin/bash

# 1. 訓練
echo "Training models..."
python train_v2.py

# 2. 找出最新模型
LONG_MODEL=$(ls -t models_output/catboost_long_v2_*.pkl | head -1)
SHORT_MODEL=$(ls -t models_output/catboost_short_v2_*.pkl | head -1)

echo "Long:  $LONG_MODEL"
echo "Short: $SHORT_MODEL"

# 3. 驗證
echo "\nVerifying models..."
python verify_model_metadata.py --long "$LONG_MODEL" --short "$SHORT_MODEL"

if [ $? -eq 0 ]; then
    echo "\n✅ Verification passed!"
    
    # 4. 回測
    echo "\nRunning backtest..."
    python run_backtest.py \
        --long "$LONG_MODEL" \
        --short "$SHORT_MODEL" \
        --prob_long 0.10 \
        --prob_short 0.10
else
    echo "\n❌ Verification failed!"
    exit 1
fi
```

---

## 版本記錄

### v1.1.0 (2026-02-24)

**新增**:
- ✅ 模型儲存時包含 metadata
- ✅ 回測時自動讀取正確特徵
- ✅ 新增模型驗證工具

**修復**:
- ✅ 修復 V2 模型 fallback 到 V1 特徵的問題

**後续計劃**:
- [ ] 自動檢測模型版本不匹配
- [ ] Web UI 顯示模型 metadata
- [ ] 支援更多 metadata (訓練參數, metrics 等)

---

## 聯絡與支援

如果遇到問題:
1. 檢查 `logs/agent_backtester.log`
2. 執行 `verify_model_metadata.py`
3. 查看本文檔的故障排除章節

---

**最後更新**: 2026-02-24  
**作者**: Zong  
**版本**: 1.1.0