# GUI 使用指南 - 模型 Metadata 功能

## 啟動 GUI

```bash
streamlit run main.py
```

瀏覽器會自動開啟 `http://localhost:8501`

---

## 新增功能: 模型管理標籤

### 位置

在主界面的最右側標籤: **📦 模型管理**

### 功能概覽

模型管理標籤分為三個子標籤:

1. **🔍 模型檢查** - 檢查單個模型的 metadata
2. **⚖️ 模型比較** - 比較 Long/Short 模型的一致性
3. **🗑️ 模型管理** - 刪除舊模型

---

## 1. 模型檢查

### 用途
驗證模型是否包含正確的 metadata,確保回測時使用正確的特徵。

### 使用步驟

1. 進入「📦 模型管理」標籤
2. 選擇「🔍 模型檢查」子標籤
3. 從下拉選單選擇要檢查的模型
4. 點擊「🔍 檢查模型」

### 結果解讀

#### ✅ 正常結果 (V2 模型)

```
✅ 模型載入成功

📊 模型信息
  版本:    V2
  特徵數量: 15
  檔案大小: 2.34 MB
  修改時間: 2026-02-24 10:45
  模型類型: Long

✅ 此模型包含完整 metadata,可以正常使用

📋 特徵列表
  1. efficiency_ratio
  2. extreme_time_diff
  3. vol_imbalance_ratio
  ...
  15. momentum_5m
```

#### ⚠️ 警告結果 (舊模型)

```
✅ 模型載入成功

⚠️ 此模型缺少 metadata,使用 V1 fallback 特徵
💡 建議重新訓練以獲取 V2 metadata
```

### 匯出功能

點擊「💾 匯出 Metadata 為 JSON」可以下載模型的完整 metadata:

```json
{
  "model_name": "catboost_long_v2_20260224_104530.pkl",
  "version": "v2",
  "feature_count": 15,
  "features": [
    "efficiency_ratio",
    "extreme_time_diff",
    ...
  ],
  "file_size_mb": 2.34,
  "modified_time": "2026-02-24T10:45:30"
}
```

---

## 2. 模型比較

### 用途
確保 Long 和 Short 模型使用相同的特徵,避免特徵不一致導致的問題。

### 使用步驟

1. 進入「⚖️ 模型比較」子標籤
2. 左側選擇 Long 模型
3. 右側選擇 Short 模型
4. 點擊「⚖️ 比較模型」

### 結果解讀

#### ✅ 特徵完全一致

```
📊 基本信息
              Long 模型    Short 模型
版本          V2          V2
特徵數        15          15
檔案大小      2.34 MB     2.31 MB
修改時間      2026-02-24  2026-02-24

🔍 特徵一致性檢查
✅ 特徵完全一致!
ℹ️ 兩個模型都使用 15 個相同特徵

🏷️ 版本一致性檢查
✅ 兩個模型都是 V2 版本
```

#### ❌ 特徵不一致

```
❌ 特徵不一致!

  共同特徵: 12
  僅 Long 有: 3
  僅 Short 有: 2

僅 Long 模型有的特徵:
  - feature_a
  - feature_b
  - feature_c

僅 Short 模型有的特徵:
  - feature_x
  - feature_y

💡 建議重新訓練並使用同一批次的模型
```

---

## 3. 模型管理

### 用途
刪除舊的或不需要的模型,釋放磁碟空間。

### 使用步驟

1. 進入「🗑️ 模型管理」子標籤
2. 查看所有模型列表
3. 勾選要刪除的模型 (可多選)
4. 勾選「我確定要刪除這些模型」
5. 點擊「🗑️ 確定刪除」

### ⚠️ 注意事項

- 刪除後無法恢復
- 請確保不要刪除正在使用的模型
- 建議保留最近的 2-3 個版本

---

## 工作流程建議

### 訓練新模型後的檢查流程

```
1. 訓練模型
   └─> 使用「🚀 V2 模型訓練」標籤
   
2. 檢查模型 metadata
   └─> 進入「📦 模型管理」-> "🔍 模型檢查"
   └─> 確認版本為 V2
   └─> 確認特徵數正確 (15個)
   
3. 比較 Long/Short 模型
   └─> 進入「⚖️ 模型比較」
   └─> 確認特徵完全一致
   └─> 確認版本一致
   
4. 執行回測
   └─> 進入「📈 策略回測」標籤
   └─> 選擇剛訓練的模型
   └─> 確認 LOG 顯示:
       "✅ Using features from model metadata"
```

### 如果發現模型 metadata 有問題

```
問題: ⚠️ 此模型缺少 metadata,使用 V1 fallback 特徵

解決方案:
1. 重新訓練模型
   └─> python train_v2.py
   
2. 或使用 GUI 訓練
   └─> 「🚀 V2 模型訓練」標籤
   
3. 訓練完成後再次檢查
   └─> 應該看到 "✅ 此模型包含完整 metadata"
```

---

## 回測標籤中的 Metadata 顯示

### 模型選擇時的版本標示

在「📈 策略回測」標籤中:

```
選擇 Long Oracle 模型:
  [ V2 (推薦) ]  [ V1 ]
  
下拉選單:
  catboost_long_v2_20260224_104530.pkl  <-- V2 模型
  catboost_long_20260223_120000.pkl     <-- V1 模型
```

### 回測時的 Metadata 資訊

回測執行時,LOG 會顯示:

```
✅ Loaded v2 model from models_output/catboost_long_v2_xxx.pkl
   Features (15): ['efficiency_ratio', 'extreme_time_diff', ...]

🚀 Backtester initialized
   Long:  v2 with 15 features
   Short: v2 with 15 features
   Using: ['efficiency_ratio', ...]
```

---

## 常見問題

### Q1: 為什麼我的模型顯示 V1?

**原因**: 模型是在 metadata fix 之前訓練的。

**解決**: 重新訓練模型
```bash
python train_v2.py
# 或使用 GUI 的 V2 訓練標籤
```

### Q2: Long/Short 模型特徵不一致怎麼辦?

**原因**: 使用了不同批次訓練的模型。

**解決**: 一次性訓練兩個模型
```bash
python train_v2.py  # 會同時訓練 Long 和 Short
```

### Q3: 刪除模型後能恢復嗎?

**答**: 無法恢復。請在刪除前確認。

**建議**: 使用備份或匯出 metadata

### Q4: 舊模型能用嗎?

**答**: 可以用,但會自動 fallback 到 V1 特徵。

**影響**: 
- 使用錯誤的特徵集
- 回測結果可能不準確
- 建議重新訓練

---

## 最佳實踐

### 模型命名規則

系統自動生成的模型名稱:
```
catboost_long_v2_20260224_104530.pkl
  │       │   │        │       └─ 時間 (HH:MM:SS)
  │       │   │        └─ 日期 (YYYYMMDD)
  │       │   └─ 版本
  │       └─ 方向 (long/short)
  └─ 模型類型
```

### 模型保留策略

建議保留:
- ✅ 最近 2-3 個版本的 V2 模型
- ✅ 性能最好的模型 (標記檔名)
- ❌ 刪除 V1 舊模型 (如果不再使用)
- ❌ 刪除測試用的模型

### 定期檢查

每週檢查一次:
1. 模型總數 (避免累積過多)
2. 模型版本 (確保使用 V2)
3. 特徵一致性 (Long/Short 匹配)

---

## 鍵盤快捷鍵

Streamlit GUI 支援以下快捷鍵:

- `R` - 重新載入頁面
- `Ctrl + K` - 打開命令面板
- `Ctrl + /` - 顯示快捷鍵幫助

---

## 疑難排解

### GUI 無法啟動

```bash
# 檢查 Streamlit 安裝
pip install streamlit

# 清除快取
streamlit cache clear

# 重新啟動
streamlit run main.py
```

### 模型管理標籤空白

```bash
# 確認 models_output 目錄存在
ls -la models_output/

# 確認有模型檔案
ls models_output/*.pkl

# 如果沒有,先訓練模型
python train_v2.py
```

### 無法匯出 JSON

**原因**: 瀏覽器阻擋下載

**解決**: 
1. 允許瀏覽器下載
2. 或右鍵另存為

---

## 進階技巧

### 批量驗證模型

使用命令行工具:
```bash
# 驗證單個模型
python verify_model_metadata.py --single models_output/catboost_long_v2_xxx.pkl

# 驗證模型對
python verify_model_metadata.py \
  --long models_output/catboost_long_v2_xxx.pkl \
  --short models_output/catboost_short_v2_xxx.pkl
```

### 自動化檢查腳本

```bash
#!/bin/bash
# check_models.sh

echo "Checking all V2 models..."

for model in models_output/catboost_*_v2_*.pkl; do
    echo "Checking $model"
    python verify_model_metadata.py --single "$model"
done
```

---

## 相關文檔

- [MODEL_METADATA_FIX.md](MODEL_METADATA_FIX.md) - Metadata 修復詳細說明
- [CHANGELOG.md](CHANGELOG.md) - 版本更新記錄
- [V2_INTEGRATION_COMPLETE.md](V2_INTEGRATION_COMPLETE.md) - V2 系統整合文檔

---

**最後更新**: 2026-02-24  
**版本**: 1.1.0  
**作者**: Zong