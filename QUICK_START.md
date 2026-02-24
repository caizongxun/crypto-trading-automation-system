# 🚀 快速啟動指南

## ✨ 最新更新 (v1.2.0)

🆕 **GUI 模型管理** 功能已上線!

- 🔍 一鍵檢查模型 metadata
- ⚖️ 驗證 Long/Short 模型一致性
- 🗑️ 圖形化管理模型

---

## 💻 安裝

### 1. Clone 倉庫

```bash
git clone https://github.com/caizongxun/crypto-trading-automation-system.git
cd crypto-trading-automation-system
```

### 2. 安裝依賴

```bash
pip install -r requirements.txt
```

### 3. 設定環境變數

```bash
# 建立 .env 檔
cp .env.example .env

# 編輯 .env,填入你的 HuggingFace token
HF_TOKEN=your_token_here
HF_REPO_ID=your_username/your_repo
```

---

## 📊 啟動 GUI

```bash
streamlit run main.py
```

瀏覽器會自動開啟: `http://localhost:8501`

---

## 🎯 5 分鐘快速使用

### 步驟 1: 抓取數據 (約 2 分鐘)

1. 點擊 **📊 K棒資料抽取** 標籤
2. 選擇 `BTCUSDT`
3. 選擇時間範圍 (建議 30 天)
4. 點擊 **下載 1m K棒**

### 步驟 2: 訓練模型 (約 1-2 小時)

1. 點擊 **🚀 V2 模型訓練** 標籤
2. 選擇訓練模式:
   - 快速測試: 30-60 分鐘
   - 完整訓練: 2-4 小時
3. 點擊 **開始訓練**
4. 等待訓練完成

### 步驟 3: 驗證模型 (新功能!)

1. 點擊 **📦 模型管理** 標籤
2. 選擇 **🔍 模型檢查** 子標籤
3. 選擇剛訓練好的模型
4. 點擊 **檢查模型**
5. 確認顯示: ✅ 此模型包含完整 metadata

### 步驟 4: 比較模型

1. 選擇 **⚖️ 模型比較** 子標籤
2. 左邊選 Long 模型,右邊選 Short 模型
3. 點擊 **比較模型**
4. 確認: ✅ 特徵完全一致

### 步驟 5: 執行回測

1. 點擊 **📈 策略回測** 標籤
2. 選擇剛驗證的 Long/Short 模型
3. 選擇回測引擎:
   - 標準雙向智能體
   - 自適應智能體 (推薦)
4. 設定參數 (使用預設即可)
5. 點擊 **執行回測**
6. 查看結果和報告

---

## 📚 重要文檔

| 文檔 | 說明 |
|------|------|
| [GUI_USAGE_GUIDE.md](GUI_USAGE_GUIDE.md) | GUI 詳細使用指南 |
| [MODEL_METADATA_FIX.md](MODEL_METADATA_FIX.md) | Metadata 修復說明 |
| [CHANGELOG.md](CHANGELOG.md) | 版本更新記錄 |
| [V2_INTEGRATION_COMPLETE.md](V2_INTEGRATION_COMPLETE.md) | V2 系統整合 |

---

## 🔧 命令行工具 (選用)

### 訓練模型

```bash
# 快速測試 (30-60 分鐘)
python train_v2.py \
  --mode quick_test \
  --enable_advanced True \
  --enable_ml True

# 完整訓練 (2-4 小時)
python train_v2.py \
  --mode full_training \
  --enable_advanced True \
  --enable_ml True
```

### 驗證模型 Metadata

```bash
# 驗證單個模型
python verify_model_metadata.py \
  --single models_output/catboost_long_v2_xxx.pkl

# 驗證模型對
python verify_model_metadata.py \
  --long models_output/catboost_long_v2_xxx.pkl \
  --short models_output/catboost_short_v2_xxx.pkl
```

### 執行回測

```bash
python run_backtest.py \
  --long models_output/catboost_long_v2_xxx.pkl \
  --short models_output/catboost_short_v2_xxx.pkl \
  --prob_long 0.10 \
  --prob_short 0.10
```

---

## ❓ 常見問題

### Q1: 模型訓練失敗?

**解決方法**:

```bash
# 1. 檢查 log
tail -f logs/train_v2.log

# 2. 確認數據存在
ls -la data/BTCUSDT/

# 3. 如果沒有數據,重新下載
# 使用 GUI 的 K棒資料抽取功能
```

### Q2: 回測結果與預期不符?

**檢查清單**:

```bash
# 1. 驗證模型 metadata
python verify_model_metadata.py \
  --long models_output/catboost_long_v2_xxx.pkl \
  --short models_output/catboost_short_v2_xxx.pkl

# 2. 檢查回測 log
tail -f logs/agent_backtester.log

# 3. 確認模型版本
# 應該看到: "✅ Using features from model metadata"
```

### Q3: GUI 無法啟動?

```bash
# 1. 確認安裝
pip install streamlit

# 2. 清除快取
streamlit cache clear

# 3. 檢查 port 是否被佔用
lsof -i :8501

# 4. 使用其他 port
streamlit run main.py --server.port 8502
```

### Q4: 模型版本識別錯誤?

如果你的模型是在 v1.1.0 之前訓練的:

```bash
# 重新訓練以獲取 metadata
python train_v2.py

# 或使用 GUI
streamlit run main.py
# 然後到 V2 模型訓練標籤
```

---

## 📊 GUI 功能概覽

### V1 模式 (6 個標籤)

1. 📊 **K棒資料抽取** - 下載歷史數據
2. 🔧 **特徵工程** - 生成特徵
3. 🎯 **模型訓練** - 訓練 V1 模型
4. 📈 **策略回測** - 測試策略
5. 🤖 **自動交易** - 實盤交易
6. 📦 **模型管理** - 管理模型 metadata ✨

### V2 模式 (7 個標籤)

在 V1 基礎上增加:
- 🚀 **V2 模型訓練** - 訓練進階模型 (44-54 特徵)

---

## 🛡️ 最佳實踐

### 模型管理

1. **保留最近 2-3 個版本**
   - 使用 📦 模型管理標籤刪除舊模型

2. **每次訓練後驗證**
   - 進入 🔍 模型檢查
   - 確認 metadata 存在

3. **回測前比較模型**
   - 進入 ⚖️ 模型比較
   - 確保 Long/Short 一致

### 工作流程

```
訓練 → 檢查 → 比較 → 回測 → 產線
  │      │      │      │      │
  └──────┴──────┴──────┴──────┴───> 監控
         │
         └───────────> 定期檢查 (每週)
```

---

## 🚀 進階功能

### 批次處理模型

```bash
# 檢查所有 V2 模型
for model in models_output/catboost_*_v2_*.pkl; do
    echo "Checking $model"
    python verify_model_metadata.py --single "$model"
done
```

### 自動化測試

```bash
#!/bin/bash
# auto_test.sh

# 1. 訓練模型
echo "訓練模型..."
python train_v2.py --mode quick_test

# 2. 驗證 metadata
echo "驗證 metadata..."
python verify_model_metadata.py \
  --long models_output/catboost_long_v2_*.pkl \
  --short models_output/catboost_short_v2_*.pkl

# 3. 執行回測
echo "執行回測..."
python run_backtest.py \
  --long models_output/catboost_long_v2_*.pkl \
  --short models_output/catboost_short_v2_*.pkl \
  --prob_long 0.10 \
  --prob_short 0.10

echo "测試完成!"
```

---

## 🔗 相關連結

- [GitHub Repository](https://github.com/caizongxun/crypto-trading-automation-system)
- [HuggingFace Dataset](https://huggingface.co/datasets/your_username/your_repo)
- [Issues](https://github.com/caizongxun/crypto-trading-automation-system/issues)

---

## 💬 支援

如有問題,請:

1. 查看 [GUI_USAGE_GUIDE.md](GUI_USAGE_GUIDE.md)
2. 查看 [MODEL_METADATA_FIX.md](MODEL_METADATA_FIX.md)
3. 在 GitHub 提交 Issue

---

**版本**: 1.2.0  
**最後更新**: 2026-02-24  
**作者**: Zong

---

⭐ **如果這個項目對你有幫助,請給一個 Star!**