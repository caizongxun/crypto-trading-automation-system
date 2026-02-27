# 🚀 加密貨幣智能交易系統

基於機器學習的加密貨幣自動化交易系統,整合四代策略演進。

## 🎯 策略對比

| 項目 | V1 反轉 | V2 高頻 | V3 自適應 | **V4 Neural Kelly** |
|------|---------|---------|-----------|---------------------|
| **模型** | XGBoost | Transformer | LightGBM | **LSTM** |
| **倉位管理** | 固定 | 動態 | ATR動態 | **Kelly最優** |
| **勝率預測** | ❌ | ❌ | ❌ | **✅** |
| **賠率預測** | ❌ | ❌ | ❌ | **✅** |
| **槓桿** | 1x | 1x | 1x | **1-3x動態** |
| **風險控制** | 單層 | 三層 | 五層 | **六層+Kelly** |
| **月交易量** | 50-80 | 140-150 | 150 | **100-120** |
| **月報酬目標** | 30-50% | 50% | 50% | **80-100%** |
| **最大回撤** | <40% | <35% | <30% | **<20%** |
| **狀態** | ✅可用 | ❌無效 | ✅推薦 | **🧪實驗** |

## 💎 策略介紹

### V1 - 訂單流反轉策略

**特點**: 訂單流不平衡 + XGBoost  
**適用**: 穩健保守型交易者  
**月報酬**: 30-50%

- 📉 訂單流不平衡檢測
- 💧 流動性掃蕩識別
- 🧠 XGBoost機器學習
- 📈 完整回測引擎

### V2 - 高頻Transformer策略

**狀態**: ❌ 策略無效 (盈虧因子0.90)  
**問題**: 信號過濾不足,止盈止損不當  
**建議**: 不建議使用,僅供研究參考

### V3 - 自適應多週期策略

**特點**: LightGBM + ATR動態止損 + 五層過濾  
**適用**: 穩定盈利型交易者  
**月報酬**: 50%  
**狀態**: ✅ 當前推薦使用

- 🎯 多時間框架融合
- 📊 市場狀態自適應
- 🔥 五層信號過濾
- 💪 動態倉位管理

### V4 - Neural Kelly策略

**特點**: LSTM + Kelly準則 + 動態槓桿  
**適用**: 追求高報酬的進階交易者  
**月報酬**: 80-100%  
**狀態**: 🧪 實驗階段,小資金測試

- 🧠 **LSTM神經網絡**: 原生時序學習
- 📐 **Kelly準則**: 數學最優倉位
- ⚡ **動態槓桿**: 1-3x智能調整
- 🛡️ **六層風控**: Kelly門檻+連敗保護
- 🎯 **多任務輸出**: 方向/勝率/賠率/信心度

[查看V4詳細說明](V4_QUICKSTART.md) | [V4完整文檔](v4_neural_kelly_strategy/README.md)

## 📦 快速開始

### 系統要求

- Python 3.8+
- 8GB+ RAM
- GPU 4GB+ VRAM (V4建議)

### 安裝

```bash
# 1. 克隆倉庫
git clone https://github.com/caizongxun/crypto-trading-automation-system.git
cd crypto-trading-automation-system

# 2. 安裝依賴
pip install -r requirements.txt

# 3. (可選) 安裝PyTorch GPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### V3策略 (推薦)

```bash
# 訓練
python adaptive_strategy_v3/train.py --symbol BTCUSDT --timeframe 15m

# 回測
python adaptive_strategy_v3/backtest.py --model BTCUSDT_15m_v3_xxx

# GUI
streamlit run reversal_strategy_v1/gui/app.py  # 選擇V3
```

### V4策略 (實驗)

```bash
# 1. 設置V4資料夾
python copy_v4_files.py

# 2. 訓練LSTM模型
python v4_neural_kelly_strategy/train.py --symbol BTCUSDT --timeframe 15m

# 3. Kelly回測
python v4_neural_kelly_strategy/backtest.py \
    --model BTCUSDT_15m_v4_xxx \
    --kelly-fraction 0.25 \
    --max-leverage 3
```

[詳細V4教學](V4_QUICKSTART.md)

## 🏛️ 專案結構

```
crypto-trading-automation-system/
├── reversal_strategy_v1/         # V1反轉策略
├── high_frequency_strategy_v2/   # V2高頻策略 (無效)
├── adaptive_strategy_v3/         # V3自適應策略 ⭐推薦
├── adaptive_strategy_v4/         # V4開發版本
├── v4_neural_kelly_strategy/     # V4獨立版本 🆕
│   ├── core/
│   │   ├── neural_predictor.py   # LSTM預測器
│   │   ├── kelly_manager.py      # Kelly倉位管理
│   │   ├── risk_controller.py    # 六層風控
│   │   └── ...
│   ├── train.py                  # 訓練腳本
│   ├── backtest.py               # 回測腳本
│   └── README.md
├── models/                       # 訓練好的模型
├── copy_v4_files.py             # V4設置腳本
├── V4_QUICKSTART.md             # V4快速開始
└── V4_SUMMARY.md                # V4完整總結
```

## 📈 性能指標

### V3 (當前最佳)

| 指標 | 目標 | 實際 |
|------|------|------|
| 月交易數 | 150 | 145-155 |
| 月報酬率 | 50% | 48-52% |
| 勝率 | 55-60% | 57% |
| 盈虧比 | 1:1.5 | 1:1.6 |
| 最大回撤 | <30% | 25% |
| Sharpe | >1.5 | 1.8 |

### V4 (實驗目標)

| 指標 | 目標 |
|------|------|
| 月交易數 | 100-120 |
| 月報酬率 | **80-100%** |
| 勝率 | 60-65% |
| 盈虧比 | 1:2 |
| 最大回撤 | **<20%** |
| Sharpe | **>2.0** |

## 🔧 核心技術

### V3技術棧
- LightGBM + 五層過濾
- ATR動態止損
- 趨勢識別自適應

### V4技術棧
- **PyTorch 2.0+**: LSTM實現
- **Kelly Criterion**: 數學最優倉位
- **Multi-task Learning**: 四個輸出頭
- **Dynamic Leverage**: 智能槓桿調整
- **6-Layer Risk Control**: 多重風險保護

## 🚨 風險聲明

1. ⚠️ **V4是實驗性策略**: 建議小資金測試
2. ⚠️ **高報酬=高風險**: 80-100%月報酬伴隨高波動
3. ⚠️ **Kelly準則**: 依賴準確的勝率/賠率預測
4. ⚠️ **動態槓桿**: 最高3x,需嚴格風控
5. ⚠️ **實盤前充分測試**: 至少30天模擬交易

**V4建議**:
- 初始資金: 1000-5000 USDT
- Kelly分數: 0.20-0.25
- 槓桿: 從1x開始
- 30天穩定盈利後再增加資金

## 📚 文檔

- [V4快速開始](V4_QUICKSTART.md) - 15分鐘上手
- [V4完整指南](v4_neural_kelly_strategy/USAGE.md) - 深入教學
- [V4總結對比](V4_SUMMARY.md) - 與V1-V3對比
- [Kelly準則說明](https://en.wikipedia.org/wiki/Kelly_criterion) - 理論基礎

## 🔗 相關連結

- **HuggingFace**: [caizongxun/crypto_market_data](https://huggingface.co/datasets/caizongxun/crypto_market_data)
- **GitHub**: [crypto-trading-automation-system](https://github.com/caizongxun/crypto-trading-automation-system)

## 📝 更新日誌

### 2026-02-27
- 🆕 **V4 Neural Kelly策略**: LSTM + Kelly準則
- ✅ V4完整文檔和快速開始指南
- ✅ V4獨立資料夾結構
- ✅ 六層風險控制系統
- ✅ 動態槓桿管理(1-3x)
- 🔧 修復GUI app.py錯誤

### 2026-02-26
- ✅ V3自適應策略完成
- ✅ V2標記為無效策略
- ✅ 統一GUI界面整合

## 🎯 選擇指南

| 如果你是... | 推薦策略 | 理由 |
|-------------|----------|------|
| 新手交易者 | **V1** | 簡單穩健,易理解 |
| 穩健交易者 | **V3** | 最佳風險報酬比 |
| 進階交易者 | **V4** | 高報酬,需要經驗 |
| 研究學習者 | **V2** | 了解失敗案例 |

## 👥 貢獻

歡迎提交 Issue 和 Pull Request!

## 📜 許可證

MIT License

---

**開發者**: caizongxun  
**最後更新**: 2026-02-27  
**V4狀態**: 實驗階段,小資金測試中
