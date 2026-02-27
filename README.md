# 🚀 加密貨幣智能交易系統

基於機器學習的加密貨幣自動化交易系統,整合V1反轉策略與V2高頻Transformer策略。

## 🎯 策略比較

| 項目 | V1 反轉策略 | V2 高頻策略 |
|------|--------------|------------|
| **模型** | XGBoost | Transformer + LightGBM |
| **時序學習** | ❌ | ✅ (100根K線) |
| **集成學習** | ❌ | ✅ |
| **信號過濾** | 單層 | 三層 |
| **風險管理** | 固定 | 動態 |
| **月交易量** | 50-80筆 | 140-150筆 |
| **月報酬目標** | 30-50% | 50%+ |
| **訓練時間** | 5-10分鐘 | 10-20分鐘 |
| **GPU需求** | 不需要 | 建議 |

## 💎 功能特點

### V1 - 訂單流反轉策略

- 📉 **訂單流不平衡檢測**: 識別買賣壓力失衡
- 💧**流動性掃蕩識別**: 捕捉大資金誘騙行為
- 🧠 **XGBoost機器學習**: 高效信號驗證
- 📈 **回測引擎**: 完整的歷史數據驗證

### V2 - 高頻Transformer策略

- ⚡ **Transformer時序模型**: 多頭注意力機制
- 📊 **集成學習**: Transformer + LightGBM
- 🎯 **三層信號過濾**: 置信度 + 狀態 + 技術
- 🔄 **市場自適應**: 動態調整風險參數
- 🔥 **高頻交易**: 140-150筆/月

## 📦 安裝

### 系統要求

- Python 3.8+
- 8GB+ RAM
- GPU 4GB+ VRAM (可選,V2建議)

### 快速開始

```bash
# 1. 克隆倉庫
git clone https://github.com/caizongxun/crypto-trading-automation-system.git
cd crypto-trading-automation-system

# 2. 安裝V1依賴
cd reversal_strategy_v1
pip install -r requirements.txt

# 3. 安裝V2依賴 (如果使用V2)
cd ../high_frequency_strategy_v2
pip install -r requirements.txt

# 4. 安裝TA-Lib
# Windows: 下載.whl從 https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# Linux: sudo apt-get install ta-lib
# Mac: brew install ta-lib

# 5. 啟動統一界面
cd ..
streamlit run main_app.py
```

## 📚 使用指南

### 方法1: 統一界面 (推薦)

```bash
streamlit run main_app.py
```

在界面中選擇V1或V2策略,即可進行:
- 模型訓練
- 回測分析
- 策略比較
- 系統狀態查看

### 方法2: V1獨立界面

```bash
streamlit run reversal_strategy_v1/gui/app.py
```

### 方法3: V2獨立界面

```bash
streamlit run high_frequency_strategy_v2/gui/app.py
```

### 命令行訓練

**V1訓練:**
```bash
cd reversal_strategy_v1
python train_model.py --symbol BTCUSDT --timeframe 15m
```

**V2訓練:**
```bash
cd high_frequency_strategy_v2
python train_model.py --symbol BTCUSDT --timeframe 15m --sequence_length 100
```

## 🏛️ 架構

```
crypto-trading-automation-system/
├── main_app.py                    # 統一GUI界面
├── reversal_strategy_v1/         # V1反轉策略
│   ├── core/
│   │   ├── signal_detector.py    # 信號檢測
│   │   ├── feature_engineer.py   # 特徵工程
│   │   ├── ml_predictor.py       # XGBoost模型
│   │   └── risk_manager.py       # 風險管理
│   ├── backtest/
│   │   └── engine.py             # 回測引擎
│   ├── gui/
│   │   └── app.py                # V1界面
│   └── train_model.py            # V1訓練腳本
├── high_frequency_strategy_v2/   # V2高頻策略
│   ├── core/
│   │   ├── transformer_model.py  # Transformer
│   │   ├── ensemble_predictor.py # 集成模型
│   │   ├── signal_filter.py      # 信號過濾
│   │   ├── market_classifier.py  # 市場分類
│   │   └── risk_manager.py       # 動態風險
│   ├── gui/
│   │   └── app.py                # V2界面
│   └── train_model.py            # V2訓練腳本
└── models/                       # 訓練完成的模型
```

## 📈 性能目標

### V1 目標

| 指標 | 目標值 |
|------|--------|
| 月交易數 | 50-80 |
| 月報酬率 | 30-50% |
| 勝率 | 55-60% |
| 盈虧比 | 1:2 |
| 最大回撤 | <20% |
| Sharpe比率 | >1.5 |

### V2 目標

| 指標 | 目標值 |
|------|--------|
| 月交易數 | 140-150 |
| 月報酬率 | 50%+ |
| 勝率 | 60%+ |
| 盈虧比 | 1:1.5 |
| 最大回撤 | <20% |
| Sharpe比率 | >2.0 |

## 🔧 核心技術

### V1 技術棧

- **機器學習**: XGBoost
- **特徵工程**: 50+技術指標
- **訂單流分析**: OFI (Order Flow Imbalance)
- **技術分析**: TA-Lib
- **數據源**: HuggingFace + Binance API

### V2 技術棧

- **深度學習**: PyTorch 2.0+
- **Transformer**: Multi-head Attention
- **集成學習**: Transformer + LightGBM
- **時序模型**: 100根K線序列
- **動態調整**: 市場狀態自適應

## 🚨 風險聲明

**重要提醒**:

1. 此系統僅供學習和研究使用
2. 加密貨幣交易具有高風險
3. 歷史表現不代表未來結果
4. 請勿投入無法承受損失的資金
5. 實盤交易前請充分測試

## 🔗 相關連結

- **HuggingFace 數據集**: [caizongxun/crypto_market_data](https://huggingface.co/datasets/caizongxun/crypto_market_data)
- **GitHub 倉庫**: [crypto-trading-automation-system](https://github.com/caizongxun/crypto-trading-automation-system)

## 📝 更新日誌

### 2026-02-27
- ✅ 創建V2高頻Transformer策略
- ✅ 創建統一GUI界面
- ✅ 整合V1和V2策略
- ✅ 添加策略對比功能

### 2026-02-26
- ✅ V1固定百分比止損止盈
- ✅ V1回測引擎完善
- ✅ HuggingFace數據集整合

## 👥 貢獻

歡迎提交 Issue 和 Pull Request!

## 📜 許可證

MIT License

---

**開發者**: caizongxun  
**最後更新**: 2026-02-27
