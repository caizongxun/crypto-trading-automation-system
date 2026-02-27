# V4 Quick Start Guide
V4策略快速開始指南

## 步驟 1: 設置V4資料夾

### 方法A: 執行複製腳本 (推薦)

```bash
# 在專案根目錄執行
python copy_v4_files.py
```

這會將所有V4檔案從`adaptive_strategy_v4/`複製到`v4_neural_kelly_strategy/`

### 方法B: 手動複製

```bash
# Linux/Mac
bash copy_v4_files.sh

# Windows (PowerShell)
# 手動複製以下檔案:
# adaptive_strategy_v4/core/* -> v4_neural_kelly_strategy/core/
# adaptive_strategy_v4/data/* -> v4_neural_kelly_strategy/data/
# adaptive_strategy_v4/backtest/* -> v4_neural_kelly_strategy/backtest/
# adaptive_strategy_v4/*.py -> v4_neural_kelly_strategy/
```

## 步驟 2: 修改import路徑

如果使用獨立資料夾,需要更新import:

**train.py 和 backtest.py中:**
```python
# 修改前:
from adaptive_strategy_v4.core.neural_predictor import NeuralPredictor

# 修改後:
from v4_neural_kelly_strategy.core.neural_predictor import NeuralPredictor
```

或者保持原有import不變,使用符號連結:
```bash
# Linux/Mac
ln -s adaptive_strategy_v4 v4_neural_kelly_strategy

# Windows (管理員權限)
mklink /D v4_neural_kelly_strategy adaptive_strategy_v4
```

## 步驟 3: 訓練第一個模型

```bash
python v4_neural_kelly_strategy/train.py \
    --symbol BTCUSDT \
    --timeframe 15m \
    --epochs 50
```

**輸出示例:**
```
============================================================
V4 Training - BTCUSDT 15m
============================================================

[1/6] 加載數據...
[OK] 加載: 50000 筆

[2/6] 特徵工程...
[OK] 特徵數: 87

[3/6] 生成標籤...
[OK] 做多標籤: 5234 (10.5%)
[OK] 做空標籤: 4892 (9.8%)
[OK] 有效標籤率: 20.3%

[4/6] 準備訓練數據...
[OK] 訓練集: 35000
[OK] 驗證集: 7500

[5/6] 訓練LSTM模型...
Epoch 1/50 - Loss: 0.8234
Epoch 10/50 - Loss: 0.4521
Epoch 50/50 - Loss: 0.2134
[OK] 訓練完成

[6/6] 保存模型...
[V4 完成] BTCUSDT_15m_v4_20260227_154532
```

## 步驟 4: 回測驗證

```bash
python v4_neural_kelly_strategy/backtest.py \
    --model BTCUSDT_15m_v4_20260227_154532 \
    --capital 10000 \
    --kelly-fraction 0.25 \
    --max-leverage 3
```

**輸出示例:**
```
============================================================
V4 Backtest - BTCUSDT_15m_v4_20260227_154532
============================================================

[Trading Performance]
  Total Trades: 112
  Winning Trades: 68
  Win Rate: 60.71%
  Total Return: 87.34%
  Total PnL: $8734.21

[Risk Metrics]
  Avg Win: $215.43
  Avg Loss: $98.21
  Profit Factor: 2.19
  Sharpe Ratio: 2.34
  Max Drawdown: -18.32%

[Kelly Metrics]
  Avg Kelly: 12.45%
  Avg Leverage: 1.87x

[Assessment]
  ✓ [Excellent] Win rate ≥60% and PF ≥2.0
  ✓ [Excellent] Sharpe ≥2.0
  ✓ [Good] Max DD ≤20%
```

## 步驟 5: 調整參數優化

### 標籤配置優化

如果有效標籤率太低(<15%),修改`label_config`:

```python
# 在train.py中
label_config = {
    'forward_window': 8,
    'atr_profit_multiplier': 0.7,  # 降低 -> 更多標籤
    'atr_loss_multiplier': 1.5,
    'min_volume_ratio': 0.7,       # 降低 -> 更多標籤
    'min_trend_strength': 0.15,    # 降低 -> 更多標籤
    'max_atr_ratio': 0.08
}
```

### Kelly參數優化

根據回測結果調整Kelly分數:

```bash
# 保守 (回撤<15%)
python v4_neural_kelly_strategy/backtest.py \
    --model MODEL_NAME \
    --kelly-fraction 0.20 \
    --max-leverage 1

# 平衡 (推薦)
python v4_neural_kelly_strategy/backtest.py \
    --model MODEL_NAME \
    --kelly-fraction 0.25 \
    --max-leverage 3

# 激進 (高風險)
python v4_neural_kelly_strategy/backtest.py \
    --model MODEL_NAME \
    --kelly-fraction 0.30 \
    --max-leverage 5
```

## 目錄結構

設置完成後的結構:

```
crypto-trading-automation-system/
├── v4_neural_kelly_strategy/     # V4獨立資料夾
│   ├── README.md
│   ├── USAGE.md
│   ├── train.py                  # 訓練腳本
│   ├── backtest.py               # 回測腳本
│   ├── gui_app.py                # 獨立GUI (待開發)
│   ├── core/
│   │   ├── kelly_manager.py
│   │   ├── neural_predictor.py
│   │   ├── risk_controller.py
│   │   ├── feature_engineer.py
│   │   └── label_generator.py
│   ├── data/
│   │   ├── hf_loader.py
│   │   └── binance_loader.py
│   └── backtest/
│       └── engine.py
│
├── adaptive_strategy_v4/         # V4原始開發資料夾
├── adaptive_strategy_v3/         # V3策略
├── models/                       # 訓練好的模型
├── copy_v4_files.py             # 複製腳本
└── V4_SUMMARY.md                # V4完整總結
```

## 常見問題

### Q: import錯誤

**錯誤:** `ModuleNotFoundError: No module named 'v4_neural_kelly_strategy'`

**解決:**
```bash
# 方法1: 添加到PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 方法2: 在專案根目錄執行
python -m v4_neural_kelly_strategy.train --symbol BTCUSDT --timeframe 15m

# 方法3: 使用符號連結 (保持原有import)
ln -s adaptive_strategy_v4 v4_neural_kelly_strategy
```

### Q: 訓練太慢

**解決:**
```bash
# 減少epochs
python v4_neural_kelly_strategy/train.py --epochs 30

# 使用GPU
# 確保已安裝 torch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Q: 回測沒有交易

**原因:** Kelly值或信心度太低

**解決:**
```bash
# 降低Kelly門檻 (在backtest.py中修改)
backtest_config = {
    'min_kelly': 0.05,  # 從0.10降到0.05
}

# 或降低信心度過濾 (在signal_filter中)
min_confidence = 0.45  # 從0.55降到0.45
```

## 更多資源

- [V4完整說明](v4_neural_kelly_strategy/README.md)
- [詳細使用指南](v4_neural_kelly_strategy/USAGE.md)
- [V4總結對比](V4_SUMMARY.md)
- [Kelly準則維基](https://en.wikipedia.org/wiki/Kelly_criterion)

## 支援

遇到問題?
1. 檢查錯誤訊息和traceback
2. 確認所有dependencies已安裝
3. 查看USAGE.md中的疑難排解
4. 參考V1/V3策略的類似實現
