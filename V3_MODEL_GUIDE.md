# 🚀 V3 Model Complete Guide

## 🎯 Overview

V3 is a complete redesign of the trading model with focus on:

1. **Higher Hit Rate** - More aggressive labels (1.2% TP vs 2%)
2. **Better Probability Calibration** - Ensure usable probability outputs
3. **Practical Features** - 30 optimized features vs 54
4. **Independent Training** - Separate Long/Short with different characteristics

---

## 📊 Key Improvements from V2

### Label Definition

```
V2 Labels:
- TP: 2.0% (太遠)
- SL: 1.0%
- Lookahead: 480 bars (8h)
- 結果: 標籤太少,機率偏低

V3 Labels:
- TP: 1.2% (更實際)
- SL: 0.8% (更緊)
- Lookahead: 240 bars (4h)
- 額外: 部分盈利條件 (2h 上漲 0.8%)
- 結果: 更多標籤,機率分佈合理
```

### Feature Count

```
V2: 54 個特徵 (太多,有噪音)
V3: 30 個特徵 (精選,高質量)
```

### Expected Results

```
V2 機率分佈:
Max: 0.21 (Long), 0.42 (Short)
> 0.15: 只有 0.2-0.4%

V3 預期機率分佈:
Max: 0.6-0.8
> 0.15: 5-10%
> 0.20: 2-5%
```

---

## 🛠️ V3 Feature List

### 1. Price Momentum (6 features)

```python
'returns_5m'         # 5分鐘報酬
'returns_15m'        # 15分鐘報酬
'returns_30m'        # 30分鐘報酬
'returns_1h'         # 1小時報酬
'price_position_1h'  # 1h 價格位置 (0-1)
'price_position_4h'  # 4h 價格位置 (0-1)
```

### 2. Volatility (4 features)

```python
'atr_pct_14'        # 14分鐘 ATR %
'atr_pct_60'        # 60分鐘 ATR %
'vol_ratio'         # 波動率比例
'vol_expanding'     # 波動擴張標誌 (0/1)
```

### 3. Trend (5 features)

```python
'trend_9_21'        # EMA9 vs EMA21 趨勢
'trend_21_50'       # EMA21 vs EMA50 趨勢
'above_ema9'        # 價格 > EMA9 (0/1)
'above_ema21'       # 價格 > EMA21 (0/1)
'above_ema50'       # 價格 > EMA50 (0/1)
```

### 4. Volume (3 features)

```python
'volume_ratio'      # 當前量 vs MA20
'volume_trend'      # MA10 vs MA30
'high_volume'       # 高量標誌 (0/1)
```

### 5. Microstructure (3 features)

```python
'body_pct'          # K線實體占比
'bullish_candle'    # 看漲 K線 (0/1)
'bearish_candle'    # 看跌 K線 (0/1)
```

### 6. Directional Pressure (2 features)

```python
'pressure_ratio_30m'  # 30分鐘看漲/看跌比例
'green_streak'        # 連續看漨根數
```

### 7. Oscillators (3 features)

```python
'rsi_14'            # 14分鐘 RSI
'rsi_oversold'      # RSI 超賣 (0/1)
'rsi_overbought'    # RSI 超買 (0/1)
```

### 8. Market Regime (3 features)

```python
'is_asian'          # 亞洲時段 (0/1)
'is_london'         # 倫敦時段 (0/1)
'is_nyc'            # 紐約時段 (0/1)
```

**Total: 30 features** (每個都經過精心選擇)

---

## 💻 How to Train V3

### Quick Start

```bash
# 1. Pull 最新代碼
git pull origin main

# 2. 執行 V3 訓練
python train_v3.py

# 3. 等待完成 (30-60分鐘)
# 會生成:
# - models_output/catboost_long_v3_YYYYMMDD_HHMMSS.pkl
# - models_output/catboost_short_v3_YYYYMMDD_HHMMSS.pkl
# - training_reports/v3_training_report_YYYYMMDD_HHMMSS.json
```

### Training Process

```
V3 Training Pipeline:

1. Load Data
   │
   ├─ 從 HuggingFace 載入 BTCUSDT 1m 數據
   └─ 約 500k-1M 筆

2. Feature Engineering
   │
   ├─ 30 個優化特徵
   ├─ TP: 1.2%, SL: 0.8%
   ├─ Lookahead: 240 bars (4h)
   └─ 預期 5-10% 標籤率

3. Train Long Model
   │
   ├─ CatBoost Classifier
   ├─ 500 iterations
   ├─ Early stopping
   └─ Isotonic calibration

4. Train Short Model
   │
   ├─ 独立訓練
   └─ 不同特徵權重

5. Save & Report
   │
   ├─ 保存模型檔
   └─ 生成評估報告
```

---

## 📊 Expected Training Results

### Good Performance

```json
{
  "long_model": {
    "auc": 0.65-0.72,
    "precision": 0.55-0.65,
    "recall": 0.50-0.65,
    "probability_stats": {
      "max": 0.60-0.80,
      "p95": 0.25-0.35,
      "p75": 0.12-0.18
    }
  },
  "short_model": {
    "auc": 0.65-0.72,
    "precision": 0.55-0.65,
    "recall": 0.50-0.65,
    "probability_stats": {
      "max": 0.60-0.80,
      "p95": 0.25-0.35,
      "p75": 0.12-0.18
    }
  }
}
```

### Label Distribution

```
好的標籤分佈:
- Positive rate: 5-10%
- 太低 (< 3%): 標籤太嚴格
- 太高 (> 15%): 標籤太寬鬆
```

---

## 🛡️ Backtesting V3

### GUI Integration

V3 模型會自動被 GUI 識別:

```
在 GUI 回測標籤:

模型選擇:
├─ V1: 舊版模型
├─ V2: 當前版本 (有問題)
└─ V3: 新版本 (推薦!) ←

點擊 V3 → 自動載入最新 V3 模型
```

### Recommended Settings

```
回測設定:
├─ 回測天數: 90-180
└─ 槓桿: 1-3x

閾值設定:
├─ Long 閾值: 0.12-0.15
└─ Short 閾值: 0.12-0.15

資金管理:
├─ TP: 1.5-2.0%
└─ SL: 0.8-1.0%
```

### Expected Backtest Results

```
好的 V3 回測結果:

交易數: 150-300 筆 (90天)
勝率: 45-55%
Profit Factor: 1.5-2.5
總報酬: 5-15% (90天, 無槓桿)

加入 3x 槓桿:
總報酬: 15-45% (90天)
年化: 60-180%
```

---

## ⚠️ Troubleshooting

### Problem 1: Label Rate Too Low

```
症狀:
Positive samples: < 3%

原因:
- TP 設太高
- Lookahead 太短

解決:
在 train_v3.py 修改:
tp_target=0.010,   # 1.2% → 1.0%
lookahead_bars=300 # 240 → 300
```

### Problem 2: Max Probability Too Low

```
症狀:
Max probability < 0.5

原因:
- 模型不確定
- 特徵不夠強

解決:
- 檢查 AUC 是否 > 0.65
- 如果 AUC 太低,需要重新設計特徵
```

### Problem 3: Still No Trades in Backtest

```
症狀:
訓練完但回測還是 0 筆交易

檢查:
1. 確認 GUI 選了 V3 模型
2. 查看 max probability
3. 降低閾值到 0.10

如果還是沒有:
- 提供 training log
- 提供 backtest log
```

---

## 📝 Training Log Example

正常的 V3 訓練 log 應該看起來像:

```
[V3] Creating features from 800,000 1m bars
[V3] Label config: TP=1.2%, SL=0.8%, Lookahead=240
[V3] Computing core price features...
[V3] Computing volatility features...
[V3] Computing trend features...
[V3] Computing volume features...
[V3] Computing microstructure features...
[V3] Computing directional pressure...
[V3] Computing oscillators...
[V3] Computing market regime...
[V3] Generating labels...
[V3] Long signals: 45,821 (5.73%)  ← 希望 5-10%
[V3] Short signals: 52,104 (6.51%) ← 希望 5-10%
[V3] Features created: 32 columns

TRAINING LONG MODEL
Features: 30
Positive samples: 45,821 (5.73%)
Train set: 640,000 samples
Test set: 160,000 samples

Training CatBoost model...
[100] AUC: 0.6823
[200] AUC: 0.6945
[300] AUC: 0.6998
Best iteration: 312

Calibrating probabilities...

Evaluating model...
AUC: 0.7021 ← > 0.65 就 OK

Classification Report:
Precision: 0.6124 ← > 0.55 就 OK
Recall:    0.5845
F1-Score:  0.5981

Probability Distribution:
Min:  0.0234
25%:  0.0856
50%:  0.1123
75%:  0.1589
95%:  0.3421 ← 超過 0.20 就好
Max:  0.7234 ← 希望 > 0.60

Precision @ Thresholds:
  @ 0.10: Precision=58.2%, Coverage=12.5%
  @ 0.15: Precision=64.1%, Coverage=6.2%  ← 這個好
  @ 0.20: Precision=69.5%, Coverage=3.1%
  @ 0.25: Precision=73.2%, Coverage=1.4%
```

---

## 🎯 Quick Comparison

```
| 指標 | V2 | V3 |
|------|----|----|---|
| 特徵數 | 54 | 30 |
| TP 目標 | 2.0% | 1.2% |
| SL 限制 | 1.0% | 0.8% |
| Lookahead | 480 (8h) | 240 (4h) |
| 標籤率 | < 2% | 5-10% |
| Max Prob | 0.21 | 0.60-0.80 |
| 預期勝率 | 35% | 45-55% |
| 預期 PF | 1.0-1.2 | 1.5-2.5 |
```

---

## 🚀 Next Steps

### 1. 立即訓練

```bash
git pull origin main
python train_v3.py
```

### 2. 檢查結果

```bash
# 查看訓練 log
tail -100 logs/train_v3.log

# 查看報告
cat training_reports/v3_training_report_*.json
```

### 3. 回測測試

```
啟動 GUI:
streamlit run main.py

回測設定:
- 選擇 V3 模型
- 閾值: 0.15
- 90天
- 1x 槓桿

執行標準回測
```

### 4. 優化參數

根據回測結果調整:
- 閾值 (0.12-0.20)
- 槓桿 (1-5x)
- TP/SL

---

## 📚 Related Documents

- [Strategy Optimization Guide](STRATEGY_OPTIMIZATION_GUIDE.md)
- [GUI Usage Guide](GUI_USAGE_GUIDE.md)
- [Quick Start](QUICK_START.md)

---

**Version**: 3.0.0  
**Date**: 2026-02-25  
**Author**: Zong

---

**🎯 核心改進**: V3 的目標是讓機率分佈合理,讓你可以在 0.10-0.20 的閾值範圍內找到足夠的交易機會,同時保持 45%+ 的勝率!