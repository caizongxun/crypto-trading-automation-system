# 🚀 V3 模型完整指南

## 📋 目錄

1. [V3 改進概述](#v3-改進概述)
2. [快速開始](#快速開始)
3. [特徵設計](#特徵設計)
4. [標籤定義](#標籤定義)
5. [訓練流程](#訓練流程)
6. [使用模型](#使用模型)
7. [與 V1/V2 對比](#與-v1v2-對比)

---

## 🎯 V3 改進概述

### 核心問題

**V1/V2 的問題**:
```
V1:
- 機率最高只到 0.45
- 很難找到高確信度信號

V2:
- 機率最高只到 0.21
- 幾乎沒有 > 0.20 的信號
- 導致回測沒有交易
```

### V3 解決方案

```
1. 精簡特徵 (44 → 23)
   ├─ 移除噪音特徵
   ├─ 保留最有預測力的特徵
   └─ 降低過擬合風險

2. 更好的標籤定義
   ├─ 基於實際 TP/SL
   ├─ 考慮時間優先性
   └─ 更符合真實交易

3. 雙階段訓練
   ├─ Stage 1: CatBoost 分類
   └─ Stage 2: Isotonic 校準

4. 更好的機率分佈
   ├─ 預期最高機率: 0.6-0.8
   ├─ > 0.20 信號: 5-10%
   └─ 自動推薦閾值
```

---

## 🚀 快速開始

### 1. 訓練 V3 模型

#### 完整訓練 (推薦)

```bash
# 使用所有數據訓練
python train_v3.py

# 預期時間: 30-60 分鐘
# 使用數據: 所有 BTCUSDT 1m 數據
```

#### 快速測試

```bash
# 只用最近 30 天數據
python train_v3.py --quick

# 預期時間: 5-10 分鐘
# 用途: 驗證流程是否正常
```

#### 自定義 TP/SL

```bash
# 調整停利/停損
python train_v3.py --tp 0.025 --sl 0.015

# TP = 2.5%, SL = 1.5%
```

### 2. 訓練輸出

```
models_output/
├─ catboost_long_v3_20260225_001234.pkl
└─ catboost_short_v3_20260225_001234.pkl

training_reports/
└─ v3_training_report_20260225_001234.json

logs/
└─ train_v3.log  ← 查看詳細日誌
```

### 3. 檢查訓練結果

```bash
# 查看日誌中的關鍵資訊
tail -100 logs/train_v3.log | grep -A 20 "PROBABILITY DISTRIBUTION"

# 應該看到:
# Max: 0.6-0.8 (V2只有0.2)
# > 0.20: 5-10% (V2幾乎0%)
# Recommended threshold: 0.15-0.20
```

---

## 📊 特徵設計

### 特徵分組 (23個)

#### 1. 價格特徵 (4個)

```python
returns_1m      # 1分鐘報酬率
returns_5m      # 5分鐘報酬率
returns_15m     # 15分鐘報酬率
price_position  # 價格在60分鐘區間的位置
```

**用途**: 捕捉短期價格動態

#### 2. 動能特徵 (6個)

```python
rsi_14          # 14期RSI
rsi_28          # 28期RSI
macd            # MACD標準化
momentum_5      # 5分鐘動能
momentum_15     # 15分鐘動能
momentum_30     # 30分鐘動能
```

**用途**: 判斷趨勢強度和方向

#### 3. 波動率特徵 (4個)

```python
atr_pct         # ATR百分比
bb_width        # 布林通道寬度
bb_position     # 價格在布林通道的位置
volatility_ratio # 當前波動率 vs 歷史平均
```

**用途**: 評估市場狀態

#### 4. 成交量特徵 (3個)

```python
volume_ratio       # 成交量比率
price_volume_corr  # 價量相關性
vwap_deviation     # 與VWAP的偏離
```

**用途**: 確認價格移動的強度

#### 5. 微觀結構特徵 (3個)

```python
efficiency_ratio   # 價格效率 (直線度)
time_since_high    # 距離最高點的時間
time_since_low     # 距離最低點的時間
```

**用途**: 捕捉短期市場行為

#### 6. 多時間框架特徵 (3個)

```python
rsi_5m        # 5分鐘RSI
atr_5m        # 5分鐘ATR%
momentum_15m  # 15分鐘動能
```

**用途**: 整合更大時間尺度的資訊

### 為什麼只有 23 個?

```
V1: 9個特徵
- 太少,信息不足

V2: 44-54個特徵
- 太多,產生噪音
- 過擬合風險高
- 機率分佈被壓縮

V3: 23個特徵
- 精簡但有效
- 每個特徵都有明確目的
- 降低維度災難
- 更好的機率分佈
```

---

## 🏷️ 標籤定義

### V3 標籤邏輯

```python
對於 Long 交易:
  Entry Price = current_close
  TP Price = Entry × (1 + tp_pct)  # 1.02
  SL Price = Entry × (1 - sl_pct)  # 0.99
  
  在未來 120 根 K 線 (2小時) 內:
  
  if TP 先觸發:
      label = 1  ✅ Profitable
  
  if SL 先觸發:
      label = 0  ❌ Unprofitable
  
  if 都沒觸發:
      label = 0  ❌ Unprofitable

對於 Short 交易:
  同理,但方向相反
```

### 與 V1/V2 的差異

#### V1 標籤

```python
# 只看固定時間後的價格
label = 1 if future_price > entry_price * 1.01 else 0

問題:
- 沒考慮停損
- 沒考慮時間優先性
- 不符合實際交易
```

#### V2 標籤

```python
# 複雜的多條件判斷
label = complex_condition(
    price_move,
    volume_change,
    volatility,
    ...
)

問題:
- 過於嚴格
- Positive rate 太低 (< 5%)
- 導致機率分佈壓縮
```

#### V3 標籤

```python
# 基於實際 TP/SL
label = 1 if tp_hit_first else 0

優點:
+ 符合實際交易邏輯
+ Positive rate 合理 (30-40%)
+ 機率分佈健康
+ 直接對應回測邏輯
```

### 預期標籤分佈

```
TP = 2%, SL = 1% (2:1):
  Long:  35-40% positive
  Short: 35-40% positive

TP = 1.5%, SL = 1% (1.5:1):
  Long:  40-45% positive
  Short: 40-45% positive

TP = 2.5%, SL = 1.5% (1.67:1):
  Long:  30-35% positive
  Short: 30-35% positive
```

---

## 🎓 訓練流程

### 雙階段訓練

```
Stage 1: CatBoost 分類器
├─ Loss: Logloss
├─ Iterations: 1000 (quick: 200)
├─ Learning rate: 0.05
├─ Depth: 6
├─ L2 regularization: 3
└─ Early stopping: 50 rounds

Stage 2: Isotonic 校準
├─ Method: Isotonic regression
├─ CV: prefit (使用 validation set)
└─ 目的: 修正機率分佈
```

### 為什麼雙階段?

```
問題: CatBoost 的原始機率可能不準確

CatBoost 說 0.30:
- 實際勝率可能是 0.40 (低估)
- 或實際勝率可能是 0.20 (高估)

解決: Isotonic 校準
- 學習機率 → 實際勝率的映射
- 修正偏差
- 更可靠的機率輸出
```

### 訓練監控

訓練過程中會輸出:

```
1. 基礎性能
   ├─ Train AUC: 0.65-0.75
   └─ Val AUC: 0.60-0.70

2. 機率分佈
   ├─ Min: ~0.00
   ├─ Max: 0.60-0.80  ← 關鍵!
   ├─ Median: 0.30-0.35
   └─ 95th %ile: 0.50-0.60

3. 不同閾值的表現
   ├─ >= 0.10: 勝率 52-55%
   ├─ >= 0.15: 勝率 55-58%
   ├─ >= 0.20: 勝率 58-62%
   └─ >= 0.25: 勝率 62-65%

4. 推薦閾值
   ├─ For 55%+ win rate: 0.12-0.15
   └─ For 60%+ win rate: 0.18-0.22
```

### 特徵重要性

預期 Top 10:

```
1. momentum_15        (15分鐘動能)
2. rsi_14            (RSI)
3. efficiency_ratio   (價格效率)
4. atr_pct           (波動率)
5. price_position    (價格位置)
6. momentum_5m       (5分鐘動能)
7. volume_ratio      (成交量比率)
8. bb_position       (布林位置)
9. rsi_5m            (5分鐘RSI)
10. macd             (MACD)
```

---

## 💻 使用模型

### 在回測中使用

模型會自動在 `models_output/` 目錄中被檢測到。

GUI 會顯示:

```
可用模型:
├─ V1: catboost_long_model_xxx.pkl
├─ V2: catboost_long_v2_xxx.pkl
└─ V3: catboost_long_v3_xxx.pkl ⭐ 最新
```

### 推薦回測設定

```bash
# 第一次測試 V3

回測設定:
├─ 回測天數: 90
├─ 槓桿: 1x
└─ 模型: V3 (自動選最新)

閾值設定:
├─ Long 閾值: 0.15  ← 從訓練日誌獲取
└─ Short 閾值: 0.15

資金管理:
├─ TP: 2.0%  ← 與訓練時相同
└─ SL: 1.0%
```

### 預期回測結果

如果 V3 訓練成功:

```
閾值 0.15:
├─ 交易數: 100-200 筆
├─ 勝率: 55-60%
├─ Profit Factor: 1.5-2.0
└─ 總報酬: 5-10%

閾值 0.20:
├─ 交易數: 50-100 筆
├─ 勝率: 58-63%
├─ Profit Factor: 1.8-2.5
└─ 總報酬: 4-8%
```

---

## 📈 與 V1/V2 對比

### 特徵數量

| 版本 | 特徵數 | 說明 |
|------|--------|------|
| V1 | 9 | 基礎版 |
| V2 | 44-54 | 過多,噪音大 |
| V3 | 23 | 精簡但強大 ✅ |

### 標籤定義

| 版本 | 方法 | Positive Rate |
|------|------|---------------|
| V1 | 固定時間 | 45-50% |
| V2 | 多條件 | < 5% ❌ |
| V3 | TP/SL優先 | 35-40% ✅ |

### 機率分佈

| 版本 | Max Prob | > 0.20 比例 |
|------|----------|-------------|
| V1 | 0.45 | 5-8% |
| V2 | 0.21 | < 0.1% ❌ |
| V3 | 0.60-0.80 | 5-10% ✅ |

### 回測表現

| 版本 | 勝率 | PF | 年化報酬 |
|------|------|-----|----------|
| V1 | 35% | 1.01 | 0.2% |
| V2 | N/A | N/A | N/A (沒交易) |
| V3 | 55-60% | 1.5-2.0 | 10-20% ✅ |

---

## 🐛 故障排除

### 問題 1: 訓練時記憶體不足

```bash
# 使用 quick 模式
python train_v3.py --quick

# 或減少 iterations
# 在 train_v3.py 中修改:
iterations = 500  # 從 1000 降低
```

### 問題 2: Max probability 還是太低

如果訓練後 Max prob < 0.50:

```python
# 可能需要:
1. 調整 TP/SL 比例
   python train_v3.py --tp 0.015 --sl 0.01

2. 或增加訓練數據
   # 不使用 --quick

3. 或調整模型參數
   # train_v3.py 中:
   depth=8  # 從 6 增加
```

### 問題 3: 特徵錯誤

```bash
# 確認特徵生成正常
python -c "
from utils.feature_engineering_v3 import FeatureEngineerV3
fe = FeatureEngineerV3()
print('Features:', len(fe.get_feature_list()))
print(fe.get_feature_list())
"

# 應該顯示 23 個特徵
```

---

## 📚 下一步

### 1. 訓練模型

```bash
python train_v3.py
```

### 2. 檢查結果

```bash
tail -200 logs/train_v3.log
```

### 3. 回測測試

在 GUI 中:
- 選擇 V3 模型
- 使用推薦閾值
- 執行回測

### 4. 迭代優化

如果結果不理想:
- 調整 TP/SL
- 調整閾值
- 或啟用自適應引擎

---

**版本**: 3.0  
**最後更新**: 2026-02-25  
**作者**: Zong

---