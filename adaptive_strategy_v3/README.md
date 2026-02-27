# V3 Adaptive Strategy - Adjusted for 150 Trades/Month

## Target Metrics
| 指標 | V2實際 | V3目標 (調整後) |
|------|--------|--------|
| 月交易數 | ~4700筆 | **150筆** |
| 勝率 | 51.9% | 55-60% |
| 盈虧因子 | 0.90 | >1.5 |
| 平均盈利 | 5.20 | >10 USDT |
| 平均虧損 | 6.28 | <6 USDT |
| 月報酬 | -100% | 50% |
| 最大回撤 | -100% | <15% |

---

## Adjustment Strategy (150 Trades/Month)

### Label Generation
**調整參數:**
- ATR利潤倍數: `1.5 -> 1.2` (降低要求)
- ATR虧損倍數: `0.8 -> 1.0` (放寬止損)
- 最小成交量比: `1.2x -> 1.0x` (允許正常成交量)
- 最小趨勢強度: `0.5 -> 0.3` (允許弱趨勢)
- 最大ATR比例: `5% -> 6%` (允許稍高波動)

**目標:**
- 有效標籤率: **10-15%** (原始 5%)
- 預估: 288K線/月 × 12% = 345個原始信號

### Signal Filtering
**調整參數:**
- Layer 1 (信心度): `≥0.60 -> ≥0.45` (降低閾值)
- Layer 2 (成交量): `≥1.3x -> ≥1.1x` (放寬)
- Layer 3 (趨勢強度): `≥0.5 -> ≥0.3` (放寬)
- Layer 4 (ATR波動): `<5% -> <5%` (保持)
- Layer 5 (時間): `移除黑名單時段`

**目標:**
- 過濾器通過率: **50%** (原始 20%)
- 預估: 345信號 × 50% = **~170筆交易/月**

### Trade Flow
```
2880 K線/月
  |
  v
10-15% 有效標籤 (標籤生成器)
  |
  v
288-432 原始信號
  |
  v
50% 通過過濾 (五層過濾器)
  |
  v
144-216 筆交易/月
  |
  v
目標: 150筆/月
```

---

## Core Improvements (vs V2)

### 1. Label Generation
**V2問題:**
```python
# 固定閾值 - 噪音太多
if max_profit >= 0.4% and max_loss < 0.3%:
    label = 1
```

**V3改進:**
```python
# ATR動態標籤 (調整後)
atr = df['atr_14']
min_profit = atr * 1.2 + 0.002  # ATR的1.2倍 + 0.2%成本緩衝
max_loss = atr * 1.0            # ATR的1.0倍

if (
    max_profit >= min_profit and 
    max_loss < max_loss_threshold and
    volume > avg_volume * 1.0 and      # 成交量確認 (放寬)
    trend_strength > 0.3               # 趨勢強度 (放寬)
):
    label = 1
```

### 2. Feature Engineering
**關鍵特徵 (保持不變):**
- 市場微觀結構 (價格效率、VWAP偏離、訂單流)
- 波動率特徵 (ATR標準化)
- 動量品質 (趨勢一致性)
- 統計特徵 (偏度、峰度)

### 3. Backtest Logic
**ATR動態止盈止損 (保持不變):**
```python
if trend_strength > 0.7:  # 強趨勢
    take_profit = entry_price + atr * 2.5
    stop_loss = entry_price - atr * 1.0
else:  # 弱趨勢
    take_profit = entry_price + atr * 1.5
    stop_loss = entry_price - atr * 0.8

# 移動止損
if profit > atr * 1.0:
    stop_loss = max(stop_loss, current_price - atr * 0.5)
```

### 4. Signal Filtering (調整後)
**五層過濾器:**
```python
# Layer 1: 信心度 ≥ 0.45 (降低)
# Layer 2: 成交量 ≥ 1.1x (放寬)
# Layer 3: 趨勢強度 ≥ 0.3 (放寬)
# Layer 4: ATR < 5% (保持)
# Layer 5: 無時間限制 (移除)
```

**預期過濾效果:**
- 原始信號: 100%
- Layer 1通過: ~80%
- Layer 2通過: ~70%
- Layer 3通過: ~60%
- Layer 4通過: ~55%
- Layer 5通過: ~50%

---

## Implementation Priority

1. **標籤生成** (adaptive_strategy_v3/core/label_generator.py) ✓
   - ATR動態標籤 (調整參數)
   - 成交量確認 (放寬)
   - 趨勢強度過濾 (放寬)

2. **特徵工程** (adaptive_strategy_v3/core/feature_engineer.py) ✓
   - 市場微觀結構
   - 波動率特徵
   - 動量品質

3. **回測引擎** (adaptive_strategy_v3/backtest/engine.py) ✓
   - ATR動態止盈止損
   - 移動止損
   - 分批平倉

4. **信號過濾** (adaptive_strategy_v3/core/signal_filter.py) ✓
   - 五層過濾器 (調整參數)

5. **模型訓練** (adaptive_strategy_v3/core/predictor.py) ✓
   - LightGBM + 改進參數

---

## Expected Results

### Monthly Performance
- **交易數**: 120-180筆 (目標150)
- **勝率**: 55-60%
- **盈虧因子**: >1.5
- **月報酬**: 50%
- **最大回撤**: <15%

### Quality Metrics
- **平均持倉**: 8-12根K線 (2-3小時)
- **平均盈利**: >10 USDT
- **平均虧損**: <6 USDT
- **盈虧比**: >1.5:1

---

## Parameter Tuning Guide

如果實際交易數與目標不符:

### 交易太少 (<100/月)
1. 降低 `min_confidence`: 0.45 -> 0.40
2. 降低 `min_volume_ratio`: 1.1 -> 1.0
3. 降低 `atr_profit_multiplier`: 1.2 -> 1.1

### 交易太多 (>200/月)
1. 提高 `min_confidence`: 0.45 -> 0.50
2. 提高 `min_trend_strength`: 0.3 -> 0.4
3. 提高 `atr_profit_multiplier`: 1.2 -> 1.3

### 勝率太低 (<50%)
1. 提高 `min_confidence`: 0.45 -> 0.55
2. 提高 `atr_profit_multiplier`: 1.2 -> 1.4
3. 降低 `atr_loss_multiplier`: 1.0 -> 0.9

### 盈虧因子太低 (<1.2)
1. 提高 `atr_tp_strong`: 2.5 -> 3.0
2. 降低 `atr_sl_strong`: 1.0 -> 0.9
3. 啟用 `partial_take_profit`

---

## Usage

```bash
# 更新代碼
git pull

# 啟動GUI
cd reversal_strategy_v1/gui
streamlit run app.py
```

在GUI中:
1. 選擇 **V3 - 自適應多週期策略**
2. **訓練頁籤** - 設定參數 -> 開始訓練
3. **回測頁籤** - 選擇模型 -> 執行回測
4. 查看月交易數是否在 120-180 範圍

---

## Monitoring

訓練後檢查:
1. **有效標籤率**: 應在 10-15%
2. **驗證集準確率**: >0.55
3. **信心度**: 做多/做空平均 >0.45

回測後檢查:
1. **月交易數**: 120-180筆
2. **勝率**: 55-60%
3. **盈虧因子**: >1.5
4. **最大回撤**: <15%
