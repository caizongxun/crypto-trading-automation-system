# V3 Adaptive Strategy - Simple but Effective

## Core Philosophy
V2失敗不是模型問題,是**標籤**、**特徵**、**回測邏輯**問題。

---

## 1. Label Generation (標籤生成)

### V2問題:
```python
# 簡單前瞻窗口 - 噪音太多
if max_profit >= 0.4% and max_loss < 0.3%:
    label = 1  # 做多
```
問題:
- 0.4%利潤在15m太小,容易被noise觸發
- 未考慮交易成本(手續費0.1% + 滑點0.05% = 0.15%)
- 未考慮ATR波動

### V3改進:
```python
# ATR動態標籤
atr = df['atr_14']
min_profit = atr * 1.5 + 0.002  # ATR的1.5倍 + 0.2%成本緩衝
max_loss = atr * 0.8            # ATR的0.8倍

# 品質過濾
if (
    max_profit >= min_profit and 
    max_loss < max_loss_threshold and
    volume > avg_volume * 1.2 and      # 成交量確認
    trend_strength > 0.5               # 趨勢強度
):
    label = 1
```

關鍵改進:
- **ATR動態調整** - 波動大時要求更高利潤
- **交易成本考慮** - 確保淨利潤
- **成交量確認** - 避免假突破
- **趨勢強度** - 只在明確趨勢交易

---

## 2. Feature Engineering (特徵工程)

### V2問題:
通用技術指標(RSI, MACD, BB)預測力不足

### V3關鍵特徵:

#### A. 市場微觀結構 (最重要)
```python
1. 價格效率 (Price Efficiency)
   = abs(close - open) / (high - low)
   # 衡量趨勢vs震盪

2. 訂單流不平衡 (Order Flow Imbalance)
   = (buy_volume - sell_volume) / total_volume
   # 買賣壓力

3. 成交量加權價格偏離 (VWAP Deviation)
   = (close - vwap) / vwap
   # 價格vs市場共識
```

#### B. 波動率特徵
```python
1. ATR標準化價格變動
   = (close - close.shift(1)) / atr
   # 相對波動幅度

2. 波動率比率
   = atr_14 / atr_50
   # 短期vs長期波動變化

3. 高低價差比率
   = (high - low) / close
   # 當根K線波動
```

#### C. 動量品質
```python
1. 趨勢一致性
   = sum(close > close.shift(1) for last 10 bars) / 10
   # 方向一致性

2. 支撐壓力距離
   = (close - support_level) / close
   # 價格位置

3. 成交量趨勢
   = volume_ma_5 / volume_ma_20
   # 成交量動能
```

#### D. 多週期確認
```python
1. 15m趨勢 與 1h趨勢 一致性
2. 15m RSI 與 1h RSI 背離檢測
3. 多週期MACD金叉/死叉確認
```

---

## 3. Backtest Logic (回測邏輯)

### V2問題:
```python
# 固定止盈/止損
take_profit = entry_price * 1.015  # 1.5%
stop_loss = entry_price * 0.992    # 0.8%
```

問題:
- BTC波動大時1.5%太小,波動小時太大
- 未考慮趨勢強度
- 未動態調整

### V3改進:

#### A. ATR動態止盈止損
```python
# 入場
atr = df.loc[entry_idx, 'atr_14']
trend_strength = df.loc[entry_idx, 'trend_strength']

if trend_strength > 0.7:  # 強趨勢
    take_profit = entry_price + atr * 2.5
    stop_loss = entry_price - atr * 1.0
else:  # 弱趨勢
    take_profit = entry_price + atr * 1.5
    stop_loss = entry_price - atr * 0.8
```

#### B. 移動止損 (Trailing Stop)
```python
# 每根K線更新
if position_type == 'long':
    if current_profit > atr * 1.0:  # 盈利超過1個ATR
        stop_loss = max(stop_loss, current_price - atr * 0.5)
```

#### C. 時間止損
```python
# 持倉時間過長 (15m: 最多持倉4小時 = 16根K線)
if bars_held > 16:
    exit_reason = 'time_stop'
    # 平倉
```

#### D. 分批平倉
```python
# 達到1.5個ATR利潤 -> 平倉50%
if profit > atr * 1.5:
    close_partial(position_size * 0.5)
    # 剩餘倉位用trailing stop
```

---

## 4. Signal Filtering (信號過濾)

### V2問題:
信號太多(31652筆交易),質量差

### V3過濾器:

```python
def filter_signal(df, idx):
    # Layer 1: 模型預測信心度
    if confidence < 0.6:
        return False
    
    # Layer 2: 成交量確認
    if volume < volume_ma_20 * 1.3:
        return False
    
    # Layer 3: 趨勢強度
    if abs(trend_strength) < 0.5:
        return False
    
    # Layer 4: ATR過濾 (避免極端波動)
    if atr_14 / close > 0.05:  # ATR>5% 太危險
        return False
    
    # Layer 5: 時間過濾 (避免美股收盤前1小時)
    hour = df.loc[idx, 'timestamp'].hour
    if hour in [21, 22]:  # UTC 21-22點
        return False
    
    return True
```

預期:
- 31652筆 -> 約3000-5000筆高質量交易
- 勝率: 55-60%
- 盈虧因子: >1.5

---

## 5. Model Choice (模型選擇)

### 保持簡單:
- **LightGBM** - 速度快,特徵重要性清晰
- 不需要Transformer/LSTM (增加複雜度但效果未必好)

### 訓練改進:
```python
lgb_params = {
    'objective': 'multiclass',
    'num_class': 3,
    'learning_rate': 0.03,  # 降低學習率
    'num_leaves': 31,
    'max_depth': 6,         # 限制深度防止過擬合
    'min_data_in_leaf': 100, # 增加最小葉節點樣本
    'feature_fraction': 0.7, # 特徵抽樣
    'bagging_fraction': 0.7, # 數據抽樣
    'bagging_freq': 5,
    'scale_pos_weight': 2.0  # 平衡類別權重
}
```

---

## 6. Target Metrics (目標指標)

| 指標 | V2實際 | V3目標 |
|------|--------|--------|
| 月交易數 | ~4700筆 | 100-150筆 |
| 勝率 | 51.9% | 55-60% |
| 盈虧因子 | 0.90 | >1.5 |
| 平均盈利 | 5.20 | >10 USDT |
| 平均虧損 | 6.28 | <6 USDT |
| 月報酬 | -100% | 50% |
| 最大回撤 | -100% | <15% |

---

## Implementation Priority

1. **標籤生成** (adaptive_strategy_v3/core/label_generator.py)
   - ATR動態標籤
   - 成交量確認
   - 趨勢強度過濾

2. **特徵工程** (adaptive_strategy_v3/core/feature_engineer.py)
   - 市場微觀結構
   - 波動率特徵
   - 動量品質
   - 多週期確認

3. **回測引擎** (adaptive_strategy_v3/backtest/engine.py)
   - ATR動態止盈止損
   - 移動止損
   - 時間止損
   - 分批平倉

4. **信號過濾** (adaptive_strategy_v3/core/signal_filter.py)
   - 五層過濾器

5. **模型訓練** (adaptive_strategy_v3/core/predictor.py)
   - LightGBM + 改進參數
   - 類別權重平衡
