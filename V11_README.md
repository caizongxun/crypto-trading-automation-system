# V11 - ZigZag 反轉點交易系統

## 簡介

V11 是一個基於 **ZigZag 反轉點**的交易系統,專門捕捉市場的波段轉折點。不同於 V10 的高頻剖頭皮,V11 專注於中線波段交易。

## 核心特色

### 1. ZigZag 反轉點識別

```python
# ZigZag 算法參數
門檼: 3% (可調整 1-10%)
標記: 高點 (high) / 低點 (low)
振幅: 記錄每個反轉的%
```

**優勢:**
- 排除市場噪音
- 捕捉真實的高低點
- 自動調整到市場波動

### 2. 多種反轉指標結合

#### RSI 背離
```
Bullish Divergence: 價格低點更低, RSI 低點更高
Bearish Divergence: 價格高點更高, RSI 高點更低
```

#### MACD 交叉
```
Bullish Cross: MACD 上穿信號線
Bearish Cross: MACD 下穿信號線
```

#### 布林帶反彈
```
Bounce Up: 觸碰下軌反彈
Bounce Down: 觸碰上軌回落
```

#### 量能發散
```
價量背離: 價格新高/低,但量能衰減/增加
```

#### 支撐/阻力位
```
在近期高低點附近 (2% 範圍內)
```

### 3. 提前預警系統

```python
lookahead_bars = 2  # 預設

# 在反轉點發生前 2 根K線就發出信號
# 使得進場更實用
```

### 4. 動態 TP/SL

```python
# 根據 ZigZag 振幅計算
TP = pivot_price * (1 ± zigzag_swing * tp_multiplier)
SL = pivot_price * (1 ± zigzag_swing * sl_multiplier)

# 預設倍數
tp_multiplier = 1.5  # TP = 1.5x 振幅
sl_multiplier = 0.5  # SL = 0.5x 振幅

# 自動 RR 比 = 3:1
```

## 與其他版本比較

| 特性 | V10 剖頭皮 | V11 ZigZag | V3 傳統 |
|------|-------------|-----------|----------|
| **時間框架** | 15m | 1h-4h | 1h-1d |
| **持倉時間** | 45-75分 | 2-6小時 | 8-24小時 |
| **日交易數** | 40-50 | 5-10 | 2-5 |
| **標籤率** | 5-8% | 10-20% | 5-10% |
| **目標勝率** | 55-60% | 50-60% | 45-55% |
| **TP/SL** | 固定 0.6%/0.3% | 動態 (1.5-6%) | 固定 1.2%/0.6% |
| **風格** | 高頻率 | 波段交易 | 趨勢跟隨 |
| **適合市場** | 震盪市 | 所有市場 | 趨勢市 |

## 快速開始

### 步驟 1: 啟動 GUI

```bash
streamlit run main.py
```

### 步驟 2: 選擇 V3 版本

(目前 V11 整合在 V3 版本中)

### 步驟 3: V11 模型訓練 Tab

```
⚙️ 標籤配置:
- ZigZag 門檼: 3% (建議 2-5%)
- 提前預警: 2 根K線
- TP 倍數: 1.5x
- SL 倍數: 0.5x

🔲 反轉確認:
☑️ RSI 背離確認 (必選)
☐ 量能確認 (可選)
☐ 支撐/阻力確認 (可選)

🏋️ 訓練配置:
- 交易對: BTCUSDT
- 時間框架: 1h (建議)
- 訓練天數: 180
- 訓練集比例: 0.8
```

### 步驟 4: 執行訓練

```
點擊 "🚀 開始訓練"

預期耗時: 5-10 分鐘

輸出:
- 模型: models_output/v11_long_BTCUSDT_1h_*.pkl
- 報告: training_reports/v11/v11_training_*.json
```

## 訓練結果預期

### 標籤統計

```
ZigZag 3% 門檼 (180天, 1h):
- 總K線數: ~4,320
- ZigZag 反轉點: ~150-200 個
- Long 信號: ~80-100 (3.5%)
- Short 信號: ~80-100 (3.5%)
- 總標籤率: 7-10%
```

### 模型績效

```
目標指標:
- AUC-ROC: > 0.65
- AUC-PR: > 0.25
- F1 Score: > 0.40
```

## 回測預期

### 配置

```python
# 建議回測配置
本金: $10,000
槓桿: 10x
倉位: 3%
閾值: 0.60-0.65
時間框架: 1h
回測天數: 90
```

### 預期績效 (90天)

```
保守預估:
- 交易數: 100-150
- 勝率: 50-55%
- 總報酬: +8-12%
- Sharpe: 2.0-3.0
- 最大回撤: -8-12%
- 盈虧比: 1.5-2.0

樂觀預估:
- 交易數: 150-200
- 勝率: 55-60%
- 總報酬: +15-20%
- Sharpe: 3.0-4.0
- 最大回撤: -6-10%
- 盈虧比: 2.0-2.5
```

## 優化建議

### 1. ZigZag 門檼調整

```python
震盪市: 2-3% (更多信號)
趨勢市: 4-6% (只做主要波段)
低波動: 5-8% (避免假突破)
```

### 2. 確認條件

```python
# 基礎版 (更多信號)
require_rsi_div = True
require_volume = False
require_sr = False
→ 標籤率 ~10%

# 嚴格版 (更高品質)
require_rsi_div = True
require_volume = True
require_sr = True
→ 標籤率 ~5%, 但勝率更高
```

### 3. TP/SL 倍數

```python
# 激進型 (RR 4:1)
tp_multiplier = 2.0
sl_multiplier = 0.5

# 平衡型 (RR 3:1) - 建議
tp_multiplier = 1.5
sl_multiplier = 0.5

# 保守型 (RR 2:1)
tp_multiplier = 1.0
sl_multiplier = 0.5
```

## 實盤交易

### 前置條件

1. ✅ 已訓練 V11 模型
2. ✅ 回測結果滿意 (勝率 > 50%)
3. ✅ 已測試 Paper Trading
4. ✅ Binance API 已配置

### 啟動步驟

```bash
# 1. 生成 Bot 配置
python generate_v11_bot_config.py \
    --symbol BTCUSDT \
    --timeframe 1h \
    --leverage 10 \
    --position_size 0.03

# 2. 啟動 Bot
streamlit run main.py
→ 自動交易 Tab
→ 載入 v11_bot_config.json
→ 點擊啟動
```

## 風險管理

### 1. 位置限制

```python
max_positions = 2  # 最多同時 2 個位置
max_daily_trades = 10  # 日交易限制
```

### 2. 虧損控制

```python
daily_loss_limit = 0.10  # 日虧損 10%
max_drawdown = 0.15  # 最大回撤 15%
```

### 3. 時間過濾

```python
# 只在流動性好的時段交易
active_hours = (8, 22)  # UTC 8:00-22:00
```

## 故障排除

### 問題 1: 標籤率太低 (<5%)

```
原因: ZigZag 門檼太高
解決: 降低到 2-3%
```

### 問題 2: 標籤率過高 (>20%)

```
原因: ZigZag 門檼太低
解決: 提高到 4-5%
```

### 問題 3: 模型 AUC < 0.60

```
原因: 特徵不足或標籤品質低
解決:
1. 啟用所有確認條件
2. 增加訓練數據到 365 天
3. 調整 ZigZag 門檼
```

### 問題 4: 回測無交易

```
原因: 閾值太高
解決: 降低到 0.55-0.60
```

## 進階功能

### 1. 多時間框架確認

```python
# 同時在 1h 和 4h 確認反轉
# 提高信號可靠性
```

### 2. 險中求穩

```python
# 只做與高級別趨勢一致的反轉
# 例: 4h 上升趨勢,只做 1h Long
```

### 3. 部分止盈

```python
# 到達 50% TP 時平掉 50% 倉位
# 鎖定利潤同時保留上漲空間
```

## 參考資源

- [ZigZag 指標原理](https://www.investopedia.com/terms/z/zig_zag_indicator.asp)
- [RSI 背離交易](https://www.babypips.com/learn/forex/rsi-divergence)
- [V10 vs V11 比較](V10_V11_COMPARISON.md)

## 買責聲明

❗ V11 系統僅供學習研究使用。實盤交易存在風險,可能導致資金損失。請在完全理解系統和風險後再使用。

---

**更新日期:** 2026-02-26  
**版本:** v11.0.0  
**狀態:** 測試中
