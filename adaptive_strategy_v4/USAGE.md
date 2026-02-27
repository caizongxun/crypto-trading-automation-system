# V4 Strategy Usage Guide
V4策略使用指南

## 快速開始

### 1. 訓練模型

```bash
# 基礎訓練
python adaptive_strategy_v4/train.py --symbol BTCUSDT --timeframe 15m

# 自定義參數
python adaptive_strategy_v4/train.py \
    --symbol ETHUSDT \
    --timeframe 1h \
    --epochs 100 \
    --batch-size 128 \
    --hidden-size 256 \
    --num-layers 3
```

**訓練參數:**
- `--symbol`: 交易對 (BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT)
- `--timeframe`: 時間框架 (15m, 1h)
- `--epochs`: 訓練輪數 (default: 50)
- `--batch-size`: 批次大小 (default: 64)
- `--hidden-size`: LSTM隱藏層 (default: 128)
- `--num-layers`: LSTM層數 (default: 2)

**訓練時間:**
- CPU: 15-30分鐘
- GPU: 5-10分鐘

### 2. 回測驗證

```bash
# 基礎回測
python adaptive_strategy_v4/backtest.py --model BTCUSDT_15m_v4_20260227_141234

# 自定義Kelly參數
python adaptive_strategy_v4/backtest.py \
    --model BTCUSDT_15m_v4_20260227_141234 \
    --capital 10000 \
    --kelly-fraction 0.25 \
    --max-leverage 3
```

**回測參數:**
- `--model`: 模型名稱 (必填)
- `--capital`: 初始資金 (default: 10000)
- `--kelly-fraction`: Kelly分數 (default: 0.25, 建議 0.2-0.3)
- `--max-leverage`: 最大槓桿 (default: 3)

### 3. GUI使用

```bash
streamlit run reversal_strategy_v1/gui/app.py
```

1. 左側選擇 **"V4 - Neural Kelly策略"**
2. 選擇 **"模型訓練"** 或 **"回測分析"** Tab
3. 設定參數並執行

## Kelly準則詳解

### 公式

```
Kelly% = (p × b - q) / b

其中:
p = 勝率 (LSTM預測)
q = 1 - p
b = 平均盈利/平均虧損 (LSTM預測)

實際倉位 = Kelly% × kelly_fraction
```

### Kelly分數選擇

| Kelly分數 | 風險 | 適用場景 |
|------------|------|----------|
| 0.10-0.15 | 極低 | 保守型,新手 |
| 0.20-0.25 | 低 | **建議值** |
| 0.30-0.40 | 中 | 激進型 |
| 0.50+ | 高 | 不建議 |

### 槓桿策略

V4根據Kelly值和信心度動態調整槓桿:

```python
if kelly > 0.4 and confidence > 0.7:
    leverage = 3x
elif kelly > 0.3 and confidence > 0.6:
    leverage = 2x
else:
    leverage = 1x
```

## 策略優化建議

### 1. 標籤配置

**標準配置 (建議):**
```python
label_config = {
    'forward_window': 8,
    'atr_profit_multiplier': 0.7,
    'atr_loss_multiplier': 1.5,
    'min_volume_ratio': 0.7,
    'min_trend_strength': 0.15,
    'max_atr_ratio': 0.08
}
```

**平衡配置:**
- `atr_profit_multiplier`: 0.9
- `atr_loss_multiplier`: 1.2
- 目標: 15-20%有效標籤

**激進配置:**
- `atr_profit_multiplier`: 0.7
- `atr_loss_multiplier`: 1.5
- 目標: 20-25%有效標籤

### 2. LSTM超參數

**小型模型 (GPU<8GB):**
```python
hidden_size = 64
num_layers = 2
batch_size = 32
sequence_length = 15
```

**標準模型 (建議):**
```python
hidden_size = 128
num_layers = 2
batch_size = 64
sequence_length = 20
```

**大型模型 (GPU≥8GB):**
```python
hidden_size = 256
num_layers = 3
batch_size = 128
sequence_length = 30
```

### 3. 風險管理

**保守型:**
```python
kelly_fraction = 0.20
max_position = 0.15
max_leverage = 1
min_kelly = 0.15
```

**平衡型 (建議):**
```python
kelly_fraction = 0.25
max_position = 0.20
max_leverage = 3
min_kelly = 0.10
```

**激進型:**
```python
kelly_fraction = 0.30
max_position = 0.25
max_leverage = 5
min_kelly = 0.08
```

## 績效評估標準

### 優秀策略
- 勝率 ≥60%
- 盈虧因子 ≥2.0
- Sharpe比率 ≥2.0
- 最大回撤 ≤20%
- 月報酬 ≥80%

### 良好策略
- 勝率 ≥55%
- 盈虧因子 ≥1.5
- Sharpe比率 ≥1.5
- 最大回撤 ≤30%
- 月報酬 ≥50%

### 可接受策略
- 勝率 ≥50%
- 盈虧因子 ≥1.2
- Sharpe比率 ≥1.0
- 最大回撤 ≤40%
- 月報酬 ≥30%

## 常見問題

### Q1: Kelly分數應該設多少?

**A:** 建議從0.2-0.25開始。分數Kelly(Fractional Kelly)是為了降低風險:
- Full Kelly (1.0): 理論最優,但波動大
- 1/2 Kelly (0.5): 平衡
- 1/4 Kelly (0.25): **建議,風險可控**
- 1/5 Kelly (0.2): 保守

### Q2: 為什麼交易數量少於V3?

**A:** V4有多層築選:
1. LSTM預測信心度
2. Kelly門檻 (min_kelly)
3. 風險控制器

這是正常的,質量>數量。

### Q3: 應該使用多少槓桿?

**A:** 建議:
- 新手: 1x
- 有經驗: 2-3x
- 專業: 3-5x

V4會根據Kelly值和信心度自動調整。

### Q4: 訓練時間太長怎麼辦?

**A:** 優化方法:
1. 減少epochs (50 → 30)
2. 減小hidden_size (128 → 64)
3. 減小sequence_length (20 → 15)
4. 使用GPU

### Q5: 模型過擬合怎麼辦?

**A:** 辨識和解決:
- 訓練準確率高,驗證準確率低 → 增加dropout
- 回測表現差 → 簡化模型
- 標籤過於理想化 → 調整label_config

## 實戰策略

### 第一次使用V4

1. **用小資金測試:**
   ```bash
   --capital 1000 --kelly-fraction 0.20 --max-leverage 1
   ```

2. **觀察30天:**
   - 記錄每筆交易
   - 追蹤回撤
   - 評估實際勝率vs預測勝率

3. **優化調整:**
   - 如果回撤>30%: 減小kelly_fraction
   - 如果交易太少: 降低min_kelly
   - 如果勝率低: 重新訓練模型

### 多符號組合

同時運行多個符號:
```bash
# 訓練多個模型
python adaptive_strategy_v4/train.py --symbol BTCUSDT --timeframe 15m
python adaptive_strategy_v4/train.py --symbol ETHUSDT --timeframe 15m
python adaptive_strategy_v4/train.py --symbol BNBUSDT --timeframe 15m
```

**組合策略:**
- 總資金分配: BTC 50%, ETH 30%, BNB 20%
- 每個符號獨立管理Kelly倉位
- 總曝險不超過50%

## 資源需求

### 訓練階段
- CPU: 4+ cores
- RAM: 8GB+
- GPU: 可選,但強烈建議
- 磁碟: 5GB+

### 運行階段
- CPU: 2+ cores
- RAM: 4GB+
- 磁碟: 2GB+

## 進階使用

### Python API

```python
from adaptive_strategy_v4.core.neural_predictor import NeuralPredictor
from adaptive_strategy_v4.core.kelly_manager import KellyManager

# 加載模型
predictor = NeuralPredictor(config)
predictor.load(model_dir)

# 預測
directions, win_rates, payoffs, confidences = predictor.predict(X)

# Kelly倉位管理
kelly_config = {'kelly_fraction': 0.25, 'max_position': 0.20}
kelly_manager = KellyManager(kelly_config)

position_size, leverage, reason = kelly_manager.calculate_position_size(
    predicted_win_rate=win_rates[i],
    predicted_payoff=payoffs[i],
    confidence=confidences[i],
    capital=10000
)
```

### 實盤整合

參考 `examples/live_trading_v4.py` (待建立)

## 參考資料

- Kelly Criterion: https://en.wikipedia.org/wiki/Kelly_criterion
- LSTM: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- V4 README: [../adaptive_strategy_v4/README.md](README.md)
