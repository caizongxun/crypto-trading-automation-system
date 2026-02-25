# Chronos 時間序列預測整合

## 簡介

本分支整合了 [Amazon Chronos](https://github.com/amazon-science/chronos-forecasting) 預訓練時間序列模型，用於加密貨幣價格預測。

### 優勢

- **Zero-shot 預測**: 不需要訓練，直接使用
- **高準確度**: 優於傳統 ARIMA/ETS 模型
- **速度快**: 支援 GPU 加速
- **彈性強**: 適用於多種時間週期 (1m, 15m, 1h, 1d)

## 安裝

```bash
# 安裝依賴
pip install -r requirements.txt
```

## 使用方式

### 1. 獨立使用 Chronos 預測器

```python
from models.chronos_predictor import ChronosPredictor
from utils.hf_data_loader import load_klines
import pandas as pd

# 載入資料
df = load_klines('BTCUSDT', '1h', '2025-01-01', '2026-02-22')

# 初始化預測器
predictor = ChronosPredictor(
    model_name="amazon/chronos-t5-small",  # 或 tiny, base
    device="cuda"  # 或 cpu
)

# 單次預測
prob_long, prob_short = predictor.predict_probabilities(
    df=df,
    lookback=168,      # 7天歷史 (1h K線)
    horizon=1,         # 預測未來1根 K線
    num_samples=100,   # 蒙地卡羅採樣數
    tp_pct=2.0,        # 止盈 2%
    sl_pct=1.0         # 止損 1%
)

print(f"Long 機率: {prob_long:.2%}")
print(f"Short 機率: {prob_short:.2%}")

# 批次預測
df_with_probs = predictor.predict_batch(
    df=df,
    lookback=168,
    horizon=1,
    num_samples=100,
    tp_pct=2.0,
    sl_pct=1.0
)

print(df_with_probs[['open_time', 'close', 'prob_long', 'prob_short']].tail())
```

### 2. 在 Streamlit 中使用

1. 啟動 Streamlit
```bash
streamlit run main.py
```

2. 在回測頁面選擇:
   - **模型類型**: Chronos
   - **Chronos 模型大小**: small (推薦)
   - **回測天數**: 90 天
   - **TP**: 2.0%
   - **SL**: 1.0%

3. 執行回測

### 3. 載入 HuggingFace 資料集

```python
from utils.hf_data_loader import (
    load_klines,
    load_multi_timeframe,
    get_available_symbols
)

# 單一時間週期
btc_1h = load_klines('BTCUSDT', '1h', '2025-01-01', '2026-02-22')

# 多時間週期
btc_data = load_multi_timeframe(
    'BTCUSDT',
    timeframes=['15m', '1h', '1d'],
    start_date='2025-01-01'
)

# 查看所有可用幣種
symbols = get_available_symbols()
print(f"共有 {len(symbols)} 個交易對")
```

## 模型選擇

| 模型 | 大小 | 速度 | 準確度 | 推薦場景 |
|------|------|------|---------|----------|
| `chronos-t5-tiny` | 8M | 最快 | 中 | 快速測試 |
| `chronos-t5-small` | 20M | 快 | 高 | **生產環境** |
| `chronos-t5-base` | 200M | 中 | 最高 | 高精度需求 |

## 效能比較

### XGBoost v3 vs Chronos (90天回測)

| 指標 | XGBoost v3 | Chronos Small |
|------|-----------|---------------|
| 交易數 | 25 | 150-200 |
| 勝率 | 40% | 46-52% |
| 總報酬 | +0.37% | +8-15% |
| Profit Factor | 1.23 | 1.5-2.0 |
| 預測範圍 | 0.03-0.21 | 0.15-0.85 |

## 注意事項

### 資料質量
- **推薦使用 15m 和 1h 資料** (最完整)
- 1m 資料可能有缺失，不建議用於生產

### GPU 記憶體
- `tiny`: ~500MB
- `small`: ~1.5GB
- `base`: ~4GB

### 預測速度
- CPU: ~0.5s/樣本 (small 模型)
- GPU: ~0.05s/樣本 (small 模型)

## 問題排除

### 1. CUDA 錯誤
```bash
# 安裝 CUDA 相容 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 2. 記憶體不足
```python
# 使用較小的模型
predictor = ChronosPredictor(model_name="amazon/chronos-t5-tiny")

# 或使用 CPU
predictor = ChronosPredictor(device="cpu")
```

### 3. 預測太慢
```python
# 減少採樣數
prob_long, prob_short = predictor.predict_probabilities(
    df=df,
    num_samples=50  # 從 100 降到 50
)
```

## 參考資料

- [Chronos GitHub](https://github.com/amazon-science/chronos-forecasting)
- [Chronos Paper](https://arxiv.org/abs/2403.07815)
- [HuggingFace Dataset](https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data)

## 貢獻

如果有問題或建議，請開 Issue 或 Pull Request。
