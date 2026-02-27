# V4 - Neural Network + Kelly Criterion Strategy

## 策略概述

V4結合深度學習預測與Kelly準則倉位管理,實現風險可控的高報酬交易策略。

## 核心特點

### 1. LSTM神經網絡預測
- 多時間框架價格預測
- 動態勝率估計
- 期望賠率計算
- 信心度評分

### 2. Kelly準則倉位管理
- 數學最優倉位計算
- 分數Kelly降低風險(1/4 Kelly)
- 動態槓桿調整(1-3x)
- 自適應參數優化

### 3. 多層風險控制
- 單筆最大倉位:20%
- 總倉位上限:50%
- Kelly值門檻:>10%
- 連敗保護機制

## 架構設計

```
adaptive_strategy_v4/
├── core/
│   ├── neural_predictor.py      # LSTM預測模型
│   ├── kelly_manager.py          # Kelly倉位管理
│   ├── risk_controller.py        # 風險控制
│   ├── feature_engineer.py       # 特徵工程
│   └── label_generator.py        # 標籤生成
├── backtest/
│   └── engine.py                 # 回測引擎
├── data/
│   ├── hf_loader.py              # HuggingFace數據
│   └── binance_loader.py         # Binance API
└── models/
    └── (trained models)
```

## 預期績效

| 指標 | V3 | V4目標 |
|------|----|---------|
| 勝率 | 55-60% | 60-65% |
| 月報酬 | 50% | 80-100% |
| 最大回撤 | <30% | <20% |
| Sharpe比率 | >1.5 | >2.0 |
| 月交易量 | 150筆 | 100-120筆 |

## Kelly準則公式

```
Kelly% = (p × b - q) / b

其中:
p = 勝率
q = 1 - p (敗率)
b = 平均盈利 / 平均虧損 (賠率)

實際倉位 = Kelly% × 0.25  # 分數Kelly
```

## 使用流程

### 1. 訓練模型
```bash
python adaptive_strategy_v4/train.py --symbol BTCUSDT --timeframe 15m
```

### 2. 回測驗證
```bash
python adaptive_strategy_v4/backtest.py --model models/BTCUSDT_15m_v4_xxx
```

### 3. GUI操作
```bash
streamlit run reversal_strategy_v1/gui/app.py
```
選擇「V4 - Neural Kelly Strategy」

## 與V1-V3對比

| 特性 | V1 | V2 | V3 | V4 |
|------|----|----|----|
| 模型 | XGBoost | LightGBM | LightGBM | LSTM |
| 倉位管理 | 固定 | 動態 | ATR動態 | Kelly動態 |
| 勝率預測 | ✗ | ✗ | ✗ | ✓ |
| 賠率估計 | ✗ | ✗ | ✗ | ✓ |
| 風險控制 | 單層 | 三層 | 五層 | 多層+Kelly |
| 狀態 | 可用 | 無效 | 推薦 | 開發中 |

## 風險警告

- Kelly準則假設參數準確,需持續優化
- 加密市場高波動,使用分數Kelly降低風險
- 建議從低槓桿(1x)開始測試
- 設置嚴格止損保護

## 開發進度

- [x] 架構設計
- [ ] LSTM預測模型
- [ ] Kelly倉位管理器
- [ ] 風險控制模塊
- [ ] 回測引擎
- [ ] GUI整合
- [ ] 實盤測試
