# V4 - Neural Kelly Strategy
LSTM神經網絡 + Kelly準則倉位管理策略

## 快速開始

```bash
# 1. 訓練模型
python v4_neural_kelly_strategy/train.py --symbol BTCUSDT --timeframe 15m

# 2. 回測驗證
python v4_neural_kelly_strategy/backtest.py --model BTCUSDT_15m_v4_xxx

# 3. 啟動GUI
streamlit run v4_neural_kelly_strategy/gui_app.py
```

## 核心特點

### 1. LSTM神經網絡
- 原生時序學習能力
- 多任務輸出:方向/勝率/賠率/信心度
- 記憶長期依賴關係

### 2. Kelly準則倉位管理
```python
Kelly% = (p × b - q) / b
實際倉位 = Kelly% × 0.25  # 1/4 Kelly降低風險
```

### 3. 動態槓桿系統
- Kelly > 40% + 信心度 > 70% → 3x槓桿
- Kelly > 30% + 信心度 > 60% → 2x槓桿
- 其他 → 1x槓桿

### 4. 六層風險控制
1. Kelly門檻: <10% 不交易
2. 倉位限制: 單筆≤20%, 總倉位≤50%
3. 連敗保護: 3連敗減倉50%, 5連敗暫停
4. 回撤限制: >30%減倉, >40%暫停
5. 波動率調整: 高波動自動減倉
6. 信心度過濾: <50%不交易

## 策略對比

| 項目 | V3 | V4 |
|------|----|
| 模型 | LightGBM | LSTM |
| 倉位管理 | ATR動態 | Kelly最優 |
| 勝率預測 | ✗ | ✓ |
| 賠率預測 | ✗ | ✓ |
| 槓桿 | 1x | 1-3x動態 |
| 風險控制 | 5層 | 6層+Kelly |
| 月報酬目標 | 50% | 80-100% |
| 最大回撤 | <30% | <20% |

## 文件結構

```
v4_neural_kelly_strategy/
├── README.md                # 本文件
├── USAGE.md                 # 詳細使用指南
├── train.py                 # 訓練腳本
├── backtest.py              # 回測腳本
├── gui_app.py               # 獨立GUI
├── core/
│   ├── neural_predictor.py  # LSTM預測器
│   ├── kelly_manager.py     # Kelly倉位管理
│   ├── risk_controller.py   # 風險控制
│   ├── feature_engineer.py  # 特徵工程
│   └── label_generator.py   # 標籤生成
├── backtest/
│   └── engine.py            # 回測引擎
├── data/
│   ├── hf_loader.py         # HuggingFace數據
│   └── binance_loader.py    # Binance API
└── models/                  # 訓練好的模型
```

## 預期績效

| 指標 | 目標 |
|------|------|
| 勝率 | 60-65% |
| 盈虧因子 | >2.0 |
| Sharpe比率 | >2.0 |
| 最大回撤 | <20% |
| 月報酬 | 80-100% |
| 月交易量 | 100-120筆 |

## 風險警告

1. **模型風險**: LSTM需要更多訓練數據,建議定期重訓
2. **Kelly風險**: 依賴勝率/賠率估計準確性,使用1/4 Kelly降低風險
3. **市場風險**: 加密市場高波動,嚴格執行止損

**建議**:
- 初始資金: 1000-5000 USDT
- Kelly分數: 0.20-0.25
- 槓桿: 從1x開始
- 30天穩定盈利後再增加資金

## 更多資訊

- [詳細使用指南](USAGE.md)
- [V4完整總結](../V4_SUMMARY.md)
- [Kelly準則說明](https://en.wikipedia.org/wiki/Kelly_criterion)

## 開發狀態

- [x] 核心模塊完成
- [x] 訓練/回測腳本
- [x] 獨立GUI
- [ ] 實盤測試
- [ ] 多符號組合管理
