# V2 High-Frequency Trading Strategy

## 策略目標

- **月交易量**: 140-150筆
- **月報酬率**: 50%+
- **勝率目標**: 60%+
- **時間框架**: 15m (6-7小時一筆交易)

## 核心創新

### 1. 多時間框架序列學習
- **Transformer模型**: 捕捉長短期依賴關係
- **注意力機制**: 自動學習重要時間點
- **多頭注意力**: 同時處理不同特徵維度

### 2. 市場微觀結構特徵
- **訂單簿失衡率**: 買賣壓力分析
- **大單檢測**: 異常成交量識別
- **價格動量分解**: 短中長期動量
- **波動率狀態機**: 市場狀態分類

### 3. 集成深度學習架構
```
Transformer (序列模式)
    ↓
  + LightGBM (快速決策)
    ↓  
  + CNN (價格圖像特徵)
    ↓
 集成輸出
```

### 4. 動態信號過濾
- **市場狀態分類**: 越勢/震盪/反轉
- **多重確認機制**: 3層過濾
- **動態置信度閾值**: 根據市場調整

## 模塊結構

```
high_frequency_strategy_v2/
├── core/
│   ├── feature_engineer.py      # 時序特徵提取
│   ├── transformer_model.py     # Transformer模型
│   ├── ensemble_predictor.py    # 集成預測器
│   ├── signal_filter.py         # 信號過濾器
│   ├── market_classifier.py     # 市場狀態分類
│   └── risk_manager.py          # 動態風險管理
├── data/
│   ├── hf_loader.py              # HuggingFace數據加載
│   └── preprocessor.py          # 時序數據預處理
├── backtest/
│   └── engine.py                 # 高頻回測引擎
├── configs/
│   └── strategy_config.json     # 策略配置
├── gui/
│   └── app.py                    # Streamlit界面
├── models/                       # 訓練完成的模型
└── requirements.txt              # 依賴套件
```

## 技術特點

### Transformer模型優勢
1. **長期依賴**: 可處理100+根K線的依賴關係
2. **平行計算**: 比LSTM快10倍
3. **注意力機制**: 自動學習關鍵時刻
4. **多頭設計**: 同時分析多個面向

### 高頻交易優化
1. **低延遲預測**: <100ms
2. **分批處理**: 同時處理多個信號
3. **快取機制**: 特徵計算緩存
4. **GPU加速**: 支持CUDA訓練

## 交易策略

### 進場條件(三層過濾)
1. **第一層**: Transformer信號 > 0.7
2. **第二層**: 市場狀態符合
3. **第三層**: 集成模型確認

### 出場策略
- **固定止盈**: 0.3-0.5%
- **固定止損**: 0.2-0.3%
- **時間止損**: 6-8小時未觸發強制平倉
- **跟蹤止盈**: 盈利>0.5%啓動

## 風險控制

- **單筆風險**: 1-2%資金
- **日內最大回撤**: 5%
- **最大持倉時間**: 12小時
- **同時最大交易**: 1筆

## 開始使用

```bash
# 安裝依賴
cd high_frequency_strategy_v2
pip install -r requirements.txt

# 訓練模型
python train_model.py --symbol BTCUSDT --timeframe 15m

# 運行回測
python run_backtest.py --model models/BTCUSDT_15m_v2_20260227

# 啟動GUI
streamlit run gui/app.py
```

## 效能目標

| 指標 | 目標值 |
|------|--------|
| 月交易數 | 140-150 |
| 月報酬率 | 50%+ |
| 勝率 | 60%+ |
| 盈虧比 | 1:1.5 |
| 最大回撤 | <20% |
| Sharpe比率 | >2.0 |

## 更新記錄

- 2026-02-27: 初始化V2策略架構
