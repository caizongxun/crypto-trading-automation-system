# 回測優化流程指南

## 系統架構

### 三大核心引擎

#### 1. BidirectionalAgentBacktester (標準引擎)

**特性**:
- 固定參數回測
- 雙向事件驅動狀態機
- 悉觀成交逻輯
- 非對稱成本計算

**適用場景**:
- 建立績效基線
- 驗證模型效果
- 統計顯著性測試

#### 2. AdaptiveBacktester (自適應引擎)

**特性**:
- 波動率自適應 TP/SL
- 機率分層倉位管理
- 時段差異化策略
- 風控強化 (日內虧損 + 連續停損)

**適用場景**:
- 基線建立後
- 提升穩定性
- 機構級風控

#### 3. BacktestAnalyzer (診斷分析器)

**功能**:
- 特徵重要性分析
- 時段績效分解
- 機率分層測試
- Long vs Short 對比
- 失敗案例診斷
- 自動生成優化建議

---

## 優化流程 (4 階段)

### Phase 1: 建立基線 (Week 1)

#### 目標
獲得統計顯著的樣本數 (n ≥ 100)

#### 配置

```yaml
引擎: BidirectionalAgentBacktester (標準)

參數:
  threshold: 0.15-0.16
  tp_pct: 2.0%
  sl_pct: 1.0%
  position_size: 10%
  trading_hours: 黃金時段 (09-13, 18-21 UTC)
```

#### 執行步驟

1. 啟動 Streamlit: `streamlit run main.py`
2. 切換到「策略回測」Tab
3. 選擇「標準雙向智能體」
4. 設定閾值 0.16
5. 執行回測
6. 查看 `logs/agent_backtester.log`

#### 成功標準

```python
if total_trades >= 100:
    if profit_factor >= 0.9:
        print("✅ 系統有效, 進入 Phase 2")
    else:
        print("⚠️ 模型問題, 需要重新訓練")
else:
    print("⚠️ 降低閾值到 0.14")
```

#### 常見結果

| 結果 | 樣本數 | Profit Factor | 下一步 |
|------|---------|---------------|----------|
| 理想 | 100-150 | 0.9-1.1 | Phase 2 |
| 樣本不足 | <100 | - | 降低閾值 |
| 模型問題 | 100+ | <0.8 | 重新訓練 |
| 過度担心 | 100+ | 1.0+ | 直接 Phase 3 |

---

### Phase 2: 數據診斷 (Week 2)

#### 目標

找出真正的瓶頸

#### 使用工具

```python
from utils.backtest_analyzer import BacktestAnalyzer

analyzer = BacktestAnalyzer(
    trades_df=trades_df,
    equity_df=equity_df,
    model_long_path="models_output/catboost_long_xxx.pkl",
    model_short_path="models_output/catboost_short_xxx.pkl"
)

results = analyzer.analyze_all()
```

#### 關鍵問題

1. **時段效應**: 哪些小時賺錢? 哪些虧錢?
2. **方向偏差**: Long 和 Short 表現差距多少?
3. **機率分布**: 哪個機率區間表現最好?
4. **失敗原因**: 大多數虧損發生在哪裡?

#### 診斷示例

```
時段績效分析:
  最佳時段: 09:00 UTC (PnL: $127.50)
  最差時段: 03:00 UTC (PnL: -$89.20)
  
機率分層分析:
  0.15-0.18: 87 筆, 勝率 34.5%, PnL: $45.20
  0.18-0.22: 38 筆, 勝率 42.1%, PnL: $78.90
  0.22+:     12 筆, 勍率 25.0%, PnL: -$23.10 ⚠️
  
方向對比:
  LONG:  55 筆, 勝率 38.2%, PF: 1.15
  SHORT: 82 筆, 勝率 28.0%, PF: 0.85 ⚠️
```

#### 優化建議

根據診斷結果:

```python
# 如果 0.18-0.22 表現最好
threshold = 0.18  # 提高閾值

# 如果 Short 表現很差
if short_pf < 0.9:
    # 方案 1: 提高 Short 閾值
    prob_threshold_short = 0.20
    
    # 方案 2: 關閉 Short
    enable_short = False

# 如果某些時段賺錢
if hour_9_pf > 1.5 and hour_3_pf < 0.5:
    trading_hours = [(9, 14), (18, 22)]  # 只保留好時段
```

---

### Phase 3: 自適應優化 (Week 3)

#### 目標

實現動態參數調整

#### 使用工具

```python
from utils.adaptive_backtester import AdaptiveBacktester

backtester = AdaptiveBacktester(
    model_long_path=long_model_path,
    model_short_path=short_model_path,
    
    # 基礎參數
    initial_capital=10000,
    base_position_size_pct=0.10,
    prob_threshold_long=0.16,
    prob_threshold_short=0.16,
    base_tp_pct=0.02,
    base_sl_pct=0.01,
    
    # 自適應功能
    enable_volatility_adaptation=True,
    enable_probability_layering=True,
    enable_time_based_strategy=True,
    enable_risk_controls=True,
    
    # 風控參數
    max_daily_loss_pct=0.03,
    max_consecutive_losses=5
)
```

#### 自適應特性說明

##### 1. 波動率自適應

```python
if atr_pct_1d < 0.02:  # 低波動
    tp_pct = 0.015
    sl_pct = 0.0075
elif atr_pct_1d > 0.04:  # 高波動
    tp_pct = 0.025
    sl_pct = 0.0125
else:  # 中等波動
    tp_pct = 0.02
    sl_pct = 0.01
```

**物理意義**: 低波動時目標較小,但更容易到達

##### 2. 機率分層倉位

```python
if probability >= 0.25:      # 極高信心
    position_size = 15%
elif probability >= 0.18:    # 高信心
    position_size = 10%
else:                        # 中等信心 (0.15-0.18)
    position_size = 5%
```

**物理意義**: 高機率時加大倉位,而非完全放棄低機率交易

##### 3. 時段差異化

```python
# 歐洲時段 (09-13 UTC): 趨勢性較弱
if 9 <= hour <= 13:
    threshold *= 0.95  # 降低 5%

# 美國時段 (18-21 UTC): 波動更大
if 18 <= hour <= 21:
    threshold *= 1.05  # 提高 5%
```

##### 4. 風控強化

```python
# 最大日內虧損
if daily_loss > -initial_capital * 0.03:
    stop_all_trading()

# 連續停損保護
if consecutive_losses >= 5:
    position_size *= 0.5  # 減半倉位
```

---

### Phase 4: 細線調整 (Week 4)

#### 目標

達到機構級穩定性

#### 關鍵指標

```yaml
合格標準:
  - Profit Factor >= 1.0
  - Win Rate >= 33%
  - Total Return > 0%
  - Sample Size >= 100

優秀標準:
  - Profit Factor >= 1.2
  - Win Rate >= 35%
  - Total Return >= 5%
  - Max Drawdown < 10%

機構級:
  - Profit Factor >= 1.5
  - Win Rate >= 40%
  - Total Return >= 10%
  - Sharpe Ratio >= 1.0
```

#### A/B 測試

```python
# 測試不同配置
configs = [
    {'threshold': 0.16, 'tp': 0.020, 'sl': 0.010},  # 基線
    {'threshold': 0.18, 'tp': 0.020, 'sl': 0.010},  # 提高閾值
    {'threshold': 0.16, 'tp': 0.015, 'sl': 0.010},  # 壓縮 TP
    {'threshold': 0.16, 'tp': 0.015, 'sl': 0.015},  # 1:1 盈虧比
]

for config in configs:
    results = run_backtest(config)
    compare_results(results)
```

---

## 參數調優指南

### 機率閾值調整

```python
# 基礎勝率 5%, Isotonic 校準後

threshold = 0.10  # 2x Lift - 太寬鬆
threshold = 0.15  # 3x Lift - 合格
threshold = 0.16  # 3.2x Lift - 推薦 ⭐
threshold = 0.18  # 3.6x Lift - 優秀
threshold = 0.20  # 4x Lift - 很好 (但樣本少)
threshold = 0.22  # 4.4x Lift - 過度保守
threshold = 0.25  # 5x Lift - 樣本太少
```

### TP/SL 調整

```python
# 盈虧比與勝率關係

tp:sl = 2:1  # 需要勝率 36%+
tp:sl = 1.5:1  # 需要勝率 42%+
tp:sl = 1:1  # 需要勝率 52%+

# 波動率適配

if market_volatility == "LOW":
    tp, sl = 1.5%, 0.75%  # 小目標,高成功率
elif market_volatility == "HIGH":
    tp, sl = 2.5%, 1.25%  # 大目標,空間充足
```

### 倉位管理

```python
# 固定倉位
position_size = 10%  # 標準

# 動態倉位 (根據機率)
if prob >= 0.25:
    position_size = 15%  # 高機率加碼
elif prob >= 0.18:
    position_size = 10%
else:
    position_size = 5%   # 低機率減碼

# 動態倉位 (根據連續停損)
if consecutive_losses >= 5:
    position_size *= 0.5  # 減半
```

---

## 常見問題解答

### Q1: 為什麼回測是 0 筆交易?

**原因**:
1. 機率閾值太高 (>0.22)
2. 特徵全是 NaN
3. 模型未載入

**解決**:
```bash
# 查看 Log
tail -f logs/agent_backtester.log

# 確認機率分布
🔍 PROBABILITY DISTRIBUTION ANALYSIS
Long prob: max=0.2847, mean=0.0617  # 正常

# 如果 max 和 mean 很接近
⚠️ 特徵遺失，模型成為瞎子
```

### Q2: 為什麼勝率太低?

**原因**:
1. TP 設太大 (>2.5%)
2. SL 設太小 (<0.75%)
3. 在低波動時段交易

**解決**:
```python
# 壓縮 TP/SL
tp_pct = 0.015  # 1.5%
sl_pct = 0.010  # 1.0%

# 或改為 1:1
tp_pct = 0.015
sl_pct = 0.015

# 啟用波動率自適應
enable_volatility_adaptation = True
```

### Q3: 為什麼 Profit Factor < 1.0?

**診斷**:
```python
# 查看詳細分析
analyzer.analyze_all()

# 檢查:
# 1. 時段效應 - 哪些時段虧錢?
# 2. 方向偏差 - Long 或 Short 表現很差?
# 3. 機率分布 - 哪個區間賺錢?
```

**解決**:
```python
# 如果某時段表現很差
trading_hours = [(9, 14), (18, 22)]  # 只保留好的

# 如果 Short 表現很差
prob_threshold_short = 0.20  # 提高閾值

# 如果 0.22+ 表現很差
threshold = 0.18  # 降低閾值,避開極端區間
```

### Q4: 樣本數不足怎麼辦?

**目標**: 至少 100 筆交易

**解決**:
```python
# 1. 降低閾值
threshold = 0.14  # 從 0.16 降到 0.14

# 2. 啟用 24/7 交易
trading_hours = [(0, 24)]

# 3. 檢查機率分布
tail -f logs/agent_backtester.log
> 0.16: 2,345 bars (1.56%)

# 如果 > 0.16 的 bar 太少
# 考慮重新訓練模型
```

---

## 最佳實踐

### 1. 先建立基線

```python
# 永遠先跑標準引擎
engine = BidirectionalAgentBacktester
threshold = 0.16
tp_pct = 0.02
sl_pct = 0.01

# 確認樣本數 >= 100
assert total_trades >= 100

# 確認 PF 接近 1.0
assert 0.9 <= profit_factor <= 1.1
```

### 2. 數據驅動決策

```python
# 不要盲目調參
# 先看診斷分析

analyzer = BacktestAnalyzer(...)
results = analyzer.analyze_all()

# 根據數據作決定
if results['direction_comparison']['short']['profit_factor'] < 0.8:
    # Short 表現很差
    prob_threshold_short = 0.20
else:
    # 兩個方向都好
    enable_adaptive = True
```

### 3. 漸進式優化

```python
# 一次只改一個參數

# 錯誤做法
threshold = 0.22  # 改
tp_pct = 0.015    # 改
sl_pct = 0.015    # 改
# 無法知道是哪個改動導致結果變化

# 正確做法
baseline_pf = 0.95

# 測試 1: 只改閾值
threshold = 0.18
test1_pf = 1.02  # 提升！

# 測試 2: 在測試 1 的基礎上改 TP
tp_pct = 0.015
test2_pf = 1.15  # 再次提升！
```

### 4. 記錄所有測試

```python
# 建立實驗紀錄

experiments = [
    {'id': 1, 'threshold': 0.16, 'tp': 0.020, 'pf': 0.95, 'trades': 142},
    {'id': 2, 'threshold': 0.18, 'tp': 0.020, 'pf': 1.02, 'trades': 98},
    {'id': 3, 'threshold': 0.18, 'tp': 0.015, 'pf': 1.15, 'trades': 105},
]

# 找出最佳配置
best = max(experiments, key=lambda x: x['pf'])
print(f"Best config: ID {best['id']}, PF={best['pf']}")
```

---

## Troubleshooting

### 問題: 模型載入失敗

```bash
Error: Cannot load models_output/catboost_long_xxx.pkl

# 解決
1. 確認檔案存在
ls -la models_output/

2. 確認路徑正確
pwd
# 應該在專案根目錄

3. 重新訓練
python train.py
```

### 問題: 特徵不匹配

```bash
Feature mismatch: expected 9, got 7

# 解決
# 模型和回測的特徵不一致

# 查看模型特徵
import joblib
model = joblib.load('models_output/catboost_long_xxx.pkl')
print(model.estimator.feature_names_)

# 確認回測時生成相同特徵
```

### 問題: 回測太慢

```python
# 原因: 逐筆預測

# 不好
for row in df:
    prob = model.predict_proba(row)  # 慢 100x

# 好
X = df[features].fillna(0).values
probs = model.predict_proba(X)  # 快 100x
```

---

## 總結

### 核心原則

1. **數據驅動**: 先診斷,再優化
2. **漸進式**: 一次改一個參數
3. **統計顯著**: 樣本數 >= 100
4. **記錄追蹤**: 所有實驗都要記錄

### 優化流程

```
Phase 1: 建立基線 (n >= 100, PF ~ 1.0)
   ↓
Phase 2: 數據診斷 (找出瓶頸)
   ↓
Phase 3: 自適應優化 (動態參數)
   ↓
Phase 4: 細線調整 (PF > 1.2)
```

### 目標指標

```yaml
最低目標:
  Profit Factor: >= 1.0
  Win Rate: >= 33%
  Sample Size: >= 100

優秀目標:
  Profit Factor: >= 1.2
  Win Rate: >= 38%
  Total Return: >= 5%

機構級:
  Profit Factor: >= 1.5
  Win Rate: >= 42%
  Total Return: >= 10%
  Sharpe Ratio: >= 1.0
```

---

**現在就開始你的優化之旅吧！**

```bash
git pull origin main
streamlit run main.py
```