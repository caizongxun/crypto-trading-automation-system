# 🔍 Model Version Comparison: V1 vs V2 vs V3

## Quick Summary

| Feature | V1 | V2 | V3 (Latest) |
|---------|----|----|-------------|
| **Status** | 舊版 | 有問題 | **推薦** |
| **Release** | 2026-02 | 2026-02-23 | 2026-02-25 |
| **Features** | 9 | 54 | **30** |
| **TP Target** | 2.0% | 2.0% | **1.2%** |
| **SL Stop** | 1.0% | 1.0% | **0.8%** |
| **Lookahead** | 480 (8h) | 480 (8h) | **240 (4h)** |
| **Label Rate** | 2-3% | < 2% | **5-10%** |
| **Max Prob** | 0.45 | 0.21 | **0.60-0.80** |
| **Expected Win Rate** | 35-40% | 30-35% | **45-55%** |
| **Expected PF** | 1.0-1.3 | 1.0-1.2 | **1.5-2.5** |

---

## 📊 Detailed Comparison

### V1 - Original Model

#### Design

```yaml
Features: 9 basic features
  - efficiency_ratio
  - extreme_time_diff  
  - vol_imbalance_ratio
  - z_score
  - bb_width_pct
  - rsi
  - atr_pct
  - z_score_1h
  - atr_pct_1d

Labels:
  TP: 2.0%
  SL: 1.0%
  Lookahead: 480 bars (8h)
  Label Rate: 2-3%

Model: CatBoost + Isotonic Calibration
```

#### Results

```yaml
Training:
  AUC: 0.62-0.68
  Max Prob: 0.45-0.50
  
Backtest (180 days):
  Trades: 150-250
  Win Rate: 35-40%
  Profit Factor: 1.0-1.3
  Total Return: 0.1-2% (no leverage)
```

#### Pros & Cons

```
✅ Pros:
- Simple, fast
- Stable probability distribution
- Worked initially

❌ Cons:
- Too few features (9)
- Win rate too low (35%)
- Label target too high (2%)
- Not enough signals
```

---

### V2 - Extended Features

#### Design

```yaml
Features: 54 features (!)
  Price: 12 features
  Volatility: 8 features
  Trend: 10 features
  Volume: 8 features
  Microstructure: 6 features
  Market Regime: 4 features
  Multi-timeframe: 6 features

Labels:
  TP: 2.0% (沒改)
  SL: 1.0% (沒改)
  Lookahead: 480 bars (8h)
  Label Rate: < 2% (!)

Model: CatBoost + Isotonic Calibration
```

#### Results

```yaml
Training:
  AUC: 0.60-0.65 (下降!)
  Max Prob: 0.21 (Long), 0.42 (Short) (太低!)
  > 0.15: only 0.2-0.4% (太少!)
  
Backtest (180 days):
  Trades: 0-50 (太少!)
  Threshold 0.16: No trades
  Threshold 0.10: Still few trades
```

#### Pros & Cons

```
✅ Pros:
- Rich features
- Multi-timeframe awareness
- Market regime detection

❌ Cons:
- Too many features (54) - overfitting risk
- Probability distribution broken
- Max prob too low (0.21)
- Label rate too low (< 2%)
- Almost no usable signals
- **Critical Issue**: Can't be used in practice
```

#### Why V2 Failed

```
1. 標籤定義太嚴格
   - 2% TP 太遠
   - 導致 label rate < 2%
   
2. 機率校準過度保守
   - Max prob 只有 0.21
   - 0.16 threshold 沒交易
   - 0.10 threshold 也很少
   
3. 特徵太多
   - 54 個特徵可能有噪音
   - 模型太複雜
```

---

### V3 - Optimized Design (Recommended)

#### Design

```yaml
Features: 30 optimized features
  Price Momentum: 6
  Volatility: 4
  Trend: 5
  Volume: 3
  Microstructure: 3
  Directional Pressure: 2
  Oscillators: 3
  Market Regime: 3
  
Labels: ⭐ KEY IMPROVEMENT
  TP: 1.2% (更實際)
  SL: 0.8% (更緊)
  Lookahead: 240 bars (4h)
  Partial Profit: +0.8% @ 2h
  Label Rate: 5-10% (target)

Model: CatBoost + Isotonic Calibration
```

#### Expected Results

```yaml
Training (Expected):
  AUC: 0.65-0.72
  Precision @ 0.15: 60-65%
  Max Prob: 0.60-0.80
  > 0.15: 5-10%
  > 0.20: 2-5%
  
Backtest (90 days, Expected):
  Trades: 150-300
  Win Rate: 45-55%
  Profit Factor: 1.5-2.5
  Total Return: 5-15% (no leverage)
  With 3x Leverage: 15-45%
```

#### Pros & Cons

```
✅ Pros:
- Balanced feature count (30)
- More aggressive labels (1.2% TP)
- Higher label rate (5-10%)
- Better probability calibration
- Practical for trading
- Higher expected win rate (45-55%)

⚠️ Cons:
- Need to retrain
- Unproven yet (need testing)
```

#### Why V3 Should Work

```
1. 更實際的標籤
   - 1.2% TP 更容易達成
   - 部分盈利條件
   - 預期 label rate 5-10%
   
2. 精簡特徵
   - 30 個精選特徵
   - 移除噪音
   - 保留高質量
   
3. 更短 lookahead
   - 4h vs 8h
   - 更快反饋
   - 更多信號
```

---

## 📊 Probability Distribution Comparison

### V1 Probability

```
Long Model:
  Max:  0.4571
  95%:  0.0903
  > 0.15: 4,108 bars (1.96%)
  > 0.20: 1,847 bars (0.88%)
  
Short Model:
  Max:  0.3660
  95%:  0.1295
  > 0.15: 2,698 bars (1.29%)
  > 0.20: 23 bars (0.01%)
  
Result: 可用,但信號少
```

### V2 Probability (Problem!)

```
Long Model:
  Max:  0.2141 ❌
  95%:  0.0856
  > 0.15: 390 bars (0.19%) ❌
  > 0.20: 5 bars (0.00%) ❌
  
Short Model:
  Max:  0.4245
  95%:  0.1179  
  > 0.15: 891 bars (0.42%)
  > 0.20: 23 bars (0.01%)
  
Result: 不可用!
```

### V3 Probability (Expected)

```
Long Model (Expected):
  Max:  0.60-0.80 ✅
  95%:  0.25-0.35 ✅
  > 0.15: 5-10% ✅
  > 0.20: 2-5% ✅
  
Short Model (Expected):
  Max:  0.60-0.80 ✅
  95%:  0.25-0.35 ✅
  > 0.15: 5-10% ✅
  > 0.20: 2-5% ✅
  
Result: 應該可用!
```

---

## 🎯 Recommended Thresholds

```
| Model | Threshold | Expected Trades | Expected Win Rate |
|-------|-----------|-----------------|-------------------|
| V1    | 0.16-0.18 | 100-200 (180d) | 35-40% |
| V2    | N/A       | 0-50 (180d)     | Can't test |
| V3    | 0.12-0.18 | 200-400 (180d) | 45-55% |
```

---

## 🚀 Migration Guide

### From V1 to V3

```bash
# 1. Train V3
python train_v3.py

# 2. Backtest comparison
# V1 settings:
# - Threshold: 0.16
# - Days: 180
# - Leverage: 1x

# V3 settings:
# - Threshold: 0.15
# - Days: 180  
# - Leverage: 1x

# 3. Compare results
# V3 should have:
# - More trades
# - Higher win rate
# - Better profit factor
```

### From V2 to V3

```bash
# V2 can't be used, so just:

# 1. Train V3
python train_v3.py

# 2. Backtest V3
# - Threshold: 0.15
# - Days: 90-180
# - Leverage: 1-3x

# 3. Should see actual trades now!
```

---

## 📊 Feature Engineering Comparison

### Feature Count

```
V1: 9 features
  - 太少
  - 缺少關鍵信息
  
V2: 54 features  
  - 太多
  - 可能 overfitting
  - 有噪音
  
V3: 30 features
  - 平衡
  - 精選
  - 高質量
```

### Feature Categories

```
| Category | V1 | V2 | V3 |
|----------|----|----|----|
| Price | 3 | 12 | 6 |
| Volatility | 2 | 8 | 4 |
| Trend | 1 | 10 | 5 |
| Volume | 1 | 8 | 3 |
| Microstructure | 0 | 6 | 3 |
| Directional | 0 | 0 | 2 |
| Oscillators | 1 | 4 | 3 |
| Market Regime | 0 | 4 | 3 |
| Multi-TF | 1 | 2 | 0 |
| **Total** | **9** | **54** | **30** |
```

---

## ✅ Recommendation

### For New Users

```
Use V3 directly:
1. python train_v3.py
2. Backtest with threshold 0.15
3. Start with 1-2x leverage
```

### For Existing V1 Users

```
Test V3 alongside V1:
1. Keep V1 model
2. Train V3
3. Run parallel backtests
4. Compare results
5. Switch to better one
```

### For V2 Users

```
Abandon V2, use V3:
1. V2 has critical issues
2. Can't be fixed by tuning
3. Need redesign (= V3)
4. Just train V3
```

---

## 📚 References

- [V3 Model Guide](V3_MODEL_GUIDE.md) - 完整 V3 文檔
- [Strategy Optimization](STRATEGY_OPTIMIZATION_GUIDE.md) - 策略優化
- [GUI Usage](GUI_USAGE_GUIDE.md) - GUI 使用
- [Quick Start](QUICK_START.md) - 快速啟動

---

**Version**: 1.0  
**Date**: 2026-02-25  
**Author**: Zong

**🎯 TL;DR**: Use V3. V2 is broken, V1 is outdated.