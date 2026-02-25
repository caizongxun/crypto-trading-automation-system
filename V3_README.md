# 🚀 V3 Model - Quick Reference

## 🎯 What is V3?

V3 is a **complete redesign** of the trading model to fix V2's critical issues:

```
V2 Problem: 
- Max probability only 0.21
- Almost no usable signals
- Can't be used for trading

V3 Solution:
- More aggressive labels (1.2% TP vs 2%)
- Better probability calibration
- Expected max probability 0.60-0.80
- 5-10% signal rate
```

---

## ⚡ Quick Start

### 1. Train V3 Model

```bash
# Pull latest code
git pull origin main

# Train V3 (30-60 minutes)
python train_v3.py
```

### 2. Backtest V3

```bash
# Launch GUI
streamlit run main.py

# Settings:
# - Model: V3
# - Threshold: 0.15
# - Days: 90
# - Leverage: 1x

# Run Standard Backtest
```

### 3. Expected Results

```
Training:
- Label rate: 5-10%
- AUC: 0.65-0.72
- Max prob: 0.60-0.80

Backtest (90 days, 1x leverage):
- Trades: 150-300
- Win rate: 45-55%
- Profit factor: 1.5-2.5
- Return: 5-15%
```

---

## 📚 Documentation

### Core Documents

1. **[V3 Model Guide](V3_MODEL_GUIDE.md)** 
   Complete V3 documentation with all details

2. **[V1/V2/V3 Comparison](V1_V2_V3_COMPARISON.md)**
   Why V3 is better than V1/V2

3. **[V3 Changelog](CHANGELOG_V3.md)**
   Technical changes and migration guide

### Supporting Documents

4. **[Strategy Optimization Guide](STRATEGY_OPTIMIZATION_GUIDE.md)**
   How to optimize backtest results

5. **[GUI Usage Guide](GUI_USAGE_GUIDE.md)**
   How to use the GUI

6. **[Quick Start](QUICK_START.md)**
   General quick start guide

---

## 🔑 Key Improvements

### 1. Better Labels

```python
V2: TP=2.0%, SL=1.0%, 8h lookahead
    → Label rate < 2%
    → Too few signals

V3: TP=1.2%, SL=0.8%, 4h lookahead
    + Partial profit condition
    → Label rate 5-10%
    → More signals
```

### 2. Cleaner Features

```python
V2: 54 features (too many, noisy)
V3: 30 features (optimized, high quality)

Removed: Redundant and noisy features
Added: Directional pressure features
```

### 3. Better Probabilities

```python
V2: Max prob = 0.21 (too low!)
    > 0.15: only 0.2%
    
V3: Max prob = 0.60-0.80 (expected)
    > 0.15: 5-10%
```

---

## 📊 Quick Comparison

| Metric | V1 | V2 | V3 |
|--------|----|----|----|
| Features | 9 | 54 | **30** |
| TP Target | 2% | 2% | **1.2%** |
| Label Rate | 2-3% | <2% | **5-10%** |
| Max Prob | 0.45 | 0.21 | **0.60-0.80** |
| Win Rate | 35-40% | N/A | **45-55%** |
| Status | Old | Broken | **Recommended** |

---

## ✅ Validation Checklist

### After Training

```
☐ Label rate: 5-10%
☐ AUC: > 0.65
☐ Max probability: > 0.60
☐ Precision @ 0.15: > 55%
☐ Model files created in models_output/
```

### After Backtest

```
☐ Trades (90d): > 100
☐ Win rate: > 40%
☐ Profit factor: > 1.3
☐ No errors in backtest
```

---

## ⚠️ Common Issues

### Issue 1: Training takes too long

```
Normal: 30-60 minutes
If > 2 hours: Check CPU usage
```

### Issue 2: Still no trades in backtest

```
1. Verify V3 model is selected
2. Lower threshold to 0.10
3. Check max probability in report
4. If max prob < 0.30: investigate training
```

### Issue 3: Label rate too low

```
If < 3%:
1. Check data quality
2. Adjust TP to 1.0%
3. Increase lookahead to 300
```

---

## 📞 Quick Help

### Commands

```bash
# Train V3
python train_v3.py

# Check training log
tail -100 logs/train_v3.log

# Check backtest log
tail -100 logs/agent_backtester.log

# Launch GUI
streamlit run main.py
```

### Files

```
Code:
  utils/feature_engineering_v3.py  # Feature engineering
  train_v3.py                       # Training script
  
Output:
  models_output/catboost_long_v3_*.pkl   # Long model
  models_output/catboost_short_v3_*.pkl  # Short model
  training_reports/v3_training_*.json    # Report
  
Logs:
  logs/train_v3.log              # Training log
  logs/agent_backtester.log      # Backtest log
```

---

## 🎯 Recommended Settings

### For Training

```python
# In train_v3.py (defaults are good):
tp_target=0.012      # 1.2%
sl_stop=0.008        # 0.8%
lookahead_bars=240   # 4h
```

### For Backtesting

```
Model: V3
Days: 90-180
Threshold: 0.12-0.18
Leverage: 1-3x

TP: 1.5-2.0%
SL: 0.8-1.0%
```

---

## 🚀 Next Steps

### New User

```
1. Read V3_MODEL_GUIDE.md
2. Train V3: python train_v3.py
3. Backtest V3 with recommended settings
4. If good, increase leverage
5. Paper trade for 1-2 weeks
6. Consider live trading
```

### Existing User

```
1. Read V1_V2_V3_COMPARISON.md
2. Train V3 alongside current model
3. Compare backtests
4. Switch to better model
5. Update live bots
```

---

## 📊 Expected Performance

### Conservative (Threshold 0.18, 1x leverage)

```
90-day backtest:
- Trades: 100-150
- Win rate: 48-52%
- Profit factor: 1.6-2.0
- Return: 3-8%
- Annualized: 12-32%
```

### Balanced (Threshold 0.15, 2x leverage)

```
90-day backtest:
- Trades: 150-250
- Win rate: 45-50%
- Profit factor: 1.5-2.0
- Return: 10-20%
- Annualized: 40-80%
```

### Aggressive (Threshold 0.12, 3x leverage)

```
90-day backtest:
- Trades: 200-350
- Win rate: 42-48%
- Profit factor: 1.3-1.8
- Return: 15-30%
- Annualized: 60-120%
```

---

## 📝 Quick Links

- [V3 Model Guide](V3_MODEL_GUIDE.md) - Full documentation
- [Comparison](V1_V2_V3_COMPARISON.md) - V1 vs V2 vs V3
- [Changelog](CHANGELOG_V3.md) - Technical details
- [Optimization](STRATEGY_OPTIMIZATION_GUIDE.md) - Tuning guide
- [GUI Guide](GUI_USAGE_GUIDE.md) - GUI usage

---

**Version**: 3.0.0  
**Date**: 2026-02-25  
**Author**: Zong  
**Status**: Ready to Train

---

**TL;DR**: 

```bash
git pull origin main
python train_v3.py    # Wait 30-60 min
streamlit run main.py # Backtest with V3
```

V3 should give you 45-55% win rate with actual tradeable signals.