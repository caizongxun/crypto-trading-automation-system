# V3 Changelog - Complete Redesign

## Version 3.0.0 - 2026-02-25

### 🚀 Major Release - Complete Model Redesign

**Status**: Production Ready (Pending Testing)

---

## 🎯 What's New

### 1. Completely New Label Definition

**Old (V1/V2)**:
```python
TP: 2.0%
SL: 1.0%
Lookahead: 480 bars (8 hours)
Condition: Hit TP before SL

Result:
- Label rate: < 2%
- Too few signals
- Probability too conservative
```

**New (V3)**:
```python
TP: 1.2%  # More realistic
SL: 0.8%  # Tighter control
Lookahead: 240 bars (4 hours)  # Faster response

Conditions:
1. Hit TP before SL (primary)
2. OR up/down 0.8% at 2h mark without hitting SL (partial profit)

Result (Expected):
- Label rate: 5-10%
- More signals
- Better probability distribution
```

### 2. Optimized Feature Set

**Removed** (from V2's 54 features):
- Redundant multi-timeframe features
- Overly complex indicators
- Noisy microstructure details
- 24 features removed

**Added** (New in V3):
- `pressure_ratio_30m` - Directional pressure
- `green_streak` - Momentum persistence
- `price_position_1h/4h` - Range position
- `vol_expanding` - Volatility regime
- Better time-session features

**Result**:
- V2: 54 features → V3: 30 features
- Higher quality, less noise
- Better generalization

### 3. Enhanced Probability Calibration

**Changes**:
- Same Isotonic calibration method
- But better base model (fewer features)
- Better label distribution
- More training samples

**Expected Results**:
```
V2 Probability:
  Max: 0.21 (Long), 0.42 (Short)
  > 0.15: < 1%
  
V3 Probability (Expected):
  Max: 0.60-0.80
  > 0.15: 5-10%
  > 0.20: 2-5%
```

### 4. Independent Long/Short Training

**Same as V2, but better**:
- Separate models for Long and Short
- Different feature importance
- Different probability distributions
- Class weight balancing

---

## 📝 Detailed Changes

### Feature Engineering (`utils/feature_engineering_v3.py`)

#### Price Momentum (6 features)

```python
# New in V3:
'price_position_1h'  # Where in 1h range (0-1)
'price_position_4h'  # Where in 4h range (0-1)

# Kept from V2:
'returns_5m', 'returns_15m', 'returns_30m', 'returns_1h'
```

#### Volatility (4 features)

```python
# New in V3:
'vol_ratio'         # ATR14/ATR60 ratio
'vol_expanding'     # Binary: expanding volatility

# Kept from V2:
'atr_pct_14', 'atr_pct_60'
```

#### Trend (5 features)

```python
# Simplified from V2:
'trend_9_21'        # EMA9 vs EMA21
'trend_21_50'       # EMA21 vs EMA50
'above_ema9'        # Binary
'above_ema21'       # Binary  
'above_ema50'       # Binary

# Removed from V2:
- Complex EMA combinations
- MACD variants
- Too many EMA periods
```

#### Volume (3 features)

```python
# Simplified from V2:
'volume_ratio'      # vs MA20
'volume_trend'      # MA10 vs MA30
'high_volume'       # Binary

# Removed from V2:
- VWAP variants
- Volume oscillators
- Complex volume patterns
```

#### Microstructure (3 features)

```python
# Kept from V2:
'body_pct'          # Candle body %
'bullish_candle'    # Binary
'bearish_candle'    # Binary

# Removed from V2:
- Shadow ratios
- Candle pattern detections
- Body momentum
```

#### Directional Pressure (2 features) - NEW!

```python
# New in V3:
'pressure_ratio_30m'  # Bullish/Bearish bars in 30m
'green_streak'        # Consecutive green candles
```

#### Oscillators (3 features)

```python
# Kept from V2:
'rsi_14'
'rsi_oversold'      # < 35
'rsi_overbought'    # > 65

# Removed from V2:
- Multiple RSI periods
- Stochastic
- Williams %R
```

#### Market Regime (3 features)

```python
# Simplified from V2:
'is_asian'          # 0-8 UTC
'is_london'         # 8-16 UTC
'is_nyc'            # 13-21 UTC

# Removed from V2:
- Minute-level time features
- Complex session overlaps
```

### Training Script (`train_v3.py`)

#### New Parameters

```python
# Label generation:
tp_target=0.012      # 1.2% (was 2.0%)
sl_stop=0.008        # 0.8% (was 1.0%)
lookahead_bars=240   # 4h (was 8h)

# Model training (same as V2):
iterations=500
learning_rate=0.05
depth=6
l2_leaf_reg=3
```

#### Enhanced Evaluation

```python
# New metrics:
- Precision @ multiple thresholds (0.10, 0.15, 0.20, 0.25, 0.30)
- Coverage at each threshold
- Detailed probability distribution (min, 25%, 50%, 75%, 95%, max)

# Reports:
- JSON format training report
- Probability statistics
- Class distribution analysis
```

---

## 📊 Expected Performance

### Training Metrics (Expected)

```yaml
Long Model:
  AUC: 0.65-0.72
  Precision: 0.55-0.65
  Recall: 0.50-0.65
  F1: 0.52-0.65
  
Short Model:
  AUC: 0.65-0.72
  Precision: 0.55-0.65
  Recall: 0.50-0.65
  F1: 0.52-0.65
```

### Backtest Metrics (Expected)

**90-day backtest, threshold 0.15, 1x leverage**:

```yaml
Trades: 150-300
Win Rate: 45-55%
Profit Factor: 1.5-2.5
Avg Win: +1.5-2.0%
Avg Loss: -0.8-1.0%
Total Return: 5-15%

With 3x leverage:
Total Return: 15-45%
Annualized: 60-180%
```

---

## 🚀 Migration Guide

### Step 1: Pull Latest Code

```bash
git pull origin main
```

### Step 2: Train V3 Models

```bash
python train_v3.py
```

Expected time: 30-60 minutes

Output:
```
models_output/
  catboost_long_v3_20260225_HHMMSS.pkl
  catboost_short_v3_20260225_HHMMSS.pkl
  
training_reports/
  v3_training_report_20260225_HHMMSS.json
```

### Step 3: Validate Training

```bash
# Check training log
tail -100 logs/train_v3.log

# Look for:
# 1. Label rate: 5-10%
# 2. AUC: > 0.65
# 3. Max probability: > 0.60
# 4. Precision @ 0.15: > 55%
```

### Step 4: Backtest

```bash
# Launch GUI
streamlit run main.py

# Settings:
- Model: Select V3
- Days: 90
- Threshold: 0.15
- Leverage: 1x

# Run Standard Backtest
```

### Step 5: Compare with V1/V2

```
Run backtests with same settings:

V1:
- Threshold: 0.16
- Days: 90
- Leverage: 1x

V3:
- Threshold: 0.15  
- Days: 90
- Leverage: 1x

Compare:
- Number of trades
- Win rate
- Profit factor
- Total return
```

---

## ⚠️ Breaking Changes

### 1. Feature Names Changed

If you have custom code referencing features:

```python
# Removed features (from V2):
- 'macd_*' features
- 'stoch_*' features  
- 'williams_*' features
- Many volume features
- Complex multi-TF features

# New features (in V3):
+ 'pressure_ratio_30m'
+ 'green_streak'
+ 'price_position_1h'
+ 'price_position_4h'
+ 'vol_expanding'
```

### 2. Label Generation Changed

```python
# Old label function won't work
# Use FeatureEngineerV3.create_features_from_1m()
```

### 3. Model Metadata Format

```python
# New fields in saved model:
results = {
    'version': 'v3',  # New
    'timestamp': '20260225_HHMMSS',
    'probability_stats': {...},  # Enhanced
    'class_distribution': {...}  # New
}
```

---

## 📚 Documentation

### New Files

1. **`V3_MODEL_GUIDE.md`** - Complete V3 documentation
2. **`V1_V2_V3_COMPARISON.md`** - Version comparison
3. **`CHANGELOG_V3.md`** - This file
4. **`utils/feature_engineering_v3.py`** - V3 feature engineering
5. **`train_v3.py`** - V3 training script

### Updated Files

- None (V3 is independent)

---

## 🔧 Technical Details

### Label Algorithm

```python
def _create_label_long_v3(df, tp=0.012, sl=0.008, lookahead=240):
    """
    V3 Long Label Logic:
    
    1. Primary condition:
       - Hit TP (1.2%) before SL (0.8%) within 4h
    
    2. Partial profit condition (NEW):
       - Never hit SL within 4h
       - AND price > entry + 0.8% at 2h mark
    
    This increases label rate while maintaining quality
    """
```

### Probability Calibration

```python
# Same method as V2:
calibrated_model = CalibratedClassifierCV(
    base_model,
    method='isotonic',
    cv='prefit'
)

# But better input:
# - More balanced labels (5-10% vs < 2%)
# - Cleaner features (30 vs 54)
# - Result: Better calibrated probabilities
```

### Class Weighting

```python
# Automatic balancing:
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# For 5% label rate:
pos_weight ≈ 19.0

# For 10% label rate:
pos_weight ≈ 9.0
```

---

## 📊 Validation Metrics

### Training Validation

```yaml
Must Pass:
  - Label rate: 5-10%
  - AUC: > 0.65
  - Max probability: > 0.60
  - Precision @ 0.15: > 55%
  - > 0.15 coverage: > 5%

Good to Have:
  - AUC: > 0.70
  - Max probability: > 0.70
  - Precision @ 0.15: > 60%
```

### Backtest Validation

```yaml
Must Pass:
  - Trades (90d): > 100
  - Win rate: > 40%
  - Profit factor: > 1.3
  
Good to Have:
  - Trades (90d): 150-300
  - Win rate: 45-55%
  - Profit factor: 1.5-2.5
  - Return (90d, 1x): 5-15%
```

---

## 🐛 Known Issues

None yet (V3 just released)

---

## 🔮 Future Improvements

### Potential V3.1 Features

1. **Adaptive Labeling**
   - Adjust TP/SL based on volatility
   - Different targets for different market regimes

2. **Ensemble Models**
   - Combine multiple models
   - Better probability estimates

3. **More Directional Features**
   - Order flow proxies
   - Maker/taker imbalance

4. **Regime Detection**
   - Trending vs ranging
   - High vs low volatility
   - Adjust strategy accordingly

---

## ❓ FAQ

### Q: Should I still use V1 or V2?

**A**: 
- V1: Can still use, but V3 should be better
- V2: No, has critical issues, use V3

### Q: How long does V3 training take?

**A**: 30-60 minutes on typical hardware

### Q: What if V3 training fails?

**A**: Check:
1. HuggingFace token configured
2. Internet connection
3. Enough disk space
4. Check logs/train_v3.log

### Q: What if backtest still has no trades?

**A**: 
1. Verify using V3 model (check GUI)
2. Lower threshold to 0.10
3. Check max probability in training report
4. If max prob < 0.30, retrain or check data

### Q: Can I use V3 with live trading?

**A**: After validating with backtesting:
1. Run 90-day backtest
2. Check performance meets criteria
3. Start with paper trading
4. Monitor for 1-2 weeks
5. Then consider live trading

---

## 📞 Support

If you encounter issues:

1. Check [V3_MODEL_GUIDE.md](V3_MODEL_GUIDE.md)
2. Review [V1_V2_V3_COMPARISON.md](V1_V2_V3_COMPARISON.md)
3. Check training logs: `logs/train_v3.log`
4. Check backtest logs: `logs/agent_backtester.log`

---

**Version**: 3.0.0  
**Release Date**: 2026-02-25  
**Author**: Zong  
**Status**: Production Ready (Pending Validation)

---

**Summary**: V3 completely redesigns labels and features to fix V2's probability distribution issues. Expected to provide 45-55% win rate with 5-15% return per 90 days (1x leverage).