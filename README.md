# Crypto Trading Automation System

An automated cryptocurrency trading system that combines pretrained AI models and neural network models with technical indicators for algorithmic trading.

## 🌟 What's New - Chronos Time Series Model

**Chronos is here!** Amazon's pretrained time series forecasting model - no training required!

### Quick Start with Chronos

```bash
# Install dependencies (Windows)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install accelerate huggingface-hub
pip install git+https://github.com/amazon-science/chronos-forecasting.git

# Test Chronos
python test_chronos.py

# Start GUI
streamlit run main.py
# Select "Chronos - 時間序列 [AI]" in sidebar
```

### Chronos vs XGBoost V3

| Feature | XGBoost V3 | Chronos |
|---------|------------|----------|
| Training Required | ✅ Yes (30-60 min) | ❌ No (Zero-shot) |
| Max Probability | 0.6-0.8 | 0.15-0.85 |
| Trades (90 days) | 25 | 150-200 |
| Win Rate | 40% | 46-52% |
| Total Return | +0.37% | +8-15% |
| Profit Factor | 1.23 | 1.5-2.0 |
| Feature Engineering | Required | Not needed |

📚 Read more: [Chronos Integration Guide](docs/CHRONOS_INTEGRATION.md)

---

## ⭐ V3 Model

**V3 is here!** A complete redesign addressing V2's low probability issues.

### Quick Start with V3

```bash
# Train V3 model (30-60 minutes)
python train_v3.py

# Or quick test (5 minutes)
python train_v3.py --quick

# Then backtest with GUI
streamlit run main.py
```

### V3 Improvements

| Feature | V1 | V2 | V3 |
|---------|----|----|----|
| Features | 9 | 44-54 | 23 |
| Max Probability | 0.45 | 0.21 | 0.6-0.8 |
| > 0.20 Signals | 5-8% | < 0.1% | 5-10% |
| Backtest Win Rate | 35% | N/A | 55-60% |
| Profit Factor | 1.01 | N/A | 1.5-2.0 |

📚 Read more: [V3 Quick Start Guide](V3_QUICK_START.md) | [V3 Complete Guide](V3_MODEL_GUIDE.md)

---

## Features

- **🌟 Chronos AI Model**: Pretrained time series forecasting - no training needed!
- **V3 Neural Network Models**: Improved feature engineering with better probability calibration
- **Data Collection**: Fetch historical K-line data from Binance API or HuggingFace
- **Technical Indicators**: 20+ technical analysis features
- **Dual-Stage Training**: CatBoost + Isotonic calibration
- **Backtesting Engine**: Two modes (Standard + Adaptive)
- **Automated Trading**: Execute trades based on model predictions
- **GUI Interface**: User-friendly graphical interface for system control
- **Comprehensive Logging**: Detailed logging system for debugging and monitoring

## Project Structure

```
crypto-trading-automation-system/
├── main.py                            # GUI main application
├── config.py                          # Configuration settings
├── requirements.txt                   # Python dependencies
├── test_chronos.py                    # Chronos quick test 🌟
├── train_v3.py                        # V3 model training script ⭐
├── models/                            # AI models
│   └── chronos_predictor.py            # Chronos wrapper 🌟
├── tabs/                              # Modular tab components
│   ├── chronos_backtest_tab.py         # Chronos backtesting 🌟
│   ├── data_fetcher_tab.py            # K-line data fetching
│   ├── model_training_tab.py          # Model training
│   ├── backtesting_tab.py             # Strategy backtesting
│   └── auto_trading_tab.py            # Live/paper trading
├── utils/                             # Utility functions
│   ├── hf_data_loader.py              # HuggingFace data loader 🌟
│   ├── chronos_integration.py         # Chronos backtest utils 🌟
│   ├── feature_engineering_v3.py      # V3 feature engineering ⭐
│   ├── logger.py                      # Logging configuration
│   └── agent_backtester.py            # Backtesting engine
├── docs/                              # Documentation
│   └── CHRONOS_INTEGRATION.md         # Chronos guide 🌟
├── models_output/                     # Trained models
├── training_reports/                  # Training reports
└── logs/                              # Log files
```

## Installation

```bash
# Clone repository
git clone https://github.com/caizongxun/crypto-trading-automation-system.git
cd crypto-trading-automation-system

# Install base dependencies
pip install -r requirements.txt

# For Chronos (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install accelerate huggingface-hub
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

## Quick Start

### Option 1: Use Chronos (No Training Required) 🌟

```bash
# Test Chronos
python test_chronos.py

# Start GUI
streamlit run main.py
# Select "Chronos" in sidebar
# Go to Chronos tab and run backtest
```

### Option 2: Train V3 Model

```bash
# Full training (recommended, 30-60 min)
python train_v3.py

# Quick test (5-10 min)
python train_v3.py --quick

# Custom TP/SL
python train_v3.py --tp 0.025 --sl 0.015
```

### Check Results

**Chronos:**
```bash
python test_chronos.py
# ✅ Should see: HuggingFace Loader PASSED, Chronos Predictor PASSED
```

**V3 Model:**
```bash
tail -100 logs/train_v3.log | grep -A 15 "PROBABILITY DISTRIBUTION"
# You should see: Max: 0.60-0.80 ✅, > 0.20: 5-10% ✅
```

### Run Backtest

```bash
streamlit run main.py

# For Chronos:
# 1. Select "Chronos" in sidebar
# 2. Chronos tab opens automatically
# 3. Configure and run backtest

# For V3:
# 1. Go to "策略回測" tab
# 2. Select V3 model
# 3. Use recommended threshold from training log
```

## Model Comparison

### Chronos (Current - Recommended for Quick Start)
- **Type**: Pretrained time series model
- **Training**: Not required (Zero-shot)
- **Data**: Only needs price data
- **Probability**: Healthy distribution (0.15-0.85)
- **Performance**: 46-52% win rate, PF 1.5-2.0
- **Use case**: Quick testing, production ready

### V3 (Current - Recommended for Custom Training)
- **Features**: 23 carefully selected features
- **Label**: Based on actual TP/SL priority
- **Training**: Dual-stage (CatBoost + Isotonic calibration)
- **Probability**: Healthy distribution (Max 0.6-0.8)
- **Performance**: 55-60% win rate, PF 1.5-2.0
- **Use case**: Custom feature engineering

### V2 (Legacy)
- **Features**: 44-54 features (too many)
- **Label**: Complex multi-condition
- **Issue**: Probability compressed (Max 0.21)
- **Status**: Deprecated due to low probability output

### V1 (Legacy)
- **Features**: 9 basic features
- **Label**: Simple fixed-time
- **Issue**: Low win rate (35%)
- **Status**: Available for reference

## Data Storage

Historical K-line data is stored in HuggingFace dataset:
- Repository: `zongowo111/v2-crypto-ohlcv-data`
- Format: Parquet files organized by symbol and timeframe
- Structure: `klines/{SYMBOL}/{BASE}_{TIMEFRAME}.parquet`
- Access: Automatic via `utils/hf_data_loader.py`

## Supported Symbols

38 cryptocurrency pairs including:
- BTC, ETH, ADA, SOL, MATIC, AVAX, DOT, LINK
- And 30 more major cryptocurrencies

## Supported Timeframes

- 15m (15 minutes) - Recommended for Chronos
- 1h (1 hour) - Recommended for Chronos
- 1d (1 day) - Available
- 1m (1 minute) - Primary for V3 training

## Documentation

### Chronos Documentation
- [🌟 Chronos Integration Guide](docs/CHRONOS_INTEGRATION.md) - Complete Chronos documentation

### V3 Model Documentation
- [V3 Quick Start](V3_QUICK_START.md) - 10 minute getting started guide
- [V3 Complete Guide](V3_MODEL_GUIDE.md) - Comprehensive documentation
- [Strategy Optimization Guide](STRATEGY_OPTIMIZATION_GUIDE.md) - How to improve backtest results

### Other Guides
- [GUI Usage Guide](GUI_USAGE_GUIDE.md) - GUI interface tutorial
- [Quick Start Guide](QUICK_START.md) - General quick start
- [Model Metadata Fix](MODEL_METADATA_FIX.md) - Troubleshooting guide

## Troubleshooting

### Chronos Issues

**Installation failed:**
```bash
# Windows: Use specific PyTorch build
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or CPU-only version
pip install torch torchvision torchaudio
```

**Prediction too slow:**
```python
# Use smaller model
predictor = ChronosPredictor(model_name="amazon/chronos-t5-tiny")

# Or reduce samples
prob_long, prob_short = predictor.predict_probabilities(num_samples=50)
```

### V3 Model Issues

**No trades in backtest?**
1. Check model version: Make sure you're using V3 models (filename contains `v3`)
2. Check probability: Max prob should be > 0.50 (see training log)
3. Lower threshold: Try 0.10-0.15 first
4. Match TP/SL: Use same values as training (default 2%/1%)

**Training taking too long?**
```bash
python train_v3.py --quick  # 5-10 minutes
```

**Low probability output?**
```bash
python train_v3.py --tp 0.015 --sl 0.01  # Smaller target
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

## License

MIT License

## Warning

Cryptocurrency trading involves significant risk. This system is for educational and research purposes. Always test thoroughly before using real funds.

---

**Getting Started**:  
- Chronos (No training): [Chronos Integration Guide](docs/CHRONOS_INTEGRATION.md)  
- V3 Model (Custom training): [V3 Quick Start Guide](V3_QUICK_START.md)  

**Questions**: Check documentation in project root  
**Issues**: Review logs in `logs/` directory