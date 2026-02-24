# Crypto Trading Automation System

An automated cryptocurrency trading system that combines neural network learning models with technical indicators for algorithmic trading.

## ⭐ What's New - V3 Model

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

- **V3 Neural Network Models**: Improved feature engineering with better probability calibration
- **Data Collection**: Fetch historical K-line data from Binance API
- **Technical Indicators**: 20+ technical analysis features
- **Dual-Stage Training**: CatBoost + Isotonic calibration
- **Backtesting Engine**: Two modes (Standard + Adaptive)
- **Automated Trading**: Execute trades based on model predictions
- **GUI Interface**: User-friendly graphical interface for system control
- **Comprehensive Logging**: Detailed logging system for debugging and monitoring

## Project Structure

```
crypto-trading-automation-system/
├── main.py                        # GUI main application
├── config.py                      # Configuration settings
├── requirements.txt               # Python dependencies
├── train_v3.py                    # V3 model training script ⭐
├── V3_QUICK_START.md              # V3 quick start guide ⭐
├── V3_MODEL_GUIDE.md              # V3 complete documentation ⭐
├── tabs/                          # Modular tab components
│   ├── data_fetcher_tab.py        # K-line data fetching
│   ├── model_training_tab.py      # Model training
│   ├── backtesting_tab.py         # Strategy backtesting
│   └── auto_trading_tab.py        # Live/paper trading
├── utils/                         # Utility functions
│   ├── feature_engineering_v3.py  # V3 feature engineering ⭐
│   ├── logger.py                  # Logging configuration
│   └── agent_backtester.py        # Backtesting engine
├── models_output/                 # Trained models
├── training_reports/              # Training reports
└── logs/                          # Log files
```

## Installation

```bash
# Clone repository
git clone https://github.com/caizongxun/crypto-trading-automation-system.git
cd crypto-trading-automation-system

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train V3 Model

```bash
# Full training (recommended, 30-60 min)
python train_v3.py

# Quick test (5-10 min)
python train_v3.py --quick

# Custom TP/SL
python train_v3.py --tp 0.025 --sl 0.015
```

### 2. Check Training Results

```bash
# View probability distribution
tail -100 logs/train_v3.log | grep -A 15 "PROBABILITY DISTRIBUTION"

# You should see:
# Max: 0.60-0.80 ✅
# > 0.20: 5-10% ✅
```

### 3. Run Backtest

```bash
# Start GUI
streamlit run main.py

# Go to "Strategy Backtest" tab
# Select V3 model
# Use recommended threshold from training log
# Run backtest
```

### Expected Results

```
With threshold 0.15:
- Trades: 100-200
- Win Rate: 55-58%
- Profit Factor: 1.5-1.8
- Total Return: 5-8% (90 days)

With 3x leverage:
- Total Return: 15-24%
```

## Configuration

Edit `config.py` to set up:
- HuggingFace token (required for data storage)
- Binance API credentials (optional, for live trading)
- Trading parameters
- Model settings

## Documentation

### V3 Model Documentation
- [V3 Quick Start](V3_QUICK_START.md) - 10 minute getting started guide
- [V3 Complete Guide](V3_MODEL_GUIDE.md) - Comprehensive documentation
- [Strategy Optimization Guide](STRATEGY_OPTIMIZATION_GUIDE.md) - How to improve backtest results

### Other Guides
- [GUI Usage Guide](GUI_USAGE_GUIDE.md) - GUI interface tutorial
- [Quick Start Guide](QUICK_START.md) - General quick start
- [Model Metadata Fix](MODEL_METADATA_FIX.md) - Troubleshooting guide

## Model Versions

### V3 (Current - Recommended)
- **Features**: 23 carefully selected features
- **Label**: Based on actual TP/SL priority
- **Training**: Dual-stage (CatBoost + Isotonic calibration)
- **Probability**: Healthy distribution (Max 0.6-0.8)
- **Performance**: 55-60% win rate, PF 1.5-2.0

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

## Supported Symbols

38 cryptocurrency pairs including:
- BTC, ETH, ADA, SOL, MATIC, AVAX, DOT, LINK
- And 30 more major cryptocurrencies

## Supported Timeframes

- 1m (1 minute) - Primary timeframe for training
- 15m (15 minutes) - Multi-timeframe features
- 1h (1 hour) - Not currently used
- 1d (1 day) - Multi-timeframe features

## Logging

The system includes comprehensive logging:
- Console output: Real-time logs in terminal
- File output: Detailed logs saved in `logs/` directory
- Log files for each component:
  - `logs/train_v3.log` - V3 model training
  - `logs/agent_backtester.log` - Backtesting operations
  - `logs/data_fetcher.log` - Data fetching
  - And more...

## Troubleshooting

### No trades in backtest?

1. **Check model version**: Make sure you're using V3 models (filename contains `v3`)
2. **Check probability**: Max prob should be > 0.50 (see training log)
3. **Lower threshold**: Try 0.10-0.15 first
4. **Match TP/SL**: Use same values as training (default 2%/1%)

See [Strategy Optimization Guide](STRATEGY_OPTIMIZATION_GUIDE.md) for detailed troubleshooting.

### Training taking too long?

```bash
# Use quick mode for testing
python train_v3.py --quick

# Only uses last 30 days of data
# Completes in 5-10 minutes
```

### Low probability output?

```bash
# Try adjusting TP/SL ratio
python train_v3.py --tp 0.015 --sl 0.01  # Smaller target

# Or use full dataset (not --quick)
python train_v3.py
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

## Modular Architecture

Each tab is implemented as a separate module for easy maintenance:
- Independent functionality per module
- Easy to extend and modify
- Centralized logging system
- Clean separation of concerns

## License

MIT License

## Warning

Cryptocurrency trading involves significant risk. This system is for educational and research purposes. Always test thoroughly before using real funds.

---

**Getting Started**: Read [V3 Quick Start Guide](V3_QUICK_START.md)  
**Questions**: Check documentation in project root  
**Issues**: Review logs in `logs/` directory