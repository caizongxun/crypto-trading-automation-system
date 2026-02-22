# Crypto Trading Automation System

An automated cryptocurrency trading system that combines neural network learning models with technical indicators for algorithmic trading.

## Features

- **Data Collection**: Fetch historical K-line data from Binance API
- **Neural Network Models**: Machine learning models for price prediction
- **Technical Indicators**: Integration with various technical analysis tools
- **Automated Trading**: Execute trades based on model predictions
- **GUI Interface**: User-friendly graphical interface for system control
- **Comprehensive Logging**: Detailed logging system for debugging and monitoring

## Project Structure

```
crypto-trading-automation-system/
├── main.py              # GUI main application
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
├── tabs/                # Modular tab components
│   ├── __init__.py
│   ├── data_fetcher_tab.py      # K-line data fetching module
│   ├── model_training_tab.py    # Model training module
│   ├── backtesting_tab.py       # Backtesting module
│   └── auto_trading_tab.py      # Auto trading module
├── utils/               # Utility functions
│   ├── __init__.py
│   └── logger.py        # Logging configuration
├── models/              # Neural network models (future)
├── strategies/          # Trading strategies (future)
└── logs/                # Log files directory
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to set up:
- HuggingFace token (required for data storage)
- Binance API credentials (optional, for live trading)
- Trading parameters
- Model settings

## Usage

Run the GUI application:

```bash
python main.py
```

Or using Streamlit directly:

```bash
streamlit run main.py
```

## Logging

The system includes comprehensive logging:
- Console output: Real-time logs in terminal
- File output: Detailed logs saved in `logs/` directory
- Separate log files for each module:
  - `logs/main.log` - Main application logs
  - `logs/data_fetcher.log` - Data fetching operations
  - `logs/model_training.log` - Model training logs
  - `logs/backtesting.log` - Backtesting logs
  - `logs/auto_trading.log` - Trading operations logs

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

- 1m (1 minute)
- 15m (15 minutes)
- 1h (1 hour)
- 1d (1 day)

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