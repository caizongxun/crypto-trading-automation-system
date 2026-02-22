# Crypto Trading Automation System

An automated cryptocurrency trading system that combines neural network learning models with technical indicators for algorithmic trading.

## Features

- **Data Collection**: Fetch historical K-line data from Binance API
- **Neural Network Models**: Machine learning models for price prediction
- **Technical Indicators**: Integration with various technical analysis tools
- **Automated Trading**: Execute trades based on model predictions
- **GUI Interface**: User-friendly graphical interface for system control

## Project Structure

```
crypto-trading-automation-system/
├── main.py              # GUI main application
├── data_fetcher.py      # K-line data fetching module
├── models/              # Neural network models
├── strategies/          # Trading strategies
├── utils/               # Utility functions
├── config.py            # Configuration settings
└── requirements.txt     # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to set up:
- Binance API credentials
- HuggingFace token
- Trading parameters
- Model settings

## Usage

Run the GUI application:

```bash
python main.py
```

## Data Storage

Historical K-line data is stored in HuggingFace dataset:
- Repository: `zongowo111/v2-crypto-ohlcv-data`
- Format: Parquet files organized by symbol and timeframe
- Structure: `klines/{SYMBOL}/{BASE}_{TIMEFRAME}.parquet`

## Supported Symbols

38 cryptocurrency pairs including BTC, ETH, ADA, SOL, and more.

## Supported Timeframes

- 1m (1 minute)
- 15m (15 minutes)
- 1h (1 hour)
- 1d (1 day)

## License

MIT License

## Warning

Cryptocurrency trading involves significant risk. This system is for educational and research purposes. Always test thoroughly before using real funds.