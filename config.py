class Config:
    # HuggingFace settings
    HF_TOKEN = ""  # Add your HuggingFace token here
    HF_REPO_ID = "zongowo111/v2-crypto-ohlcv-data"
    
    # Binance API settings (optional, for live trading)
    BINANCE_API_KEY = ""
    BINANCE_API_SECRET = ""
    
    # Trading settings
    DEFAULT_LEVERAGE = 10
    DEFAULT_POSITION_SIZE = 0.01  # in BTC equivalent
    
    # Model settings
    MODEL_UPDATE_INTERVAL = 3600  # seconds
    PREDICTION_THRESHOLD = 0.6  # confidence threshold
    
    # Data settings
    DATA_UPDATE_INTERVAL = 60  # seconds