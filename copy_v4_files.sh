#!/bin/bash
# Script to copy V4 files to standalone folder

echo "Copying V4 files to v4_neural_kelly_strategy/"

# Copy core modules
cp adaptive_strategy_v4/core/kelly_manager.py v4_neural_kelly_strategy/core/
cp adaptive_strategy_v4/core/neural_predictor.py v4_neural_kelly_strategy/core/
cp adaptive_strategy_v4/core/risk_controller.py v4_neural_kelly_strategy/core/
cp adaptive_strategy_v4/core/feature_engineer.py v4_neural_kelly_strategy/core/
cp adaptive_strategy_v4/core/label_generator.py v4_neural_kelly_strategy/core/

# Copy data loaders
cp adaptive_strategy_v4/data/hf_loader.py v4_neural_kelly_strategy/data/
cp adaptive_strategy_v4/data/binance_loader.py v4_neural_kelly_strategy/data/

# Copy backtest engine
cp adaptive_strategy_v4/backtest/engine.py v4_neural_kelly_strategy/backtest/

# Copy scripts
cp adaptive_strategy_v4/train.py v4_neural_kelly_strategy/
cp adaptive_strategy_v4/backtest.py v4_neural_kelly_strategy/

# Copy documentation
cp adaptive_strategy_v4/USAGE.md v4_neural_kelly_strategy/

echo "Done! All V4 files copied."
echo ""
echo "Next steps:"
echo "1. Update import paths in copied files"
echo "2. Test training: python v4_neural_kelly_strategy/train.py --symbol BTCUSDT --timeframe 15m"
echo "3. Test backtest: python v4_neural_kelly_strategy/backtest.py --model MODEL_NAME"
