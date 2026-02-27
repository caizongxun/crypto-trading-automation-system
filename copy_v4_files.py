#!/usr/bin/env python3
"""
V4 Setup Script
將adaptive_strategy_v4的所有檔案複製到v4_neural_kelly_strategy
"""
import shutil
from pathlib import Path

def main():
    print("Setting up V4 standalone folder...\n")
    
    # Define paths
    src_root = Path('adaptive_strategy_v4')
    dst_root = Path('v4_neural_kelly_strategy')
    
    # Files to copy
    files_to_copy = [
        # Core modules
        ('core/kelly_manager.py', 'core/kelly_manager.py'),
        ('core/neural_predictor.py', 'core/neural_predictor.py'),
        ('core/risk_controller.py', 'core/risk_controller.py'),
        ('core/feature_engineer.py', 'core/feature_engineer.py'),
        ('core/label_generator.py', 'core/label_generator.py'),
        
        # Data loaders
        ('data/hf_loader.py', 'data/hf_loader.py'),
        ('data/binance_loader.py', 'data/binance_loader.py'),
        
        # Backtest
        ('backtest/engine.py', 'backtest/engine.py'),
        
        # Scripts
        ('train.py', 'train.py'),
        ('backtest.py', 'backtest.py'),
        
        # Docs
        ('USAGE.md', 'USAGE.md'),
    ]
    
    # Copy files
    copied_count = 0
    for src_path, dst_path in files_to_copy:
        src_file = src_root / src_path
        dst_file = dst_root / dst_path
        
        if src_file.exists():
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            print(f"✓ Copied: {src_path}")
            copied_count += 1
        else:
            print(f"✗ Missing: {src_path}")
    
    print(f"\nDone! Copied {copied_count}/{len(files_to_copy)} files.")
    print("\nNext steps:")
    print("1. Update import paths in train.py and backtest.py")
    print("   Change: adaptive_strategy_v4 -> v4_neural_kelly_strategy")
    print("\n2. Test training:")
    print("   python v4_neural_kelly_strategy/train.py --symbol BTCUSDT --timeframe 15m")
    print("\n3. Test backtest:")
    print("   python v4_neural_kelly_strategy/backtest.py --model MODEL_NAME")

if __name__ == '__main__':
    main()
