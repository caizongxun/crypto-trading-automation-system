"""
V4 Training Script
LSTM + Kelly 策略訓練腥本
"""
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from adaptive_strategy_v4.data.hf_loader import HFDataLoader
from adaptive_strategy_v4.core.feature_engineer import FeatureEngineer
from adaptive_strategy_v4.core.label_generator import LabelGenerator
from adaptive_strategy_v4.core.neural_predictor import NeuralPredictor

def train_v4_model(symbol: str, timeframe: str, config: dict):
    """
    訓練V4模型
    
    Args:
        symbol: 交易對 (e.g., 'BTCUSDT')
        timeframe: 時間框架 (e.g., '15m')
        config: 訓練配置
    """
    print("="*60)
    print(f"V4 Training - {symbol} {timeframe}")
    print("="*60)
    
    # 1. 加載數據
    print("\n[1/6] 加載數據...")
    loader = HFDataLoader()
    df = loader.load_klines(symbol, timeframe)
    print(f"[OK] 加載: {len(df)} 筆")
    
    # 2. 特徵工程
    print("\n[2/6] 特徵工程...")
    feature_config = config.get('feature_config', {})
    feature_engineer = FeatureEngineer(feature_config)
    df = feature_engineer.create_features(df)
    print(f"[OK] 特徵數: {len(df.columns)}")
    
    # 3. 生成標籤
    print("\n[3/6] 生成標籤...")
    label_config = config.get('label_config', {
        'forward_window': 8,
        'atr_profit_multiplier': 0.7,
        'atr_loss_multiplier': 1.5,
        'min_volume_ratio': 0.7,
        'min_trend_strength': 0.15,
        'max_atr_ratio': 0.08
    })
    label_generator = LabelGenerator(label_config)
    df = label_generator.generate_labels(df)
    
    # 4. 準備訓練數據
    print("\n[4/6] 準備訓練數據...")
    exclude_cols = ['timestamp', 'label', 'target_win_rate', 'target_payoff',
                   'open', 'high', 'low', 'close', 'volume']
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols 
                   and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
    
    # 處理無限值和NaN
    for col in feature_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols + ['label'])
    
    # 劃分訓練/驗證/測試集
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    df_train = df.iloc[:train_size]
    df_val = df.iloc[train_size:train_size+val_size]
    df_test = df.iloc[train_size+val_size:]
    
    X_train = df_train[feature_cols].values
    y_train = df_train['label'].values
    X_val = df_val[feature_cols].values
    y_val = df_val['label'].values
    
    print(f"[OK] 訓練集: {len(X_train)}")
    print(f"[OK] 驗證集: {len(X_val)}")
    print(f"[OK] 測試集: {len(df_test)}")
    print(f"[OK] 特徵數: {len(feature_cols)}")
    
    # 5. 訓練LSTM模型
    print("\n[5/6] 訓練LSTM模型...")
    predictor_config = {
        'input_size': len(feature_cols),
        'hidden_size': config.get('hidden_size', 128),
        'num_layers': config.get('num_layers', 2),
        'dropout': config.get('dropout', 0.2),
        'sequence_length': config.get('sequence_length', 20)
    }
    
    predictor = NeuralPredictor(predictor_config)
    train_results = predictor.train(
        X_train, y_train,
        X_val, y_val,
        epochs=config.get('epochs', 50),
        batch_size=config.get('batch_size', 64),
        learning_rate=config.get('learning_rate', 0.001)
    )
    
    print(f"\n[OK] 訓練完成")
    print(f"  - Final Accuracy: {train_results['final_accuracy']:.3f}")
    print(f"  - Best Val Loss: {train_results['best_val_loss']:.4f}")
    
    # 6. 保存模型
    print("\n[6/6] 保存模型...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"{symbol}_{timeframe}_v4_{timestamp}"
    model_dir = project_root / 'models' / model_name
    
    predictor.save(model_dir)
    
    # 保存完整配置
    model_config = {
        'symbol': symbol,
        'timeframe': timeframe,
        'model_version': 'v4',
        'training_date': timestamp,
        'data_samples': len(df),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(df_test),
        'feature_count': len(feature_cols),
        'feature_names': feature_cols,
        'label_config': label_config,
        'predictor_config': predictor_config,
        'train_results': {
            'final_accuracy': float(train_results['final_accuracy']),
            'best_val_loss': float(train_results['best_val_loss'])
        }
    }
    
    with open(model_dir / 'model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"\n[V4 完成] {model_name}")
    print(f"[Path] {model_dir}")
    print("="*60)
    
    return model_name

def main():
    parser = argparse.ArgumentParser(description='V4 Model Training')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                       help='Trading pair (default: BTCUSDT)')
    parser.add_argument('--timeframe', type=str, default='15m',
                       help='Timeframe (default: 15m)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='LSTM hidden size (default: 128)')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='LSTM layers (default: 2)')
    
    args = parser.parse_args()
    
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': 0.2,
        'sequence_length': 20,
        'learning_rate': 0.001
    }
    
    train_v4_model(args.symbol, args.timeframe, config)

if __name__ == '__main__':
    main()
