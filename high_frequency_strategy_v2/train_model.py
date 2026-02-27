"""
V2 Model Training Script
V2模型訓練腳本
"""
import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import json

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.feature_engineer import FeatureEngineer
from core.ensemble_predictor import EnsemblePredictor
from data.hf_loader import HFDataLoader

def create_labels(df: pd.DataFrame, 
                 forward_window: int = 8,
                 profit_threshold: float = 0.004,  # 0.4%
                 stop_loss: float = 0.003) -> pd.DataFrame:  # 0.3%
    """生成交易標籤"""
    df = df.copy()
    df['label'] = 0
    
    for i in range(len(df) - forward_window):
        current_price = df.iloc[i]['close']
        future_prices = df.iloc[i+1:i+forward_window+1]['close']
        
        # 最高/最低價
        max_price = future_prices.max()
        min_price = future_prices.min()
        
        # 最大潛在盈虧/虧損
        max_profit = (max_price - current_price) / current_price
        max_loss = (current_price - min_price) / current_price
        
        # 做多標籤: 可以盈利且不會觸發止損
        if max_profit >= profit_threshold and max_loss < stop_loss:
            df.loc[df.index[i], 'label'] = 1
        
        # 做空標籤
        short_profit = (current_price - min_price) / current_price
        short_loss = (max_price - current_price) / current_price
        
        if short_profit >= profit_threshold and short_loss < stop_loss:
            df.loc[df.index[i], 'label'] = -1
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Train V2 Model')
    parser.add_argument('--symbol', type=str, default='BTCUSDT')
    parser.add_argument('--timeframe', type=str, default='15m')
    parser.add_argument('--sequence_length', type=int, default=100)
    args = parser.parse_args()
    
    print("="*60)
    print(f"V2 High-Frequency Trading Model Training")
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print("="*60)
    
    # 1. 加載數據
    print("\n[1/6] 加載歷史數據...")
    loader = HFDataLoader()
    df = loader.load_klines(args.symbol, args.timeframe)
    print(f"加載完成: {len(df)} 筆數據")
    
    # 2. 特徵工程
    print("\n[2/6] 提取特徵...")
    feature_config = {
        'sequence_length': args.sequence_length,
        'use_orderbook_features': False,  # HF數據沒有訂單簿
        'use_microstructure': True,
        'use_momentum': True,
        'lookback_periods': [5, 10, 20, 50]
    }
    
    feature_engineer = FeatureEngineer(feature_config)
    df = feature_engineer.create_features(df)
    print(f"特徵提取完成: {len(df)} 筆")
    
    # 3. 生成標籤
    print("\n[3/6] 生成交易標籤...")
    df = create_labels(df)
    
    long_signals = (df['label'] == 1).sum()
    short_signals = (df['label'] == -1).sum()
    neutral = (df['label'] == 0).sum()
    
    print(f"做多標籤: {long_signals}")
    print(f"做空標籤: {short_signals}")
    print(f"中性: {neutral}")
    
    # 4. 準備數據
    print("\n[4/6] 準備訓練數據...")
    
    # 獲取特徵名稱(排除非數值列)
    exclude_cols = ['timestamp', 'label', 'bb_upper', 'bb_lower']
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
    
    print(f"使用 {len(feature_cols)} 個特徵")
    
    # 分割數據
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)
    
    df_train = df.iloc[:train_size]
    df_val = df.iloc[train_size:train_size+val_size]
    df_test = df.iloc[train_size+val_size:]
    
    # 準備特徵和標籤
    X_train = df_train[feature_cols].values
    y_train = df_train['label'].values
    X_val = df_val[feature_cols].values
    y_val = df_val['label'].values
    
    # 準備序列數據用於Transformer
    print("準備時序數據...")
    X_train_seq = feature_engineer.prepare_sequences(df_train, feature_cols)
    X_val_seq = feature_engineer.prepare_sequences(df_val, feature_cols)
    y_train_seq = df_train['label'].values[args.sequence_length:]
    y_val_seq = df_val['label'].values[args.sequence_length:]
    
    print(f"訓練集: {len(X_train)} (序列: {len(X_train_seq)})")
    print(f"驗證集: {len(X_val)} (序列: {len(X_val_seq)})")
    print(f"測試集: {len(df_test)}")
    
    # 5. 訓練模型
    print("\n[5/6] 訓練集成模型...")
    ensemble_config = {
        'use_transformer': True,
        'use_lgb': True,
        'ensemble_method': 'weighted_avg',
        'weights': {
            'transformer': 0.5,
            'lgb': 0.5
        }
    }
    
    predictor = EnsemblePredictor(ensemble_config)
    results = predictor.train(
        X_train, y_train,
        X_val, y_val,
        X_train_seq, X_val_seq
    )
    
    # 6. 保存模型
    print("\n[6/6] 保存模型...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"{args.symbol}_{args.timeframe}_v2_{timestamp}"
    model_dir = Path('models') / model_name
    
    predictor.save(model_dir)
    
    # 保存配置
    model_config = {
        'symbol': args.symbol,
        'timeframe': args.timeframe,
        'model_version': 'v2',
        'training_date': timestamp,
        'data_samples': len(df),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(df_test),
        'long_signals': int(long_signals),
        'short_signals': int(short_signals),
        'feature_count': len(feature_cols),
        'sequence_length': args.sequence_length,
        'feature_config': feature_config,
        'ensemble_config': ensemble_config
    }
    
    with open(model_dir / 'model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # 保存特徵名稱
    with open(model_dir / 'feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_cols))
    
    print(f"\n✅ 訓練完成!")
    print(f"模型保存至: {model_dir}")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
