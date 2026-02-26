"""
完整的訓練流程腦本
"""
import sys
from pathlib import Path
import json
from datetime import datetime

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.signal_detector import SignalDetector
from core.feature_engineer import FeatureEngineer
from core.ml_predictor import MLPredictor
from data.hf_loader import HFDataLoader

def run_training(symbol='BTCUSDT', timeframe='15m', config_path='configs/strategy_config.json'):
    """執行完整訓練流程"""
    
    # 1. 加載配置
    print(f"\n{'='*60}")
    print(f"開始訓練: {symbol} {timeframe}")
    print(f"{'='*60}\n")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 2. 加載數據
    print("[1/5] 加載歷史數據...")
    loader = HFDataLoader()
    df = loader.load_klines(symbol, timeframe)
    
    if df.empty:
        print("錯誤: 無法加載數據")
        return None
    
    print(f"加載完成: {len(df)} 筆數據")
    
    # 3. 信號檢測
    print("\n[2/5] 檢測反轉信號...")
    signal_detector = SignalDetector(config['signal_detection'])
    df = signal_detector.detect_signals(df)
    
    long_signals = df['signal_long'].sum()
    short_signals = df['signal_short'].sum()
    print(f"做多信號: {long_signals} | 做空信號: {short_signals}")
    
    # 4. 特徵工程
    print("\n[3/5] 生成ML特徵...")
    feature_engineer = FeatureEngineer(config['feature_engineering'])
    df = feature_engineer.create_features(df)
    df = feature_engineer.create_labels(
        df,
        forward_window=config['label_generation']['forward_window'],
        profit_threshold=config['label_generation']['profit_threshold'],
        stop_loss=config['label_generation']['stop_loss']
    )
    
    feature_cols = feature_engineer.get_feature_names()
    print(f"特徵數量: {len(feature_cols)}")
    print(f"標籤分布 - 做多: {(df['label']==1).sum()}, 做空: {(df['label']==-1).sum()}, 中性: {(df['label']==0).sum()}")
    
    # 5. 訓練ML模型
    print("\n[4/5] 訓練機器學習模型...")
    ml_predictor = MLPredictor(config['ml_model'])
    
    train_results = ml_predictor.train(
        df,
        feature_cols,
        test_size=0.2,
        oos_size=0.1
    )
    
    # 6. 保存模型
    print("\n[5/5] 保存模型...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"{symbol}_{timeframe}_v1_{timestamp}"
    model_dir = Path('models') / model_name
    
    ml_predictor.save(model_dir)
    
    # 保存配置
    model_config = {
        'symbol': symbol,
        'timeframe': timeframe,
        'training_date': timestamp,
        'data_samples': len(df),
        'long_signals': int(long_signals),
        'short_signals': int(short_signals),
        'config': config
    }
    
    with open(model_dir / 'model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"訓練完成: {model_name}")
    print(f"{'='*60}")
    
    return model_name

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='訓練反轉策略模型')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='交易對')
    parser.add_argument('--timeframe', type=str, default='15m', help='時間框架')
    parser.add_argument('--config', type=str, default='configs/strategy_config.json', help='配置文件路徑')
    
    args = parser.parse_args()
    
    run_training(args.symbol, args.timeframe, args.config)
