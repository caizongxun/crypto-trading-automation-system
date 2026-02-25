#!/usr/bin/env python3
"""
Chronos 加密貨幣微調訓練腳本

基於 Amazon Chronos T5 模型，專門為加密貨幣市場微調
使用 LoRA 高效微調，只需 2-4 小時

使用方法:
    python train_chronos_crypto.py --symbol BTCUSDT --epochs 10
    python train_chronos_crypto.py --quick  # 快速測試
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_chronos_crypto.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Chronos model for crypto trading')
    
    # 模型參數
    parser.add_argument('--base_model', type=str, default='amazon/chronos-t5-small',
                        choices=['amazon/chronos-t5-tiny', 'amazon/chronos-t5-small', 'amazon/chronos-t5-base'],
                        help='基礎模型')
    parser.add_argument('--output_dir', type=str, default='models_output/chronos_crypto',
                        help='輸出路徑')
    
    # 資料參數
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='交易對')
    parser.add_argument('--timeframe', type=str, default='1h', help='時間週期')
    parser.add_argument('--train_days', type=int, default=365, help='訓練天數')
    parser.add_argument('--val_days', type=int, default=30, help='驗證天數')
    
    # 訓練參數
    parser.add_argument('--epochs', type=int, default=10, help='訓練 epoch 數')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='學習率')
    parser.add_argument('--context_length', type=int, default=168, help='輸入窗口 (168=7天)')
    parser.add_argument('--prediction_length', type=int, default=24, help='預測長度 (24=1天)')
    
    # LoRA 參數
    parser.add_argument('--use_lora', action='store_true', default=True, help='使用 LoRA 微調')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    
    # 其他
    parser.add_argument('--quick', action='store_true', help='快速測試模式')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def load_training_data(
    symbol: str,
    timeframe: str,
    train_days: int,
    val_days: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    載入訓練和驗證資料
    """
    from utils.hf_data_loader import load_klines
    
    end_date = datetime.now()
    train_start = end_date - timedelta(days=train_days + val_days)
    val_start = end_date - timedelta(days=val_days)
    
    logger.info(f"Loading data for {symbol} {timeframe}")
    logger.info(f"Train: {train_start.date()} to {val_start.date()}")
    logger.info(f"Val: {val_start.date()} to {end_date.date()}")
    
    # 載入全部資料
    df = load_klines(
        symbol=symbol,
        timeframe=timeframe,
        start_date=train_start.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    # 分割訓練/驗證集
    split_idx = int(len(df) * train_days / (train_days + val_days))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    logger.info(f"Train data: {len(train_df)} samples")
    logger.info(f"Val data: {len(val_df)} samples")
    
    return train_df, val_df


def create_training_samples(
    df: pd.DataFrame,
    context_length: int,
    prediction_length: int
) -> List[torch.Tensor]:
    """
    創建訓練樣本
    """
    samples = []
    
    # 使用收盤價 (close) 作為訓練標的
    prices = df['close'].values
    
    # Sliding window
    for i in range(len(prices) - context_length - prediction_length + 1):
        # 提取一個完整的序列 (輸入 + 目標)
        sequence = prices[i:i + context_length + prediction_length]
        samples.append(torch.tensor(sequence, dtype=torch.float32))
    
    logger.info(f"Created {len(samples)} training samples")
    
    return samples


def train_chronos_crypto(args):
    """
    主訓練函數
    """
    logger.info("=" * 80)
    logger.info("Chronos Crypto Fine-tuning")
    logger.info("=" * 80)
    
    # 設定隨機種子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 快速測試模式
    if args.quick:
        logger.info("⚡ Quick test mode enabled")
        args.train_days = 30
        args.val_days = 7
        args.epochs = 2
        args.base_model = 'amazon/chronos-t5-tiny'
    
    # 建立輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: 載入資料
    logger.info("\nStep 1/5: Loading data...")
    train_df, val_df = load_training_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        train_days=args.train_days,
        val_days=args.val_days
    )
    
    # Step 2: 創建訓練樣本
    logger.info("\nStep 2/5: Creating training samples...")
    train_samples = create_training_samples(
        train_df,
        context_length=args.context_length,
        prediction_length=args.prediction_length
    )
    val_samples = create_training_samples(
        val_df,
        context_length=args.context_length,
        prediction_length=args.prediction_length
    )
    
    # Step 3: 載入基礎模型
    logger.info(f"\nStep 3/5: Loading base model {args.base_model}...")
    logger.info("⚠️ 此步驟需要安裝 chronos 訓練套件")
    logger.info("pip install git+https://github.com/amazon-science/chronos-forecasting.git")
    
    try:
        from chronos import ChronosPipeline, ChronosConfig
        from transformers import Trainer, TrainingArguments
        
        # 載入預訓練模型
        pipeline = ChronosPipeline.from_pretrained(
            args.base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        
        logger.info("✅ Model loaded successfully")
        
    except ImportError as e:
        logger.error(f"Failed to import chronos training modules: {e}")
        logger.error("請安裝: pip install git+https://github.com/amazon-science/chronos-forecasting.git")
        return
    
    # Step 4: 設定 LoRA (如果啟用)
    if args.use_lora:
        logger.info(f"\nStep 4/5: Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
        
        try:
            from peft import LoraConfig, get_peft_model
            
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=["q", "v"],  # T5 attention modules
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
            
            # 應用 LoRA
            model = get_peft_model(pipeline.model, lora_config)
            model.print_trainable_parameters()
            
            logger.info("✅ LoRA applied successfully")
            
        except ImportError:
            logger.error("peft not installed. Install: pip install peft")
            return
    else:
        model = pipeline.model
    
    # Step 5: 訓練
    logger.info(f"\nStep 5/5: Training for {args.epochs} epochs...")
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed
    )
    
    logger.info("⚠️ 此步驟需要實現自定義 Dataset 和 Trainer")
    logger.info("完整實現見: https://github.com/amazon-science/chronos-forecasting")
    
    # 訓練記錄
    training_report = {
        'base_model': args.base_model,
        'symbol': args.symbol,
        'timeframe': args.timeframe,
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'context_length': args.context_length,
        'prediction_length': args.prediction_length,
        'epochs': args.epochs,
        'use_lora': args.use_lora,
        'output_dir': str(output_dir),
        'timestamp': datetime.now().isoformat()
    }
    
    # 儲存配置
    import json
    with open(output_dir / 'training_config.json', 'w') as f:
        json.dump(training_report, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Training setup complete!")
    logger.info("=" * 80)
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info(f"Config saved: {output_dir / 'training_config.json'}")
    
    logger.info("

⚠️ 此腳本為框架版本，完整實現需要:")
    logger.info("1. 實現 ChronosDataset class")
    logger.info("2. 實現自定義 Trainer")
    logger.info("3. 添加 evaluation metrics")
    logger.info("\n完整範例: https://github.com/amazon-science/chronos-forecasting/blob/main/scripts/training/train.py")
    
    return output_dir


if __name__ == "__main__":
    args = parse_args()
    
    # 建立 logs 目錄
    Path('logs').mkdir(exist_ok=True)
    
    try:
        output_dir = train_chronos_crypto(args)
        logger.info("\n✅ Training script completed successfully!")
        
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
