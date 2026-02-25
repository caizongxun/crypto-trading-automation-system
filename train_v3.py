"""
V3 Model Training Script - Optimized for High Win Rate Trading

Key Features:
- Aggressive label definitions (1.2% TP, 0.8% SL)
- 30+ optimized features
- Better probability calibration
- Independent Long/Short models
- Comprehensive validation

Author: Zong
Version: 3.0.0
Date: 2026-02-25
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, roc_auc_score, 
    precision_recall_curve, confusion_matrix
)

# Project imports
from config import Config
from utils.logger import setup_logger
from utils.feature_engineering_v3 import FeatureEngineerV3
from huggingface_hub import hf_hub_download

logger = setup_logger('train_v3', 'logs/train_v3.log')

class V3ModelTrainer:
    """
    V3 Model Trainer with Optimized Pipeline
    """
    
    def __init__(self):
        self.version = "3.0.0"
        self.fe = FeatureEngineerV3()
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Output directories
        self.models_dir = Path("models_output")
        self.models_dir.mkdir(exist_ok=True)
        
        self.reports_dir = Path("training_reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        logger.info(f"="*80)
        logger.info(f"V3 MODEL TRAINER INITIALIZED")
        logger.info(f"Version: {self.version}")
        logger.info(f"Timestamp: {self.timestamp}")
        logger.info(f"="*80)
    
    def load_data(self, symbol: str = "BTCUSDT") -> pd.DataFrame:
        """
        Load 1m kline data from HuggingFace
        """
        logger.info(f"Loading {symbol} 1m data from HuggingFace...")
        
        repo_id = Config.HF_REPO_ID
        base = symbol.replace("USDT", "")
        filename = f"{base}_1m.parquet"
        path_in_repo = f"klines/{symbol}/{filename}"
        
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=path_in_repo,
                repo_type="dataset",
                token=Config.HF_TOKEN
            )
            
            df = pd.read_parquet(local_path)
            logger.info(f"Loaded {len(df):,} rows")
            logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def train_model(self, 
                   df: pd.DataFrame, 
                   direction: str,
                   test_size: float = 0.2) -> dict:
        """
        Train a single model (long or short)
        
        Args:
            df: DataFrame with features and labels
            direction: 'long' or 'short'
            test_size: Test set ratio
        
        Returns:
            dict with model, metrics, and metadata
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING {direction.upper()} MODEL")
        logger.info(f"{'='*80}")
        
        # Get features and labels
        feature_cols = self.fe.get_feature_list()
        label_col = f'label_{direction}'
        
        logger.info(f"Features: {len(feature_cols)}")
        logger.info(f"Label: {label_col}")
        
        # Prepare data
        X = df[feature_cols].copy()
        y = df[label_col].copy()
        
        # Class distribution
        pos_rate = (y == 1).sum() / len(y) * 100
        logger.info(f"Positive samples: {(y==1).sum():,} ({pos_rate:.2f}%)")
        logger.info(f"Negative samples: {(y==0).sum():,} ({100-pos_rate:.2f}%)")
        
        # Train/test split (time-based)
        split_idx = int(len(X) * (1 - test_size))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"\nTrain set: {len(X_train):,} samples")
        logger.info(f"Test set:  {len(X_test):,} samples")
        
        # Calculate class weight
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        logger.info(f"Class weight: {pos_weight:.2f}")
        
        # ========================================
        # Train CatBoost
        # ========================================
        logger.info("\nTraining CatBoost model...")
        
        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False,
            class_weights={0: 1.0, 1: pos_weight},
            eval_metric='AUC'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=50,
            verbose=100
        )
        
        # ========================================
        # Calibrate Probabilities
        # ========================================
        logger.info("\nCalibrating probabilities...")
        
        calibrated_model = CalibratedClassifierCV(
            model, 
            method='isotonic',
            cv='prefit'
        )
        calibrated_model.fit(X_test, y_test)
        
        # ========================================
        # Evaluate
        # ========================================
        logger.info("\nEvaluating model...")
        
        # Predictions
        y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        logger.info(f"AUC: {auc:.4f}")
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        logger.info(f"\nClassification Report:")
        logger.info(f"Precision: {report['1']['precision']:.4f}")
        logger.info(f"Recall:    {report['1']['recall']:.4f}")
        logger.info(f"F1-Score:  {report['1']['f1-score']:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"TN: {cm[0,0]:,} | FP: {cm[0,1]:,}")
        logger.info(f"FN: {cm[1,0]:,} | TP: {cm[1,1]:,}")
        
        # Probability distribution
        logger.info(f"\nProbability Distribution:")
        logger.info(f"Min:  {y_pred_proba.min():.4f}")
        logger.info(f"25%:  {np.percentile(y_pred_proba, 25):.4f}")
        logger.info(f"50%:  {np.percentile(y_pred_proba, 50):.4f}")
        logger.info(f"75%:  {np.percentile(y_pred_proba, 75):.4f}")
        logger.info(f"95%:  {np.percentile(y_pred_proba, 95):.4f}")
        logger.info(f"Max:  {y_pred_proba.max():.4f}")
        
        # Precision at different thresholds
        logger.info(f"\nPrecision @ Thresholds:")
        for threshold in [0.10, 0.15, 0.20, 0.25, 0.30]:
            mask = y_pred_proba >= threshold
            if mask.sum() > 0:
                precision = (y_test[mask] == 1).sum() / mask.sum()
                coverage = mask.sum() / len(y_test) * 100
                logger.info(f"  @ {threshold:.2f}: Precision={precision:.2%}, Coverage={coverage:.2f}%")
        
        # ========================================
        # Package Results
        # ========================================
        results = {
            'model': calibrated_model,
            'feature_names': feature_cols,
            'version': 'v3',
            'direction': direction,
            'timestamp': self.timestamp,
            'metrics': {
                'auc': float(auc),
                'precision': float(report['1']['precision']),
                'recall': float(report['1']['recall']),
                'f1': float(report['1']['f1-score'])
            },
            'probability_stats': {
                'min': float(y_pred_proba.min()),
                'p25': float(np.percentile(y_pred_proba, 25)),
                'p50': float(np.percentile(y_pred_proba, 50)),
                'p75': float(np.percentile(y_pred_proba, 75)),
                'p95': float(np.percentile(y_pred_proba, 95)),
                'max': float(y_pred_proba.max())
            },
            'class_distribution': {
                'train_positive_rate': float((y_train == 1).sum() / len(y_train)),
                'test_positive_rate': float((y_test == 1).sum() / len(y_test))
            }
        }
        
        return results
    
    def save_model(self, results: dict, direction: str):
        """
        Save model with metadata
        """
        model_name = f"catboost_{direction}_v3_{self.timestamp}.pkl"
        model_path = self.models_dir / model_name
        
        # Save model package
        with open(model_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"\nModel saved: {model_path}")
        logger.info(f"Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        return model_path
    
    def save_report(self, long_results: dict, short_results: dict):
        """
        Save training report
        """
        report = {
            'version': 'v3',
            'timestamp': self.timestamp,
            'long_model': {
                'metrics': long_results['metrics'],
                'probability_stats': long_results['probability_stats'],
                'class_distribution': long_results['class_distribution']
            },
            'short_model': {
                'metrics': short_results['metrics'],
                'probability_stats': short_results['probability_stats'],
                'class_distribution': short_results['class_distribution']
            },
            'features': long_results['feature_names'],
            'feature_count': len(long_results['feature_names'])
        }
        
        report_path = self.reports_dir / f"v3_training_report_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nReport saved: {report_path}")
        return report_path
    
    def run_full_training(self):
        """
        Run complete V3 training pipeline
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING V3 FULL TRAINING PIPELINE")
        logger.info("="*80)
        
        # 1. Load data
        df_1m = self.load_data()
        
        # 2. Create features
        logger.info("\nCreating V3 features...")
        df_features = self.fe.create_features_from_1m(
            df_1m,
            label_type='both',
            tp_target=0.012,  # 1.2%
            sl_stop=0.008,    # 0.8%
            lookahead_bars=240  # 4 hours
        )
        
        # 3. Train Long model
        long_results = self.train_model(df_features, 'long')
        long_path = self.save_model(long_results, 'long')
        
        # 4. Train Short model
        short_results = self.train_model(df_features, 'short')
        short_path = self.save_model(short_results, 'short')
        
        # 5. Save report
        report_path = self.save_report(long_results, short_results)
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Long Model:  {long_path}")
        logger.info(f"Short Model: {short_path}")
        logger.info(f"Report:      {report_path}")
        logger.info("\nV3 Models ready for backtesting!")
        logger.info("="*80)

if __name__ == "__main__":
    trainer = V3ModelTrainer()
    trainer.run_full_training()