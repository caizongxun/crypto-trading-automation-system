import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
import optuna
from huggingface_hub import hf_hub_download
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))
from config import Config
from utils.logger import setup_logger
from utils.enhanced_feature_engineering import EnhancedFeatureEngineer

logger = setup_logger('train_enhanced', 'logs/train_enhanced.log')

class EnhancedModelTrainer:
    """
    增強版模型訓練器
    
    **整合所有優化**:
    - 增強特徵工程 (70+ features)
    - 集成學習 (CatBoost + XGBoost + LightGBM)
    - 超參數調優 (Optuna)
    - Walk-Forward 驗證
    - 動態樣本權重
    """
    
    def __init__(self, use_ensemble: bool = True, use_optuna: bool = False):
        self.use_ensemble = use_ensemble
        self.use_optuna = use_optuna
        self.feature_engineer = EnhancedFeatureEngineer()
        
        # 輸出目錄
        self.output_dir = Path("models_output_enhanced")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("="*80)
        logger.info("ENHANCED MODEL TRAINER INITIALIZED")
        logger.info("="*80)
        logger.info(f"Ensemble Learning: {use_ensemble}")
        logger.info(f"Optuna Optimization: {use_optuna}")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info("="*80)
    
    def load_data(self) -> pd.DataFrame:
        """
        載入 1 分鐘 K 線數據
        """
        logger.info("Loading 1m kline data from HuggingFace...")
        
        try:
            local_path = hf_hub_download(
                repo_id=Config.HF_REPO_ID,
                filename="klines/BTCUSDT/BTC_1m.parquet",
                repo_type="dataset",
                token=Config.HF_TOKEN
            )
            
            df = pd.read_parquet(local_path)
            
            if 'open_time' in df.columns:
                df['open_time'] = pd.to_datetime(df['open_time'])
                df.set_index('open_time', inplace=True)
            
            logger.info(f"  ✅ Loaded {len(df):,} records")
            logger.info(f"  Date range: {df.index[0]} to {df.index[-1]}")
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def compute_sample_weights(self, y: pd.Series, feature_df: pd.DataFrame = None) -> np.ndarray:
        """
        計算動態樣本權重
        """
        logger.info("Computing dynamic sample weights...")
        
        weights = np.ones(len(y))
        
        # 1. 類別平衡
        class_counts = np.bincount(y)
        class_weight = len(y) / (2 * class_counts)
        weights = class_weight[y]
        
        # 2. 時間衰減 (近期數據更重要)
        time_decay = np.exp(np.linspace(-1, 0, len(y)))
        weights *= time_decay
        
        # 3. 波動率加權 (高波動期更重要)
        if feature_df is not None and 'atr_pct' in feature_df.columns:
            volatility_weight = 1 + feature_df['atr_pct'].values / feature_df['atr_pct'].mean()
            weights *= volatility_weight
        
        weights = weights / weights.mean()
        
        logger.info(f"  Weight range: {weights.min():.2f} - {weights.max():.2f}")
        logger.info(f"  Weight mean: {weights.mean():.2f}")
        
        return weights
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, direction: str):
        """
        使用 Optuna 優化超參數
        """
        logger.info(f"Optimizing hyperparameters for {direction.upper()}...")
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 300, 1000),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'random_strength': trial.suggest_float('random_strength', 0, 10),
            }
            
            model = CatBoostClassifier(
                **params,
                random_seed=42,
                verbose=False,
                thread_count=-1
            )
            
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
            
            probs = model.predict_proba(X_val)[:, 1]
            
            # 自定義指標: 優化 >0.16 區間的精確度
            mask = probs >= 0.16
            if mask.sum() > 0:
                precision = y_val[mask].mean()
                recall = mask.sum() / len(y_val)
                f_score = 2 * precision * recall / (precision + recall + 1e-8)
                return f_score
            return 0
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        logger.info(f"  Best F-score: {study.best_value:.4f}")
        logger.info(f"  Best params: {study.best_params}")
        
        return study.best_params
    
    def train_single_model(
        self,
        X_train, y_train,
        X_val, y_val,
        direction: str,
        model_type: str = 'catboost',
        params: dict = None
    ):
        """
        訓練單一模型
        """
        logger.info(f"Training {model_type.upper()} for {direction.upper()}...")
        
        # 計算樣本權重
        sample_weights = self.compute_sample_weights(y_train)
        
        if model_type == 'catboost':
            if params is None:
                params = {
                    'iterations': 500,
                    'depth': 6,
                    'learning_rate': 0.05,
                    'l2_leaf_reg': 3,
                    'border_count': 128,
                    'bagging_temperature': 0.5,
                    'random_strength': 1.0,
                }
            
            model = CatBoostClassifier(
                **params,
                random_seed=42,
                verbose=False,
                thread_count=-1,
                eval_metric='AUC'
            )
            
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                sample_weight=sample_weights,
                early_stopping_rounds=50,
                verbose=False
            )
        
        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='auc'
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                sample_weight=sample_weights,
                early_stopping_rounds=50,
                verbose=False
            )
        
        elif model_type == 'lightgbm':
            model = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                sample_weight=sample_weights,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
        
        # 驗證
        probs = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, probs)
        
        mask = probs >= 0.16
        if mask.sum() > 0:
            precision = y_val[mask].mean()
            logger.info(f"  AUC: {auc:.4f}, Precision@0.16: {precision:.4f}, Samples@0.16: {mask.sum()}")
        else:
            logger.info(f"  AUC: {auc:.4f}, No samples >= 0.16")
        
        return model
    
    def train_ensemble(
        self,
        X_train, y_train,
        X_val, y_val,
        direction: str,
        best_params: dict = None
    ):
        """
        訓練集成模型
        """
        logger.info(f"Training ENSEMBLE for {direction.upper()}...")
        
        # 1. 訓練三個基礎模型
        model_cat = self.train_single_model(
            X_train, y_train, X_val, y_val,
            direction, 'catboost', best_params
        )
        
        model_xgb = self.train_single_model(
            X_train, y_train, X_val, y_val,
            direction, 'xgboost'
        )
        
        model_lgb = self.train_single_model(
            X_train, y_train, X_val, y_val,
            direction, 'lightgbm'
        )
        
        # 2. Voting Ensemble
        logger.info("Creating Voting Ensemble...")
        ensemble = VotingClassifier(
            estimators=[
                ('catboost', model_cat),
                ('xgboost', model_xgb),
                ('lightgbm', model_lgb)
            ],
            voting='soft',
            weights=[0.5, 0.25, 0.25]  # CatBoost 權重較高
        )
        
        # Fit ensemble (已經訓練過的模型不需要重新 fit)
        # ensemble.fit(X_train, y_train)  # 不需要
        
        # 3. 驗證 Ensemble
        probs_ensemble = ensemble.predict_proba(X_val)[:, 1]
        auc_ensemble = roc_auc_score(y_val, probs_ensemble)
        
        mask = probs_ensemble >= 0.16
        if mask.sum() > 0:
            precision = y_val[mask].mean()
            logger.info(f"  Ensemble AUC: {auc_ensemble:.4f}, Precision@0.16: {precision:.4f}")
        
        return ensemble
    
    def walk_forward_validation(
        self,
        df_features: pd.DataFrame,
        direction: str,
        n_splits: int = 5
    ):
        """
        Walk-Forward 驗證
        """
        logger.info("="*80)
        logger.info(f"WALK-FORWARD VALIDATION - {direction.upper()}")
        logger.info("="*80)
        
        label_col = f'label_{direction}'
        feature_cols = [col for col in df_features.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 
                                     'label_long', 'label_short']]
        
        X = df_features[feature_cols]
        y = df_features[label_col]
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"\nFold {fold+1}/{n_splits}")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # 分出驗證集
            val_size = int(len(X_train) * 0.2)
            X_train_sub, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
            y_train_sub, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]
            
            # 訓練
            if self.use_ensemble:
                model = self.train_ensemble(
                    X_train_sub, y_train_sub, X_val, y_val, direction
                )
            else:
                model = self.train_single_model(
                    X_train_sub, y_train_sub, X_val, y_val, direction
                )
                
                # Calibration
                model = CalibratedClassifierCV(model, method='isotonic', cv=3)
                model.fit(X_train_sub, y_train_sub)
            
            # 測試
            probs = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
            
            mask = probs >= 0.16
            precision = y_test[mask].mean() if mask.sum() > 0 else 0
            samples = mask.sum()
            
            results.append({
                'fold': fold + 1,
                'auc': auc,
                'precision_016': precision,
                'samples_016': samples,
                'test_size': len(y_test)
            })
            
            logger.info(f"  AUC: {auc:.4f}, Precision@0.16: {precision:.4f}, Samples: {samples}")
        
        # 統計
        results_df = pd.DataFrame(results)
        
        logger.info("="*80)
        logger.info("WALK-FORWARD VALIDATION RESULTS")
        logger.info("="*80)
        logger.info(f"Average AUC: {results_df['auc'].mean():.4f} ± {results_df['auc'].std():.4f}")
        logger.info(f"Average Precision@0.16: {results_df['precision_016'].mean():.4f}")
        logger.info(f"Total Samples@0.16: {results_df['samples_016'].sum()}")
        logger.info("="*80)
        
        return results_df
    
    def train_final_models(self):
        """
        訓練最終模型
        """
        logger.info("="*80)
        logger.info("TRAINING FINAL ENHANCED MODELS")
        logger.info("="*80)
        
        # 1. 載入數據
        df_1m = self.load_data()
        
        # 2. 生成增強特徵
        logger.info("\nGenerating enhanced features...")
        df_features = self.feature_engineer.create_enhanced_features(
            df_1m,
            use_adaptive_labels=True,
            label_type='both'
        )
        
        # 3. 切分數據
        split_idx = int(len(df_features) * 0.8)
        df_train = df_features.iloc[:split_idx]
        df_test = df_features.iloc[split_idx:]
        
        logger.info(f"\nTrain set: {len(df_train):,} samples")
        logger.info(f"Test set: {len(df_test):,} samples")
        
        # 4. 訓練 Long 模型
        logger.info("\n" + "="*80)
        logger.info("TRAINING LONG ORACLE")
        logger.info("="*80)
        
        feature_cols = [col for col in df_features.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 
                                     'label_long', 'label_short']]
        
        X_train = df_train[feature_cols]
        y_train_long = df_train['label_long']
        X_test = df_test[feature_cols]
        y_test_long = df_test['label_long']
        
        # 分出驗證集
        val_size = int(len(X_train) * 0.2)
        X_train_sub = X_train.iloc[:-val_size]
        y_train_sub_long = y_train_long.iloc[:-val_size]
        X_val = X_train.iloc[-val_size:]
        y_val_long = y_train_long.iloc[-val_size:]
        
        # Optuna 優化
        best_params_long = None
        if self.use_optuna:
            best_params_long = self.optimize_hyperparameters(
                X_train_sub, y_train_sub_long, X_val, y_val_long, 'long'
            )
        
        # 訓練
        if self.use_ensemble:
            model_long = self.train_ensemble(
                X_train_sub, y_train_sub_long, X_val, y_val_long, 'long', best_params_long
            )
        else:
            model_long = self.train_single_model(
                X_train_sub, y_train_sub_long, X_val, y_val_long, 'long', 'catboost', best_params_long
            )
            
            # Calibration
            model_long = CalibratedClassifierCV(model_long, method='isotonic', cv=3)
            model_long.fit(X_train, y_train_long)
        
        # 保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_type = 'ensemble' if self.use_ensemble else 'catboost'
        long_path = self.output_dir / f"{model_type}_long_enhanced_{timestamp}.pkl"
        joblib.dump(model_long, long_path)
        logger.info(f"\n✅ Long model saved: {long_path}")
        
        # 5. 訓練 Short 模型
        logger.info("\n" + "="*80)
        logger.info("TRAINING SHORT ORACLE")
        logger.info("="*80)
        
        y_train_short = df_train['label_short']
        y_test_short = df_test['label_short']
        y_train_sub_short = y_train_short.iloc[:-val_size]
        y_val_short = y_train_short.iloc[-val_size:]
        
        # Optuna 優化
        best_params_short = None
        if self.use_optuna:
            best_params_short = self.optimize_hyperparameters(
                X_train_sub, y_train_sub_short, X_val, y_val_short, 'short'
            )
        
        # 訓練
        if self.use_ensemble:
            model_short = self.train_ensemble(
                X_train_sub, y_train_sub_short, X_val, y_val_short, 'short', best_params_short
            )
        else:
            model_short = self.train_single_model(
                X_train_sub, y_train_sub_short, X_val, y_val_short, 'short', 'catboost', best_params_short
            )
            
            # Calibration
            model_short = CalibratedClassifierCV(model_short, method='isotonic', cv=3)
            model_short.fit(X_train, y_train_short)
        
        # 保存
        short_path = self.output_dir / f"{model_type}_short_enhanced_{timestamp}.pkl"
        joblib.dump(model_short, short_path)
        logger.info(f"\n✅ Short model saved: {short_path}")
        
        logger.info("\n" + "="*80)
        logger.info("ENHANCED MODELS TRAINING COMPLETED")
        logger.info("="*80)
        logger.info(f"Total features: {len(feature_cols)}")
        logger.info(f"Long model: {long_path}")
        logger.info(f"Short model: {short_path}")
        logger.info("="*80)
        
        return model_long, model_short, long_path, short_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Model Training')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble learning')
    parser.add_argument('--optuna', action='store_true', help='Use Optuna optimization')
    parser.add_argument('--walk-forward', action='store_true', help='Run walk-forward validation')
    
    args = parser.parse_args()
    
    trainer = EnhancedModelTrainer(use_ensemble=args.ensemble, use_optuna=args.optuna)
    
    if args.walk_forward:
        # Walk-Forward 驗證
        df_1m = trainer.load_data()
        df_features = trainer.feature_engineer.create_enhanced_features(df_1m)
        
        trainer.walk_forward_validation(df_features, 'long', n_splits=5)
        trainer.walk_forward_validation(df_features, 'short', n_splits=5)
    else:
        # 訓練最終模型
        trainer.train_final_models()