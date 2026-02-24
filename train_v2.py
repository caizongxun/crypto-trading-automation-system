import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV

from catboost import CatBoostClassifier
import xgboost as xgb

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not installed. Skipping hyperparameter optimization.")
    print("Install: pip install optuna")

from huggingface_hub import hf_hub_download

sys.path.append(str(Path(__file__).parent))
from config import Config
from utils.logger import setup_logger
from utils.feature_engineering_v2 import FeatureEngineerV2
from utils.agent_backtester import BidirectionalAgentBacktester

logger = setup_logger('train_v2', 'logs/train_v2.log')

def save_model_with_metadata(model, feature_names: list, model_type: str, 
                             version: str, output_dir: Path) -> Path:
    """
    儲存模型時包含完整 metadata
    
    Args:
        model: 訓練好的模型
        feature_names: 特徵名稱列表
        model_type: 'long' or 'short'
        version: 版本標記 (e.g., 'v2')
        output_dir: 輸出目錄
    
    Returns:
        Path: 儲存的檔案路徑
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 包裝成 dict
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'version': version,
        'timestamp': timestamp,
        'metadata': {
            'n_features': len(feature_names),
            'model_type': model_type,
            'training_date': datetime.now().isoformat()
        }
    }
    
    filename = output_dir / f"catboost_{model_type}_{version}_{timestamp}.pkl"
    
    joblib.dump(model_data, filename)
    
    logger.info(f"✅ Model saved with metadata: {filename}")
    logger.info(f"   Version: {version}")
    logger.info(f"   Features ({len(feature_names)}): {feature_names[:5]}... (showing first 5)")
    
    return filename

class AdvancedTrainer:
    """
    進階訓練器 - 整合所有優化方案
    """
    
    def __init__(self, 
                 enable_hyperopt: bool = True,
                 enable_ensemble: bool = True,
                 enable_walk_forward: bool = True,
                 n_trials: int = 50):
        
        logger.info("="*80)
        logger.info("ADVANCED TRAINER INITIALIZATION")
        logger.info("="*80)
        
        self.enable_hyperopt = enable_hyperopt and OPTUNA_AVAILABLE
        self.enable_ensemble = enable_ensemble
        self.enable_walk_forward = enable_walk_forward
        self.n_trials = n_trials
        
        self.feature_engineer = FeatureEngineerV2(
            enable_advanced_features=True,
            enable_ml_features=True
        )
        
        logger.info(f"Hyperparameter Optimization: {self.enable_hyperopt}")
        logger.info(f"Ensemble Learning: {self.enable_ensemble}")
        logger.info(f"Walk-Forward Validation: {self.enable_walk_forward}")
    
    def load_klines(self, symbol: str = "BTCUSDT", timeframe: str = "1m") -> pd.DataFrame:
        """
        從 HuggingFace 載入 K線數據
        """
        logger.info(f"Loading {symbol} {timeframe} from HuggingFace...")
        
        try:
            repo_id = Config.HF_REPO_ID
            base = symbol.replace("USDT", "")
            filename = f"{base}_{timeframe}.parquet"
            path_in_repo = f"klines/{symbol}/{filename}"
            
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=path_in_repo,
                repo_type="dataset",
                token=Config.HF_TOKEN
            )
            
            df = pd.read_parquet(local_path)
            logger.info(f"Loaded {len(df):,} records")
            return df
        
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def compute_sample_weights(self, y: np.ndarray, 
                              probs: np.ndarray = None) -> np.ndarray:
        """
        計算樣本權重
        """
        weights = np.ones(len(y))
        
        # 1. 類別平衡
        unique, counts = np.unique(y, return_counts=True)
        class_weight_dict = {}
        for cls, count in zip(unique, counts):
            class_weight_dict[cls] = len(y) / (len(unique) * count)
        
        for i, label in enumerate(y):
            weights[i] *= class_weight_dict[label]
        
        # 2. 時間衰減 (近期數據更重要)
        time_decay = np.exp(np.linspace(-1, 0, len(y)))
        weights *= time_decay
        
        # 3. 困難樣本加權
        if probs is not None:
            pred_labels = (probs > 0.5).astype(int)
            errors = np.abs(pred_labels - y)
            weights *= (1 + errors * 0.5)
        
        # 標準化
        weights = weights / weights.mean()
        
        logger.info(f"Sample weights: min={weights.min():.2f}, max={weights.max():.2f}, mean={weights.mean():.2f}")
        
        return weights
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val,
                                direction: str = 'long') -> dict:
        """
        使用 Optuna 優化超參數
        """
        if not self.enable_hyperopt:
            logger.info("Using default hyperparameters")
            return {
                'iterations': 500,
                'depth': 6,
                'learning_rate': 0.05,
                'l2_leaf_reg': 3,
                'random_strength': 1,
                'bagging_temperature': 1,
                'border_count': 128
            }
        
        logger.info(f"Optimizing hyperparameters for {direction} oracle...")
        logger.info(f"Running {self.n_trials} trials...")
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 300, 1000),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_strength': trial.suggest_float('random_strength', 0, 2),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_state': 42,
                'verbose': False
            }
            
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            
            probs = model.predict_proba(X_val)[:, 1]
            
            # 自定義指標: 優化 >0.16 區間的精確度 × 召回率
            mask = probs >= 0.16
            if mask.sum() > 0:
                precision = y_val[mask].mean()
                recall = mask.sum() / len(y_val)
                f_score = 2 * precision * recall / (precision + recall + 1e-8)
                return f_score
            return 0
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['verbose'] = False
        
        logger.info(f"Best hyperparameters found:")
        for k, v in best_params.items():
            logger.info(f"  {k}: {v}")
        logger.info(f"Best F-score: {study.best_value:.4f}")
        
        return best_params
    
    def train_single_model(self, X_train, y_train, X_val, y_val,
                          params: dict, direction: str = 'long'):
        """
        訓練單一模型
        """
        logger.info(f"Training {direction} oracle...")
        
        # 計算樣本權重
        sample_weights = self.compute_sample_weights(y_train)
        
        # 訓練 CatBoost
        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=100
        )
        
        # 校準
        calibrated_model = CalibratedClassifierCV(
            model, method='isotonic', cv=3
        )
        calibrated_model.fit(X_train, y_train)
        
        return calibrated_model
    
    def train_ensemble(self, X_train, y_train, X_val, y_val,
                      catboost_params: dict, direction: str = 'long'):
        """
        訓練集成模型
        """
        if not self.enable_ensemble:
            return self.train_single_model(X_train, y_train, X_val, y_val,
                                          catboost_params, direction)
        
        logger.info(f"Training ensemble for {direction} oracle...")
        
        # 樣本權重
        sample_weights = self.compute_sample_weights(y_train)
        
        # 1. CatBoost
        logger.info("Training CatBoost...")
        model_cat = CatBoostClassifier(**catboost_params)
        model_cat.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False
        )
        
        # 2. XGBoost
        logger.info("Training XGBoost...")
        xgb_params = {
            'n_estimators': catboost_params.get('iterations', 500),
            'max_depth': catboost_params.get('depth', 6),
            'learning_rate': catboost_params.get('learning_rate', 0.05),
            'reg_lambda': catboost_params.get('l2_leaf_reg', 3),
            'random_state': 42,
            'verbosity': 0
        }
        
        model_xgb = xgb.XGBClassifier(**xgb_params)
        model_xgb.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # 3. Voting Ensemble
        logger.info("Creating voting ensemble...")
        ensemble = VotingClassifier(
            estimators=[
                ('catboost', model_cat),
                ('xgboost', model_xgb)
            ],
            voting='soft',
            weights=[0.6, 0.4]  # CatBoost 權重較高
        )
        
        ensemble.fit(X_train, y_train)
        
        # 4. Calibration
        logger.info("Calibrating ensemble...")
        calibrated_ensemble = CalibratedClassifierCV(
            ensemble, method='isotonic', cv=3
        )
        calibrated_ensemble.fit(X_train, y_train)
        
        return calibrated_ensemble
    
    def evaluate_model(self, model, X_test, y_test, direction: str = 'long'):
        """
        評估模型性能
        """
        probs = model.predict_proba(X_test)[:, 1]
        
        # AUC
        auc = roc_auc_score(y_test, probs)
        
        # 不同閾值的精確度
        thresholds = [0.10, 0.15, 0.16, 0.18, 0.20, 0.22, 0.25]
        metrics = {}
        
        for threshold in thresholds:
            mask = probs >= threshold
            if mask.sum() > 0:
                precision = y_test[mask].mean()
                recall = mask.sum() / len(y_test)
                metrics[threshold] = {
                    'precision': precision,
                    'recall': recall * 100,
                    'samples': mask.sum()
                }
        
        logger.info(f"\n{direction.upper()} Oracle Evaluation:")
        logger.info(f"AUC: {auc:.4f}")
        logger.info(f"\nThreshold Analysis:")
        for th, m in metrics.items():
            logger.info(
                f"  {th:.2f}: Precision={m['precision']*100:5.1f}%, "
                f"Recall={m['recall']:5.2f}%, Samples={m['samples']:,}"
            )
        
        return {'auc': auc, 'threshold_metrics': metrics}
    
    def walk_forward_validation(self, df_features, n_splits: int = 5):
        """
        Walk-Forward 驗證
        """
        if not self.enable_walk_forward:
            logger.info("Walk-forward validation disabled")
            return None
        
        logger.info("="*80)
        logger.info("WALK-FORWARD VALIDATION")
        logger.info("="*80)
        
        feature_cols = self.feature_engineer.get_feature_list()
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df_features)):
            logger.info(f"\nFold {fold+1}/{n_splits}")
            
            df_train = df_features.iloc[train_idx]
            df_test = df_features.iloc[test_idx]
            
            X_train = df_train[feature_cols].fillna(0).values
            y_train_long = df_train['label_long_adaptive'].values
            X_test = df_test[feature_cols].fillna(0).values
            y_test_long = df_test['label_long_adaptive'].values
            
            # 訓練簡單模型
            params = {
                'iterations': 300,
                'depth': 6,
                'learning_rate': 0.05,
                'random_state': 42,
                'verbose': False
            }
            
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train_long, verbose=False)
            
            # 評估
            probs = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test_long, probs)
            
            mask = probs >= 0.16
            precision = y_test_long[mask].mean() if mask.sum() > 0 else 0
            
            logger.info(f"  AUC: {auc:.4f}, Precision@0.16: {precision*100:.1f}%")
            
            results.append({
                'fold': fold + 1,
                'auc': auc,
                'precision_016': precision
            })
        
        results_df = pd.DataFrame(results)
        logger.info("\n" + "="*80)
        logger.info("WALK-FORWARD SUMMARY")
        logger.info("="*80)
        logger.info(f"Average AUC: {results_df['auc'].mean():.4f} ± {results_df['auc'].std():.4f}")
        logger.info(f"Average Precision@0.16: {results_df['precision_016'].mean()*100:.1f}%")
        logger.info("="*80)
        
        return results_df
    
    def run(self):
        """
        執行完整訓練流程
        """
        logger.info("="*80)
        logger.info("STARTING ADVANCED TRAINING PIPELINE")
        logger.info("="*80)
        
        # 1. 載入數據
        df_1m = self.load_klines("BTCUSDT", "1m")
        
        # 2. 生成特徵
        logger.info("\nGenerating features...")
        df_features = self.feature_engineer.create_features_from_1m(
            df_1m,
            use_adaptive_labels=True,
            label_type='both'
        )
        
        # 3. 切分數據
        split_idx = int(len(df_features) * 0.7)
        val_idx = int(len(df_features) * 0.8)
        
        df_train = df_features.iloc[:split_idx]
        df_val = df_features.iloc[split_idx:val_idx]
        df_test = df_features.iloc[val_idx:]
        
        logger.info(f"\nData split:")
        logger.info(f"  Train: {len(df_train):,} ({len(df_train)/len(df_features)*100:.1f}%)")
        logger.info(f"  Val:   {len(df_val):,} ({len(df_val)/len(df_features)*100:.1f}%)")
        logger.info(f"  Test:  {len(df_test):,} ({len(df_test)/len(df_features)*100:.1f}%)")
        
        feature_cols = self.feature_engineer.get_feature_list()
        logger.info(f"\nTotal features: {len(feature_cols)}")
        logger.info(f"Feature list: {feature_cols}")
        
        # 4. 準備訓練數據
        X_train = df_train[feature_cols].fillna(0).values
        X_val = df_val[feature_cols].fillna(0).values
        X_test = df_test[feature_cols].fillna(0).values
        
        y_train_long = df_train['label_long_adaptive'].values
        y_val_long = df_val['label_long_adaptive'].values
        y_test_long = df_test['label_long_adaptive'].values
        
        y_train_short = df_train['label_short_adaptive'].values
        y_val_short = df_val['label_short_adaptive'].values
        y_test_short = df_test['label_short_adaptive'].values
        
        logger.info(f"\nLabel distribution:")
        logger.info(f"  Long:  {y_train_long.mean()*100:.2f}% positive")
        logger.info(f"  Short: {y_train_short.mean()*100:.2f}% positive")
        
        # 5. 超參數優化 (Long)
        logger.info("\n" + "="*80)
        logger.info("HYPERPARAMETER OPTIMIZATION - LONG ORACLE")
        logger.info("="*80)
        best_params_long = self.optimize_hyperparameters(
            X_train, y_train_long, X_val, y_val_long, 'long'
        )
        
        # 6. 超參數優化 (Short)
        logger.info("\n" + "="*80)
        logger.info("HYPERPARAMETER OPTIMIZATION - SHORT ORACLE")
        logger.info("="*80)
        best_params_short = self.optimize_hyperparameters(
            X_train, y_train_short, X_val, y_val_short, 'short'
        )
        
        # 7. 訓練 Long Oracle
        logger.info("\n" + "="*80)
        logger.info("TRAINING LONG ORACLE")
        logger.info("="*80)
        model_long = self.train_ensemble(
            X_train, y_train_long, X_val, y_val_long,
            best_params_long, 'long'
        )
        
        # 8. 訓練 Short Oracle
        logger.info("\n" + "="*80)
        logger.info("TRAINING SHORT ORACLE")
        logger.info("="*80)
        model_short = self.train_ensemble(
            X_train, y_train_short, X_val, y_val_short,
            best_params_short, 'short'
        )
        
        # 9. 評估
        logger.info("\n" + "="*80)
        logger.info("MODEL EVALUATION")
        logger.info("="*80)
        
        eval_long = self.evaluate_model(model_long, X_test, y_test_long, 'long')
        eval_short = self.evaluate_model(model_short, X_test, y_test_short, 'short')
        
        # 10. Walk-Forward 驗證
        wf_results = self.walk_forward_validation(df_features)
        
        # 11. 保存模型 (使用新的 save_model_with_metadata 函數)
        output_dir = Path("models_output")
        output_dir.mkdir(exist_ok=True)
        
        logger.info("\n" + "="*80)
        logger.info("SAVING MODELS WITH METADATA")
        logger.info("="*80)
        
        long_path = save_model_with_metadata(
            model=model_long,
            feature_names=feature_cols,
            model_type='long',
            version='v2',
            output_dir=output_dir
        )
        
        short_path = save_model_with_metadata(
            model=model_short,
            feature_names=feature_cols,
            model_type='short',
            version='v2',
            output_dir=output_dir
        )
        
        logger.info("="*80)
        logger.info("MODELS SAVED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Long Oracle:  {long_path}")
        logger.info(f"Short Oracle: {short_path}")
        
        # 12. 保存特徵列表 (額外備份)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        features_path = output_dir / f"features_v2_{timestamp}.txt"
        with open(features_path, 'w') as f:
            for feat in feature_cols:
                f.write(f"{feat}\n")
        
        logger.info(f"Features:     {features_path}")
        logger.info("="*80)
        
        return {
            'model_long': model_long,
            'model_short': model_short,
            'eval_long': eval_long,
            'eval_short': eval_short,
            'walk_forward': wf_results,
            'long_path': long_path,
            'short_path': short_path,
            'feature_cols': feature_cols
        }

if __name__ == "__main__":
    trainer = AdvancedTrainer(
        enable_hyperopt=True,
        enable_ensemble=True,
        enable_walk_forward=True,
        n_trials=50
    )
    
    results = trainer.run()
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"Long Oracle:  {results['long_path']}")
    print(f"Short Oracle: {results['short_path']}")
    print(f"Long AUC:     {results['eval_long']['auc']:.4f}")
    print(f"Short AUC:    {results['eval_short']['auc']:.4f}")
    print(f"Features:     {len(results['feature_cols'])}")
    print("="*80)