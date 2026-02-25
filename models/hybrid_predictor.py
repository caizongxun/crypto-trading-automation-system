"""
混合預測器 - 結合 Chronos + XGBoost
激進複利策略: 30天翻倉 (2x)

策略設計:
- 大量交易: 150-300 筆/30天
- 勝率目標: 55-60%
- 單筆目標: +1.2% (低 TP 高頻)
- 複利加成: 1.012^200 = 8.9x (理論值)
- 實際預期: 2-3x (30天)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class HybridPredictor:
    """
    Chronos + XGBoost 混合預測器
    激進複利策略
    """
    
    def __init__(
        self,
        chronos_model: str = "amazon/chronos-t5-tiny",
        xgboost_model_dir: str = "models_output",
        strategy: str = "aggressive"  # aggressive, moderate, conservative
    ):
        """
        Args:
            chronos_model: Chronos 模型名稱
            xgboost_model_dir: XGBoost 模型目錄
            strategy: 策略模式
        """
        self.strategy = strategy
        
        # 載入 Chronos
        logger.info(f"Loading Chronos model: {chronos_model}")
        from models.chronos_predictor import ChronosPredictor
        self.chronos = ChronosPredictor(model_name=chronos_model, device="cpu")
        
        # 載入 XGBoost
        logger.info(f"Loading XGBoost models from: {xgboost_model_dir}")
        self.xgboost_long = self._load_xgboost(xgboost_model_dir, 'long')
        self.xgboost_short = self._load_xgboost(xgboost_model_dir, 'short')
        
        # 策略參數
        self.params = self._get_strategy_params(strategy)
        
        logger.info(f"Hybrid predictor initialized with {strategy} strategy")
        logger.info(f"Target: {self.params['target_return']}x in 30 days")
    
    def _load_xgboost(self, model_dir: str, side: str):
        """載入 XGBoost 模型"""
        try:
            import pickle
            model_dir = Path(model_dir)
            
            # 尋找最新的 v3 模型
            pattern = f"catboost_{side}_v3_*.pkl"
            models = list(model_dir.glob(pattern))
            
            if not models:
                # 降級到 v1
                pattern = f"catboost_{side}_[0-9]*.pkl"
                models = list(model_dir.glob(pattern))
            
            if models:
                latest = max(models, key=lambda p: p.stat().st_mtime)
                with open(latest, 'rb') as f:
                    return pickle.load(f)
            
            logger.warning(f"No XGBoost {side} model found")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load XGBoost {side} model: {e}")
            return None
    
    def _get_strategy_params(self, strategy: str) -> Dict:
        """獲取策略參數"""
        if strategy == "aggressive":
            return {
                'target_return': 2.0,  # 2x in 30 days
                'chronos_threshold_long': 0.08,  # 極低門檻
                'chronos_threshold_short': 0.08,
                'xgboost_threshold_long': 0.12,  # 極低門檻
                'xgboost_threshold_short': 0.12,
                'require_both': False,  # OR 模式 (任一達標即可)
                'tp_pct': 1.2,  # 低 TP 高頻交易
                'sl_pct': 0.6,  # 緊止損
                'position_size': 1.0,  # 全倉 (激進)
                'max_trades_per_day': 20,  # 高頻
                'min_confidence': 0.10  # 最低信心
            }
        
        elif strategy == "moderate":
            return {
                'target_return': 1.5,  # 1.5x in 30 days
                'chronos_threshold_long': 0.12,
                'chronos_threshold_short': 0.12,
                'xgboost_threshold_long': 0.15,
                'xgboost_threshold_short': 0.15,
                'require_both': False,
                'tp_pct': 1.5,
                'sl_pct': 0.8,
                'position_size': 0.8,
                'max_trades_per_day': 10,
                'min_confidence': 0.12
            }
        
        else:  # conservative
            return {
                'target_return': 1.2,  # 1.2x in 30 days
                'chronos_threshold_long': 0.15,
                'chronos_threshold_short': 0.15,
                'xgboost_threshold_long': 0.20,
                'xgboost_threshold_short': 0.20,
                'require_both': True,  # AND 模式
                'tp_pct': 2.0,
                'sl_pct': 1.0,
                'position_size': 0.5,
                'max_trades_per_day': 5,
                'min_confidence': 0.15
            }
    
    def predict(
        self,
        df: pd.DataFrame,
        features_df: Optional[pd.DataFrame] = None
    ) -> Tuple[str, float, Dict]:
        """
        混合預測
        
        Args:
            df: K線資料 (for Chronos)
            features_df: 特徵資料 (for XGBoost)
        
        Returns:
            (signal, confidence, details)
            signal: 'LONG', 'SHORT', 'HOLD'
            confidence: 0.0-1.0
            details: 詳細資訊
        """
        # Step 1: Chronos 預測
        chronos_long, chronos_short = self.chronos.predict_probabilities(
            df=df,
            lookback=168,
            horizon=1,
            num_samples=50,
            tp_pct=self.params['tp_pct'],
            sl_pct=self.params['sl_pct']
        )
        
        # Step 2: XGBoost 預測
        xgb_long = 0.0
        xgb_short = 0.0
        
        if features_df is not None and len(features_df) > 0:
            try:
                last_features = features_df.iloc[-1:]
                
                if self.xgboost_long:
                    xgb_long = float(self.xgboost_long.predict_proba(last_features)[:, 1][0])
                
                if self.xgboost_short:
                    xgb_short = float(self.xgboost_short.predict_proba(last_features)[:, 1][0])
                    
            except Exception as e:
                logger.warning(f"XGBoost prediction failed: {e}")
        
        # Step 3: 信號融合
        signal, confidence = self._combine_signals(
            chronos_long, chronos_short,
            xgb_long, xgb_short
        )
        
        details = {
            'chronos_long': chronos_long,
            'chronos_short': chronos_short,
            'xgb_long': xgb_long,
            'xgb_short': xgb_short,
            'strategy': self.strategy,
            'tp_pct': self.params['tp_pct'],
            'sl_pct': self.params['sl_pct']
        }
        
        return signal, confidence, details
    
    def _combine_signals(
        self,
        chronos_long: float,
        chronos_short: float,
        xgb_long: float,
        xgb_short: float
    ) -> Tuple[str, float]:
        """
        信號融合邏輯
        """
        params = self.params
        
        if params['require_both']:
            # AND 模式: 兩個模型都要達標
            long_signal = (
                chronos_long > params['chronos_threshold_long'] and
                xgb_long > params['xgboost_threshold_long']
            )
            short_signal = (
                chronos_short > params['chronos_threshold_short'] and
                xgb_short > params['xgboost_threshold_short']
            )
            
            if long_signal:
                confidence = (chronos_long + xgb_long) / 2
                return 'LONG', confidence
            elif short_signal:
                confidence = (chronos_short + xgb_short) / 2
                return 'SHORT', confidence
        
        else:
            # OR 模式: 任一模型達標即可 (激進)
            long_scores = []
            short_scores = []
            
            # Chronos 信號
            if chronos_long > params['chronos_threshold_long']:
                long_scores.append(chronos_long)
            if chronos_short > params['chronos_threshold_short']:
                short_scores.append(chronos_short)
            
            # XGBoost 信號
            if xgb_long > params['xgboost_threshold_long']:
                long_scores.append(xgb_long)
            if xgb_short > params['xgboost_threshold_short']:
                short_scores.append(xgb_short)
            
            # 計算綜合信心
            long_confidence = max(long_scores) if long_scores else 0.0
            short_confidence = max(short_scores) if short_scores else 0.0
            
            # 選擇最強信號
            if long_confidence > short_confidence and long_confidence > params['min_confidence']:
                return 'LONG', long_confidence
            elif short_confidence > params['min_confidence']:
                return 'SHORT', short_confidence
        
        return 'HOLD', 0.0


def print_strategy_comparison():
    """
    列印策略比較
    """
    print("\n" + "="*80)
    print("混合策略比較")
    print("="*80)
    
    strategies = {
        'aggressive': HybridPredictor(strategy='aggressive').params,
        'moderate': HybridPredictor(strategy='moderate').params,
        'conservative': HybridPredictor(strategy='conservative').params
    }
    
    headers = ['參數', 'Aggressive', 'Moderate', 'Conservative']
    
    rows = [
        ['目標報酬 (30天)', '2.0x (100%)', '1.5x (50%)', '1.2x (20%)'],
        ['Chronos 門檻', '0.08', '0.12', '0.15'],
        ['XGBoost 門檻', '0.12', '0.15', '0.20'],
        ['組合模式', 'OR (任一)', 'OR (任一)', 'AND (兩者)'],
        ['TP', '1.2%', '1.5%', '2.0%'],
        ['SL', '0.6%', '0.8%', '1.0%'],
        ['倉位', '100%', '80%', '50%'],
        ['日交易上限', '20', '10', '5'],
        ['預期交易數/30天', '200-300', '100-150', '50-80'],
        ['預期勝率', '52-55%', '55-58%', '58-62%'],
        ['風險等級', '極高', '高', '中']
    ]
    
    # 列印表格
    col_widths = [20, 20, 20, 20]
    
    # 標題
    header_row = ''.join(h.ljust(w) for h, w in zip(headers, col_widths))
    print("\n" + header_row)
    print("-" * 80)
    
    # 資料
    for row in rows:
        print(''.join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))
    
    print("\n" + "="*80)
    print("\n⚠️  風險警告:")
    print("- Aggressive 策略風險極高,可能爆倉")
    print("- 建議先用小資金測試 (100-500 USDT)")
    print("- 30天2x需要完美執行,實際可能只有1.5-1.8x")
    print("="*80 + "\n")


if __name__ == "__main__":
    # 顯示策略比較
    print_strategy_comparison()
