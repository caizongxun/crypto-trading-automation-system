import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))
from config import Config
from utils.logger import setup_logger
from utils.feature_engineering import FeatureEngineer
from utils.enhanced_feature_engineering import EnhancedFeatureEngineer
from utils.adaptive_backtester import AdaptiveBacktester

logger = setup_logger('system_comparison', 'logs/system_comparison.log')

class SystemComparison:
    """
    系統對比工具 - 原系統 vs 增強系統
    """
    
    def __init__(self):
        self.original_fe = FeatureEngineer()
        self.enhanced_fe = EnhancedFeatureEngineer()
        
        logger.info("="*80)
        logger.info("SYSTEM COMPARISON TOOL")
        logger.info("="*80)
    
    def compare_features(self, df_1m: pd.DataFrame) -> Dict:
        """
        對比特徵工程
        """
        logger.info("\nComparing Feature Engineering...")
        
        # 原系統特徵
        logger.info("  Generating original features...")
        df_original = self.original_fe.create_features_from_1m(
            df_1m.copy(),
            base_tp_pct=0.02,
            base_sl_pct=0.01
        )
        
        original_features = [col for col in df_original.columns 
                            if col not in ['open', 'high', 'low', 'close', 'volume', 
                                          'label_long', 'label_short']]
        
        # 增強系統特徵
        logger.info("  Generating enhanced features...")
        df_enhanced = self.enhanced_fe.create_enhanced_features(
            df_1m.copy(),
            use_adaptive_labels=True,
            label_type='both'
        )
        
        enhanced_features = [col for col in df_enhanced.columns 
                            if col not in ['open', 'high', 'low', 'close', 'volume', 
                                          'label_long', 'label_short']]
        
        # 統計
        comparison = {
            'original': {
                'feature_count': len(original_features),
                'label_long_positive_rate': df_original['label_long'].mean() * 100,
                'label_short_positive_rate': df_original['label_short'].mean() * 100,
                'sample_count': len(df_original)
            },
            'enhanced': {
                'feature_count': len(enhanced_features),
                'label_long_positive_rate': df_enhanced['label_long'].mean() * 100,
                'label_short_positive_rate': df_enhanced['label_short'].mean() * 100,
                'sample_count': len(df_enhanced)
            }
        }
        
        logger.info("\n" + "="*80)
        logger.info("FEATURE ENGINEERING COMPARISON")
        logger.info("="*80)
        logger.info(f"Original System:")
        logger.info(f"  Features: {comparison['original']['feature_count']}")
        logger.info(f"  Long Positive Rate: {comparison['original']['label_long_positive_rate']:.2f}%")
        logger.info(f"  Short Positive Rate: {comparison['original']['label_short_positive_rate']:.2f}%")
        logger.info(f"\nEnhanced System:")
        logger.info(f"  Features: {comparison['enhanced']['feature_count']}")
        logger.info(f"  Long Positive Rate: {comparison['enhanced']['label_long_positive_rate']:.2f}%")
        logger.info(f"  Short Positive Rate: {comparison['enhanced']['label_short_positive_rate']:.2f}%")
        logger.info(f"\nImprovement:")
        logger.info(f"  Features: +{comparison['enhanced']['feature_count'] - comparison['original']['feature_count']} "
                   f"({(comparison['enhanced']['feature_count'] / comparison['original']['feature_count'] - 1)*100:.1f}%)")
        logger.info(f"  Long Positive Rate: +{comparison['enhanced']['label_long_positive_rate'] - comparison['original']['label_long_positive_rate']:.2f}%")
        logger.info("="*80)
        
        return comparison, df_original, df_enhanced
    
    def compare_backtest(
        self,
        df_original: pd.DataFrame,
        df_enhanced: pd.DataFrame,
        model_original_long: str,
        model_original_short: str,
        model_enhanced_long: str,
        model_enhanced_short: str
    ) -> Dict:
        """
        對比回測結果
        """
        logger.info("\n" + "="*80)
        logger.info("BACKTEST COMPARISON")
        logger.info("="*80)
        
        # 回測參數
        config = {
            'initial_capital': 10000,
            'base_position_size': 0.10,
            'base_threshold': 0.16,
            'base_tp_pct': 0.02,
            'base_sl_pct': 0.01,
            'enable_volatility_adaptation': True,
            'enable_probability_layering': True,
            'enable_time_based_strategy': True,
            'enable_risk_controls': True
        }
        
        # 原系統回測
        logger.info("\nRunning original system backtest...")
        backtester_original = AdaptiveBacktester(**config)
        results_original = backtester_original.run_backtest(
            df_original,
            model_original_long,
            model_original_short
        )
        
        # 增強系統回測
        logger.info("\nRunning enhanced system backtest...")
        backtester_enhanced = AdaptiveBacktester(**config)
        results_enhanced = backtester_enhanced.run_backtest(
            df_enhanced,
            model_enhanced_long,
            model_enhanced_short
        )
        
        # 對比
        metrics = [
            'total_trades', 'win_rate', 'profit_factor',
            'total_return_pct', 'max_drawdown_pct', 'sharpe_ratio'
        ]
        
        comparison = {
            'original': {k: results_original.get(k, 0) for k in metrics},
            'enhanced': {k: results_enhanced.get(k, 0) for k in metrics}
        }
        
        logger.info("\n" + "="*80)
        logger.info("BACKTEST RESULTS COMPARISON")
        logger.info("="*80)
        
        for metric in metrics:
            orig_val = comparison['original'][metric]
            enh_val = comparison['enhanced'][metric]
            
            if metric in ['total_trades']:
                improvement = enh_val - orig_val
                improvement_pct = (enh_val / orig_val - 1) * 100 if orig_val > 0 else 0
                logger.info(f"{metric:20s}: {orig_val:8.0f} → {enh_val:8.0f} "
                           f"(+{improvement:.0f}, +{improvement_pct:.1f}%)")
            else:
                improvement = enh_val - orig_val
                improvement_pct = (enh_val / orig_val - 1) * 100 if orig_val > 0 else 0
                logger.info(f"{metric:20s}: {orig_val:8.2f} → {enh_val:8.2f} "
                           f"(+{improvement:.2f}, +{improvement_pct:.1f}%)")
        
        logger.info("="*80)
        
        return comparison, results_original, results_enhanced
    
    def generate_report(self, feature_comp: Dict, backtest_comp: Dict):
        """
        生成完整對比報告
        """
        report = f"""
# 系統對比報告
生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 特徵工程對比

| 維度 | 原系統 | 增強系統 | 改進 |
|------|---------|----------|------|
| 特徵數 | {feature_comp['original']['feature_count']} | {feature_comp['enhanced']['feature_count']} | +{feature_comp['enhanced']['feature_count'] - feature_comp['original']['feature_count']} ({(feature_comp['enhanced']['feature_count'] / feature_comp['original']['feature_count'] - 1)*100:.1f}%) |
| Long 正樣本率 | {feature_comp['original']['label_long_positive_rate']:.2f}% | {feature_comp['enhanced']['label_long_positive_rate']:.2f}% | +{feature_comp['enhanced']['label_long_positive_rate'] - feature_comp['original']['label_long_positive_rate']:.2f}% |
| Short 正樣本率 | {feature_comp['original']['label_short_positive_rate']:.2f}% | {feature_comp['enhanced']['label_short_positive_rate']:.2f}% | +{feature_comp['enhanced']['label_short_positive_rate'] - feature_comp['original']['label_short_positive_rate']:.2f}% |

## 2. 回測性能對比

| 指標 | 原系統 | 增強系統 | 改進 |
|------|---------|----------|------|
| 交易數 | {backtest_comp['original']['total_trades']:.0f} | {backtest_comp['enhanced']['total_trades']:.0f} | +{backtest_comp['enhanced']['total_trades'] - backtest_comp['original']['total_trades']:.0f} ({(backtest_comp['enhanced']['total_trades'] / backtest_comp['original']['total_trades'] - 1)*100:.1f}%) |
| 勝率 | {backtest_comp['original']['win_rate']:.2f}% | {backtest_comp['enhanced']['win_rate']:.2f}% | +{backtest_comp['enhanced']['win_rate'] - backtest_comp['original']['win_rate']:.2f}% |
| Profit Factor | {backtest_comp['original']['profit_factor']:.2f} | {backtest_comp['enhanced']['profit_factor']:.2f} | +{backtest_comp['enhanced']['profit_factor'] - backtest_comp['original']['profit_factor']:.2f} ({(backtest_comp['enhanced']['profit_factor'] / backtest_comp['original']['profit_factor'] - 1)*100:.1f}%) |
| 總報酬 | {backtest_comp['original']['total_return_pct']:.2f}% | {backtest_comp['enhanced']['total_return_pct']:.2f}% | +{backtest_comp['enhanced']['total_return_pct'] - backtest_comp['original']['total_return_pct']:.2f}% |
| 最大回撤 | {backtest_comp['original']['max_drawdown_pct']:.2f}% | {backtest_comp['enhanced']['max_drawdown_pct']:.2f}% | {backtest_comp['enhanced']['max_drawdown_pct'] - backtest_comp['original']['max_drawdown_pct']:.2f}% |
| Sharpe Ratio | {backtest_comp['original']['sharpe_ratio']:.2f} | {backtest_comp['enhanced']['sharpe_ratio']:.2f} | +{backtest_comp['enhanced']['sharpe_ratio'] - backtest_comp['original']['sharpe_ratio']:.2f} |

## 3. 總結

### 關鍵改進

1. **交易數增加**: 從 {backtest_comp['original']['total_trades']:.0f} 筆提升到 {backtest_comp['enhanced']['total_trades']:.0f} 筆 (+{(backtest_comp['enhanced']['total_trades'] / backtest_comp['original']['total_trades'] - 1)*100:.1f}%)
   - 原因: 動態標籤增加正樣本,增強特徵捕捉更多機會

2. **勝率提升**: 從 {backtest_comp['original']['win_rate']:.2f}% 提升到 {backtest_comp['enhanced']['win_rate']:.2f}% (+{backtest_comp['enhanced']['win_rate'] - backtest_comp['original']['win_rate']:.2f}%)
   - 原因: 訂單流 + 微觀結構 + 多時間框架特徵

3. **Profit Factor 提升**: 從 {backtest_comp['original']['profit_factor']:.2f} 提升到 {backtest_comp['enhanced']['profit_factor']:.2f} (+{(backtest_comp['enhanced']['profit_factor'] / backtest_comp['original']['profit_factor'] - 1)*100:.1f}%)
   - 原因: 集成學習 + 動態權重 + 超參數優化

### 建議

{'✅ **建議使用增強系統**' if backtest_comp['enhanced']['profit_factor'] > backtest_comp['original']['profit_factor'] * 1.2 else '⚠️ **需要進一步優化**'}

"""
        
        # 保存報告
        report_path = Path('logs') / f"system_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path.write_text(report, encoding='utf-8')
        
        logger.info(f"\n✅ Report saved: {report_path}")
        
        return report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Original vs Enhanced System')
    parser.add_argument('--original-long', required=True, help='Path to original long model')
    parser.add_argument('--original-short', required=True, help='Path to original short model')
    parser.add_argument('--enhanced-long', required=True, help='Path to enhanced long model')
    parser.add_argument('--enhanced-short', required=True, help='Path to enhanced short model')
    
    args = parser.parse_args()
    
    # 實例化
    comparator = SystemComparison()
    
    # 載入數據
    from huggingface_hub import hf_hub_download
    
    local_path = hf_hub_download(
        repo_id=Config.HF_REPO_ID,
        filename="klines/BTCUSDT/BTC_1m.parquet",
        repo_type="dataset",
        token=Config.HF_TOKEN
    )
    
    df_1m = pd.read_parquet(local_path)
    
    # 1. 對比特徵
    feature_comp, df_original, df_enhanced = comparator.compare_features(df_1m)
    
    # 2. 對比回測
    backtest_comp, _, _ = comparator.compare_backtest(
        df_original, df_enhanced,
        args.original_long, args.original_short,
        args.enhanced_long, args.enhanced_short
    )
    
    # 3. 生成報告
    report = comparator.generate_report(feature_comp, backtest_comp)
    
    print("\n" + report)