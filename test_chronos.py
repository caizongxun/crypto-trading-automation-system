#!/usr/bin/env python3
"""
Chronos 快速測試腳本
用於驗證 Chronos 模型整合是否成功
"""

import sys
import logging
from pathlib import Path

# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_hf_loader():
    """測試 HuggingFace 資料載入器"""
    logger.info("=" * 60)
    logger.info("Test 1: HuggingFace Data Loader")
    logger.info("=" * 60)
    
    try:
        from utils.hf_data_loader import load_klines, get_available_symbols
        
        # 測試載入 BTCUSDT 1h 資料
        logger.info("Loading BTCUSDT 1h data...")
        df = load_klines('BTCUSDT', '1h', '2026-02-01', '2026-02-22')
        
        logger.info(f"✅ 成功載入 {len(df)} 筆 K 線")
        logger.info(f"   時間範圍: {df['open_time'].min()} ~ {df['open_time'].max()}")
        logger.info(f"   價格範圍: ${df['close'].min():.2f} ~ ${df['close'].max():.2f}")
        
        # 顯示可用幣種
        symbols = get_available_symbols()
        logger.info(f"✅ 共有 {len(symbols)} 個可用交易對")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ HuggingFace 資料載入器測試失敗: {e}")
        return False


def test_chronos_predictor():
    """測試 Chronos 預測器"""
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Chronos Predictor")
    logger.info("=" * 60)
    
    try:
        from models.chronos_predictor import ChronosPredictor
        from utils.hf_data_loader import load_klines
        
        # 載入資料
        logger.info("Loading test data...")
        df = load_klines('BTCUSDT', '1h', '2026-02-01', '2026-02-22')
        
        # 初始化預測器 (使用最小模型以加快測試)
        logger.info("Initializing Chronos predictor (tiny model)...")
        predictor = ChronosPredictor(
            model_name="amazon/chronos-t5-tiny",
            device="cpu"  # 使用 CPU 以確保兼容性
        )
        
        # 單次預測
        logger.info("Testing single prediction...")
        prob_long, prob_short = predictor.predict_probabilities(
            df=df,
            lookback=168,
            horizon=1,
            num_samples=50,  # 減少採樣數以加快測試
            tp_pct=2.0,
            sl_pct=1.0
        )
        
        logger.info(f"✅ 單次預測成功")
        logger.info(f"   Long 機率: {prob_long:.2%}")
        logger.info(f"   Short 機率: {prob_short:.2%}")
        logger.info(f"   當前價格: ${df['close'].iloc[-1]:.2f}")
        
        # 判斷信號
        if prob_long > 0.15:
            logger.info("   🟢 信號: LONG (看漲)")
        elif prob_short > 0.15:
            logger.info("   🔴 信號: SHORT (看跌)")
        else:
            logger.info("   ⚪ 信號: NEUTRAL (觀望)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Chronos 預測器測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主程式"""
    logger.info("\n" + "#" * 60)
    logger.info("# Chronos Integration Test")
    logger.info("#" * 60 + "\n")
    
    results = []
    
    # Test 1: HuggingFace Loader
    results.append(("HuggingFace Loader", test_hf_loader()))
    
    # Test 2: Chronos Predictor
    results.append(("Chronos Predictor", test_chronos_predictor()))
    
    # 總結
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        logger.info("\n🎉 所有測試通過! Chronos 整合成功!")
        logger.info("\n下一步:")
        logger.info("1. 啟動 Streamlit: streamlit run main.py")
        logger.info("2. 選擇 Chronos 模型執行回測")
        logger.info("3. 比較 XGBoost vs Chronos 效能")
        return 0
    else:
        logger.error("\n❌ 有測試失敗，請檢查錯誤訊息")
        logger.error("\n常見問題:")
        logger.error("1. 沒有安裝依賴: pip install -r requirements.txt")
        logger.error("2. PyTorch 版本不相容: pip install torch --upgrade")
        logger.error("3. HuggingFace 網路問題: 檢查網路連線")
        return 1


if __name__ == "__main__":
    sys.exit(main())
