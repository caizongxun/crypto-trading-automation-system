import torch
import pandas as pd
import numpy as np
from chronos import ChronosPipeline
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ChronosPredictor:
    """
    Amazon Chronos 時間序列預測器
    用於預測加密貨幣價格方向
    """
    
    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-small",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            model_name: Chronos 模型名稱
                - amazon/chronos-t5-tiny (最快)
                - amazon/chronos-t5-small (平衡)
                - amazon/chronos-t5-base (最準)
            device: 'cuda' 或 'cpu'
        """
        logger.info(f"Loading Chronos model: {model_name} on {device}")
        
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        
        self.device = device
        logger.info("Chronos model loaded successfully")
    
    
    def predict_probabilities(
        self,
        df: pd.DataFrame,
        lookback: int = 168,
        horizon: int = 1,
        num_samples: int = 100,
        tp_pct: float = 2.0,
        sl_pct: float = 1.0
    ) -> Tuple[float, float]:
        """
        預測上漲/下跌機率
        
        Args:
            df: K線資料 (必須包含 'close' 欄位)
            lookback: 歷史窗口 (預設168=7天1h K線)
            horizon: 預測未來幾根 K線
            num_samples: 蒙地卡羅採樣數
            tp_pct: 止盈百分比
            sl_pct: 止損百分比
        
        Returns:
            (prob_long, prob_short): 上漲機率, 下跌機率
        """
        if len(df) < lookback:
            logger.warning(f"Insufficient data: {len(df)} < {lookback}")
            return 0.05, 0.05
        
        # 準備輸入資料
        context = torch.tensor(
            df['close'].values[-lookback:],
            dtype=torch.float32
        )
        
        # 預測 (直接傳入 context,不用關鍵字參數)
        try:
            forecast = self.pipeline.predict(
                context.unsqueeze(0),  # 移除 context= 關鍵字
                prediction_length=horizon,
                num_samples=num_samples
            )
            
            # forecast shape: (1, num_samples, horizon)
            future_prices = forecast[0, :, -1].cpu().numpy()
            
        except Exception as e:
            logger.error(f"Chronos prediction failed: {e}")
            return 0.05, 0.05
        
        # 計算機率
        current_price = df['close'].iloc[-1]
        
        # Long: 價格上漲 >= tp_pct
        tp_price = current_price * (1 + tp_pct / 100)
        prob_long = (future_prices >= tp_price).mean()
        
        # Short: 價格下跌 >= tp_pct
        sl_price = current_price * (1 - tp_pct / 100)
        prob_short = (future_prices <= sl_price).mean()
        
        # 限制範圍 [0.01, 0.99]
        prob_long = np.clip(prob_long, 0.01, 0.99)
        prob_short = np.clip(prob_short, 0.01, 0.99)
        
        return float(prob_long), float(prob_short)
    
    
    def predict_batch(
        self,
        df: pd.DataFrame,
        lookback: int = 168,
        horizon: int = 1,
        num_samples: int = 100,
        tp_pct: float = 2.0,
        sl_pct: float = 1.0
    ) -> pd.DataFrame:
        """
        批次預測整個資料集
        
        Args:
            df: K線資料
            lookback: 歷史窗口
            horizon: 預測未來幾根 K線
            num_samples: 蒙地卡羅採樣數
            tp_pct: 止盈百分比
            sl_pct: 止損百分比
        
        Returns:
            df with 'prob_long' and 'prob_short' columns
        """
        prob_long_list = []
        prob_short_list = []
        
        logger.info(f"Batch predicting {len(df) - lookback} samples...")
        
        for i in range(lookback, len(df)):
            if i % 1000 == 0:
                logger.info(f"Progress: {i}/{len(df)}")
            
            window = df.iloc[i-lookback:i]
            prob_long, prob_short = self.predict_probabilities(
                window,
                lookback=lookback,
                horizon=horizon,
                num_samples=num_samples,
                tp_pct=tp_pct,
                sl_pct=sl_pct
            )
            prob_long_list.append(prob_long)
            prob_short_list.append(prob_short)
        
        # 填補前面的空值
        result = df.copy()
        result['prob_long'] = [np.nan] * lookback + prob_long_list
        result['prob_short'] = [np.nan] * lookback + prob_short_list
        
        logger.info("Batch prediction completed")
        
        return result
