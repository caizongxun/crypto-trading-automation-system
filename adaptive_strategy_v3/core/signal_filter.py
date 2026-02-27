"""
V3 Signal Filter - Adjusted for ~150 trades/month
調整為月150筆交易 (5%信號通過率)
"""
import pandas as pd
import numpy as np
from typing import Dict

class SignalFilter:
    """
    五層信號過濾器 - 調整為月150筆交易
    
    目標: 5%信號通過率 (15m: 2880根K線/月 -> 144筆)
    策略: 放寬過濾,但保持質量控制
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Layer 1: 模型信心度 - 降低閾值
        self.min_confidence = config.get('min_confidence', 0.45)  # 0.6 -> 0.45
        
        # Layer 2: 成交量確認 - 放寬
        self.min_volume_ratio = config.get('min_volume_ratio', 1.1)  # 1.3 -> 1.1
        
        # Layer 3: 趨勢強度 - 放寬
        self.min_trend_strength = config.get('min_trend_strength', 0.3)  # 0.5 -> 0.3
        
        # Layer 4: ATR波動率過濾 - 保持不變
        self.max_atr_ratio = config.get('max_atr_ratio', 0.05)  # 保持
        
        # Layer 5: 時間過濾 - 只過濾極端時段
        self.blackout_hours = config.get('blackout_hours', [])  # 移除時間過濾
        
        # 統計
        self.filter_stats = {
            'total_signals': 0,
            'passed_layer1': 0,
            'passed_layer2': 0,
            'passed_layer3': 0,
            'passed_layer4': 0,
            'passed_layer5': 0,
            'final_passed': 0
        }
    
    def filter_signals(self, df: pd.DataFrame, predictions: np.ndarray, 
                      confidences: np.ndarray = None) -> np.ndarray:
        """
        對預測信號進行五層過濾
        
        Args:
            df: K線數據
            predictions: 模型預測 (-1, 0, 1)
            confidences: 預測信心度
        
        Returns:
            filtered_predictions: 過濾後的信號
        """
        filtered = predictions.copy()
        self.filter_stats['total_signals'] = (predictions != 0).sum()
        
        for i in range(len(predictions)):
            if predictions[i] == 0:
                continue  # 已經是中立,跳過
            
            # 逐層過濾
            if not self._layer1_confidence(i, confidences):
                filtered[i] = 0
                continue
            self.filter_stats['passed_layer1'] += 1
            
            if not self._layer2_volume(df, i):
                filtered[i] = 0
                continue
            self.filter_stats['passed_layer2'] += 1
            
            if not self._layer3_trend(df, i):
                filtered[i] = 0
                continue
            self.filter_stats['passed_layer3'] += 1
            
            if not self._layer4_volatility(df, i):
                filtered[i] = 0
                continue
            self.filter_stats['passed_layer4'] += 1
            
            if not self._layer5_time(df, i):
                filtered[i] = 0
                continue
            self.filter_stats['passed_layer5'] += 1
        
        self.filter_stats['final_passed'] = (filtered != 0).sum()
        
        # 打印統計
        self._print_filter_stats()
        
        return filtered
    
    def _layer1_confidence(self, idx: int, confidences: np.ndarray) -> bool:
        """
        Layer 1: 模型信心度過濾
        
        降低閾值到0.45,允許更多信號通過
        """
        if confidences is None:
            return True  # 無信心度數據,通過
        
        if idx >= len(confidences):
            return False
        
        return confidences[idx] >= self.min_confidence
    
    def _layer2_volume(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Layer 2: 成交量確認
        
        放寬到1.1倍,但仍然過濾極低成交量
        """
        if 'volume' not in df.columns or 'volume_ma_20' not in df.columns:
            return True  # 缺少數據,通過
        
        row = df.iloc[idx]
        
        if pd.isna(row['volume_ma_20']) or row['volume_ma_20'] == 0:
            return False
        
        volume_ratio = row['volume'] / row['volume_ma_20']
        
        return volume_ratio >= self.min_volume_ratio
    
    def _layer3_trend(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Layer 3: 趨勢強度過濾
        
        放寬到0.3,允許弱趨勢交易
        """
        if 'trend_strength' not in df.columns:
            return True
        
        row = df.iloc[idx]
        
        if pd.isna(row['trend_strength']):
            return False
        
        return abs(row['trend_strength']) >= self.min_trend_strength
    
    def _layer4_volatility(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Layer 4: ATR波動率過濾
        
        保持不變 - 只過濾極端波動 (ATR>5%)
        """
        if 'atr_14' not in df.columns:
            return True
        
        row = df.iloc[idx]
        
        if pd.isna(row['atr_14']) or row['atr_14'] == 0:
            return False
        
        atr_ratio = row['atr_14'] / row['close']
        
        return atr_ratio <= self.max_atr_ratio
    
    def _layer5_time(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Layer 5: 時間過濾
        
        移除時間過濾 - 允許所有時段交易
        """
        if len(self.blackout_hours) == 0:
            return True
        
        if 'timestamp' not in df.columns:
            return True
        
        row = df.iloc[idx]
        
        try:
            timestamp = pd.to_datetime(row['timestamp'])
            hour = timestamp.hour
            
            return hour not in self.blackout_hours
        except:
            return True  # 時間格式錯誤,通過
    
    def _print_filter_stats(self):
        """
        打印過濾統計
        """
        stats = self.filter_stats
        total = stats['total_signals']
        
        if total == 0:
            print("\n[過濾統計] 無交易信號")
            return
        
        print("\n[五層過濾統計 - 調整為月150筆]")
        print(f"原始信號: {total}")
        print(f"Layer 1 (信心度≥0.45): {stats['passed_layer1']} ({stats['passed_layer1']/total*100:.1f}%)")
        print(f"Layer 2 (成交量≥1.1x): {stats['passed_layer2']} ({stats['passed_layer2']/total*100:.1f}%)")
        print(f"Layer 3 (趨勢強度≥0.3): {stats['passed_layer3']} ({stats['passed_layer3']/total*100:.1f}%)")
        print(f"Layer 4 (ATR<5%): {stats['passed_layer4']} ({stats['passed_layer4']/total*100:.1f}%)")
        print(f"Layer 5 (時間): {stats['passed_layer5']} ({stats['passed_layer5']/total*100:.1f}%)")
        print(f"\n最終通過: {stats['final_passed']} ({stats['final_passed']/total*100:.1f}%)")
        print(f"過濾效果: {(1 - stats['final_passed']/total)*100:.1f}% 信號被濾除")
        
        # 預估月交易數 (15m: 2880根K線/月)
        monthly_estimate = (stats['final_passed'] / total) * 2880
        print(f"\n[預估] 月交易數: {monthly_estimate:.0f} 筆")
        
        # 警告
        if monthly_estimate > 200:
            print("[警告] 預估月交易>200筆,可能質量不足")
        elif monthly_estimate < 80:
            print("[警告] 預估月交易<80筆,可能過於保守")
        else:
            print("[合理] 月交易數萸80-200筆,品質平衡")
    
    def get_filter_report(self) -> Dict:
        """
        獲取詳細過濾報告
        """
        stats = self.filter_stats
        total = stats['total_signals']
        
        if total == 0:
            return {'error': '無交易信號'}
        
        monthly_estimate = (stats['final_passed'] / total) * 2880
        
        return {
            'original_signals': total,
            'final_signals': stats['final_passed'],
            'filter_rate': 1 - stats['final_passed'] / total,
            'monthly_estimate': monthly_estimate,
            'layer_pass_rates': {
                'layer1': stats['passed_layer1'] / total,
                'layer2': stats['passed_layer2'] / total,
                'layer3': stats['passed_layer3'] / total,
                'layer4': stats['passed_layer4'] / total,
                'layer5': stats['passed_layer5'] / total
            },
            'bottleneck_layer': self._identify_bottleneck(),
            'filter_config': {
                'min_confidence': self.min_confidence,
                'min_volume_ratio': self.min_volume_ratio,
                'min_trend_strength': self.min_trend_strength,
                'max_atr_ratio': self.max_atr_ratio
            }
        }
    
    def _identify_bottleneck(self) -> str:
        """
        識別瓶頸層 (過濾最多的層)
        """
        stats = self.filter_stats
        
        drops = {
            'layer1': stats['total_signals'] - stats['passed_layer1'],
            'layer2': stats['passed_layer1'] - stats['passed_layer2'],
            'layer3': stats['passed_layer2'] - stats['passed_layer3'],
            'layer4': stats['passed_layer3'] - stats['passed_layer4'],
            'layer5': stats['passed_layer4'] - stats['passed_layer5']
        }
        
        return max(drops, key=drops.get)
