"""
V3 Signal Filter - Five-Layer Quality Control
五層信號過濾器
"""
import pandas as pd
import numpy as np
from typing import Dict

class SignalFilter:
    """
    五層信號過濾器
    
    目標: 將信號從數萬筆減少到3000-5000筆高質量交易
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Layer 1: 模型信心度
        self.min_confidence = config.get('min_confidence', 0.6)
        
        # Layer 2: 成交量確認
        self.min_volume_ratio = config.get('min_volume_ratio', 1.3)
        
        # Layer 3: 趨勢強度
        self.min_trend_strength = config.get('min_trend_strength', 0.5)
        
        # Layer 4: ATR波動率過濾
        self.max_atr_ratio = config.get('max_atr_ratio', 0.05)  # ATR<5%
        
        # Layer 5: 時間過濾
        self.blackout_hours = config.get('blackout_hours', [21, 22])  # UTC
        
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
        
        確保模型對預測的信心度足夠高
        """
        if confidences is None:
            return True  # 無信心度數據,通過
        
        if idx >= len(confidences):
            return False
        
        return confidences[idx] >= self.min_confidence
    
    def _layer2_volume(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Layer 2: 成交量確認
        
        確保有足夠成交量支持，避免假突破
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
        
        確保只在明確趨勢中交易
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
        
        避免在極端波動時交易
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
        
        避免在特定時間段交易 (例如美股收盤前)
        """
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
        
        print("\n[五層過濾統計]")
        print(f"原始信號: {total}")
        print(f"Layer 1 (模型信心度): {stats['passed_layer1']} ({stats['passed_layer1']/total*100:.1f}%)")
        print(f"Layer 2 (成交量確認): {stats['passed_layer2']} ({stats['passed_layer2']/total*100:.1f}%)")
        print(f"Layer 3 (趨勢強度): {stats['passed_layer3']} ({stats['passed_layer3']/total*100:.1f}%)")
        print(f"Layer 4 (ATR波動率): {stats['passed_layer4']} ({stats['passed_layer4']/total*100:.1f}%)")
        print(f"Layer 5 (時間過濾): {stats['passed_layer5']} ({stats['passed_layer5']/total*100:.1f}%)")
        print(f"\n最終通過: {stats['final_passed']} ({stats['final_passed']/total*100:.1f}%)")
        print(f"過濾效果: {(1 - stats['final_passed']/total)*100:.1f}% 信號被濾除")
        
        # 警告
        if stats['final_passed'] / total > 0.5:
            print("[警告] 過濾後仍有>50%信號,可能需要更嚴格的過濾")
        
        if stats['final_passed'] / total < 0.05:
            print("[警告] 過濾後<5%信號,可能過於嚴格")
    
    def get_filter_report(self) -> Dict:
        """
        獲取詳細過濾報告
        """
        stats = self.filter_stats
        total = stats['total_signals']
        
        if total == 0:
            return {'error': '無交易信號'}
        
        return {
            'original_signals': total,
            'final_signals': stats['final_passed'],
            'filter_rate': 1 - stats['final_passed'] / total,
            'layer_pass_rates': {
                'layer1': stats['passed_layer1'] / total,
                'layer2': stats['passed_layer2'] / total,
                'layer3': stats['passed_layer3'] / total,
                'layer4': stats['passed_layer4'] / total,
                'layer5': stats['passed_layer5'] / total
            },
            'bottleneck_layer': self._identify_bottleneck()
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
