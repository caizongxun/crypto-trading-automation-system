"""
V3 HuggingFace Data Loader
重用V1的數據加載器
"""
import sys
from pathlib import Path

# 動態引入V1的數據加載器
project_root = Path(__file__).parent.parent.parent
v1_root = project_root / 'reversal_strategy_v1'
sys.path.insert(0, str(v1_root))

from reversal_strategy_v1.data.hf_loader import HFDataLoader

# 直接導出
HFDataLoader = HFDataLoader
