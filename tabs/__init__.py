"""
Tabs package
"""

from . import data_fetcher_tab
from . import feature_engineering_tab
from . import model_training_tab
from . import backtesting_tab
from . import auto_trading_tab
from . import model_management_tab
from . import model_training_v2_tab
from . import model_training_v3_tab
from . import chronos_backtest_tab
from . import v10_scalping_tab

__all__ = [
    'data_fetcher_tab',
    'feature_engineering_tab',
    'model_training_tab',
    'backtesting_tab',
    'auto_trading_tab',
    'model_management_tab',
    'model_training_v2_tab',
    'model_training_v3_tab',
    'chronos_backtest_tab',
    'v10_scalping_tab',
]
