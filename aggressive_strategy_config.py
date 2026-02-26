#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
激進策略配置 - 目標月報酬 50%
起始資金: $10 USDT
最大槓桿: 20x
風險等級: 高
"""

import json
from pathlib import Path

# ========================================
# 三階段成長策略
# ========================================

STRATEGY_PHASES = {
    "phase_1_bootstrap": {
        "name": "啟動階段",
        "capital_range": [10, 50],
        "target": "$10 → $50 (2-3週)",
        "leverage": 20,
        "position_size": 0.05,  # 5% 每筆
        "threshold": 0.70,  # 更高閾值,只做高質量信號
        "tp_pct": 0.012,  # 1.2%
        "sl_pct": 0.006,  # 0.6%
        "max_positions": 1,  # 只做單邊
        "daily_loss_limit": 0.15,  # 日虧損限制 15%
        "strategy": "超短線突破 + 高勝率優先",
        "notes": """
        - 小資金階段最危險,需要極度謹慎
        - 每筆交易名目: $10 × 0.05 × 20 = $10
        - 單次最大虧損: $0.60 (6%)
        - 需要 8-10 筆成功交易才能達標
        - 重點: 保護本金 > 追求利潤
        """
    },
    
    "phase_2_growth": {
        "name": "成長階段",
        "capital_range": [50, 200],
        "target": "$50 → $200 (3-4週)",
        "leverage": 15,  # 降低槓桿
        "position_size": 0.04,  # 4%
        "threshold": 0.65,
        "tp_pct": 0.010,  # 1.0%
        "sl_pct": 0.005,  # 0.5%
        "max_positions": 2,  # 可做多空對沖
        "daily_loss_limit": 0.12,  # 日虧損限制 12%
        "strategy": "均衡型 + 複利加速",
        "notes": """
        - 資金開始有安全邊際
        - 每筆交易名目: $100 × 0.04 × 15 = $60
        - 可以承受連續虧損
        - 開始使用動態 TP/SL
        - 重點: 穩定成長 + 風控
        """
    },
    
    "phase_3_compound": {
        "name": "複利階段",
        "capital_range": [200, 1000],
        "target": "$200 → $500+ (4-6週)",
        "leverage": 10,  # 標準槓桿
        "position_size": 0.03,  # 3%
        "threshold": 0.60,  # 可接受更多信號
        "tp_pct": 0.008,  # 0.8%
        "sl_pct": 0.004,  # 0.4%
        "max_positions": 3,  # 多標的分散
        "daily_loss_limit": 0.10,  # 日虧損限制 10%
        "strategy": "標準策略 + 多標的",
        "notes": """
        - 資金充足,可以正常操作
        - 使用優化後的 v10 參數
        - 啟用所有優化方案
        - 可交易多個交易對分散風險
        - 重點: 持續複利 + 回撤控制
        """
    }
}

# ========================================
# 激進優化配置
# ========================================

AGGRESSIVE_OPTIMIZATIONS = {
    "enable_dynamic_tpsl": True,  # 根據波動性調整
    "enable_quality_sizing": True,  # 信號強度分級倉位
    "enable_trailing_stop": True,  # 移動止損保護利潤
    "enable_time_filter": True,  # 避開低波動時段
    "enable_strict_filter": False,  # 不要太嚴格,需要交易頻率
    
    # 時間過濾 (UTC)
    "active_hours": {
        "start": 8,   # 8:00 UTC (亞洲開盤)
        "end": 22,    # 22:00 UTC (美洲收盤)
        "reason": "高波動時段,流動性好"
    },
    
    # 質量分級倉位
    "quality_sizing": {
        "low": {"threshold": [0.60, 0.70], "size_multiplier": 0.7},
        "medium": {"threshold": [0.70, 0.80], "size_multiplier": 1.0},
        "high": {"threshold": [0.80, 1.00], "size_multiplier": 1.3}
    },
    
    # 動態 TP/SL
    "dynamic_tpsl": {
        "atr_multiplier_tp": 2.0,  # ATR × 2
        "atr_multiplier_sl": 1.0,  # ATR × 1
        "min_rr_ratio": 1.5,  # 最低風險報酬比
    },
    
    # 移動止損
    "trailing_stop": {
        "activation_pct": 0.005,  # 獲利 0.5% 啟動
        "trail_pct": 0.003,  # 回撤 0.3% 止盈
    }
}

# ========================================
# 風險管理規則
# ========================================

RISK_MANAGEMENT = {
    "max_daily_trades": 15,  # 日交易次數限制
    "max_daily_loss_pct": 0.15,  # 日最大虧損 15%
    "max_drawdown_pct": 0.25,  # 最大回撤 25%
    "consecutive_loss_limit": 3,  # 連續虧損 3 次暫停
    "pause_duration_hours": 4,  # 暫停 4 小時
    "max_position_hold_bars": 20,  # 最多持有 20 根 K 線 (15m × 20 = 5 小時)
    
    # 動態止損升級
    "stop_loss_escalation": {
        "after_2_losses": {"action": "reduce_position", "multiplier": 0.8},
        "after_3_losses": {"action": "pause_trading", "duration_hours": 4},
        "after_5_losses_in_day": {"action": "stop_today", "resume": "next_day"}
    },
    
    # 利潤保護
    "profit_protection": {
        "lock_profit_after_pct": 0.20,  # 獲利 20% 後鎖定一半
        "lock_ratio": 0.5,  # 鎖定 50%
        "trailing_after_pct": 0.10,  # 獲利 10% 後啟動移動止損
    }
}

# ========================================
# 多標的配置 (階段3使用)
# ========================================

MULTI_SYMBOL_CONFIG = {
    "symbols": [
        {"symbol": "BTCUSDT", "weight": 0.4, "priority": 1},
        {"symbol": "ETHUSDT", "weight": 0.3, "priority": 2},
        {"symbol": "BNBUSDT", "weight": 0.15, "priority": 3},
        {"symbol": "SOLUSDT", "weight": 0.15, "priority": 4},
    ],
    "correlation_filter": True,  # 避免高相關性同時開倉
    "max_correlated_positions": 2,
    "rebalance_daily": True
}

# ========================================
# 績效目標與檢查點
# ========================================

PERFORMANCE_TARGETS = {
    "week_1": {
        "target_capital": 15,
        "min_win_rate": 0.50,
        "min_trades": 10,
        "action_if_fail": "降低倉位到 3%, 提高閾值到 0.75"
    },
    "week_2": {
        "target_capital": 25,
        "min_win_rate": 0.48,
        "min_trades": 20,
        "action_if_fail": "重新訓練模型或切換到保守模式"
    },
    "week_3": {
        "target_capital": 50,
        "min_win_rate": 0.47,
        "milestone": "完成階段1,進入階段2"
    },
    "week_4": {
        "target_capital": 100,
        "min_win_rate": 0.46,
        "milestone": "階段2進行中"
    },
    "month_1": {
        "target_capital": 15,  # 50% = $10 → $15 (保守)
        "stretch_target": 20,  # 100% = $10 → $20 (樂觀)
        "min_sharpe": 1.5,
        "max_drawdown": 0.25
    }
}

# ========================================
# 實戰建議
# ========================================

PRACTICAL_ADVICE = """
===========================================
激進策略 50% 月報酬實戰指南
===========================================

【現實評估】
✓ 可行性: 中等 (需要優秀執行 + 市場配合)
✓ 風險: 高 (25-40% 最大回撤)
✓ 時間投入: 高 (需要密切監控)
✓ 適合對象: 有經驗的交易者

【關鍵成功因素】
1. 嚴格遵守風控 (最重要!)
2. 不追求每筆都贏,接受連續虧損
3. 在虧損期間減小倉位,獲利期間適度加倉
4. 使用複利但要鎖定部分利潤
5. 市場條件不佳時果斷暫停

【預期月度分布】
- 順利月: +40% ~ +70%
- 普通月: +20% ~ +40%
- 困難月: +0% ~ +20%
- 災難月: -10% ~ -20% (需要避免!)

【起始階段 ($10-$50) 最大挑戰】
❌ 手續費佔比高 (4-5%)
❌ 無法分散風險
❌ 心理壓力大
❌ 容易因小虧損放棄

💡 建議: 考慮增加起始資金到 $50-100
   - 手續費影響降低
   - 心理壓力減輕
   - 倉位管理更靈活
   - 成功率顯著提升

【如果堅持 $10 起始】
第1週目標: 不虧錢,學習系統
第2週目標: +20% ($12)
第3週目標: +50% ($15)
第4週目標: +100% ($20)

然後:
- 提取 $10 (回收本金)
- 用剩餘 $10 繼續
- 心理壓力歸零,可以更大膽

【推薦執行方案】
1. 先在模擬盤測試 2 週
2. 確認勝率 > 45% 才實盤
3. 實盤前 3 天用最小倉位 (2%)
4. 確認系統穩定後才加大倉位
5. 達到 $50 後提取本金 $10

【監控指標】
每日檢查:
- 當日 PnL
- 勝率
- 最大回撤
- 連續虧損次數

每週檢查:
- 是否達成階段目標
- Sharpe Ratio
- 盈虧比
- 需要調整參數?

【緊急熔斷機制】
觸發以下任一條件立即停止交易:
1. 單日虧損 > 15%
2. 連續 3 筆虧損
3. 週回撤 > 20%
4. 月回撤 > 25%
5. 勝率連續 3 天 < 40%

暫停後:
- 分析失敗原因
- 重新訓練模型
- 調整參數
- 模擬盤驗證
- 再次實盤

===========================================
記住: 風險管理 > 利潤最大化
寧可錯過機會,不要承擔過大風險
===========================================
"""

def save_configs():
    """保存配置到 JSON 文件"""
    output_dir = Path('aggressive_configs')
    output_dir.mkdir(exist_ok=True)
    
    configs = {
        'strategy_phases': STRATEGY_PHASES,
        'aggressive_optimizations': AGGRESSIVE_OPTIMIZATIONS,
        'risk_management': RISK_MANAGEMENT,
        'multi_symbol_config': MULTI_SYMBOL_CONFIG,
        'performance_targets': PERFORMANCE_TARGETS
    }
    
    output_file = output_dir / 'aggressive_50pct_monthly.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)
    
    print(f"配置已保存到: {output_file}")
    
    # 保存實戰建議
    advice_file = output_dir / 'practical_advice.txt'
    with open(advice_file, 'w', encoding='utf-8') as f:
        f.write(PRACTICAL_ADVICE)
    
    print(f"實戰建議已保存到: {advice_file}")
    print("\n" + PRACTICAL_ADVICE)

if __name__ == '__main__':
    save_configs()
