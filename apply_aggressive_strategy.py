#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
應用激進策略 - 自動調整 Bot 配置
"""

import json
import sys
from pathlib import Path
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from aggressive_strategy_config import STRATEGY_PHASES, AGGRESSIVE_OPTIMIZATIONS, RISK_MANAGEMENT


def get_current_phase(current_capital):
    """根據當前資金決定階段"""
    if current_capital < 50:
        return 'phase_1_bootstrap', STRATEGY_PHASES['phase_1_bootstrap']
    elif current_capital < 200:
        return 'phase_2_growth', STRATEGY_PHASES['phase_2_growth']
    else:
        return 'phase_3_compound', STRATEGY_PHASES['phase_3_compound']


def generate_bot_config(current_capital, test_mode=False):
    """生成 Bot 配置文件"""
    
    phase_name, phase_config = get_current_phase(current_capital)
    
    print("="*80)
    print(f"當前資金: ${current_capital:.2f}")
    print(f"當前階段: {phase_config['name']} ({phase_name})")
    print(f"階段目標: {phase_config['target']}")
    print("="*80)
    
    # 基礎配置
    bot_config = {
        'metadata': {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_capital': current_capital,
            'phase': phase_name,
            'phase_name': phase_config['name'],
            'target': phase_config['target'],
            'test_mode': test_mode
        },
        
        # 交易參數
        'trading': {
            'leverage': phase_config['leverage'],
            'position_size': phase_config['position_size'],
            'threshold': phase_config['threshold'],
            'tp_pct': phase_config['tp_pct'],
            'sl_pct': phase_config['sl_pct'],
            'max_positions': phase_config['max_positions'],
        },
        
        # 優化方案
        'optimizations': AGGRESSIVE_OPTIMIZATIONS,
        
        # 風險管理
        'risk_management': {
            'daily_loss_limit': phase_config['daily_loss_limit'],
            'max_daily_trades': RISK_MANAGEMENT['max_daily_trades'],
            'max_drawdown_pct': RISK_MANAGEMENT['max_drawdown_pct'],
            'consecutive_loss_limit': RISK_MANAGEMENT['consecutive_loss_limit'],
            'pause_duration_hours': RISK_MANAGEMENT['pause_duration_hours'],
            'max_position_hold_bars': RISK_MANAGEMENT['max_position_hold_bars'],
        },
        
        # 安全機制
        'safety': {
            'enable_stop_loss_escalation': True,
            'enable_profit_protection': True,
            'enable_circuit_breaker': True,
        }
    }
    
    # 根據階段調整
    if phase_name == 'phase_1_bootstrap':
        print("\n[階段1] 啟動階段特殊配置:")
        print(f"  - 槓桿: {phase_config['leverage']}x (最高)")
        print(f"  - 倉位: {phase_config['position_size']*100}% (較大)")
        print(f"  - 閾值: {phase_config['threshold']} (極高,只做高質量)")
        print(f"  - TP/SL: {phase_config['tp_pct']*100:.1f}% / {phase_config['sl_pct']*100:.1f}%")
        print(f"  - 每筆名目: ${current_capital * phase_config['position_size'] * phase_config['leverage']:.2f}")
        print(f"  \n  ⚠️  此階段最危險,建議先模擬盤測試!")
        
        # 啟動階段特殊設定
        bot_config['trading']['long_only'] = False
        bot_config['trading']['short_only'] = False
        bot_config['optimizations']['enable_strict_filter'] = False
        
    elif phase_name == 'phase_2_growth':
        print("\n[階段2] 成長階段配置:")
        print(f"  - 槓桿: {phase_config['leverage']}x (降低)")
        print(f"  - 倉位: {phase_config['position_size']*100}%")
        print(f"  - 閾值: {phase_config['threshold']}")
        print(f"  - 每筆名目: ${current_capital * phase_config['position_size'] * phase_config['leverage']:.2f}")
        print(f"  - 可同時持有: {phase_config['max_positions']} 個位置")
        print(f"  \n  ✅ 可以開始使用優化方案")
        
        bot_config['optimizations']['enable_dynamic_tpsl'] = True
        bot_config['optimizations']['enable_quality_sizing'] = True
        
    else:  # phase_3_compound
        print("\n[階段3] 複利階段配置:")
        print(f"  - 槓桿: {phase_config['leverage']}x (標準)")
        print(f"  - 倉位: {phase_config['position_size']*100}%")
        print(f"  - 閾值: {phase_config['threshold']}")
        print(f"  - 每筆名目: ${current_capital * phase_config['position_size'] * phase_config['leverage']:.2f}")
        print(f"  - 可同時持有: {phase_config['max_positions']} 個位置")
        print(f"  \n  ✅ 啟用所有優化方案 + 多標的")
        
        # 啟用所有優化
        for key in bot_config['optimizations']:
            if key.startswith('enable_'):
                bot_config['optimizations'][key] = True
    
    # 保存配置
    config_dir = Path('bot_configs')
    config_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_file = config_dir / f'aggressive_{phase_name}_{timestamp}.json'
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(bot_config, f, indent=2, ensure_ascii=False)
    
    print(f"\n配置已保存: {config_file}")
    
    # 顯示使用說明
    print("\n" + "="*80)
    print("使用說明")
    print("="*80)
    print(f"""
1. 模擬盤測試 (強烈建議!):
   streamlit run main.py
   - 選擇 "Paper Trading Bot"
   - 載入配置: {config_file.name}
   - 運行 3-7 天
   - 確認勝率 > 45%

2. 實盤交易:
   - 確認 Binance API 已配置
   - 確認帳戶有 ${current_capital:.2f} USDT
   - 選擇 "Live Trading Bot"
   - 載入配置: {config_file.name}
   - 點擊啟動

3. 監控與調整:
   - 每天檢查 PnL 與勝率
   - 達到階段目標後重新生成配置:
     python apply_aggressive_strategy.py --capital <新資金>
   - 觸發熔斷機制後分析原因再啟動

4. 階段升級:
   當前: {phase_config['name']}
   下一階段: 資金達到 ${phase_config['capital_range'][1]}
    """)
    
    return bot_config, config_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='應用激進策略配置')
    parser.add_argument('--capital', type=float, default=10, help='當前資金 (USDT)')
    parser.add_argument('--test', action='store_true', help='測試模式')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("激進策略配置生成器")
    print("目標: 月報酬 50%")
    print("="*80)
    
    # 生成配置
    bot_config, config_file = generate_bot_config(args.capital, args.test)
    
    # 警告訊息
    print("\n" + "="*80)
    print("⚠️  重要警告")
    print("="*80)
    print("""
此策略屬於高風險策略:

1. 可能的最大回撤: 25-40%
2. 連續虧損是正常現象
3. 不是每個月都能達成 50%
4. 市場条件不佳時可能虧錢

建議:
- 只用你能承受全損的資金
- 先在模擬盤測試至少 1-2 週
- 嚴格遵守風控規則
- 達到日虧損限制立即停止
- 定期提取利潤鎖定成果

開始前請確認:
☐ 我理解這是高風險策略
☐ 我只用可以全損的資金
☐ 我會遵守風控規則
☐ 我已在模擬盤測試並確認可行
    """)
    
    print("\n" + "="*80)
    print("祝你交易順利! 🚀")
    print("="*80)


if __name__ == '__main__':
    main()
