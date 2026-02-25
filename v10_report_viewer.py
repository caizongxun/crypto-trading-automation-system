#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v10 交互式報告查看器

可視化 v10 回測結果，包括:
- 資金曲線
- 交易明細
- 統計分析
"""

import sys
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import rcParams
from pathlib import Path
import json

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
rcParams['axes.unicode_minus'] = False


class V10ReportViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("v10 剝頭皮策略 - 回測報告")
        self.root.geometry("1400x900")
        
        self.trades_df = None
        self.equity_df = None
        self.summary = None
        
        self.setup_ui()
        self.load_latest_report()
    
    def setup_ui(self):
        # 主容器
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左側: 摘要與統計
        left_frame = ttk.Frame(main_frame, width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        # 標題
        title_label = ttk.Label(left_frame, text="v10 剝頭皮策略", font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # 配置資訊
        config_frame = ttk.LabelFrame(left_frame, text="配置參數", padding=10)
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.config_text = tk.Text(config_frame, height=6, width=45, font=('Consolas', 9))
        self.config_text.pack()
        
        # 摘要指標
        summary_frame = ttk.LabelFrame(left_frame, text="績效摘要", padding=10)
        summary_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.summary_text = tk.Text(summary_frame, height=15, width=45, font=('Consolas', 10))
        self.summary_text.pack()
        
        # 按鈕
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="重新載入", command=self.load_latest_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="查看交易", command=self.show_trades_window).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="生成報告", command=self.generate_new_report).pack(side=tk.LEFT, padx=5)
        
        # 分析分頁
        analysis_frame = ttk.LabelFrame(left_frame, text="詳細分析", padding=10)
        analysis_frame.pack(fill=tk.BOTH, expand=True)
        
        self.analysis_text = tk.Text(analysis_frame, height=10, width=45, font=('Consolas', 9), wrap=tk.WORD)
        analysis_scroll = ttk.Scrollbar(analysis_frame, command=self.analysis_text.yview)
        self.analysis_text.config(yscrollcommand=analysis_scroll.set)
        self.analysis_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        analysis_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 右側: 圖表
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 分頁籤
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: 資金曲線
        self.equity_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.equity_tab, text="資金曲線")
        
        # Tab 2: PnL 分析
        self.pnl_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.pnl_tab, text="PnL 分析")
        
        # Tab 3: 時間分析
        self.time_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.time_tab, text="時間分析")
        
        # Tab 4: 方向分析
        self.side_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.side_tab, text="Long vs Short")
    
    def load_latest_report(self):
        """ 載入最新報告 """
        results_dir = Path('backtest_results/v10_detailed')
        
        if not results_dir.exists():
            messagebox.showwarning("警告", "未找到 v10 報告目錄，請先執行 generate_v10_report.py")
            return
        
        # 找最新檔案
        trade_files = sorted(results_dir.glob('trades_*.csv'))
        equity_files = sorted(results_dir.glob('equity_curve_*.csv'))
        summary_files = sorted(results_dir.glob('summary_*.json'))
        
        if not trade_files or not equity_files or not summary_files:
            messagebox.showwarning("警告", "未找到報告檔案")
            return
        
        # 載入數據
        self.trades_df = pd.read_csv(trade_files[-1])
        self.equity_df = pd.read_csv(equity_files[-1])
        
        with open(summary_files[-1], 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.summary = data['summary']
            self.config = data['config']
        
        # 更新顯示
        self.update_config_display()
        self.update_summary_display()
        self.update_analysis_display()
        self.plot_charts()
        
        messagebox.showinfo("成功", f"已載入報告\n交易數: {len(self.trades_df)}")
    
    def update_config_display(self):
        self.config_text.delete('1.0', tk.END)
        self.config_text.insert('1.0', 
            f"策略: {self.config['strategy']}\n"
            f"時間框架: {self.config['timeframe']}\n"
            f"Threshold: {self.config['threshold']}\n"
            f"TP: {self.config['tp_pct']}%\n"
            f"SL: {self.config['sl_pct']}%\n"
            f"RR 比: {self.config['tp_pct']/self.config['sl_pct']:.2f}"
        )
        self.config_text.config(state=tk.DISABLED)
    
    def update_summary_display(self):
        self.summary_text.delete('1.0', tk.END)
        s = self.summary
        
        annual_return = s['total_return_pct'] * (365 / 234)  # 估算年化
        
        text = (
            f"═══ 總體績效 ═══\n\n"
            f"交易數:     {s['total_trades']:.0f} 筆\n"
            f"勝率:       {s['win_rate']*100:.2f}%\n"
            f"總報酬:     {s['total_return_pct']*100:.2f}%\n"
            f"總 PnL:      ${s['total_pnl']:.2f}\n\n"
            f"═══ 風險指標 ═══\n\n"
            f"盈虧比:     {s['profit_factor']:.2f}\n"
            f"Sharpe:      {s['sharpe_ratio']:.2f}\n"
            f"最大回撤:   {s['max_drawdown']*100:.2f}%\n\n"
            f"═══ 平均指標 ═══\n\n"
            f"平均獲利:   ${s['avg_win']:.2f}\n"
            f"平均虧損:   ${s['avg_loss']:.2f}\n\n"
            f"═══ 預估指標 ═══\n\n"
            f"年化報酬:   {annual_return*100:.1f}%\n"
            f"每日交易:   {s['total_trades']/234:.1f} 筆"
        )
        
        self.summary_text.insert('1.0', text)
        self.summary_text.config(state=tk.DISABLED)
    
    def update_analysis_display(self):
        self.analysis_text.delete('1.0', tk.END)
        
        trades = self.trades_df
        trades['entry_time'] = pd.to_datetime(trades['entry_time'])
        trades['hour'] = trades['entry_time'].dt.hour
        trades['weekday'] = trades['entry_time'].dt.dayofweek
        
        # 最佳時段
        hour_stats = trades.groupby('hour').agg({'pnl': 'sum'}).sort_values('pnl', ascending=False)
        best_hour = hour_stats.index[0]
        best_hour_pnl = hour_stats.iloc[0]['pnl']
        
        # Long vs Short
        side_stats = trades.groupby('side').agg({
            'win': ['count', lambda x: x.sum()/len(x)*100],
            'pnl': 'sum'
        })
        
        long_wr = side_stats.loc['long', ('win', '<lambda>')]
        short_wr = side_stats.loc['short', ('win', '<lambda>')]
        long_pnl = side_stats.loc['long', ('pnl', 'sum')]
        short_pnl = side_stats.loc['short', ('pnl', 'sum')]
        
        # 連勝/連敗
        trades['win_int'] = trades['win'].astype(int)
        trades['streak'] = (trades['win_int'] != trades['win_int'].shift()).cumsum()
        streaks = trades.groupby(['streak', 'win']).size()
        
        win_streaks = streaks[streaks.index.get_level_values(1) == True]
        loss_streaks = streaks[streaks.index.get_level_values(1) == False]
        
        max_win_streak = win_streaks.max() if len(win_streaks) > 0 else 0
        max_loss_streak = loss_streaks.max() if len(loss_streaks) > 0 else 0
        
        text = (
            f"● 最佳交易時段: {best_hour:02d}:00\n"
            f"  → PnL: ${best_hour_pnl:.2f}\n\n"
            
            f"● Long 表現:\n"
            f"  → 勝率: {long_wr:.1f}%\n"
            f"  → PnL: ${long_pnl:.2f}\n\n"
            
            f"● Short 表現:\n"
            f"  → 勝率: {short_wr:.1f}%\n"
            f"  → PnL: ${short_pnl:.2f}\n\n"
            
            f"● 連勝紀錄: {max_win_streak} 筆\n"
            f"● 連敗紀錄: {max_loss_streak} 筆\n\n"
            
            f"● 平均持有: {trades['bars_held'].mean():.1f} 根K線\n"
            f"  ({trades['bars_held'].mean()*15:.0f} 分鐘)\n\n"
            
            f"● TP 達成率: "
            f"{len(trades[trades['exit_reason']=='TP'])/len(trades)*100:.1f}%\n"
            f"● SL 觸發率: "
            f"{len(trades[trades['exit_reason']=='SL'])/len(trades)*100:.1f}%"
        )
        
        self.analysis_text.insert('1.0', text)
    
    def plot_charts(self):
        """ 繪製各種圖表 """
        # 清除舊圖
        for widget in self.equity_tab.winfo_children():
            widget.destroy()
        for widget in self.pnl_tab.winfo_children():
            widget.destroy()
        for widget in self.time_tab.winfo_children():
            widget.destroy()
        for widget in self.side_tab.winfo_children():
            widget.destroy()
        
        # Tab 1: 資金曲線
        self.plot_equity_curve()
        
        # Tab 2: PnL 分析
        self.plot_pnl_analysis()
        
        # Tab 3: 時間分析
        self.plot_time_analysis()
        
        # Tab 4: 方向分析
        self.plot_side_analysis()
    
    def plot_equity_curve(self):
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        equity = self.equity_df.copy()
        equity['time'] = pd.to_datetime(equity['time'])
        
        ax.plot(equity['time'], equity['equity'], linewidth=2, color='#2E86DE', label='資金')
        ax.fill_between(equity['time'], 10000, equity['equity'], alpha=0.3, color='#2E86DE')
        ax.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_title('資金曲線', fontsize=14, fontweight='bold')
        ax.set_xlabel('時間')
        ax.set_ylabel('資金 (USD)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, self.equity_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_pnl_analysis(self):
        fig = Figure(figsize=(10, 6))
        gs = fig.add_gridspec(2, 2)
        
        trades = self.trades_df
        
        # 累積 PnL
        ax1 = fig.add_subplot(gs[0, :])
        cumulative = trades['pnl'].cumsum()
        ax1.plot(cumulative.values, linewidth=2, color='#2E86DE')
        ax1.fill_between(range(len(cumulative)), 0, cumulative.values, alpha=0.3, color='#2E86DE')
        ax1.set_title('累積 PnL', fontweight='bold')
        ax1.set_ylabel('PnL (USD)')
        ax1.grid(True, alpha=0.3)
        
        # PnL 分佈
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(trades['pnl'], bins=50, color='#2E86DE', alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_title('PnL 分佈', fontweight='bold')
        ax2.set_xlabel('PnL (USD)')
        ax2.set_ylabel('次數')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 勝率趋勢
        ax3 = fig.add_subplot(gs[1, 1])
        win_rate_ma = trades['win'].rolling(100).mean() * 100
        ax3.plot(win_rate_ma.values, linewidth=2, color='#26de81')
        ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax3.set_title('勝率趋勢 (100MA)', fontweight='bold')
        ax3.set_xlabel('交易編號')
        ax3.set_ylabel('勝率 (%)')
        ax3.grid(True, alpha=0.3)
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, self.pnl_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_time_analysis(self):
        fig = Figure(figsize=(10, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        trades = self.trades_df.copy()
        trades['entry_time'] = pd.to_datetime(trades['entry_time'])
        trades['hour'] = trades['entry_time'].dt.hour
        trades['weekday'] = trades['entry_time'].dt.dayofweek
        
        # 按小時
        hour_pnl = trades.groupby('hour')['pnl'].sum()
        colors1 = ['#26de81' if x > 0 else '#fc5c65' for x in hour_pnl]
        ax1.bar(hour_pnl.index, hour_pnl.values, color=colors1, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax1.set_title('按小時 PnL', fontweight='bold')
        ax1.set_xlabel('小時')
        ax1.set_ylabel('PnL (USD)')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 按星期
        weekday_pnl = trades.groupby('weekday')['pnl'].sum()
        weekday_names = ['一', '二', '三', '四', '五', '六', '日']
        colors2 = ['#26de81' if x > 0 else '#fc5c65' for x in weekday_pnl]
        ax2.bar(range(len(weekday_pnl)), weekday_pnl.values, color=colors2, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_title('按星期 PnL', fontweight='bold')
        ax2.set_xlabel('星期')
        ax2.set_ylabel('PnL (USD)')
        ax2.set_xticks(range(len(weekday_pnl)))
        ax2.set_xticklabels([weekday_names[i] for i in weekday_pnl.index])
        ax2.grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, self.time_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_side_analysis(self):
        fig = Figure(figsize=(10, 6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        trades = self.trades_df
        
        # PnL 對比
        side_pnl = trades.groupby('side')['pnl'].sum()
        colors = ['#2E86DE', '#26de81']
        ax1.bar(range(len(side_pnl)), side_pnl.values, color=colors, alpha=0.7)
        ax1.set_title('PnL 對比', fontweight='bold')
        ax1.set_ylabel('PnL (USD)')
        ax1.set_xticks(range(len(side_pnl)))
        ax1.set_xticklabels([x.upper() for x in side_pnl.index])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 勝率對比
        side_wr = trades.groupby('side')['win'].apply(lambda x: x.sum()/len(x)*100)
        ax2.bar(range(len(side_wr)), side_wr.values, color=colors, alpha=0.7)
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('勝率對比', fontweight='bold')
        ax2.set_ylabel('勝率 (%)')
        ax2.set_xticks(range(len(side_wr)))
        ax2.set_xticklabels([x.upper() for x in side_wr.index])
        ax2.grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, self.side_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_trades_window(self):
        """ 顯示交易明細視窗 """
        if self.trades_df is None:
            messagebox.showwarning("警告", "請先載入報告")
            return
        
        window = tk.Toplevel(self.root)
        window.title("交易明細")
        window.geometry("1200x600")
        
        # 創建 TreeView
        frame = ttk.Frame(window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tree = ttk.Treeview(frame, columns=(
            'entry_time', 'side', 'entry_price', 'exit_price', 
            'exit_reason', 'pnl', 'return_pct', 'bars_held'
        ), show='headings', height=20)
        
        tree.heading('entry_time', text='進場時間')
        tree.heading('side', text='方向')
        tree.heading('entry_price', text='進場價')
        tree.heading('exit_price', text='出場價')
        tree.heading('exit_reason', text='出場原因')
        tree.heading('pnl', text='PnL')
        tree.heading('return_pct', text='報酬%')
        tree.heading('bars_held', text='持有K線')
        
        for col in tree['columns']:
            tree.column(col, width=120, anchor=tk.CENTER)
        
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 填充數據
        for _, row in self.trades_df.iterrows():
            values = (
                row['entry_time'],
                row['side'].upper(),
                f"{row['entry_price']:.2f}",
                f"{row['exit_price']:.2f}",
                row['exit_reason'],
                f"${row['pnl']:.2f}",
                f"{row['return_pct']*100:.2f}%",
                f"{row['bars_held']:.0f}"
            )
            
            tag = 'win' if row['win'] else 'loss'
            tree.insert('', tk.END, values=values, tags=(tag,))
        
        tree.tag_configure('win', background='#e8f5e9')
        tree.tag_configure('loss', background='#ffebee')
    
    def generate_new_report(self):
        """ 生成新報告 """
        result = messagebox.askyesno(
            "確認", 
            "將執行 generate_v10_report.py 生成新報告\n\n"
            "這可能需要幾分鐘，繼續？"
        )
        
        if result:
            import subprocess
            try:
                subprocess.Popen([sys.executable, 'generate_v10_report.py'])
                messagebox.showinfo("提示", "已啟動報告生成程序\n請稍後重新載入")
            except Exception as e:
                messagebox.showerror("錯誤", f"啟動失敗: {e}")


if __name__ == '__main__':
    root = tk.Tk()
    app = V10ReportViewer(root)
    root.mainloop()
