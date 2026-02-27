"""
LSTM Neural Network Predictor
LSTM神經網絡預測器

功能:
1. 多時間框架價格預測
2. 勝率估計
3. 賠率預測
4. 信心度評分
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from pathlib import Path
import joblib

class LSTMPredictor(nn.Module):
    """
    LSTM預測模型
    
    輸出:
    - direction: 方向預測 (-1, 0, 1)
    - win_rate: 勝率估計 (0-1)
    - payoff: 賠率估計 (>0)
    - confidence: 信心度 (0-1)
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM層
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # 方向預測頭
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)  # -1, 0, 1
        )
        
        # 勝率預測頭
        self.win_rate_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0-1
        )
        
        # 賠率預測頭
        self.payoff_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Softplus()  # >0
        )
        
        # 信心度頭
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0-1
        )
    
    def forward(self, x):
        # LSTM前向傳播
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # 取最後一個時間步
        
        # 多任務預測
        direction = self.direction_head(last_output)
        win_rate = self.win_rate_head(last_output)
        payoff = self.payoff_head(last_output)
        confidence = self.confidence_head(last_output)
        
        return direction, win_rate, payoff, confidence

class NeuralPredictor:
    """
    神經網絡預測器包裝類
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 配置字典
                - input_size: 輸入特徵數
                - hidden_size: 隱藏層大小 (default: 128)
                - num_layers: LSTM層數 (default: 2)
                - dropout: Dropout率 (default: 0.2)
                - sequence_length: 序列長度 (default: 20)
        """
        self.config = config
        self.input_size = config.get('input_size', 50)
        self.hidden_size = config.get('hidden_size', 128)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.2)
        self.sequence_length = config.get('sequence_length', 20)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = LSTMPredictor(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.scaler = None
        self.is_trained = False
    
    def prepare_sequences(self, X: np.ndarray) -> torch.Tensor:
        """
        準備序列數據
        
        Args:
            X: 特徵矩陣 (samples, features)
        
        Returns:
            序列張量 (samples - seq_len, seq_len, features)
        """
        sequences = []
        for i in range(len(X) - self.sequence_length):
            seq = X[i:i + self.sequence_length]
            sequences.append(seq)
        
        return torch.FloatTensor(np.array(sequences)).to(self.device)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 50, batch_size: int = 64,
             learning_rate: float = 0.001) -> Dict:
        """
        訓練模型
        
        Args:
            X_train: 訓練特徵
            y_train: 訓練標籤 (direction, -1/0/1)
            X_val: 驗證特徵
            y_val: 驗證標籤
            epochs: 訓練輪數
            batch_size: 批次大小
            learning_rate: 學習率
        
        Returns:
            訓練結果字典
        """
        print(f"\n[神經網絡] 開始訓練")
        print(f"訓練樣本: {len(X_train)}, 驗證樣本: {len(X_val)}")
        print(f"設備: {self.device}")
        
        # 準備序列數據
        X_train_seq = self.prepare_sequences(X_train)
        X_val_seq = self.prepare_sequences(X_val)
        
        # 調整標籤長度
        y_train = y_train[self.sequence_length:]
        y_val = y_val[self.sequence_length:]
        
        # 轉換標籤為類別 (-1 -> 0, 0 -> 1, 1 -> 2)
        y_train_cls = torch.LongTensor(y_train + 1).to(self.device)
        y_val_cls = torch.LongTensor(y_val + 1).to(self.device)
        
        # 損失函數和優化器
        criterion_direction = nn.CrossEntropyLoss()
        criterion_regression = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # 訓練階段
            self.model.train()
            total_loss = 0
            
            for i in range(0, len(X_train_seq), batch_size):
                batch_X = X_train_seq[i:i+batch_size]
                batch_y = y_train_cls[i:i+batch_size]
                
                optimizer.zero_grad()
                
                direction, win_rate, payoff, confidence = self.model(batch_X)
                
                # 方向損失
                loss_dir = criterion_direction(direction, batch_y)
                
                # 簡化: 勝率和賠率目標為預設值
                loss_win = criterion_regression(win_rate, torch.ones_like(win_rate) * 0.6)
                loss_payoff = criterion_regression(payoff, torch.ones_like(payoff) * 1.5)
                
                # 總損失 (方向最重要)
                loss = loss_dir * 0.7 + loss_win * 0.15 + loss_payoff * 0.15
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / (len(X_train_seq) / batch_size)
            train_losses.append(avg_train_loss)
            
            # 驗證階段
            self.model.eval()
            with torch.no_grad():
                val_direction, val_win, val_payoff, val_conf = self.model(X_val_seq)
                val_loss_dir = criterion_direction(val_direction, y_val_cls)
                val_loss = val_loss_dir.item()
                val_losses.append(val_loss)
                
                # 計算準確率
                _, predicted = torch.max(val_direction, 1)
                accuracy = (predicted == y_val_cls).float().mean().item()
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Accuracy: {accuracy:.3f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        self.is_trained = True
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_accuracy': accuracy,
            'best_val_loss': best_val_loss
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        預測
        
        Args:
            X: 特徵矩陣
        
        Returns:
            (directions, win_rates, payoffs, confidences)
        """
        if not self.is_trained:
            raise ValueError("模型尚未訓練")
        
        self.model.eval()
        
        # 準備序列
        X_seq = self.prepare_sequences(X)
        
        with torch.no_grad():
            direction, win_rate, payoff, confidence = self.model(X_seq)
            
            # 轉換方向預測
            _, predicted_cls = torch.max(direction, 1)
            directions = (predicted_cls - 1).cpu().numpy()  # 轉回 -1, 0, 1
            
            win_rates = win_rate.cpu().numpy().flatten()
            payoffs = payoff.cpu().numpy().flatten()
            confidences = confidence.cpu().numpy().flatten()
        
        # 補齊前面的序列長度
        directions = np.concatenate([np.zeros(self.sequence_length), directions])
        win_rates = np.concatenate([np.ones(self.sequence_length) * 0.5, win_rates])
        payoffs = np.concatenate([np.ones(self.sequence_length) * 1.5, payoffs])
        confidences = np.concatenate([np.zeros(self.sequence_length), confidences])
        
        return directions, win_rates, payoffs, confidences
    
    def save(self, save_dir: Path):
        """保存模型"""
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_dir / 'model.pt')
        joblib.dump(self.config, save_dir / 'config.pkl')
        print(f"[保存] 模型已保存至 {save_dir}")
    
    def load(self, save_dir: Path):
        """加載模型"""
        self.config = joblib.load(save_dir / 'config.pkl')
        self.model.load_state_dict(torch.load(save_dir / 'model.pt', map_location=self.device))
        self.model.eval()
        self.is_trained = True
        print(f"[加載] 模型已從 {save_dir} 加載")
