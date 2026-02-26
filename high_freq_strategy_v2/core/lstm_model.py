"""
LSTM Model
長短期記憶模型 - 捕捉長期依賴
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from pathlib import Path

class PriceLSTM(nn.Module):
    """
    價格LSTM模型
    輸入: (batch, sequence_length, features)
    輸出: (batch, 3) - [LONG, NEUTRAL, SHORT]
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.input_dim = config.get('input_dim', 100)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.2)
        self.bidirectional = config.get('bidirectional', True)
        
        # LSTM層
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        
        # 輸出維度
        lstm_output_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        
        # 全連接層
        self.fc1 = nn.Linear(lstm_output_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向傳播"""
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 取最後一個時間步
        last_output = lstm_out[:, -1, :]
        
        # 全連接層
        x = self.layer_norm(last_output)
        x = torch.relu(self.fc1(x))
        x = self.dropout_layer(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout_layer(x)
        logits = self.fc3(x)
        
        return logits, lstm_out

class LSTMPredictor:
    """
LSTM預測器封裝"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
    
    def build_model(self, input_dim: int):
        config = self.config.copy()
        config['input_dim'] = input_dim
        self.model = PriceLSTM(config).to(self.device)
        print(f"LSTM模型已建立, 參數: {sum(p.numel() for p in self.model.parameters())}")
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=0.001):
        from sklearn.preprocessing import StandardScaler
        from torch.utils.data import TensorDataset, DataLoader
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(
            X_train.reshape(-1, X_train.shape[-1])
        ).reshape(X_train.shape)
        X_val_scaled = self.scaler.transform(
            X_val.reshape(-1, X_val.shape[-1])
        ).reshape(X_val.shape)
        
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        class_counts = np.bincount(y_train)
        class_weights = torch.FloatTensor(1.0 / (class_counts + 1e-6)).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        patience = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                logits, _ = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            self.model.eval()
            with torch.no_grad():
                val_logits, _ = self.model(X_val_tensor)
                val_loss = criterion(val_logits, y_val_tensor).item()
                val_acc = (torch.argmax(val_logits, -1) == y_val_tensor).float().mean().item()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            scheduler.step(val_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= 10:
                    break
        
        return history
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(X_tensor)
            probs = torch.softmax(logits, -1).cpu().numpy()
        return np.argmax(probs, axis=1), probs
    
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config,
            'scaler': self.scaler
        }, path / 'lstm_model.pth')
    
    def load(self, path: Path):
        checkpoint = torch.load(path / 'lstm_model.pth', map_location=self.device)
        self.config = checkpoint['config']
        self.scaler = checkpoint['scaler']
        self.model = PriceLSTM(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
