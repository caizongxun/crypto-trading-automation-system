"""
Transformer Model for Time-Series Prediction
Transformer時序預測模型
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import joblib
from pathlib import Path

class PositionalEncoding(nn.Module):
    """位置編碼"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerPredictor(nn.Module):
    """Transformer預測模型"""
    def __init__(self, config: Dict):
        super().__init__()
        
        self.feature_dim = config.get('feature_dim', 50)
        self.d_model = config.get('d_model', 128)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 4)
        self.dim_feedforward = config.get('dim_feedforward', 512)
        self.dropout = config.get('dropout', 0.1)
        
        # 輸入投影
        self.input_projection = nn.Linear(self.feature_dim, self.d_model)
        
        # 位置編碼
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # 輸出層
        self.fc1 = nn.Linear(self.d_model, 64)
        self.dropout1 = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(self.dropout)
        self.fc3 = nn.Linear(32, 3)  # 3類: 做多/中性/做空
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # 取最後一個時間步
        x = x[:, -1, :]
        
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

class TransformerTrainer:
    """訓練Transformer模型"""
    def __init__(self, config: Dict):
        self.config = config
        self.model = TransformerPredictor(config)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0.01)
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 50, batch_size: int = 32) -> Dict:
        """訓練模型"""
        # 轉換為Tensor
        X_train = torch.FloatTensor(X_train).to(self.model.device)
        y_train = torch.LongTensor(y_train + 1).to(self.model.device)  # 0,1,2
        X_val = torch.FloatTensor(X_val).to(self.model.device)
        y_val = torch.LongTensor(y_val + 1).to(self.model.device)
        
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # 訓練
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 驗證
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = self.criterion(val_outputs, y_val)
                val_pred = torch.argmax(val_outputs, dim=1)
                val_acc = (val_pred == y_val).float().mean()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss.item())
            history['val_acc'].append(val_acc.item())
            
            self.scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss={train_loss:.4f}, "
                      f"Val Loss={val_loss:.4f}, "
                      f"Val Acc={val_acc:.4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """預測"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.model.device)
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1) - 1  # 轉回-1,0,1
        
        return predictions.cpu().numpy(), probs.cpu().numpy()
    
    def save(self, path: Path):
        """保存模型"""
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config
        }, path / 'transformer_model.pt')
    
    def load(self, path: Path):
        """加載模型"""
        checkpoint = torch.load(path / 'transformer_model.pt', 
                               map_location=self.model.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
