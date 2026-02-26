"""
Transformer Model
基於Transformer的時序模式識別模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import joblib
from pathlib import Path

class PositionalEncoding(nn.Module):
    """位置編碼 - 為序列加入位置信息"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class PriceTransformer(nn.Module):
    """
    價格Transformer模型
    輸入: (batch, sequence_length, features)
    輸出: (batch, 3) - [LONG, NEUTRAL, SHORT]機率分布
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.input_dim = config.get('input_dim', 100)
        self.d_model = config.get('d_model', 128)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 4)
        self.dim_feedforward = config.get('dim_feedforward', 512)
        self.dropout = config.get('dropout', 0.1)
        self.sequence_length = config.get('sequence_length', 30)
        
        # 輸入投影
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        
        # 位置編碼
        self.pos_encoder = PositionalEncoding(
            self.d_model, 
            max_len=self.sequence_length,
            dropout=self.dropout
        )
        
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
        
        # 輸出頭
        self.fc1 = nn.Linear(self.d_model, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # LONG, NEUTRAL, SHORT
        
        self.dropout_layer = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播
        輸入: x (batch, seq_len, features)
        輸出: logits (batch, 3), attention_weights
        """
        # 輸入投影
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # 位置編碼
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # Transformer編碼
        encoded = self.transformer_encoder(x, mask=mask)  # (batch, seq_len, d_model)
        
        # 取最後一個時間步的輸出
        last_output = encoded[:, -1, :]  # (batch, d_model)
        
        # 全連接層
        x = self.layer_norm(last_output)
        x = F.relu(self.fc1(x))
        x = self.dropout_layer(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_layer(x)
        logits = self.fc3(x)  # (batch, 3)
        
        return logits, encoded
    
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """預測機率分布"""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = F.softmax(logits, dim=-1)
        return probs.cpu().numpy()
    
    def get_attention_weights(self, x: torch.Tensor) -> np.ndarray:
        """獲取注意力權重 - 用於解釋性"""
        self.eval()
        with torch.no_grad():
            _, encoded = self.forward(x)
        return encoded.cpu().numpy()

class TransformerPredictor:
    """
Transformer預測器封裝類
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        
    def build_model(self, input_dim: int):
        """\u5efa立模型"""
        config = self.config.copy()
        config['input_dim'] = input_dim
        
        self.model = PriceTransformer(config).to(self.device)
        print(f"模型已建立在 {self.device}")
        print(f"參數數量: {sum(p.numel() for p in self.model.parameters())}")
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001) -> Dict:
        """訓練模型"""
        from sklearn.preprocessing import StandardScaler
        from torch.utils.data import TensorDataset, DataLoader
        
        # 標準化
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(
            X_train.reshape(-1, X_train.shape[-1])
        ).reshape(X_train.shape)
        X_val_scaled = self.scaler.transform(
            X_val.reshape(-1, X_val.shape[-1])
        ).reshape(X_val.shape)
        
        # 轉換Tensor
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        # DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 優化器
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 損失函數 - 使用類別權重處理不平衡
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # 訓練迴圈
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 訓練
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                logits, _ = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 驗證
            self.model.eval()
            with torch.no_grad():
                val_logits, _ = self.model(X_val_tensor)
                val_loss = criterion(val_logits, y_val_tensor).item()
                val_preds = torch.argmax(val_logits, dim=-1)
                val_acc = (val_preds == y_val_tensor).float().mean().item()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
            
            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"早停於 Epoch {epoch+1}")
                    break
        
        return history
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """預測"""
        X_scaled = self.scaler.transform(
            X.reshape(-1, X.shape[-1])
        ).reshape(X.shape)
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        probs = self.model.predict_proba(X_tensor)
        preds = np.argmax(probs, axis=1)
        
        return preds, probs
    
    def save(self, path: Path):
        """保存模型"""
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config,
            'scaler': self.scaler
        }, path / 'transformer_model.pth')
        print(f"Transformer模型已保存至 {path}")
    
    def load(self, path: Path):
        """加載模型"""
        checkpoint = torch.load(path / 'transformer_model.pth', map_location=self.device)
        self.config = checkpoint['config']
        self.scaler = checkpoint['scaler']
        
        input_dim = self.config['input_dim']
        self.model = PriceTransformer(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        print(f"Transformer模型已加載從 {path}")
