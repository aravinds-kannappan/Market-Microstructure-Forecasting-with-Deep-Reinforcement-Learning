"""
advanced_models.py
==================
State-of-the-art deep learning architectures for LOB forecasting.
Implements DeepLOB (CNN-LSTM), TransLOB (Transformer), and enhanced TCN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict, List
import math

class LOBDataset(Dataset):
    """PyTorch dataset for LOB sequences"""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels + 1)  # Shift to 0-2 for CrossEntropy
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class DeepLOB(nn.Module):
    """
    DeepLOB architecture from Zhang et al.
    Combines CNN for spatial features with LSTM for temporal dynamics.
    """
    
    def __init__(self, 
                 n_features: int = 40,  # FI-2010 uses 40 raw features
                 seq_length: int = 100,
                 n_classes: int = 3):
        super().__init__()
        
        # Convolutional blocks for spatial feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 10)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        
        # Inception module for multi-scale features
        self.inception = InceptionModule(32, 64)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(64, n_classes)
        
    def forward(self, x):
        # Reshape to (batch, 1, seq_len, features)
        x = x.unsqueeze(1)
        
        # CNN feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Inception module
        x = self.inception(x)
        
        # Reshape for LSTM: (batch, seq, features)
        x = x.squeeze(2).permute(0, 2, 1)
        
        # LSTM
        x, _ = self.lstm(x)
        
        # Take last timestep
        x = x[:, -1, :]
        
        # Classification
        return self.fc(x)


class InceptionModule(nn.Module):
    """Inception module for multi-scale feature extraction"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        
        # 1x1 -> 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(3, 1), padding=(1, 0))
        )
        
        # 1x1 -> 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(5, 1), padding=(2, 0))
        )
        
        # MaxPool -> 1x1 convolution branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        )
        
    def forward(self, x):
        outputs = [
            self.branch1x1(x),
            self.branch3x3(x),
            self.branch5x5(x),
            self.branch_pool(x)
        ]
        return torch.cat(outputs, dim=1)


class TransLOB(nn.Module):
    """
    Transformer-based architecture for LOB modeling.
    Inspired by TransLOB paper - uses attention mechanisms for temporal dependencies.
    """
    
    def __init__(self,
                 n_features: int = 144,
                 seq_length: int = 100,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 1024,
                 dropout: float = 0.1,
                 n_classes: int = 3):
        super().__init__()
        
        self.seq_length = seq_length
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout, seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Attention pooling
        self.attention_pool = AttentionPooling(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
    def forward(self, x):
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Attention pooling instead of just taking last timestep
        x = self.attention_pool(x)
        
        # Classification
        return self.classifier(x)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class AttentionPooling(nn.Module):
    """Attention-based pooling layer"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x shape: (batch, seq, d_model)
        weights = F.softmax(self.attention(x), dim=1)
        return torch.sum(weights * x, dim=1)


class EnhancedTCN(nn.Module):
    """
    Enhanced Temporal Convolutional Network with residual blocks and attention.
    Improved version of the basic TCN with ideas from WaveNet.
    """
    
    def __init__(self,
                 n_features: int = 144,
                 n_classes: int = 3,
                 num_channels: List[int] = None,
                 kernel_size: int = 7,
                 dropout: float = 0.2,
                 attention: bool = True):
        super().__init__()
        
        if num_channels is None:
            num_channels = [64, 128, 256, 256, 128, 64]
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = n_features if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size,
                    dropout=dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
        
        # Global attention layer
        self.attention = AttentionPooling(num_channels[-1]) if attention else None
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1] // 2, n_classes)
        )
        
    def forward(self, x):
        # x shape: (batch, seq, features)
        x = x.transpose(1, 2)  # (batch, features, seq)
        x = self.network(x)
        x = x.transpose(1, 2)  # (batch, seq, features)
        
        if self.attention:
            x = self.attention(x)
        else:
            x = x[:, -1, :]  # Take last timestep
            
        return self.classifier(x)


class TemporalBlock(nn.Module):
    """Residual temporal block for TCN"""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
        
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
            
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove trailing padding"""
    
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class LOBCASTModel(nn.Module):
    """
    LOBCAST-inspired architecture combining CNN, LSTM with attention mechanisms.
    Includes volatility and market regime awareness.
    """
    
    def __init__(self,
                 n_features: int = 144,
                 seq_length: int = 100,
                 n_classes: int = 3,
                 hidden_dim: int = 128,
                 n_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        # Feature extraction CNN
        self.feature_cnn = nn.Sequential(
            nn.Conv1d(n_features, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Bi-LSTM with attention
        self.lstm = nn.LSTM(
            hidden_dim * 2,
            hidden_dim,
            n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            hidden_dim * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Market regime detector (volatility clustering)
        self.regime_detector = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Low, Medium, High volatility regimes
        )
        
        # Final classifier with regime conditioning
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract volatility regime
        regime_features = self.regime_detector(x.mean(dim=1))
        regime_probs = F.softmax(regime_features, dim=-1)
        
        # CNN feature extraction
        x_cnn = x.transpose(1, 2)  # (batch, features, seq)
        x_cnn = self.feature_cnn(x_cnn)
        x_cnn = x_cnn.transpose(1, 2)  # (batch, seq, features)
        
        # LSTM encoding
        x_lstm, _ = self.lstm(x_cnn)
        
        # Self-attention
        x_att, _ = self.self_attention(x_lstm, x_lstm, x_lstm)
        
        # Global pooling
        x_pooled = x_att.mean(dim=1)
        
        # Combine with regime information
        x_combined = torch.cat([x_pooled, regime_probs], dim=-1)
        
        # Classification
        return self.classifier(x_combined)


class ModelTrainer:
    """Unified trainer for all LOB models"""
    
    def __init__(self,
                 model: nn.Module,
                 device: str = None,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=1e-6
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
        return total_loss / len(dataloader), correct / total
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float, Dict]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import f1_score, confusion_matrix
        f1 = f1_score(all_labels, all_preds, average='macro')
        cm = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels)
        }
        
        return total_loss / len(dataloader), correct / total, metrics
    
    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int = 50,
            early_stopping_patience: int = 10):
        """Full training loop with early stopping"""
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_metrics = self.validate(val_loader)
            
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_metrics['f1_score']:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        return self.model
