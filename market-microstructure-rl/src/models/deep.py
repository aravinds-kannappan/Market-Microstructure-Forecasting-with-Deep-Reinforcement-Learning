"""
deep.py
=======
Temporal Convolutional Network (TCN) classifier for LOB sequences.
"""
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.seq_len = seq_len
    def __len__(self): return max(0, self.X.shape[0] - self.seq_len)
    def __getitem__(self, idx):
        x = self.X[idx: idx+self.seq_len].T
        label = self.y[idx + self.seq_len - 1]
        return torch.from_numpy(x), torch.tensor(label)

class Chomp1d(nn.Module):
    def __init__(self, s): super().__init__(); self.s = s
    def forward(self, x): return x[:, :, :-self.s].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, d, dropout):
        super().__init__()
        pad = (k - 1) * d
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d), Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d), Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x):
        out = self.net(x); res = x if self.down is None else self.down(x); return torch.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_features: int, channels=(64,64,64), kernel_size=3, dropout=0.1, num_classes=3):
        super().__init__()
        layers = []; in_ch = num_features
        for i, ch in enumerate(channels):
            layers.append(TemporalBlock(in_ch, ch, kernel_size, dilation=2**i, dropout=dropout)); in_ch = ch
        self.tcn = nn.Sequential(*layers)
        self.cls = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(in_ch, num_classes))
    def forward(self, x):
        return self.cls(self.tcn(x))

def train_tcn(X: pd.DataFrame, y: pd.Series, seq_len: int = 64, epochs: int = 10, batch_size: int = 128, lr: float = 1e-3, device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    split = int(0.8 * len(X))
    Xtr, Xte = X.iloc[:split].values, X.iloc[split:].values
    ytr, yte = y.iloc[:split].values, y.iloc[split:].values
    dtr, dte = SequenceDataset(Xtr, ytr, seq_len), SequenceDataset(Xte, yte, seq_len)
    ltr = DataLoader(dtr, batch_size=batch_size, shuffle=True, drop_last=True)
    lte = DataLoader(dte, batch_size=batch_size, shuffle=False, drop_last=False)
    model = TCN(num_features=X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train(); tot = 0.0
        for xb, yb in ltr:
            xb, yb = xb.to(device), (yb + 1).to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += loss.item()
        # simple validation
        model.eval(); correct = n = 0
        with torch.no_grad():
            for xb, yb in lte:
                xb = xb.to(device); logits = model(xb); preds = logits.argmax(dim=1) - 1
                correct += (preds.cpu() == yb).sum().item(); n += yb.numel()
        print(f"[TCN] epoch {ep+1}/{epochs} loss={tot/max(1,len(ltr)):.4f} val_acc={correct/max(1,n):.4f}")
    return model
