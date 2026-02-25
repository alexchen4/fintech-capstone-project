# =========================================================
# PURE PyTorch S4-style + Transformer
# No mamba, no extra install
# Save best_ssm_tf.pt
# =========================================================

import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_everything(42)

# =========================================================
# Dataset
# =========================================================
class FactorSequenceDataset(Dataset):
    def __init__(self, df, feature_cols,
                 target_col="ret_mid_t1",
                 asset_col="SecuCode",
                 day_col="TradingDay",
                 time_col="TimeEnd",
                 lookback=32,
                 stats=None):

        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lookback = lookback

        df = df[[asset_col, day_col, time_col, target_col] + feature_cols].copy()
        df.sort_values([asset_col, day_col, time_col], inplace=True)
        df.dropna(inplace=True)

        feats = df[feature_cols].values.astype(np.float32)

        if stats is None:
            mu = feats.mean(0, keepdims=True)
            std = feats.std(0, keepdims=True)
            std = np.where(std < 1e-6, 1.0, std)
            self.stats = (mu.astype(np.float32), std.astype(np.float32))
        else:
            self.stats = stats
            mu, std = stats

        feats = (feats - mu) / std
        df[feature_cols] = feats

        X_list, y_list = [], []

        for _, g in df.groupby(asset_col):
            g = g.reset_index(drop=True)
            Xg = g[feature_cols].values
            yg = g[target_col].values

            if len(g) < lookback:
                continue

            for i in range(lookback-1, len(g)):
                X_list.append(Xg[i-lookback+1:i+1])
                y_list.append(yg[i])

        self.X = np.array(X_list, dtype=np.float32)
        self.y = np.array(y_list, dtype=np.float32).reshape(-1,1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# =========================================================
# Positional Encoding
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) *
                        (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# =========================================================
# Simple S4-style State Space Layer
# =========================================================
class SimpleSSM(nn.Module):
    """
    Discrete linear state space:
        h_{t+1} = A h_t + B x_t
        y_t = C h_t
    Implemented with parallel scan approximation
    """
    def __init__(self, d_model):
        super().__init__()
        self.A = nn.Parameter(torch.randn(d_model))
        self.B = nn.Linear(d_model, d_model)
        self.C = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, L, D)
        Bsz, L, D = x.shape
        h = torch.zeros(Bsz, D, device=x.device)

        outputs = []
        for t in range(L):
            h = torch.tanh(self.A * h + self.B(x[:,t]))
            outputs.append(self.C(h).unsqueeze(1))

        y = torch.cat(outputs, dim=1)
        return y


# =========================================================
# S4 + Transformer Model
# =========================================================
class SSM_Transformer(nn.Module):
    def __init__(self, n_features,
                 d_model=128,
                 ssm_layers=2,
                 tf_layers=2,
                 nhead=4):

        super().__init__()

        self.config = dict(
            d_model=d_model,
            ssm_layers=ssm_layers,
            tf_layers=tf_layers,
            nhead=nhead,
        )

        self.in_proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)

        self.ssm_stack = nn.Sequential(*[
            SimpleSSM(d_model) for _ in range(ssm_layers)
        ])

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True,
            norm_first=True
        )

        self.tf = nn.TransformerEncoder(enc_layer,
                                        num_layers=tf_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model,1)
        )

    def forward(self, x):
        x = self.in_proj(x)
        x = self.pos(x)
        x = self.ssm_stack(x)
        x = self.tf(x)
        return self.head(x[:,-1])


# =========================================================
# Training
# =========================================================
def train_ssm_model(df_train, feature_cols,
                    target_col="ret_mid_t1",
                    lookback=32,
                    epochs=10,
                    batch_size=2048):

    ds = FactorSequenceDataset(df_train,
                               feature_cols,
                               target_col=target_col,
                               lookback=lookback)

    dl = DataLoader(ds, batch_size=batch_size,
                    shuffle=True, num_workers=0)

    model = SSM_Transformer(len(feature_cols)).to(DEVICE)

    loss_fn = nn.SmoothL1Loss()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    best_loss = 1e18

    for ep in range(1, epochs+1):
        model.train()
        total = 0

        pbar = tqdm(dl, desc=f"Epoch {ep}/{epochs}")

        for X,y in pbar:
            X,y = X.to(DEVICE), y.to(DEVICE)

            optim.zero_grad()
            yhat = model(X)
            loss = loss_fn(yhat, y)
            loss.backward()
            optim.step()

            total += loss.item()*X.size(0)
            pbar.set_postfix(loss=loss.item())

        epoch_loss = total/len(ds)
        print("Epoch loss:", epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                "model_state": model.state_dict(),
                "stats": ds.stats,
                "feature_cols": feature_cols,
                "config": model.config
            }, "best_ssm_tf.pt")
            print("✅ Saved best_ssm_tf.pt")

    return model


# =========================================================
# RUN
# =========================================================
feature_cols = [c for c in df_train.columns if c.startswith("f_")]

model = train_ssm_model(
    df_train=df_train,
    feature_cols=feature_cols,
    target_col="ret_mid_t1",
    lookback=32,
    epochs=10,
    batch_size=2048
)