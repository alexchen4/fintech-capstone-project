# =========================================================
# CNN + LSTM + Transformer Hybrid (PyTorch, GPU)
# For df_train (long 15min bars with factors) + Progress Bar
# =========================================================
import os
import math
import random
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# -----------------------------
# Reproducibility
# -----------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # faster
seed_everything(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

def gpu_mem_gb():
    if DEVICE != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / 1024**3


# =========================================================
# 1) Build sequence dataset from df_train
# =========================================================
class FactorSequenceDataset(Dataset):
    """
    Build samples:
      X: (lookback, n_features)
      y: scalar (target is already aligned with current row, e.g. ret_mid_t1)

    For a window ending at index i (current time i), label is y[i].
    X uses rows [i-lookback+1 ... i].
    """
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols,
        target_col="ret_mid_t1",
        asset_col="SecuCode",
        day_col="TradingDay",
        time_col="TimeEnd",
        lookback=32,
        dropna=True,
        standardize=True,
        stats=None,  # (mu, std) computed on train only
    ):
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.lookback = int(lookback)
        self.standardize = standardize

        use_cols = [asset_col, day_col, time_col, target_col] + self.feature_cols
        df = df[use_cols].copy()

        # sort inside each asset
        df.sort_values([asset_col, day_col, time_col], inplace=True)

        # optional drop NA
        if dropna:
            df = df.dropna(subset=self.feature_cols + [target_col]).copy()

        # standardization (fit on train only)
        feats = df[self.feature_cols].to_numpy(dtype=np.float32)
        if self.standardize:
            if stats is None:
                mu = feats.mean(axis=0, keepdims=True)
                std = feats.std(axis=0, keepdims=True)
                std = np.where(std < 1e-6, 1.0, std)
                self.mu = mu.astype(np.float32)
                self.std = std.astype(np.float32)
            else:
                self.mu, self.std = stats
            feats = (feats - self.mu) / self.std
            df.loc[:, self.feature_cols] = feats

        # group by asset -> sliding windows
        X_list, y_list = [], []
        for _, g in df.groupby(asset_col, sort=False):
            g = g.reset_index(drop=True)
            Xg = g[self.feature_cols].to_numpy(dtype=np.float32)
            yg = g[target_col].to_numpy(dtype=np.float32)

            if len(g) < self.lookback:
                continue

            for i in range(self.lookback - 1, len(g)):
                X_list.append(Xg[i - self.lookback + 1 : i + 1])  # (L, F)
                y_list.append(yg[i])                               # scalar

        self.X = np.stack(X_list, axis=0)  # (N, L, F)
        self.y = np.asarray(y_list, dtype=np.float32).reshape(-1, 1)

        # keep stats for reuse
        if self.standardize and stats is None:
            self.stats = (self.mu, self.std)
        else:
            self.stats = stats

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


def time_split_by_day(df, day_col="TradingDay", val_ratio=0.1, test_ratio=0.1):
    """
    Split by unique day to avoid leakage across time.
    """
    days = np.array(sorted(df[day_col].unique()))
    n = len(days)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))

    test_days = set(days[-n_test:]) if n_test > 0 else set()
    val_days = set(days[-(n_test + n_val):-n_test]) if n_val > 0 else set()
    train_days = set(days[: -(n_test + n_val)]) if (n_test + n_val) > 0 else set(days)

    df_tr = df[df[day_col].isin(train_days)].copy()
    df_va = df[df[day_col].isin(val_days)].copy()
    df_te = df[df[day_col].isin(test_days)].copy()
    return df_tr, df_va, df_te


# =========================================================
# 2) Model: CNN -> LSTM -> Transformer -> Head
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return self.dropout(x)


class CNN_LSTM_Transformer(nn.Module):
    """
    Input:  X (B, L, F)
    Output: y_hat (B, 1)
    """
    def __init__(
        self,
        n_features: int,
        cnn_channels: int = 64,
        cnn_kernel: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        attn_dim: int = 128,
        nhead: int = 4,
        tf_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=cnn_channels, kernel_size=cnn_kernel, padding=cnn_kernel//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=cnn_kernel, padding=cnn_kernel//2),
            nn.GELU(),
        )

        self.cnn_to_lstm = nn.Linear(cnn_channels, attn_dim)

        self.lstm = nn.LSTM(
            input_size=attn_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.proj = nn.Linear(lstm_hidden, attn_dim)
        self.pos = PositionalEncoding(attn_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attn_dim,
            nhead=nhead,
            dim_feedforward=4 * attn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tf = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(attn_dim),
            nn.Linear(attn_dim, attn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, x):
        # x: (B, L, F)
        x = x.transpose(1, 2)        # (B, F, L)
        x = self.cnn(x)              # (B, C, L)
        x = x.transpose(1, 2)        # (B, L, C)
        x = self.cnn_to_lstm(x)      # (B, L, attn_dim)

        x, _ = self.lstm(x)          # (B, L, lstm_hidden)
        x = self.proj(x)             # (B, L, attn_dim)
        x = self.pos(x)              # (B, L, attn_dim)
        x = self.tf(x)               # (B, L, attn_dim)

        x_last = x[:, -1, :]         # (B, attn_dim)
        return self.head(x_last)     # (B, 1)


# =========================================================
# 3) Train / Eval loops (AMP + GPU) + Progress
# =========================================================
@torch.no_grad()
def eval_epoch(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    n = 0
    preds_all, y_all = [], []
    for X, y in loader:
        X = X.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        y_hat = model(X)
        loss = loss_fn(y_hat, y)

        bs = X.size(0)
        total_loss += loss.item() * bs
        n += bs

        preds_all.append(y_hat.detach().cpu())
        y_all.append(y.detach().cpu())

    preds = torch.cat(preds_all, dim=0).numpy().reshape(-1)
    ys = torch.cat(y_all, dim=0).numpy().reshape(-1)

    mse = float(((preds - ys) ** 2).mean())
    mae = float(np.abs(preds - ys).mean())
    corr = float(np.corrcoef(preds, ys)[0, 1]) if len(preds) > 3 else np.nan
    return total_loss / max(n, 1), {"mse": mse, "mae": mae, "corr": corr}


def train_model(
    df_train: pd.DataFrame,
    feature_cols,
    target_col="ret_mid_t1",
    lookback=32,
    batch_size=2048,
    num_workers=0,     # ✅ Windows/Notebook 更稳；你要加速再调 2/4/8
    lr=3e-4,
    weight_decay=1e-4,
    epochs=10,
    val_ratio=0.1,
    test_ratio=0.1,
):
    # split by day
    df_tr, df_va, df_te = time_split_by_day(df_train, val_ratio=val_ratio, test_ratio=test_ratio)

    # datasets (fit scaler on train only)
    ds_tr = FactorSequenceDataset(df_tr, feature_cols, target_col=target_col, lookback=lookback, standardize=True)
    ds_va = FactorSequenceDataset(df_va, feature_cols, target_col=target_col, lookback=lookback, standardize=True, stats=ds_tr.stats)
    ds_te = FactorSequenceDataset(df_te, feature_cols, target_col=target_col, lookback=lookback, standardize=True, stats=ds_tr.stats)

    pin = (DEVICE == "cuda")
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    model = CNN_LSTM_Transformer(n_features=len(feature_cols)).to(DEVICE)

    loss_fn = nn.SmoothL1Loss(beta=1e-3)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_val = float("inf")
    best_path = "best_hybrid.pt"

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0

        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        pbar = tqdm(dl_tr, desc=f"Epoch {ep:02d}/{epochs} [train]", leave=True)

        for step, (X, y) in enumerate(pbar, start=1):
            X = X.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                y_hat = model(X)
                loss = loss_fn(y_hat, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()

            bs = X.size(0)
            total += loss.item() * bs
            n += bs

            # live stats
            lr_now = optim.param_groups[0]["lr"]
            avg_loss = total / max(n, 1)

            if DEVICE == "cuda":
                mem = gpu_mem_gb()
                pbar.set_postfix(loss=f"{loss.item():.4g}", avg=f"{avg_loss:.4g}", lr=f"{lr_now:.2e}", mem=f"{mem:.2f}GB")
            else:
                pbar.set_postfix(loss=f"{loss.item():.4g}", avg=f"{avg_loss:.4g}", lr=f"{lr_now:.2e}")

        scheduler.step()

        tr_loss = total / max(n, 1)
        va_loss, va_metrics = eval_epoch(model, dl_va, loss_fn)

        dt = time.time() - t0
        speed = n / max(dt, 1e-9)

        print(
            f"Epoch {ep:02d} done | time={dt:.1f}s | speed={speed:.0f} samples/s "
            f"| train_loss={tr_loss:.6f} | val_loss={va_loss:.6f} "
            f"| val_mse={va_metrics['mse']:.6g} val_mae={va_metrics['mae']:.6g} val_corr={va_metrics['corr']:.4f}"
        )

        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(), "stats": ds_tr.stats, "feature_cols": list(feature_cols)}, best_path)
            print(f"  ✅ saved best -> {best_path}")

    # test best
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    te_loss, te_metrics = eval_epoch(model, dl_te, loss_fn)
    print(f"[TEST] loss={te_loss:.6f} | mse={te_metrics['mse']:.6g} mae={te_metrics['mae']:.6g} corr={te_metrics['corr']:.4f}")

    return model, ckpt, (ds_tr, ds_va, ds_te)


# =========================================================
# 4) USAGE
# =========================================================
feature_cols = [c for c in df_train.columns if c.startswith("f_")]
target_col = "ret_mid_t1"   # or "log_mid_ret_t1"

model, ckpt, dsets = train_model(
    df_train=df_train,
    feature_cols=feature_cols,
    target_col=target_col,
    lookback=32,
    batch_size=2048,
    num_workers=0,   # 
    lr=3e-4,
    weight_decay=1e-4,
    epochs=10,
    val_ratio=0.1,
    test_ratio=0.1,
)