# =========================================================
# Smaller CNN + LSTM + Transformer Hybrid (anti-overfit)
# Train on df_trainsub only: split into train/valid
# Save full checkpoint + reload best model
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
    torch.backends.cudnn.benchmark = True

seed_everything(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)


def gpu_mem_gb():
    if DEVICE != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / 1024**3


# =========================================================
# 1) Dataset
# =========================================================
class FactorSequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols,
        target_col="ret_mid_t1",
        asset_col="SecuCode",
        day_col="TradingDay",
        time_col="TimeEnd",
        lookback=20,
        dropna=True,
        standardize=True,
        stats=None,
    ):
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.lookback = int(lookback)
        self.standardize = standardize

        use_cols = [asset_col, day_col, time_col, target_col] + self.feature_cols
        df = df[use_cols].copy()
        df.sort_values([asset_col, day_col, time_col], inplace=True)

        if dropna:
            df = df.dropna(subset=self.feature_cols + [target_col]).copy()

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

        X_list, y_list = [], []
        for _, g in df.groupby(asset_col, sort=False):
            g = g.reset_index(drop=True)
            Xg = g[self.feature_cols].to_numpy(dtype=np.float32)
            yg = g[target_col].to_numpy(dtype=np.float32)

            if len(g) < self.lookback:
                continue

            for i in range(self.lookback - 1, len(g)):
                X_list.append(Xg[i - self.lookback + 1:i + 1])
                y_list.append(yg[i])

        if len(X_list) == 0:
            raise ValueError("No training samples were built. Check lookback / NA / input data.")

        self.X = np.stack(X_list, axis=0)
        self.y = np.asarray(y_list, dtype=np.float32).reshape(-1, 1)

        if self.standardize and stats is None:
            self.stats = (self.mu, self.std)
        else:
            self.stats = stats

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


def time_split_by_day(df, day_col="TradingDay", val_ratio=0.1):
    days = np.array(sorted(df[day_col].unique()))
    n = len(days)
    n_val = int(round(n * val_ratio))

    val_days = set(days[-n_val:]) if n_val > 0 else set()
    train_days = set(days[:-n_val]) if n_val > 0 else set(days)

    df_tr = df[df[day_col].isin(train_days)].copy()
    df_va = df[df[day_col].isin(val_days)].copy()
    return df_tr, df_va


# =========================================================
# 2) Model
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return self.dropout(x)


class CNN_LSTM_Transformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        cnn_channels: int = 32,
        cnn_kernel: int = 3,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        attn_dim: int = 64,
        nhead: int = 4,
        tf_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=n_features,
                out_channels=cnn_channels,
                kernel_size=cnn_kernel,
                padding=cnn_kernel // 2,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=cnn_channels,
                out_channels=cnn_channels,
                kernel_size=cnn_kernel,
                padding=cnn_kernel // 2,
            ),
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
        x = x.transpose(1, 2)   # (B, F, L)
        x = self.cnn(x)         # (B, C, L)
        x = x.transpose(1, 2)   # (B, L, C)
        x = self.cnn_to_lstm(x) # (B, L, attn_dim)

        x, _ = self.lstm(x)     # (B, L, lstm_hidden)
        x = self.proj(x)        # (B, L, attn_dim)
        x = self.pos(x)         # (B, L, attn_dim)
        x = self.tf(x)          # (B, L, attn_dim)

        x_last = x[:, -1, :]
        return self.head(x_last)


# =========================================================
# 3) Eval
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


# =========================================================
# 4) Save / Load helpers
# =========================================================
def save_hybrid_checkpoint(
    path,
    model,
    stats,
    feature_cols,
    target_col,
    lookback,
    model_config,
    extra_info=None,
):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "stats": stats,
        "feature_cols": list(feature_cols),
        "target_col": target_col,
        "lookback": int(lookback),
        "extra_info": extra_info if extra_info is not None else {},
    }
    torch.save(ckpt, path)


def load_hybrid_checkpoint(path, map_location=None):
    if map_location is None:
        map_location = DEVICE

    ckpt = torch.load(path, map_location=map_location)

    model = CNN_LSTM_Transformer(**ckpt["model_config"]).to(map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return {
        "model": model,
        "stats": ckpt["stats"],
        "feature_cols": ckpt["feature_cols"],
        "target_col": ckpt["target_col"],
        "lookback": ckpt["lookback"],
        "model_config": ckpt["model_config"],
        "extra_info": ckpt.get("extra_info", {}),
        "raw_ckpt": ckpt,
    }


# =========================================================
# 5) Train on df_trainsub only (train + valid)
# =========================================================
def train_model_trainvalid_only(
    df_trainsub: pd.DataFrame,
    feature_cols,
    target_col="ret_mid_t1",
    lookback=20,
    batch_size=2048,
    num_workers=0,
    lr=3e-4,
    weight_decay=5e-4,
    epochs=8,
    val_ratio=0.1,
    save_path="best_hybrid_trainsub_small.pt",
    patience=2,
):
    df_tr, df_va = time_split_by_day(df_trainsub, val_ratio=val_ratio)

    print("Train period:",
          df_tr["TradingDay"].min(), "->", df_tr["TradingDay"].max(),
          "| days =", df_tr["TradingDay"].nunique())
    print("Valid period:",
          df_va["TradingDay"].min(), "->", df_va["TradingDay"].max(),
          "| days =", df_va["TradingDay"].nunique())

    ds_tr = FactorSequenceDataset(
        df_tr,
        feature_cols,
        target_col=target_col,
        lookback=lookback,
        standardize=True,
    )
    ds_va = FactorSequenceDataset(
        df_va,
        feature_cols,
        target_col=target_col,
        lookback=lookback,
        standardize=True,
        stats=ds_tr.stats,
    )

    pin = (DEVICE == "cuda")
    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    model_config = {
        "n_features": len(feature_cols),
        "cnn_channels": 32,
        "cnn_kernel": 3,
        "lstm_hidden": 64,
        "lstm_layers": 1,
        "attn_dim": 64,
        "nhead": 4,
        "tf_layers": 1,
        "dropout": 0.2,
    }

    model = CNN_LSTM_Transformer(**model_config).to(DEVICE)

    loss_fn = nn.SmoothL1Loss(beta=1e-3)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_val = float("inf")
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0

        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        pbar = tqdm(dl_tr, desc=f"Epoch {ep:02d}/{epochs} [train]", leave=True)

        for X, y in pbar:
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

            lr_now = optim.param_groups[0]["lr"]
            avg_loss = total / max(n, 1)

            if DEVICE == "cuda":
                mem = gpu_mem_gb()
                pbar.set_postfix(
                    loss=f"{loss.item():.4g}",
                    avg=f"{avg_loss:.4g}",
                    lr=f"{lr_now:.2e}",
                    mem=f"{mem:.2f}GB"
                )
            else:
                pbar.set_postfix(
                    loss=f"{loss.item():.4g}",
                    avg=f"{avg_loss:.4g}",
                    lr=f"{lr_now:.2e}"
                )

        scheduler.step()

        tr_loss = total / max(n, 1)
        va_loss, va_metrics = eval_epoch(model, dl_va, loss_fn)

        dt = time.time() - t0
        speed = n / max(dt, 1e-9)

        print(
            f"Epoch {ep:02d} done | time={dt:.1f}s | speed={speed:.0f} samples/s "
            f"| train_loss={tr_loss:.6f} | val_loss={va_loss:.6f} "
            f"| val_mse={va_metrics['mse']:.6g} "
            f"val_mae={va_metrics['mae']:.6g} "
            f"val_corr={va_metrics['corr']:.4f}"
        )

        if va_loss < best_val:
            best_val = va_loss
            bad_epochs = 0

            save_hybrid_checkpoint(
                path=save_path,
                model=model,
                stats=ds_tr.stats,
                feature_cols=feature_cols,
                target_col=target_col,
                lookback=lookback,
                model_config=model_config,
                extra_info={
                    "best_val_loss": float(best_val),
                    "train_days": int(df_tr["TradingDay"].nunique()),
                    "valid_days": int(df_va["TradingDay"].nunique()),
                    "epoch": ep,
                },
            )
            print(f"  ✅ saved best -> {save_path}")
        else:
            bad_epochs += 1
            print(f"  no improvement. bad_epochs = {bad_epochs}/{patience}")

        if bad_epochs >= patience:
            print("  early stopping triggered.")
            break

    loaded = load_hybrid_checkpoint(save_path, map_location=DEVICE)
    model_loaded = loaded["model"]

    va_loss_best, va_metrics_best = eval_epoch(model_loaded, dl_va, loss_fn)
    print(
        f"[RELOADED VALID] loss={va_loss_best:.6f} | "
        f"mse={va_metrics_best['mse']:.6g} "
        f"mae={va_metrics_best['mae']:.6g} "
        f"corr={va_metrics_best['corr']:.4f}"
    )

    return model_loaded, loaded, (ds_tr, ds_va), (df_tr, df_va)