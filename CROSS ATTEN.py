import math
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)


# =========================================================
# 0) Seed
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)


# =========================================================
# 1) Model
# =========================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CrossAttentionBlock(nn.Module):
    """
    Cross-sectional self-attention over all stocks at the same time slice.
    Input: (B, D)
    """
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, D)
        z = self.norm1(x).unsqueeze(0)   # (1, B, D)
        attn_out, _ = self.attn(z, z, z, need_weights=False)
        x = x + attn_out.squeeze(0)

        z2 = self.norm2(x)
        x = x + self.ffn(z2)
        return x


class ResidualMLPHead(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x):
        z = self.norm(x)
        h = self.fc1(z)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        z = z + h
        return self.out(z)


class TransformerCross(nn.Module):
    """
    Input:
        X: (B, L, F)
        B = number of stocks in the cross-section at the same time point
        L = lookback window length
        F = number of features

    Pipeline:
        1) Temporal transformer (time dimension)
        2) Learnable pooling (last token + mean)
        3) 2-layer cross-sectional cross-attention
        4) Residual MLP head
    """
    def __init__(
        self,
        n_features,
        d_model=64,
        nhead=4,
        time_layers=1,
        cross_heads=4,
        cross_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.config = dict(
            n_features=n_features,
            d_model=d_model,
            nhead=nhead,
            time_layers=time_layers,
            cross_heads=cross_heads,
            cross_layers=cross_layers,
            dropout=dropout,
        )

        self.in_proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.time_tf = nn.TransformerEncoder(enc_layer, num_layers=time_layers)

        self.pool_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(
                d_model=d_model,
                nhead=cross_heads,
                dropout=dropout,
            )
            for _ in range(cross_layers)
        ])

        self.head = ResidualMLPHead(d_model=d_model, dropout=dropout)

    def forward(self, X):
        # X: (B, L, F)
        x = self.in_proj(X)
        x = self.pos(x)
        x = self.time_tf(x)

        last = x[:, -1, :]
        mean = x.mean(dim=1)
        alpha = self.pool_gate(torch.cat([last, mean], dim=-1))
        x = alpha * last + (1.0 - alpha) * mean

        for blk in self.cross_blocks:
            x = blk(x)

        y = self.head(x)
        return y


# =========================================================
# 2) Utils
# =========================================================
def split_train_val_by_day(df, day_col="TradingDay", val_ratio=0.2):
    days = np.array(sorted(df[day_col].unique()))
    n_total = len(days)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_train = n_total - n_val

    train_days = set(days[:n_train])
    val_days = set(days[n_train:])

    df_train = df[df[day_col].isin(train_days)].copy()
    df_val   = df[df[day_col].isin(val_days)].copy()

    print(f"train day range: {min(train_days)} -> {max(train_days)} ({len(train_days)} days)")
    print(f"val   day range: {min(val_days)} -> {max(val_days)} ({len(val_days)} days)")

    return df_train, df_val


def safe_corr_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


def safe_corr_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    xr = pd.Series(x).rank(method="average").values
    yr = pd.Series(y).rank(method="average").values
    return safe_corr_pearson(xr, yr)


# =========================================================
# 3) Dataset: Train / Valid
# Input window = [t-lookback+1, ..., t]
# =========================================================
class TimeConsistentSeqDataset(Dataset):
    def __init__(
        self,
        df,
        feature_cols,
        target_col="ret_mid_t1",
        lookback=32,
        asset_col="SecuCode",
        day_col="TradingDay",
        time_col="TimeEnd",
        cs_zscore=True,
        eps=1e-6,
    ):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lookback = lookback
        self.asset_col = asset_col
        self.day_col = day_col
        self.time_col = time_col

        df = df.copy()

        if cs_zscore:
            grp = df.groupby([day_col, time_col], sort=False)
            for col in feature_cols:
                mu = grp[col].transform("mean")
                sd = grp[col].transform("std")
                sd = sd.where(sd > 0, eps)
                df[col] = (df[col] - mu) / sd

        df = df.sort_values([asset_col, day_col, time_col]).reset_index(drop=True)
        self.df = df

        self.assets = df[asset_col].to_numpy()
        self.days   = df[day_col].to_numpy()
        self.times  = df[time_col].to_numpy()
        self.X_all  = df[feature_cols].to_numpy(np.float32)
        self.y_all  = df[target_col].to_numpy(np.float32)

        _, start_idx = np.unique(self.assets, return_index=True)
        start_idx = np.sort(start_idx)
        ends = np.r_[start_idx[1:], len(df)]
        self.asset_slices = list(zip(start_idx.tolist(), ends.tolist()))

        self.valid_indices = []
        self.bucket = {}

        start_for_row = np.empty(len(df), dtype=np.int32)
        for s, e in self.asset_slices:
            start_for_row[s:e] = s

        for i in range(len(df)):
            s = start_for_row[i]

            # Window [i-lookback+1, ..., i]
            if i - lookback + 1 < s:
                continue

            y = self.y_all[i]
            if not np.isfinite(y):
                continue

            X_win = self.X_all[i - lookback + 1 : i + 1]
            if X_win.shape[0] != lookback:
                continue
            if not np.isfinite(X_win).all():
                continue

            ds_idx = len(self.valid_indices)
            self.valid_indices.append(i)

            key = (int(self.days[i]), int(self.times[i]))
            self.bucket.setdefault(key, []).append(ds_idx)

        self.keys = list(self.bucket.keys())
        print(
            f"[Train/Val Dataset] df_rows={len(df)} "
            f"valid_samples={len(self.valid_indices)} "
            f"unique_time_slices={len(self.keys)}"
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        lb = self.lookback

        X = self.X_all[i - lb + 1 : i + 1]
        y = self.y_all[i]
        day = self.days[i]
        time_ = self.times[i]

        return (
            torch.from_numpy(X),
            torch.tensor([y], dtype=torch.float32),
            int(day),
            int(time_),
        )


# =========================================================
# 4) Dataset: Test with train padding
# When test period has insufficient history at the start, pad with the last history of the same stock from train
# =========================================================
class FactorSequenceDatasetTestWithTrainPad(Dataset):
    """
    Constructs a lookback-length window for each row in df_test:
        [t-lookback+1, ..., t]

    If test history is insufficient, pad from the last records of the same stock in df_train.
    Only predicts on test data; labels (y) are not required.
    """
    def __init__(
        self,
        df_train,
        df_test,
        feature_cols,
        lookback=32,
        asset_col="SecuCode",
        day_col="TradingDay",
        time_col="TimeEnd",
        cs_zscore=True,
        eps=1e-6,
    ):
        self.feature_cols = feature_cols
        self.lookback = lookback
        self.asset_col = asset_col
        self.day_col = day_col
        self.time_col = time_col

        df_train = df_train.copy()
        df_test = df_test.copy()

        # Apply cross-sectional z-score independently on train and test
        if cs_zscore:
            grp_tr = df_train.groupby([day_col, time_col], sort=False)
            for col in feature_cols:
                mu = grp_tr[col].transform("mean")
                sd = grp_tr[col].transform("std")
                sd = sd.where(sd > 0, eps)
                df_train[col] = (df_train[col] - mu) / sd

            grp_te = df_test.groupby([day_col, time_col], sort=False)
            for col in feature_cols:
                mu = grp_te[col].transform("mean")
                sd = grp_te[col].transform("std")
                sd = sd.where(sd > 0, eps)
                df_test[col] = (df_test[col] - mu) / sd

        df_train = df_train.sort_values([asset_col, day_col, time_col]).reset_index(drop=True)
        df_test  = df_test.sort_values([asset_col, day_col, time_col]).reset_index(drop=True)

        self.df_train = df_train
        self.df_test = df_test

        # Store the last lookback history for each stock in train
        self.train_hist = {}
        for secu, g in df_train.groupby(asset_col, sort=False):
            arr = g[feature_cols].to_numpy(np.float32)
            arr = arr[np.isfinite(arr).all(axis=1)]
            if len(arr) > 0:
                self.train_hist[secu] = arr

        # Store the full test sequence for each stock
        self.test_groups = {}
        for secu, g in df_test.groupby(asset_col, sort=False):
            self.test_groups[secu] = g.copy().reset_index()

        self.samples = []
        self.bucket = {}

        for secu, g in self.test_groups.items():
            Xg = g[feature_cols].to_numpy(np.float32)
            row_ids = g["index"].to_numpy()

            for j in range(len(g)):
                cur_x = Xg[j]
                if not np.isfinite(cur_x).all():
                    continue

                need_hist = lookback - 1
                left = max(0, j - need_hist)
                test_hist = Xg[left : j + 1]   # includes current row j

                # How many rows to pad from train
                lack = lookback - len(test_hist)
                if lack > 0:
                    train_arr = self.train_hist.get(secu, None)
                    if train_arr is None or len(train_arr) < lack:
                        continue
                    pad = train_arr[-lack:]
                    X_win = np.concatenate([pad, test_hist], axis=0)
                else:
                    X_win = test_hist[-lookback:]

                if X_win.shape != (lookback, len(feature_cols)):
                    continue
                if not np.isfinite(X_win).all():
                    continue

                ds_idx = len(self.samples)
                self.samples.append({
                    "X": X_win.astype(np.float32),
                    "row_idx": int(row_ids[j]),
                    "day": int(g.loc[j, day_col]),
                    "time": int(g.loc[j, time_col]),
                })

                key = (int(g.loc[j, day_col]), int(g.loc[j, time_col]))
                self.bucket.setdefault(key, []).append(ds_idx)

        self.keys = list(self.bucket.keys())
        print(
            f"[Test Dataset] test_rows={len(df_test)} "
            f"predictable_samples={len(self.samples)} "
            f"unique_time_slices={len(self.keys)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.from_numpy(s["X"]),
            torch.tensor(s["row_idx"], dtype=torch.long),
            int(s["day"]),
            int(s["time"]),
        )


# =========================================================
# 5) Sampler
# =========================================================
class TimeSliceBatchSampler(Sampler):
    """
    Each batch = all samples from the same (TradingDay, TimeEnd) time slice (or chunked subset)
    """
    def __init__(self, bucket, shuffle_slices=True, max_batch=512, drop_small=10):
        self.bucket = bucket
        self.shuffle_slices = shuffle_slices
        self.max_batch = max_batch
        self.drop_small = drop_small
        self.keys = [k for k in bucket.keys() if len(bucket[k]) >= drop_small]

    def __iter__(self):
        keys = self.keys.copy()
        if self.shuffle_slices:
            np.random.shuffle(keys)

        for k in keys:
            inds = self.bucket[k]
            if len(inds) <= self.max_batch:
                yield inds
            else:
                inds = inds.copy()
                np.random.shuffle(inds)
                for j in range(0, len(inds), self.max_batch):
                    yield inds[j:j+self.max_batch]

    def __len__(self):
        n = 0
        for k in self.keys:
            m = len(self.bucket[k])
            n += int(np.ceil(m / self.max_batch))
        return n


# =========================================================
# 6) Evaluate
# =========================================================
@torch.no_grad()
def evaluate_cross_sectional(model, loader, loss_fn):
    model.eval()

    total = 0.0
    n_obs = 0
    ic_list = []
    rankic_list = []

    for X, y, day, time_ in loader:
        X = X.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        yhat = model(X)
        loss = loss_fn(yhat, y)

        bs = X.size(0)
        total += loss.item() * bs
        n_obs += bs

        pred = yhat.squeeze(-1).detach().cpu().numpy()
        true = y.squeeze(-1).detach().cpu().numpy()

        ic = safe_corr_pearson(pred, true)
        rankic = safe_corr_spearman(pred, true)

        ic_list.append(ic)
        rankic_list.append(rankic)

    ic_arr = np.asarray(ic_list, dtype=float)
    rankic_arr = np.asarray(rankic_list, dtype=float)

    out = {
        "loss": total / max(n_obs, 1),
        "ic_mean": np.nanmean(ic_arr) if len(ic_arr) else np.nan,
        "ic_std": np.nanstd(ic_arr) if len(ic_arr) else np.nan,
        "rankic_mean": np.nanmean(rankic_arr) if len(rankic_arr) else np.nan,
        "rankic_std": np.nanstd(rankic_arr) if len(rankic_arr) else np.nan,
        "rankic_ir": (
            np.nanmean(rankic_arr) / (np.nanstd(rankic_arr) + 1e-12)
            if len(rankic_arr) else np.nan
        ),
        "rankic_pos_ratio": (
            np.nanmean(rankic_arr > 0)
            if len(rankic_arr) else np.nan
        ),
        "n_slices": len(ic_list),
    }
    return out


# =========================================================
# 7) Train with train/valid split inside df_trainsub
# =========================================================
def train_cross_model_time_consistent(
    df_trainsub,
    feature_cols,
    target_col="ret_mid_t1",
    lookback=32,
    epochs=10,
    val_ratio=0.2,
    max_batch=256,
    drop_small=10,
    num_workers=0,
    d_model=64,
    nhead=4,
    time_layers=1,
    cross_heads=4,
    cross_layers=2,
    dropout=0.1,
    lr=3e-4,
    grad_clip=1.0,
    weight_decay=1e-4,
    scheduler_patience=2,
    scheduler_factor=0.5,
    early_stop_patience=5,
    save_path="best_tf_cross.pt",
):
    df_tr, df_val = split_train_val_by_day(
        df_trainsub,
        day_col="TradingDay",
        val_ratio=val_ratio,
    )

    train_ds = TimeConsistentSeqDataset(
        df=df_tr,
        feature_cols=feature_cols,
        target_col=target_col,
        lookback=lookback,
        cs_zscore=True,
    )
    val_ds = TimeConsistentSeqDataset(
        df=df_val,
        feature_cols=feature_cols,
        target_col=target_col,
        lookback=lookback,
        cs_zscore=True,
    )

    train_sampler = TimeSliceBatchSampler(
        bucket=train_ds.bucket,
        shuffle_slices=True,
        max_batch=max_batch,
        drop_small=drop_small,
    )
    val_sampler = TimeSliceBatchSampler(
        bucket=val_ds.bucket,
        shuffle_slices=False,
        max_batch=max_batch,
        drop_small=1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = TransformerCross(
        n_features=len(feature_cols),
        d_model=d_model,
        nhead=nhead,
        time_layers=time_layers,
        cross_heads=cross_heads,
        cross_layers=cross_layers,
        dropout=dropout,
    ).to(DEVICE)

    loss_fn = nn.SmoothL1Loss()

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="max",
        factor=scheduler_factor,
        patience=scheduler_patience,
    )

    use_amp = (DEVICE == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_rankic = -np.inf
    best_state = None
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n_obs = 0

        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")

        for X, y, day, time_ in pbar:
            X = X.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                yhat = model(X)
                loss = loss_fn(yhat, y)

            scaler.scale(loss).backward()

            if grad_clip is not None:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optim)
            scaler.update()

            bs = X.size(0)
            total += loss.item() * bs
            n_obs += bs

            pbar.set_postfix(
                train_loss=float(loss.item()),
                B=int(bs),
                lr=float(optim.param_groups[0]["lr"]),
            )

        train_loss = total / max(n_obs, 1)
        val_metrics = evaluate_cross_sectional(model, val_loader, loss_fn)

        scheduler.step(val_metrics["rankic_mean"])

        print(
            f"Epoch {ep} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_ic={val_metrics['ic_mean']:.6f} | "
            f"val_rankic={val_metrics['rankic_mean']:.6f} | "
            f"rankic_std={val_metrics['rankic_std']:.6f} | "
            f"rankic_ir={val_metrics['rankic_ir']:.6f} | "
            f"rankic_pos={val_metrics['rankic_pos_ratio']:.4f} | "
            f"n_slices={val_metrics['n_slices']} | "
            f"lr={optim.param_groups[0]['lr']:.2e}"
        )

        improved = (
            np.isfinite(val_metrics["rankic_mean"])
            and val_metrics["rankic_mean"] > best_rankic
        )

        if improved:
            best_rankic = val_metrics["rankic_mean"]
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0

            torch.save({
                "model_state": best_state,
                "feature_cols": feature_cols,
                "config": model.config,
                "lookback": lookback,
                "best_rankic": best_rankic,
                "best_val_metrics": val_metrics,
            }, save_path)
            print(f"✅ Saved best -> {save_path} | rankic={best_rankic:.6f}")
        else:
            bad_epochs += 1
            print(f"No improvement. bad_epochs={bad_epochs}/{early_stop_patience}")
            if bad_epochs >= early_stop_patience:
                print("⏹ Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# =========================================================
# 8) Predict on df_testsub using train padding
# =========================================================
@torch.no_grad()
def predict_test_with_train_pad(
    model,
    df_train,
    df_test,
    feature_cols,
    lookback=32,
    batch_size=256,
    num_workers=0,
    pred_col="signal_tf_cross",
    drop_small=1,
):
    model.eval()

    ds_test = FactorSequenceDatasetTestWithTrainPad(
        df_train=df_train,
        df_test=df_test,
        feature_cols=feature_cols,
        lookback=lookback,
        cs_zscore=True,
    )

    sampler = TimeSliceBatchSampler(
        bucket=ds_test.bucket,
        shuffle_slices=False,
        max_batch=batch_size,
        drop_small=drop_small,
    )

    dl_test = DataLoader(
        ds_test,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    preds = np.full(len(df_test), np.nan, dtype=np.float32)

    for X, row_idx, day, time_ in tqdm(dl_test, desc="Predict test"):
        X = X.to(DEVICE, non_blocking=True)
        yhat = model(X).squeeze(-1).detach().cpu().numpy().astype(np.float32)
        preds[row_idx.numpy()] = yhat

    out = df_test.copy()
    out[pred_col] = preds
    return out


# =========================================================
# 9) Load best
# =========================================================
def load_trained_transformer_cross(model_path, map_location=None):
    if map_location is None:
        map_location = DEVICE

    ckpt = torch.load(model_path, map_location=map_location)
    config = ckpt["config"]

    model = TransformerCross(**config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt