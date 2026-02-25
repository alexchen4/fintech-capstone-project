# =========================================================
# ✅ FIXED: GAT + Transformer (Spatiotemporal) with Corr Graph
# - Fix duplicate pivot index (TradingDay+TimeEnd)
# - Deduplicate (bar_time, SecuCode) via last
# - Fill missing values (no "drop all if any nan")
# - Align stocks order between X and adj
# - Save best_gat_tf.pt
# =========================================================

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)


# -----------------------------
# 1) Build unique time key + dedup
# -----------------------------
def preprocess_long_df(
    df: pd.DataFrame,
    feature_cols,
    target_col="ret_mid_t1",
    asset_col="SecuCode",
    day_col="TradingDay",
    time_col="TimeEnd",
    agg="last",
):
    """
    Returns df2 with columns: [bar_time, SecuCode, target, features...]
    bar_time = TradingDay*10000 + TimeEnd  (int)
    Dedup duplicates on (bar_time, SecuCode) by agg (last/mean/median)
    """
    need = [asset_col, day_col, time_col, target_col] + list(feature_cols)
    df2 = df[need].copy()

    # make sure numeric
    df2[day_col] = df2[day_col].astype(np.int64)
    df2[time_col] = df2[time_col].astype(np.int64)

    df2["bar_time"] = df2[day_col] * 10000 + df2[time_col]

    # sort to make "last" deterministic
    df2.sort_values(["bar_time", asset_col], inplace=True)

    keys = ["bar_time", asset_col]
    if agg == "last":
        df2 = df2.groupby(keys, as_index=False).last()
    elif agg == "mean":
        df2 = df2.groupby(keys, as_index=False).mean(numeric_only=True)
    elif agg == "median":
        df2 = df2.groupby(keys, as_index=False).median(numeric_only=True)
    else:
        raise ValueError("agg must be one of: last/mean/median")

    return df2


# -----------------------------
# 2) Fill NaNs (robust + fast)
# -----------------------------
def fill_panel_nan(X: np.ndarray, method="ts_median_then_0"):
    """
    X: (T, N, F)
    Fill missing without dropping whole samples.
    method:
      - ts_median_then_0: per time t, per feature f, fill across stocks by median; remaining fill 0
    """
    if method == "none":
        return X

    T, N, F = X.shape
    Xf = X.copy()

    # fill per (t,f) across N
    for t in range(T):
        Xt = Xf[t]  # (N,F)
        # median across N for each feature
        med = np.nanmedian(Xt, axis=0)  # (F,)
        # if med is nan (all nan), replace with 0
        med = np.where(np.isfinite(med), med, 0.0)
        inds = np.isnan(Xt)
        if inds.any():
            Xt[inds] = np.take(med, np.where(inds)[1])
        Xf[t] = Xt

    # any leftover nan -> 0
    Xf = np.nan_to_num(Xf, nan=0.0, posinf=0.0, neginf=0.0)
    return Xf


# -----------------------------
# 3) Build adjacency from correlation (TopK)
# -----------------------------
def build_adj_from_corr(corr: np.ndarray, topk: int = 20, self_loop=True):
    """
    corr: (N,N)
    """
    N = corr.shape[0]
    adj = np.zeros((N, N), dtype=np.float32)

    # handle nan corr
    corr2 = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    for i in range(N):
        idx = np.argsort(-np.abs(corr2[i]))[: topk + 1]
        adj[i, idx] = 1.0

    if self_loop:
        np.fill_diagonal(adj, 1.0)

    adj = np.maximum(adj, adj.T)
    return torch.tensor(adj, dtype=torch.float32)


# -----------------------------
# 4) Spatiotemporal Dataset
# -----------------------------
class SpatioTemporalDataset(Dataset):
    """
    Build samples from long df:
      X: (T, N, F)
      y: (N, 1)
    Uses bar_time as unique time index.
    """

    def __init__(
        self,
        df_long: pd.DataFrame,
        feature_cols,
        target_col="ret_mid_t1",
        asset_col="SecuCode",
        lookback=32,
        fill_method="ts_median_then_0",
        min_names_per_time=30,   # prevent extremely sparse times
        standardize=True,
        stats=None,              # (mu,std) for features (global)
        universe=None,           # optional fixed stock list
    ):
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.asset_col = asset_col
        self.lookback = int(lookback)

        # choose universe (stocks)
        if universe is None:
            # keep stocks with enough observations
            cnt = df_long.groupby(asset_col)["bar_time"].nunique()
            universe = cnt.sort_values(ascending=False).index.tolist()
        self.stocks = list(universe)

        # filter to universe
        dfu = df_long[df_long[asset_col].isin(self.stocks)].copy()

        # time index
        times = np.array(sorted(dfu["bar_time"].unique()), dtype=np.int64)
        self.times = times

        # pivot features: (time, stock, feat)
        # Use pivot_table with aggfunc='last' to avoid duplicate errors safely (even after preprocess)
        feat_mats = []
        for col in self.feature_cols:
            mat = (
                dfu.pivot_table(index="bar_time", columns=asset_col, values=col, aggfunc="last")
                   .reindex(index=times, columns=self.stocks)
            )
            feat_mats.append(mat.to_numpy(dtype=np.float32))
        X_all = np.stack(feat_mats, axis=-1)  # (T_total, N, F)

        y_mat = (
            dfu.pivot_table(index="bar_time", columns=asset_col, values=target_col, aggfunc="last")
               .reindex(index=times, columns=self.stocks)
        )
        y_all = y_mat.to_numpy(dtype=np.float32)  # (T_total, N)

        # standardize features (global, fit on available values)
        if standardize:
            if stats is None:
                flat = X_all.reshape(-1, X_all.shape[-1])
                mu = np.nanmean(flat, axis=0, keepdims=True).astype(np.float32)
                std = np.nanstd(flat, axis=0, keepdims=True).astype(np.float32)
                std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
                self.stats = (mu, std)
            else:
                self.stats = stats
            mu, std = self.stats
            X_all = (X_all - mu) / std
        else:
            self.stats = stats

        # build samples
        self.samples = []
        T_total, N, Fdim = X_all.shape

        # valid time mask: require enough non-nan names in y
        valid_names = np.sum(~np.isnan(y_all), axis=1)  # (T_total,)
        valid_time_mask = valid_names >= min_names_per_time

        for t in range(self.lookback, T_total):
            if not valid_time_mask[t]:
                continue

            X = X_all[t - self.lookback : t]   # (lookback, N, F)
            y = y_all[t]                       # (N,)

            # fill NaNs
            X = fill_panel_nan(X, method=fill_method)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            self.samples.append((X.astype(np.float32), y.reshape(N, 1)))

        if len(self.samples) == 0:
            raise RuntimeError(
                "Dataset became empty. Try:\n"
                "- reduce min_names_per_time\n"
                "- change fill_method\n"
                "- ensure df has TradingDay + TimeEnd and enough overlap stocks\n"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        return torch.from_numpy(X), torch.from_numpy(y)


# -----------------------------
# 5) Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# -----------------------------
# 6) Dense GAT Layer (adj fixed, attn learned)
# -----------------------------
class DenseGATLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.dropout = nn.Dropout(dropout)

        self.W = nn.Linear(d_model, d_model, bias=False)
        self.a_src = nn.Parameter(torch.randn(nhead, self.d_head))
        self.a_dst = nn.Parameter(torch.randn(nhead, self.d_head))

        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, adj):
        # x: (B, N, D) ; adj: (N, N)
        B, N, D = x.shape

        h = self.W(x).view(B, N, self.nhead, self.d_head).transpose(1, 2)  # (B,H,N,Dh)
        src = (h * self.a_src.view(1, self.nhead, 1, self.d_head)).sum(-1) # (B,H,N)
        dst = (h * self.a_dst.view(1, self.nhead, 1, self.d_head)).sum(-1) # (B,H,N)

        e = F.leaky_relu(src.unsqueeze(-1) + dst.unsqueeze(-2), 0.2)        # (B,H,N,N)

        mask = (adj > 0).unsqueeze(0).unsqueeze(0)                           # (1,1,N,N)
        e = e.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(e, dim=-1)
        attn = self.dropout(attn)

        out = attn @ h                                                       # (B,H,N,Dh)
        out = out.transpose(1, 2).contiguous().view(B, N, D)                 # (B,N,D)
        out = self.out_proj(out)

        return self.norm(x + self.dropout(out))


# -----------------------------
# 7) GAT + Transformer Model
# -----------------------------
class GAT_Transformer(nn.Module):
    def __init__(self, n_features, d_model=128, time_layers=2, gat_layers=2, time_heads=4, gat_heads=4, dropout=0.1):
        super().__init__()
        self.config = dict(
            d_model=d_model,
            time_layers=time_layers,
            gat_layers=gat_layers,
            time_heads=time_heads,
            gat_heads=gat_heads,
            dropout=dropout,
        )

        self.in_proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=time_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.time_tf = nn.TransformerEncoder(enc_layer, num_layers=time_layers)

        self.gat = nn.ModuleList([DenseGATLayer(d_model, nhead=gat_heads, dropout=dropout) for _ in range(gat_layers)])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, X, adj):
        # X: (B, T, N, F)
        B, T, N, Fdim = X.shape
        x = self.in_proj(X)                         # (B,T,N,D)

        x = x.permute(0, 2, 1, 3).contiguous()       # (B,N,T,D)
        x = x.view(B * N, T, -1)                     # (B*N,T,D)
        x = self.pos(x)
        x = self.time_tf(x)                          # (B*N,T,D)

        node_emb = x[:, -1, :].view(B, N, -1)        # (B,N,D)

        for layer in self.gat:
            node_emb = layer(node_emb, adj)

        return self.head(node_emb)                   # (B,N,1)


# -----------------------------
# 8) Train + Save best
# -----------------------------
def train_gat_tf(
    df_train: pd.DataFrame,
    feature_cols,
    target_col="ret_mid_t1",
    lookback=32,
    epochs=10,
    batch_size=8,
    lr=3e-4,
    weight_decay=1e-4,
    topk=20,
    min_names_per_time=30,
    save_path="best_gat_tf.pt",
):
    # preprocess (FIX duplicates)
    df2 = preprocess_long_df(
        df_train,
        feature_cols=feature_cols,
        target_col=target_col,
        asset_col="SecuCode",
        day_col="TradingDay",
        time_col="TimeEnd",
        agg="last",
    )

    # build dataset (stocks order determined here)
    dataset = SpatioTemporalDataset(
        df_long=df2,
        feature_cols=feature_cols,
        target_col=target_col,
        asset_col="SecuCode",
        lookback=lookback,
        fill_method="ts_median_then_0",
        min_names_per_time=min_names_per_time,
        standardize=True,
        stats=None,
        universe=None,
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # build corr on the SAME stocks order
    y_panel = (
        df2.pivot_table(index="bar_time", columns="SecuCode", values=target_col, aggfunc="last")
           .reindex(columns=dataset.stocks)
    )
    # fill for correlation compute (avoid all-nan columns)
    y_panel = y_panel.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    corr = y_panel.corr().to_numpy(dtype=np.float32)
    adj = build_adj_from_corr(corr, topk=topk, self_loop=True).to(DEVICE)

    model = GAT_Transformer(n_features=len(feature_cols)).to(DEVICE)
    loss_fn = nn.SmoothL1Loss(beta=1e-3)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0

        pbar = tqdm(loader, desc=f"Epoch {ep}/{epochs}")
        for X, y in pbar:
            X = X.to(DEVICE, non_blocking=True)               # (B,T,N,F)
            y = y.to(DEVICE, non_blocking=True)               # (B,N,1)

            optim.zero_grad(set_to_none=True)
            y_hat = model(X, adj)
            loss = loss_fn(y_hat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            bs = X.size(0)
            total += loss.item() * bs
            n += bs
            pbar.set_postfix(loss=f"{loss.item():.4g}", avg=f"{(total/max(n,1)):.4g}")

        ep_loss = total / max(n, 1)
        print(f"Epoch {ep} | loss={ep_loss:.6f}")

        if ep_loss < best:
            best = ep_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": model.config,
                    "feature_cols": list(feature_cols),
                    "stats": dataset.stats,            # (mu,std)
                    "stocks": list(dataset.stocks),    # stock order
                    "adj": adj.detach().cpu(),
                    "epoch": ep,
                    "best_loss": best,
                },
                save_path,
            )
            print(f"✅ Saved best -> {save_path}")

    return model


# -----------------------------
# 9) RUN
# -----------------------------
feature_cols = [c for c in df_train.columns if c.startswith("f_")]

model = train_gat_tf(
    df_train=df_train,
    feature_cols=feature_cols,
    target_col="ret_mid_t1",
    lookback=32,
    epochs=10,
    batch_size=8,
    topk=20,
    min_names_per_time=30,
    save_path="best_gat_tf.pt",
)