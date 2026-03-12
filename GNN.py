# =========================================================
# GAT + Transformer (Spatiotemporal) - Balanced Version
# Updated:
# 1) time-based train/valid split
# 2) global feature standardization using TRAIN ONLY
# 3) weighted static graph only
# 4) deeper but simple temporal encoder
# 5) single graph pass at the end
# 6) light residual MLP head
# 7) strict timing: X[t-lookback:t] -> y[t]
# 8) save by validation RankIC
# 9) test predicted separately and merged back
# =========================================================

import math
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
# 1) Read data
# =========================================================
df_trainsub = pd.read_parquet("df_trainsub.parquet")
df_testsub = pd.read_parquet("df_testsub.parquet")
df_testsub_full = pd.read_parquet("df_testsub_full.parquet")

print("train shape:", df_trainsub.shape)
print("test shape:", df_testsub.shape)
print("full shape:", df_testsub_full.shape)


# =========================================================
# 2) Preprocess long df
# =========================================================
def preprocess_long_df(
    df: pd.DataFrame,
    feature_cols,
    target_col="ret_mid_t1",
    asset_col="SecuCode",
    day_col="TradingDay",
    time_col="TimeEnd",
    agg="last",
):
    need = [asset_col, day_col, time_col, target_col] + list(feature_cols)
    df2 = df[need].copy()

    df2[day_col] = df2[day_col].astype(np.int64)
    df2[time_col] = df2[time_col].astype(np.int64)
    df2["bar_time"] = df2[day_col] * 10000 + df2[time_col]

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

    df2["TradingDay"] = (df2["bar_time"] // 10000).astype(np.int64)
    df2["TimeEnd"] = (df2["bar_time"] % 10000).astype(np.int64)
    return df2


# =========================================================
# 3) Robust fill
# =========================================================
def fill_panel_nan(X: np.ndarray, method="ts_median_then_0"):
    """
    X: (T, N, F)
    """
    if method == "none":
        return X

    Xf = X.copy()
    T, N, Fdim = Xf.shape

    for t in range(T):
        Xt = Xf[t]
        med = np.nanmedian(Xt, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        inds = np.isnan(Xt)
        if inds.any():
            Xt[inds] = np.take(med, np.where(inds)[1])
        Xf[t] = Xt

    Xf = np.nan_to_num(Xf, nan=0.0, posinf=0.0, neginf=0.0)
    return Xf


# =========================================================
# 4) Build static graph from TRAIN ONLY
# =========================================================
def build_weighted_adj_from_corr(corr: np.ndarray, topk: int = 20, self_loop=True):
    N = corr.shape[0]
    adj = np.zeros((N, N), dtype=np.float32)

    corr2 = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    for i in range(N):
        idx = np.argsort(-np.abs(corr2[i]))[: topk + 1]
        adj[i, idx] = np.abs(corr2[i, idx])

    if self_loop:
        np.fill_diagonal(adj, 1.0)

    adj = np.maximum(adj, adj.T)

    row_sum = adj.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum < 1e-6, 1.0, row_sum)
    adj = adj / row_sum

    return torch.tensor(adj, dtype=torch.float32)


# =========================================================
# 5) Time split helper
# =========================================================
def time_based_split(
    df_long: pd.DataFrame,
    val_ratio=0.2,
    time_col="bar_time",
):
    times = np.array(sorted(df_long[time_col].unique()))
    n_total = len(times)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    train_times = set(times[:n_train])
    val_times = set(times[n_train:])

    df_train = df_long[df_long[time_col].isin(train_times)].copy()
    df_val = df_long[df_long[time_col].isin(val_times)].copy()

    print(
        f"time split | total_times={n_total}, train_times={len(train_times)}, val_times={len(val_times)}"
    )
    print(
        f"train time range: {min(train_times)} -> {max(train_times)} | "
        f"val time range: {min(val_times)} -> {max(val_times)}"
    )

    return df_train, df_val, sorted(train_times), sorted(val_times)


# =========================================================
# 6) Stats fit on train only
# =========================================================
def fit_feature_stats_from_long_df(df_long: pd.DataFrame, feature_cols):
    flat = df_long[feature_cols].to_numpy(dtype=np.float32)
    mu = np.nanmean(flat, axis=0, keepdims=True).astype(np.float32)
    std = np.nanstd(flat, axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mu, std


# =========================================================
# 7) Dataset
# =========================================================
class SpatioTemporalDataset(Dataset):
    """
    sample at t uses X[t-lookback:t] -> y[t]
    """
    def __init__(
        self,
        df_long: pd.DataFrame,
        feature_cols,
        target_col="ret_mid_t1",
        asset_col="SecuCode",
        time_col="bar_time",
        lookback=32,
        fill_method="ts_median_then_0",
        min_names_per_time=30,
        standardize=True,
        stats=None,
        universe=None,
        times=None,
        keep_row_index=False
    ):
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.asset_col = asset_col
        self.time_col = time_col
        self.lookback = int(lookback)
        self.keep_row_index = keep_row_index

        dfu = df_long.copy()

        if universe is None:
            cnt = dfu.groupby(asset_col)[time_col].nunique()
            universe = cnt.sort_values(ascending=False).index.tolist()
        self.stocks = list(universe)

        dfu = dfu[dfu[asset_col].isin(self.stocks)].copy()

        if times is None:
            times = np.array(sorted(dfu[time_col].unique()), dtype=np.int64)
        else:
            times = np.array(sorted(times), dtype=np.int64)
        self.times = times

        if keep_row_index and "_orig_row_id" not in dfu.columns:
            raise ValueError("keep_row_index=True requires '_orig_row_id' in df_long")

        feat_mats = []
        for col in self.feature_cols:
            mat = (
                dfu.pivot_table(index=time_col, columns=asset_col, values=col, aggfunc="last")
                   .reindex(index=times, columns=self.stocks)
            )
            feat_mats.append(mat.to_numpy(dtype=np.float32))
        X_all = np.stack(feat_mats, axis=-1)  # (T, N, F)

        y_mat = (
            dfu.pivot_table(index=time_col, columns=asset_col, values=target_col, aggfunc="last")
               .reindex(index=times, columns=self.stocks)
        )
        y_all = y_mat.to_numpy(dtype=np.float32)  # (T, N)

        if keep_row_index:
            rowid_mat = (
                dfu.pivot_table(index=time_col, columns=asset_col, values="_orig_row_id", aggfunc="last")
                   .reindex(index=times, columns=self.stocks)
            )
            rowid_all = rowid_mat.to_numpy(dtype=np.float32)
        else:
            rowid_all = None

        if standardize:
            if stats is None:
                raise ValueError("standardize=True requires train-fitted stats")
            mu, std = stats
            X_all = (X_all - mu.reshape(1, 1, -1)) / std.reshape(1, 1, -1)

        self.samples = []
        T_total, N, Fdim = X_all.shape
        valid_names = np.sum(~np.isnan(y_all), axis=1)
        valid_time_mask = valid_names >= min_names_per_time

        for t in range(self.lookback, T_total):
            if not valid_time_mask[t]:
                continue

            X = X_all[t - self.lookback:t]
            y = y_all[t]

            X = fill_panel_nan(X, method=fill_method)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            if keep_row_index:
                row_ids = rowid_all[t]
                self.samples.append(
                    (X.astype(np.float32), y.reshape(N, 1), row_ids.astype(np.float32), int(times[t]))
                )
            else:
                self.samples.append((X.astype(np.float32), y.reshape(N, 1), int(times[t])))

        if len(self.samples) == 0:
            raise RuntimeError("Dataset became empty. Try reducing min_names_per_time or check overlap.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        if self.keep_row_index:
            X, y, row_ids, bar_time = item
            return (
                torch.from_numpy(X),
                torch.from_numpy(y),
                torch.from_numpy(row_ids),
                torch.tensor(bar_time, dtype=torch.long),
            )
        else:
            X, y, bar_time = item
            return (
                torch.from_numpy(X),
                torch.from_numpy(y),
                torch.tensor(bar_time, dtype=torch.long),
            )


# =========================================================
# 8) Positional Encoding
# =========================================================
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
        return x + self.pe[:, :x.size(1)]


# =========================================================
# 9) Simple weighted static GAT layer
# =========================================================
class StaticWeightedGATLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.dropout = nn.Dropout(dropout)

        self.W = nn.Linear(d_model, d_model, bias=False)
        self.a_src = nn.Parameter(torch.randn(nhead, self.d_head) * 0.02)
        self.a_dst = nn.Parameter(torch.randn(nhead, self.d_head) * 0.02)

        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, adj_static):
        """
        x: (B, N, D)
        adj_static: (N, N) weighted static graph
        """
        B, N, D = x.shape

        h = self.W(x).view(B, N, self.nhead, self.d_head).transpose(1, 2)   # (B,H,N,Dh)

        src = (h * self.a_src.view(1, self.nhead, 1, self.d_head)).sum(-1)  # (B,H,N)
        dst = (h * self.a_dst.view(1, self.nhead, 1, self.d_head)).sum(-1)  # (B,H,N)
        e = F.leaky_relu(src.unsqueeze(-1) + dst.unsqueeze(-2), 0.2)         # (B,H,N,N)

        A = adj_static.unsqueeze(0).unsqueeze(0).clamp(min=1e-8)             # (1,1,N,N)
        e = e + torch.log(A)

        mask = (adj_static > 0).unsqueeze(0).unsqueeze(0)
        e = e.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(e, dim=-1)
        attn = self.dropout(attn)

        out = attn @ h                                                        # (B,H,N,Dh)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)

        return self.norm(x + self.dropout(out))


# =========================================================
# 10) Light residual MLP head
# =========================================================
class LightResidualHead(nn.Module):
    def __init__(self, d_model=128, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: (B,N,D)
        """
        z = self.norm(x)
        h = self.fc1(z)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        z = z + h
        return self.out(z)


# =========================================================
# 11) Balanced model:
# deeper temporal -> last token -> single graph -> light residual head
# =========================================================
class GAT_Transformer_Balanced(nn.Module):
    def __init__(
        self,
        n_features,
        d_model=128,
        tf_layers=3,
        time_heads=4,
        gat_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        self.config = dict(
            n_features=n_features,
            d_model=d_model,
            tf_layers=tf_layers,
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
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=tf_layers)

        self.graph = StaticWeightedGATLayer(
            d_model=d_model,
            nhead=gat_heads,
            dropout=dropout,
        )

        self.head = LightResidualHead(d_model=d_model, dropout=dropout)

    def forward(self, X, adj_static):
        """
        X: (B, T, N, F)
        """
        B, T, N, Fdim = X.shape

        x = self.in_proj(X)                            # (B,T,N,D)
        x = x.permute(0, 2, 1, 3).contiguous()        # (B,N,T,D)
        x = x.view(B * N, T, -1)                      # (B*N,T,D)

        x = self.pos(x)
        x = self.temporal_encoder(x)

        node_emb = x[:, -1, :].view(B, N, -1)         # last token
        node_emb = self.graph(node_emb, adj_static)   # single graph pass

        out = self.head(node_emb)                     # (B,N,1)
        return out


# =========================================================
# 12) IC helpers
# =========================================================
def safe_corr_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.nan
    xs = np.nanstd(x)
    ys = np.nanstd(y)
    if xs < 1e-12 or ys < 1e-12:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


def safe_corr_spearman(x, y):
    xr = pd.Series(x).rank(method="average").values
    yr = pd.Series(y).rank(method="average").values
    return safe_corr_pearson(xr, yr)


@torch.no_grad()
def evaluate_gat_tf_metrics(model, loader, adj, loss_fn):
    model.eval()
    total = 0.0
    n = 0

    rows = []

    for X, y, bar_time in loader:
        X = X.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        y_hat = model(X, adj)
        loss = loss_fn(y_hat, y)

        bs = X.size(0)
        total += loss.item() * bs
        n += bs

        pred_np = y_hat.squeeze(-1).detach().cpu().numpy()   # (B,N)
        y_np = y.squeeze(-1).detach().cpu().numpy()          # (B,N)
        bt_np = bar_time.numpy()

        for b in range(pred_np.shape[0]):
            rows.append({
                "bar_time": int(bt_np[b]),
                "pred": pred_np[b].copy(),
                "y": y_np[b].copy(),
            })

    val_loss = total / max(n, 1)

    ic_list = []
    rankic_list = []

    for row in rows:
        pred = row["pred"]
        y = row["y"]

        mask = np.isfinite(pred) & np.isfinite(y)
        if mask.sum() < 5:
            continue

        pred2 = pred[mask]
        y2 = y[mask]

        ic = safe_corr_pearson(pred2, y2)
        rankic = safe_corr_spearman(pred2, y2)

        ic_list.append(ic)
        rankic_list.append(rankic)

    metrics = {
        "val_loss": val_loss,
        "ic_mean": np.nanmean(ic_list) if len(ic_list) else np.nan,
        "rankic_mean": np.nanmean(rankic_list) if len(rankic_list) else np.nan,
        "n_obs": len(ic_list),
    }
    return metrics


# =========================================================
# 13) Train
# =========================================================
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
    val_ratio=0.2,
    d_model=128,
    tf_layers=3,
    time_heads=4,
    gat_heads=4,
    dropout=0.1,
    save_path="best_gat_tf.pt",
):
    df2 = preprocess_long_df(
        df_train,
        feature_cols=feature_cols,
        target_col=target_col,
        asset_col="SecuCode",
        day_col="TradingDay",
        time_col="TimeEnd",
        agg="last",
    )

    df_train_long, df_val_long, train_times, val_times = time_based_split(
        df2, val_ratio=val_ratio, time_col="bar_time"
    )

    stock_cnt = df_train_long.groupby("SecuCode")["bar_time"].nunique()
    universe = stock_cnt.sort_values(ascending=False).index.tolist()

    stats = fit_feature_stats_from_long_df(df_train_long, feature_cols)

    train_ds = SpatioTemporalDataset(
        df_long=df_train_long,
        feature_cols=feature_cols,
        target_col=target_col,
        asset_col="SecuCode",
        time_col="bar_time",
        lookback=lookback,
        fill_method="ts_median_then_0",
        min_names_per_time=min_names_per_time,
        standardize=True,
        stats=stats,
        universe=universe,
        times=train_times,
        keep_row_index=False,
    )

    val_ds = SpatioTemporalDataset(
        df_long=df_val_long,
        feature_cols=feature_cols,
        target_col=target_col,
        asset_col="SecuCode",
        time_col="bar_time",
        lookback=lookback,
        fill_method="ts_median_then_0",
        min_names_per_time=min_names_per_time,
        standardize=True,
        stats=stats,
        universe=universe,
        times=val_times,
        keep_row_index=False,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # weighted static graph from TRAIN ONLY
    y_panel_train = (
        df_train_long.pivot_table(index="bar_time", columns="SecuCode", values=target_col, aggfunc="last")
                   .reindex(columns=universe)
    )
    y_panel_train = y_panel_train.ffill().bfill().fillna(0.0)
    corr_train = y_panel_train.corr().to_numpy(dtype=np.float32)
    adj = build_weighted_adj_from_corr(corr_train, topk=topk, self_loop=True).to(DEVICE)

    model = GAT_Transformer_Balanced(
        n_features=len(feature_cols),
        d_model=d_model,
        tf_layers=tf_layers,
        time_heads=time_heads,
        gat_heads=gat_heads,
        dropout=dropout,
    ).to(DEVICE)

    loss_fn = nn.SmoothL1Loss(beta=1e-3)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_rankic = -np.inf
    best_state = None
    best_metrics = None

    print(f"train samples: {len(train_ds)} | val samples: {len(val_ds)} | n_stocks: {len(universe)}")

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")
        for X, y, bar_time in pbar:
            X = X.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            y_hat = model(X, adj)
            loss = loss_fn(y_hat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            bs = X.size(0)
            total += loss.item() * bs
            n += bs
            pbar.set_postfix(train_loss=f"{loss.item():.4g}", train_avg=f"{(total/max(n,1)):.4g}")

        train_loss = total / max(n, 1)

        val_metrics = evaluate_gat_tf_metrics(model, val_loader, adj, loss_fn)
        val_loss = val_metrics["val_loss"]
        val_ic = val_metrics["ic_mean"]
        val_rankic = val_metrics["rankic_mean"]

        print(
            f"Epoch {ep} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_ic={val_ic:.6f} | "
            f"val_rankic={val_rankic:.6f} | "
            f"n_obs={val_metrics['n_obs']}"
        )

        if np.isfinite(val_rankic) and val_rankic > best_rankic:
            best_rankic = val_rankic
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = val_metrics.copy()

            torch.save(
                {
                    "model_state": best_state,
                    "config": model.config,
                    "feature_cols": list(feature_cols),
                    "stats": stats,
                    "stocks": list(universe),
                    "adj": adj.detach().cpu(),
                    "lookback": lookback,
                    "epoch": ep,
                    "best_rankic": best_rankic,
                    "best_val_metrics": best_metrics,
                },
                save_path,
            )
            print(f"✅ Saved best -> {save_path} | rankic={best_rankic:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "stats": stats,
        "stocks": universe,
        "adj": adj.detach().cpu(),
        "lookback": lookback,
        "feature_cols": list(feature_cols),
        "best_rankic": best_rankic,
        "best_val_metrics": best_metrics,
    }


# =========================================================
# 14) Build test dataframe for prediction
# =========================================================
def preprocess_test_long_df(
    df: pd.DataFrame,
    feature_cols,
    target_col="ret_mid_t1",
    asset_col="SecuCode",
    day_col="TradingDay",
    time_col="TimeEnd",
    agg="last",
):
    need = [asset_col, day_col, time_col] + list(feature_cols)
    if target_col in df.columns:
        need = [asset_col, day_col, time_col, target_col] + list(feature_cols)

    df2 = df[need].copy()
    df2[day_col] = df2[day_col].astype(np.int64)
    df2[time_col] = df2[time_col].astype(np.int64)
    df2["bar_time"] = df2[day_col] * 10000 + df2[time_col]

    df2["_orig_row_id"] = np.arange(len(df2), dtype=np.int64)

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

    df2["TradingDay"] = (df2["bar_time"] // 10000).astype(np.int64)
    df2["TimeEnd"] = (df2["bar_time"] % 10000).astype(np.int64)
    return df2


# =========================================================
# 15) Test prediction
# =========================================================
@torch.no_grad()
def predict_gat_tf_on_test(
    model,
    df_train,
    df_test,
    feature_cols,
    train_artifacts,
    target_col="ret_mid_t1",
    batch_size=8,
    min_names_per_time=30,
    pred_col="signal_gat_tf",
):
    lookback = train_artifacts["lookback"]
    stats = train_artifacts["stats"]
    universe = train_artifacts["stocks"]
    adj = train_artifacts["adj"].to(DEVICE)

    train_long = preprocess_long_df(
        df_train,
        feature_cols=feature_cols,
        target_col=target_col,
        asset_col="SecuCode",
        day_col="TradingDay",
        time_col="TimeEnd",
        agg="last",
    )

    test_long = preprocess_test_long_df(
        df_test,
        feature_cols=feature_cols,
        target_col=target_col if target_col in df_test.columns else "ret_mid_t1",
        asset_col="SecuCode",
        day_col="TradingDay",
        time_col="TimeEnd",
        agg="last",
    )

    train_long = train_long[train_long["SecuCode"].isin(universe)].copy()
    test_long = test_long[test_long["SecuCode"].isin(universe)].copy()

    pad_len = lookback
    train_times_sorted = np.array(sorted(train_long["bar_time"].unique()))
    hist_times = train_times_sorted[-pad_len:] if len(train_times_sorted) >= pad_len else train_times_sorted

    hist_long = train_long[train_long["bar_time"].isin(hist_times)].copy()
    hist_long["_orig_row_id"] = np.nan

    combo_long = pd.concat([hist_long, test_long], axis=0, ignore_index=True)
    combo_times = np.array(sorted(combo_long["bar_time"].unique()), dtype=np.int64)

    test_ds = SpatioTemporalDataset(
        df_long=combo_long,
        feature_cols=feature_cols,
        target_col=target_col if target_col in combo_long.columns else "ret_mid_t1",
        asset_col="SecuCode",
        time_col="bar_time",
        lookback=lookback,
        fill_method="ts_median_then_0",
        min_names_per_time=min_names_per_time,
        standardize=True,
        stats=stats,
        universe=universe,
        times=combo_times,
        keep_row_index=True,
    )

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    preds_by_row = {}

    for X, _, row_ids, bar_time in tqdm(test_loader, desc="Predict test"):
        X = X.to(DEVICE, non_blocking=True)
        y_hat = model(X, adj).squeeze(-1).detach().cpu().numpy()
        row_ids = row_ids.numpy()

        B, N = y_hat.shape
        for b in range(B):
            for n in range(N):
                rid = row_ids[b, n]
                if np.isnan(rid):
                    continue
                preds_by_row[int(rid)] = float(y_hat[b, n])

    out = df_test.copy()
    out[pred_col] = np.nan

    valid_rids = np.array(list(preds_by_row.keys()), dtype=np.int64)
    valid_vals = np.array([preds_by_row[k] for k in valid_rids], dtype=np.float32)

    mask = (valid_rids >= 0) & (valid_rids < len(out))
    out.loc[valid_rids[mask], pred_col] = valid_vals[mask]

    return out


# =========================================================
# 16) RUN TRAIN
# =========================================================
feature_cols = [c for c in df_trainsub.columns if c.startswith("f_")]
print("n_features:", len(feature_cols))

model, train_artifacts = train_gat_tf(
    df_train=df_trainsub,
    feature_cols=feature_cols,
    target_col="ret_mid_t1",
    lookback=32,
    epochs=8,
    batch_size=8,
    lr=3e-4,
    weight_decay=1e-4,
    topk=20,
    min_names_per_time=30,
    val_ratio=0.2,
    d_model=128,
    tf_layers=3,
    time_heads=4,
    gat_heads=4,
    dropout=0.1,
    save_path="best_gat_tf.pt",
)


# =========================================================
# 17) RUN TEST PREDICT
# =========================================================
df_testsub_pred = predict_gat_tf_on_test(
    model=model,
    df_train=df_trainsub,
    df_test=df_testsub,
    feature_cols=feature_cols,
    train_artifacts=train_artifacts,
    target_col="ret_mid_t1",
    batch_size=8,
    min_names_per_time=30,
    pred_col="signal_gat_tf",
)

print(df_testsub_pred[["signal_gat_tf"]].describe())
print("non-null signal_gat_tf in test:", df_testsub_pred["signal_gat_tf"].notna().sum())


# =========================================================
# 18) MERGE BACK TO FULL
# =========================================================
df_testsub_full = df_testsub_full.copy()

merge_cols = ["SecuCode", "TradingDay", "TimeEnd"]
merge_pred = df_testsub_pred[merge_cols + ["signal_gat_tf"]].copy()

df_testsub_full = df_testsub_full.merge(
    merge_pred,
    on=merge_cols,
    how="left",
    suffixes=("", "_new"),
)

if "signal_gat_tf" in df_testsub_full.columns and "signal_gat_tf_new" in df_testsub_full.columns:
    df_testsub_full["signal_gat_tf"] = df_testsub_full["signal_gat_tf_new"].combine_first(
        df_testsub_full["signal_gat_tf"]
    )
    df_testsub_full.drop(columns=["signal_gat_tf_new"], inplace=True)
elif "signal_gat_tf_new" in df_testsub_full.columns:
    df_testsub_full.rename(columns={"signal_gat_tf_new": "signal_gat_tf"}, inplace=True)

print("non-null signal_gat_tf in full:", df_testsub_full["signal_gat_tf"].notna().sum())