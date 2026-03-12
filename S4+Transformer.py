import math
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)


# =========================================================
# Read data
# =========================================================
df_trainsub = pd.read_parquet("df_trainsub.parquet")
df_testsub = pd.read_parquet("df_testsub.parquet")
df_testsub_full = pd.read_parquet("df_testsub_full.parquet")

print("train shape:", df_trainsub.shape)
print("test shape :", df_testsub.shape)
print("full shape :", df_testsub_full.shape)


# =========================================================
# Helpers
# =========================================================
def cs_zscore_features(df, feature_cols, day_col="TradingDay", time_col="TimeEnd", clip_value=5.0):
    df = df.copy()
    grp = df.groupby([day_col, time_col])

    for c in feature_cols:
        mu = grp[c].transform("mean")
        std = grp[c].transform("std")
        df[c] = ((df[c] - mu) / (std + 1e-6)).clip(-clip_value, clip_value)

    return df


def add_cs_target(
    df,
    target_col="ret_mid_t1",
    day_col="TradingDay",
    time_col="TimeEnd",
    clip_value=5.0,
    out_col=None,
):
    df = df.copy()
    if out_col is None:
        out_col = f"{target_col}_cs"

    grp = df.groupby([day_col, time_col])
    mu = grp[target_col].transform("mean")
    std = grp[target_col].transform("std")
    df[out_col] = ((df[target_col] - mu) / (std + 1e-6)).clip(-clip_value, clip_value)

    return df


def safe_corr_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.nan
    x_std = x.std()
    y_std = y.std()
    if x_std < 1e-12 or y_std < 1e-12:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


def safe_corr_spearman(x, y):
    x = pd.Series(x).rank(method="average").values
    y = pd.Series(y).rank(method="average").values
    return safe_corr_pearson(x, y)


def compute_group_metrics_from_df(
    df,
    pred_col="pred",
    raw_target_col="ret_mid_t1",
    day_col="TradingDay",
    time_col="TimeEnd",
    n_bins=10,
):
    ic_list = []
    rankic_list = []
    spread_list = []

    for _, g in df.groupby([day_col, time_col], sort=False):
        g = g[[pred_col, raw_target_col]].dropna()
        if len(g) < max(5, n_bins):
            continue

        pred = g[pred_col].values
        y = g[raw_target_col].values

        ic = safe_corr_pearson(pred, y)
        rankic = safe_corr_spearman(pred, y)

        try:
            bins = pd.qcut(
                g[pred_col].rank(method="first"),
                q=n_bins,
                labels=False,
                duplicates="drop",
            )
            g2 = g.copy()
            g2["bin"] = bins.values
            bin_ret = g2.groupby("bin")[raw_target_col].mean()

            if len(bin_ret) >= 2:
                spread = bin_ret.iloc[-1] - bin_ret.iloc[0]
            else:
                spread = np.nan
        except Exception:
            spread = np.nan

        ic_list.append(ic)
        rankic_list.append(rankic)
        spread_list.append(spread)

    return {
        "ic_mean": np.nanmean(ic_list) if len(ic_list) else np.nan,
        "rankic_mean": np.nanmean(rankic_list) if len(rankic_list) else np.nan,
        "tb_spread_mean": np.nanmean(spread_list) if len(spread_list) else np.nan,
        "n_groups": len(ic_list),
    }


def batch_corr_loss(pred, y, eps=1e-8):
    pred = pred.squeeze(-1)
    y = y.squeeze(-1)

    pred = pred - pred.mean()
    y = y - y.mean()

    pred_std = torch.sqrt((pred ** 2).mean() + eps)
    y_std = torch.sqrt((y ** 2).mean() + eps)

    corr = (pred * y).mean() / (pred_std * y_std + eps)
    return -corr


# =========================================================
# Sampler
# =========================================================
class TimeBatchSampler(Sampler):
    def __init__(self, group_ids, shuffle_groups=True, seed=42):
        self.group_ids = np.asarray(group_ids)
        self.shuffle_groups = shuffle_groups
        self.seed = seed

        self.group_to_indices = {}
        for idx, gid in enumerate(self.group_ids):
            self.group_to_indices.setdefault(gid, []).append(idx)

        self.unique_groups = list(self.group_to_indices.keys())

    def __iter__(self):
        groups = self.unique_groups.copy()
        if self.shuffle_groups:
            rng = np.random.default_rng(self.seed + np.random.randint(0, 1_000_000))
            rng.shuffle(groups)

        for gid in groups:
            yield self.group_to_indices[gid]

    def __len__(self):
        return len(self.unique_groups)


# =========================================================
# Dataset: train / val
# =========================================================
class FactorSequenceDatasetByPredWindow(Dataset):
    def __init__(
        self,
        history_df,
        pred_df,
        feature_cols,
        raw_target_col="ret_mid_t1",
        target_cs_col="ret_mid_t1_cs",
        asset_col="SecuCode",
        day_col="TradingDay",
        time_col="TimeEnd",
        lookback=32,
        apply_cs_zscore=True,
    ):
        self.feature_cols = feature_cols
        self.raw_target_col = raw_target_col
        self.target_cs_col = target_cs_col
        self.lookback = lookback
        self.asset_col = asset_col
        self.day_col = day_col
        self.time_col = time_col

        history_use = [asset_col, day_col, time_col] + feature_cols
        pred_use = [asset_col, day_col, time_col, raw_target_col, target_cs_col] + feature_cols

        hist = history_df[history_use].copy()
        pred = pred_df[pred_use].copy()

        hist.sort_values([asset_col, day_col, time_col], inplace=True)
        pred.sort_values([asset_col, day_col, time_col], inplace=True)

        hist.dropna(subset=feature_cols, inplace=True)
        pred.dropna(subset=feature_cols + [raw_target_col, target_cs_col], inplace=True)

        if apply_cs_zscore:
            hist = cs_zscore_features(hist, feature_cols, day_col=day_col, time_col=time_col)
            pred = cs_zscore_features(pred, feature_cols, day_col=day_col, time_col=time_col)

        hist_map = {}
        pad_len = lookback - 1

        for asset, g in hist.groupby(asset_col, sort=False):
            g = g.reset_index(drop=True)
            hist_map[asset] = g.tail(pad_len).copy() if pad_len > 0 else g.iloc[0:0].copy()

        X_list, y_cs_list, y_raw_list, day_list, time_list, gid_list = [], [], [], [], [], []

        unique_ts = pred[[day_col, time_col]].drop_duplicates().sort_values([day_col, time_col])
        ts_to_gid = {
            (r[day_col], r[time_col]): i
            for i, r in unique_ts.reset_index(drop=True).iterrows()
        }

        for asset, g_pred in pred.groupby(asset_col, sort=False):
            g_pred = g_pred.reset_index(drop=True)

            g_hist = hist_map.get(asset, hist.iloc[0:0].copy()).copy()
            g_hist[raw_target_col] = np.nan
            g_hist[target_cs_col] = np.nan

            g_all = pd.concat([g_hist, g_pred], axis=0, ignore_index=True)

            Xg = g_all[feature_cols].values.astype(np.float32)
            y_raw_g = g_all[raw_target_col].values
            y_cs_g = g_all[target_cs_col].values
            day_g = g_all[day_col].values
            time_g = g_all[time_col].values

            start_pred_idx = len(g_hist)

            for i in range(start_pred_idx, len(g_all)):
                if i < lookback - 1:
                    continue

                X_list.append(Xg[i - lookback + 1:i + 1])
                y_cs_list.append(np.float32(y_cs_g[i]))
                y_raw_list.append(np.float32(y_raw_g[i]))
                day_list.append(day_g[i])
                time_list.append(time_g[i])
                gid_list.append(ts_to_gid[(day_g[i], time_g[i])])

        self.X = np.array(X_list, dtype=np.float32)
        self.y_cs = np.array(y_cs_list, dtype=np.float32).reshape(-1, 1)
        self.y_raw = np.array(y_raw_list, dtype=np.float32).reshape(-1, 1)
        self.day = np.array(day_list)
        self.time = np.array(time_list)
        self.group_ids = np.array(gid_list, dtype=np.int64)

        print("Dataset samples:", len(self.X))
        print("Num timestamp groups:", len(np.unique(self.group_ids)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y_cs[idx], dtype=torch.float32),
            torch.tensor(self.y_raw[idx], dtype=torch.float32),
            int(self.day[idx]),
            int(self.time[idx]),
            int(self.group_ids[idx]),
        )


# =========================================================
# Dataset: test with train padding
# =========================================================
class FactorSequenceDatasetTestWithTrainPad(Dataset):
    def __init__(
        self,
        df_train,
        df_test,
        feature_cols,
        asset_col="SecuCode",
        day_col="TradingDay",
        time_col="TimeEnd",
        lookback=32,
        apply_cs_zscore=True,
    ):
        self.feature_cols = feature_cols
        self.lookback = lookback

        train_use = [asset_col, day_col, time_col] + feature_cols
        test_use = [asset_col, day_col, time_col] + feature_cols

        train_df = df_train[train_use].copy()
        test_df = df_test[test_use].copy()

        train_df.sort_values([asset_col, day_col, time_col], inplace=True)
        test_df.sort_values([asset_col, day_col, time_col], inplace=True)

        train_df.dropna(subset=feature_cols, inplace=True)

        test_df = test_df.copy()
        test_df["_orig_index"] = np.arange(len(test_df))
        test_df.dropna(subset=feature_cols, inplace=True)

        if apply_cs_zscore:
            train_df = cs_zscore_features(train_df, feature_cols, day_col=day_col, time_col=time_col)
            test_df = cs_zscore_features(test_df, feature_cols, day_col=day_col, time_col=time_col)

        hist_map = {}
        pad_len = lookback - 1

        for asset, g in train_df.groupby(asset_col, sort=False):
            g = g.reset_index(drop=True)
            hist_map[asset] = g.tail(pad_len).copy() if pad_len > 0 else g.iloc[0:0].copy()

        X_list, row_idx_list = [], []

        for asset, g_test in test_df.groupby(asset_col, sort=False):
            g_test = g_test.reset_index(drop=True)
            g_hist = hist_map.get(asset, train_df.iloc[0:0].copy()).copy()
            g_hist["_orig_index"] = -1

            g_all = pd.concat([g_hist, g_test], axis=0, ignore_index=True)
            Xg = g_all[feature_cols].values.astype(np.float32)
            idxg = g_all["_orig_index"].values

            for i in range(len(g_all)):
                if idxg[i] < 0:
                    continue
                if i < lookback - 1:
                    continue

                X_list.append(Xg[i - lookback + 1:i + 1])
                row_idx_list.append(idxg[i])

        self.X = np.array(X_list, dtype=np.float32)
        self.row_idx = np.array(row_idx_list, dtype=np.int64)

        print("Test sequences with train padding:", len(self.X))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), int(self.row_idx[idx])


# =========================================================
# Model
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


class GatedSSMBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.A = nn.Parameter(torch.randn(d_model) * 0.1)
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.norm(x)
        B, L, D = z.shape
        h = torch.zeros(B, D, device=z.device)
        outs = []

        for t in range(L):
            u, g = self.in_proj(z[:, t]).chunk(2, dim=-1)
            g = torch.sigmoid(g)
            h = g * h + (1.0 - g) * torch.tanh(self.A * h + u)
            outs.append(self.out_proj(h).unsqueeze(1))

        out = torch.cat(outs, dim=1)
        return x + self.dropout(out)


class SSMTransformer(nn.Module):
    def __init__(
        self,
        n_features,
        d_model=128,
        ssm_layers=2,
        tf_layers=2,
        nhead=4,
        dropout=0.1,
    ):
        super().__init__()

        self.config = dict(
            n_features=n_features,
            d_model=d_model,
            ssm_layers=ssm_layers,
            tf_layers=tf_layers,
            nhead=nhead,
            dropout=dropout,
        )

        self.in_proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)

        self.ssm_stack = nn.Sequential(*[
            GatedSSMBlock(d_model=d_model, dropout=dropout)
            for _ in range(ssm_layers)
        ])

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.tf = nn.TransformerEncoder(enc_layer, num_layers=tf_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        x = self.in_proj(x)
        x = self.pos(x)
        x = self.ssm_stack(x)
        x = self.tf(x)

        last = x[:, -1, :]
        mean = x.mean(dim=1)
        pooled = 0.5 * last + 0.5 * mean

        return self.head(pooled)


# =========================================================
# Evaluation
# =========================================================
@torch.no_grad()
def evaluate_on_loader(model, dl, raw_target_col="ret_mid_t1"):
    model.eval()

    total_loss = 0.0
    total_n = 0
    rows = []

    for X, y_cs, y_raw, day, time_, gid in dl:
        X = X.to(DEVICE)
        y_cs = y_cs.to(DEVICE)

        pred = model(X)
        loss = batch_corr_loss(pred, y_cs)

        total_loss += loss.item() * X.size(0)
        total_n += X.size(0)

        pred_np = pred.squeeze(-1).detach().cpu().numpy()
        ycs_np = y_cs.squeeze(-1).detach().cpu().numpy()
        yraw_np = y_raw.squeeze(-1).detach().cpu().numpy()

        for i in range(len(pred_np)):
            rows.append({
                "TradingDay": int(day[i]),
                "TimeEnd": int(time_[i]),
                "pred": float(pred_np[i]),
                "target_cs": float(ycs_np[i]),
                raw_target_col: float(yraw_np[i]),
            })

    val_loss = total_loss / max(total_n, 1)
    eval_df = pd.DataFrame(rows)

    metrics = compute_group_metrics_from_df(
        eval_df,
        pred_col="pred",
        raw_target_col=raw_target_col,
        day_col="TradingDay",
        time_col="TimeEnd",
        n_bins=10,
    )
    metrics["loss"] = val_loss

    return metrics, eval_df


# =========================================================
# Predict
# =========================================================
@torch.no_grad()
def predict_test_with_train_pad(
    model,
    df_train,
    df_test,
    feature_cols,
    lookback=32,
    batch_size=4096,
    pred_col="signal_ssm_tf",
):
    model.eval()

    ds_test = FactorSequenceDatasetTestWithTrainPad(
        df_train=df_train,
        df_test=df_test,
        feature_cols=feature_cols,
        lookback=lookback,
        apply_cs_zscore=True,
    )

    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0)

    preds = np.full(len(df_test), np.nan, dtype=np.float32)

    for X, row_idx in tqdm(dl_test, desc="Predict test"):
        X = X.to(DEVICE)
        yhat = model(X).squeeze(-1).detach().cpu().numpy()
        preds[np.asarray(row_idx)] = yhat.astype(np.float32)

    out = df_test.copy()
    out[pred_col] = preds
    return out


# =========================================================
# Train
# =========================================================
def train_ssm_model(
    df_train,
    feature_cols,
    raw_target_col="ret_mid_t1",
    lookback=32,
    epochs=8,
    lr=3e-4,
    weight_decay=1e-4,
    d_model=128,
    ssm_layers=2,
    tf_layers=2,
    nhead=4,
    dropout=0.1,
    val_ratio=0.2,
):
    target_cs_col = f"{raw_target_col}_cs"
    df_train = add_cs_target(
        df_train,
        target_col=raw_target_col,
        day_col="TradingDay",
        time_col="TimeEnd",
        clip_value=5.0,
        out_col=target_cs_col,
    )

    unique_days = np.sort(df_train["TradingDay"].unique())
    n_days = len(unique_days)
    n_val_days = max(1, int(round(n_days * val_ratio)))
    n_train_days = n_days - n_val_days

    train_days = unique_days[:n_train_days]
    val_days = unique_days[n_train_days:]

    df_train_part = df_train[df_train["TradingDay"].isin(train_days)].copy()
    df_val_part = df_train[df_train["TradingDay"].isin(val_days)].copy()

    print("Train day range:", train_days[0], "->", train_days[-1], "n_days =", len(train_days))
    print("Val   day range:", val_days[0], "->", val_days[-1], "n_days =", len(val_days))

    train_ds = FactorSequenceDatasetByPredWindow(
        history_df=df_train_part,
        pred_df=df_train_part,
        feature_cols=feature_cols,
        raw_target_col=raw_target_col,
        target_cs_col=target_cs_col,
        lookback=lookback,
        apply_cs_zscore=True,
    )

    val_ds = FactorSequenceDatasetByPredWindow(
        history_df=df_train_part,
        pred_df=df_val_part,
        feature_cols=feature_cols,
        raw_target_col=raw_target_col,
        target_cs_col=target_cs_col,
        lookback=lookback,
        apply_cs_zscore=True,
    )

    train_sampler = TimeBatchSampler(train_ds.group_ids, shuffle_groups=True, seed=42)
    val_sampler = TimeBatchSampler(val_ds.group_ids, shuffle_groups=False, seed=42)

    train_dl = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=0)
    val_dl = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=0)

    model = SSMTransformer(
        n_features=len(feature_cols),
        d_model=d_model,
        ssm_layers=ssm_layers,
        tf_layers=tf_layers,
        nhead=nhead,
        dropout=dropout,
    ).to(DEVICE)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_rankic = -np.inf
    best_state = None

    print("Train groups:", len(train_sampler))
    print("Val groups  :", len(val_sampler))

    for ep in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_n = 0

        pbar = tqdm(train_dl, desc=f"Epoch {ep}/{epochs}")
        for X, y_cs, y_raw, day, time_, gid in pbar:
            X = X.to(DEVICE)
            y_cs = y_cs.to(DEVICE)

            optim.zero_grad()
            pred = model(X)
            loss = batch_corr_loss(pred, y_cs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            train_loss_total += loss.item() * X.size(0)
            train_n += X.size(0)
            pbar.set_postfix(loss=float(loss.item()), batch_n=int(X.size(0)))

        train_loss = train_loss_total / max(train_n, 1)

        val_metrics, _ = evaluate_on_loader(
            model,
            val_dl,
            raw_target_col=raw_target_col,
        )

        print(
            f"Epoch {ep}: "
            f"train_loss={train_loss:.8f}, "
            f"val_loss={val_metrics['loss']:.8f}, "
            f"val_ic={val_metrics['ic_mean']:.6f}, "
            f"val_rankic={val_metrics['rankic_mean']:.6f}, "
            f"val_tb_spread={val_metrics['tb_spread_mean']:.8f}, "
            f"val_groups={val_metrics['n_groups']}"
        )

        score = val_metrics["rankic_mean"]
        if np.isfinite(score) and score > best_rankic:
            best_rankic = score
            best_state = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "model_state": best_state,
                    "feature_cols": feature_cols,
                    "config": model.config,
                    "best_rankic": best_rankic,
                    "best_val_metrics": val_metrics,
                    "lookback": lookback,
                    "raw_target_col": raw_target_col,
                    "target_cs_col": target_cs_col,
                    "train_days": train_days,
                    "val_days": val_days,
                },
                "best_ssm_tf.pt",
            )
            print(f"✅ Saved best_ssm_tf.pt (best_rankic={best_rankic:.6f})")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model



