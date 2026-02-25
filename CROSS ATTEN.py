import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

class TimeConsistentSeqDataset(Dataset):
    """
    每个样本 = (某只股票在某个时刻t的lookback序列, 该时刻t的y)
    但 batch 由 BatchSampler 保证：同一个 (TradingDay, TimeEnd) 的样本聚在一起。
    """
    def __init__(self, df, feature_cols, target_col="ret_mid_t1",
                 lookback=32,
                 asset_col="SecuCode",
                 day_col="TradingDay",
                 time_col="TimeEnd"):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lookback = lookback

        # 排序保证时间序列正确
        df = df.sort_values([asset_col, day_col, time_col]).reset_index(drop=True)

        # 转成 numpy（节省内存 + 取数快）
        self.assets = df[asset_col].to_numpy()
        self.days   = df[day_col].to_numpy()
        self.times  = df[time_col].to_numpy()
        self.X_all  = df[feature_cols].to_numpy(np.float32)
        self.y_all  = df[target_col].to_numpy(np.float32)

        # 记录每只股票在 df 中的连续区间（因为已按 SecuCode 排序）
        # start_end: list[(start_idx, end_idx_exclusive)]
        self.asset_slices = []
        uniq_assets, start_idx = np.unique(self.assets, return_index=True)
        order = np.argsort(start_idx)
        start_idx = start_idx[order]
        uniq_assets = uniq_assets[order]
        ends = np.r_[start_idx[1:], len(df)]
        self.asset_slices = list(zip(start_idx.tolist(), ends.tolist()))

        # 构造“可用样本”索引：i 必须满足在同一股票内部 i-lookback >= start
        # 同时按 (day,time) 分桶，供 batch sampler 使用
        self.valid_indices = []
        self.bucket = {}  # key=(day,time) -> list of dataset indices (not df indices)

        # 为了 O(1) 判断股票段起点：给每个 df 行标注所属 slice 的 start
        start_for_row = np.empty(len(df), dtype=np.int32)
        for s, e in self.asset_slices:
            start_for_row[s:e] = s

        # 扫一遍 df 行：符合 lookback 的就加入
        for i in range(len(df)):
            s = start_for_row[i]
            if i - lookback < s:
                continue  # 不够lookback

            # 可以加：过滤 y nan
            y = self.y_all[i]
            if not np.isfinite(y):
                continue

            ds_idx = len(self.valid_indices)
            self.valid_indices.append(i)

            key = (int(self.days[i]), int(self.times[i]))
            self.bucket.setdefault(key, []).append(ds_idx)

        self.keys = list(self.bucket.keys())

        print(f"[Dataset] df_rows={len(df)} valid_samples={len(self.valid_indices)} unique_time_slices={len(self.keys)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]  # df row index
        lb = self.lookback
        X = self.X_all[i-lb:i]               # (L,F)
        y = self.y_all[i]                    # scalar
        return torch.from_numpy(X), torch.tensor([y], dtype=torch.float32)


class TimeSliceBatchSampler(Sampler):
    """
    每个 batch = 同一个 (TradingDay, TimeEnd) 的所有样本（或截断到 max_batch）
    """
    def __init__(self, bucket, shuffle_slices=True, max_batch=512, drop_small=10):
        self.bucket = bucket
        self.shuffle_slices = shuffle_slices
        self.max_batch = max_batch
        self.drop_small = drop_small
        self.keys = list(bucket.keys())

        # 过滤太小的横截面（可选）
        self.keys = [k for k in self.keys if len(bucket[k]) >= drop_small]

    def __iter__(self):
        keys = self.keys.copy()
        if self.shuffle_slices:
            np.random.shuffle(keys)

        for k in keys:
            inds = self.bucket[k]
            # 横截面过大就切块，避免显存爆
            if len(inds) <= self.max_batch:
                yield inds
            else:
                # 打乱横截面内部再切块（可选）
                inds = inds.copy()
                np.random.shuffle(inds)
                for j in range(0, len(inds), self.max_batch):
                    yield inds[j:j+self.max_batch]

    def __len__(self):
        # 估算 batches 数
        n = 0
        for k in self.keys:
            m = len(self.bucket[k])
            n += int(np.ceil(m / self.max_batch))
        return n
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def train_cross_model_time_consistent(
    df_train,
    feature_cols,
    target_col="ret_mid_t1",
    lookback=32,
    epochs=5,
    max_batch=256,          # 控制横截面batch最大大小，防止爆显存
    drop_small=30,          # 太小的横截面不要（可调）
    num_workers=0,
    d_model=64,
    nhead=4,
    time_layers=1,
    cross_heads=4,
    lr=3e-4,
    grad_clip=1.0,
):

    dataset = TimeConsistentSeqDataset(
        df_train,
        feature_cols,
        target_col=target_col,
        lookback=lookback
    )

    batch_sampler = TimeSliceBatchSampler(
        bucket=dataset.bucket,
        shuffle_slices=True,
        max_batch=max_batch,
        drop_small=drop_small
    )

    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,   # ✅ 关键：不用 batch_size/shuffle
        num_workers=num_workers,
        pin_memory=True
    )

    model = TransformerCross(
        n_features=len(feature_cols),
        d_model=d_model,
        nhead=nhead,
        time_layers=time_layers,
        cross_heads=cross_heads,
        dropout=0.1
    ).to(DEVICE)

    loss_fn = nn.SmoothL1Loss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    use_amp = (DEVICE == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_loss = 1e18

    for ep in range(1, epochs+1):
        model.train()
        total = 0.0
        n_obs = 0

        pbar = tqdm(loader, desc=f"Epoch {ep}/{epochs}")

        for X, y in pbar:
            X = X.to(DEVICE, non_blocking=True)  # (B,L,F) 同一time slice
            y = y.to(DEVICE, non_blocking=True)  # (B,1)

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                yhat = model(X)                  # (B,1)
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
            pbar.set_postfix(loss=float(loss.item()), B=int(bs))

        epoch_loss = total / max(n_obs, 1)
        print("Epoch loss:", epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                "model_state": model.state_dict(),
                "feature_cols": feature_cols,
                "config": model.config
            }, "best_tf_cross.pt")
            print("✅ Saved best_tf_cross.pt")

    return model

feature_cols = sorted([c for c in df_train.columns if c.startswith("f_")])
print(len(feature_cols), feature_cols[:5], feature_cols[-5:])

model = train_cross_model_time_consistent(
    df_train=df_train,
    feature_cols=feature_cols,
    target_col="ret_mid_t1",
    lookback=32,
    epochs=5,
    max_batch=256,      # 先小点，稳
    drop_small=30,      # A股每bar股票很多，30够了
    d_model=64,
    nhead=4,
    time_layers=1,
    cross_heads=4,
)