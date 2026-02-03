import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 一体化：从15min长表df -> 日频矩阵 -> IC/分层/净值/回撤/画图
# =========================================================
def fast_factor_eval_from_long_df(
    df: pd.DataFrame,
    close_col: str = "mid_price",
    factor_col: str = "vpin_like_10",
    asset_col: str = "SecuCode",
    day_col: str = "TradingDay",
    time_col: str = "TimeEnd",
    n_groups: int = 10,
    freq: int = 252,
    min_names_per_day: int = 30,
    ic_method: str = "spearman",     # "spearman" or "pearson"
    dd_style: str = "fill",          # "fill" or "twin"
    is_plot: bool = True,
    use_last_per_day: bool = True,   # True=日末last；False=你可自己改成日均/日中等
):
    """
    输入：15min长表df + 列名
    输出：dict，包含 close/factor矩阵、IC序列、分层收益、净值、回撤、metrics 等。

    注意：
    - “日收益”来自 close_matrix 的 pct_change().shift(-1)，即日末 close-to-close forward return。
    - 因子用同一天日末 last（不会lookahead）。
    """

    # -------------------------
    # 0) 检查列
    # -------------------------
    need = {asset_col, day_col, time_col, close_col, factor_col}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"缺列: {miss}")

    d = df[[asset_col, day_col, time_col, close_col, factor_col]].copy()
    d = d.dropna(subset=[close_col, factor_col])

    # TradingDay -> datetime index
    d["date"] = pd.to_datetime(d[day_col].astype(str), format="%Y%m%d")

    # -------------------------
    # 1) 抽日末 last（每股每天一条）
    # -------------------------
    d = d.sort_values([asset_col, "date", time_col])
    if use_last_per_day:
        d = d.groupby([asset_col, "date"], sort=False).tail(1)

    # 横截面过滤（每天至少 min_names_per_day）
    cs = d.groupby("date")[asset_col].nunique()
    keep_dates = cs[cs >= min_names_per_day].index
    d = d[d["date"].isin(keep_dates)]

    # -------------------------
    # 2) pivot 成矩阵：date x asset
    # -------------------------
    close = d.pivot(index="date", columns=asset_col, values=close_col).sort_index()
    factor = d.pivot(index="date", columns=asset_col, values=factor_col).sort_index()

    # align
    idx = close.index.intersection(factor.index)
    cols = close.columns.intersection(factor.columns)
    close = close.loc[idx, cols].copy()
    factor = factor.loc[idx, cols].copy()

    # -------------------------
    # 3) forward return (t->t+1)  日频
    # -------------------------
    fwd_ret = close.pct_change(fill_method=None).shift(-1)

    # 额外防御：pivot 后仍可能出现某些日期整行 factor 极稀疏 / 全 NaN
    valid_cs = factor.notna().sum(axis=1)
    keep = valid_cs >= max(min_names_per_day, n_groups)   # 至少能分组
    close = close.loc[keep]
    factor = factor.loc[keep]
    fwd_ret = fwd_ret.loc[keep]

    # =========================================================
    # 4) IC / RankIC（向量化 row-wise corr）
    # =========================================================
    X = factor.to_numpy(dtype=float)
    Y = fwd_ret.to_numpy(dtype=float)

    def row_corr(A, B):
        A0 = A.copy()
        B0 = B.copy()

        mask = ~np.isnan(A0) & ~np.isnan(B0)
        A0[~mask] = np.nan
        B0[~mask] = np.nan

        cnt = np.sum(~np.isnan(A0), axis=1)
        empty = cnt == 0

        Am = np.nanmean(A0, axis=1, keepdims=True)
        Bm = np.nanmean(B0, axis=1, keepdims=True)
        Ac = A0 - Am
        Bc = B0 - Bm

        cov = np.nansum(Ac * Bc, axis=1)
        va  = np.nansum(Ac * Ac, axis=1)
        vb  = np.nansum(Bc * Bc, axis=1)
        denom = np.sqrt(va * vb)

        out = cov / np.where(denom == 0, np.nan, denom)
        out[empty] = np.nan
        return out

    # Pearson IC
    ic_pearson = pd.Series(row_corr(X, Y), index=factor.index, name="IC_pearson")

    # Spearman IC (= RankIC)
    rankX = factor.rank(axis=1, method="average", pct=False).to_numpy(float)
    rankY = fwd_ret.rank(axis=1, method="average", pct=False).to_numpy(float)
    ic_spearman = pd.Series(row_corr(rankX, rankY), index=factor.index, name="IC_spearman")

    # 选择主 IC
    ic_main = (ic_pearson if ic_method.lower() == "pearson" else ic_spearman).rename("IC")

    # 全样本 IC 统计（纯向量化）
    ic_vals = ic_main.to_numpy(dtype=float)
    m = ~np.isnan(ic_vals)
    ic_vals = ic_vals[m]
    if ic_vals.size > 1:
        ic_mean = float(ic_vals.mean())
        ic_std  = float(ic_vals.std(ddof=1))
        ic_ir   = float(ic_mean / ic_std * np.sqrt(freq)) if ic_std > 0 else np.nan
    else:
        ic_mean = np.nan
        ic_std  = np.nan
        ic_ir   = np.nan

    # 累计 IC
    ic_cum = ic_main.fillna(0.0).cumsum().rename("IC_cum")

    # 年度 IC 统计（纯向量化，无 groupby）
    years = ic_main.index.year.astype(np.int32)     # (T,)
    vals  = ic_main.to_numpy(dtype=float)           # (T,)
    m = ~np.isnan(vals)
    years = years[m]
    vals  = vals[m]

    if vals.size > 0:
        uniq_years, inv = np.unique(years, return_inverse=True)
        K_year = uniq_years.size

        cnt_y  = np.bincount(inv, minlength=K_year).astype(np.int32)
        sum_y  = np.bincount(inv, weights=vals, minlength=K_year)
        sum2_y = np.bincount(inv, weights=vals * vals, minlength=K_year)

        mean_y = sum_y / cnt_y
        var_y  = sum2_y / cnt_y - mean_y * mean_y
        std_y  = np.sqrt(np.maximum(var_y, 0.0))
        ir_y   = np.where(std_y > 0, mean_y / std_y * np.sqrt(freq), np.nan)

        metrics_yearly = pd.DataFrame(
            {"IC_mean": mean_y, "IC_std": std_y, "IC_IR": ir_y, "obs": cnt_y},
            index=uniq_years
        )
        metrics_yearly.index.name = "year"
    else:
        metrics_yearly = pd.DataFrame(columns=["IC_mean", "IC_std", "IC_IR", "obs"])
        metrics_yearly.index.name = "year"

    # =========================================================
    # 5) 分组（向量化）：pct-rank -> 1..G
    # =========================================================
    pct = factor.rank(axis=1, method="first", pct=True)
    grp_float = np.ceil(pct * n_groups).clip(1, n_groups)
    grp = grp_float.fillna(0).astype(np.int16)   # 0..K

    R = fwd_ret.to_numpy(dtype=float)            # (T,N)
    G = grp.to_numpy(dtype=np.int16)             # (T,N)  0表示无组
    T, N = R.shape
    K = n_groups

    group_ret_np = np.full((T, K), np.nan, dtype=float)
    for t in range(T):
        gt = G[t]
        rt = R[t]
        valid = (gt > 0) & ~np.isnan(rt)
        if not np.any(valid):
            continue

        g = gt[valid]   # 1..K
        x = rt[valid]

        cnt = np.bincount(g, minlength=K+1).astype(float)
        s   = np.bincount(g, weights=x, minlength=K+1)

        mean = s[1:] / np.where(cnt[1:] == 0, np.nan, cnt[1:])
        group_ret_np[t, :] = mean

    group_ret = pd.DataFrame(group_ret_np, index=factor.index, columns=range(1, K+1))

    # -------------------------
    # 5.5) NAV / Long-Short / Drawdown
    # -------------------------
    group_nav = (1.0 + group_ret.fillna(0.0)).cumprod()

    hedge_ret = (group_ret[1] - group_ret[K]).rename(f"Hedge(L1-S{K})")
    hedge_nav = (1.0 + hedge_ret.fillna(0.0)).cumprod().rename("hedge_nav")

    peak = hedge_nav.cummax()
    dd = (hedge_nav / peak - 1.0).rename("drawdown")

    # -------------------------
    # 6) metrics
    # -------------------------
    r = hedge_ret.dropna()
    if len(r):
        avg_ret = float(r.mean())
        hedge_total = float(hedge_nav.iloc[-1] - 1.0)
        n = len(r)
        annual_ret = float((1.0 + hedge_total) ** (freq / n) - 1.0)

        mdd = float(dd.min())
        mar = float(annual_ret / abs(mdd)) if mdd < 0 else np.nan

        vol = float(r.std(ddof=1)) if len(r) > 1 else np.nan
        sharpe = float((avg_ret / vol) * np.sqrt(freq)) if (vol and vol > 0) else np.nan

        win_ratio = float((r > 0).mean())

        trough_date = dd.idxmin()
        peak_date = hedge_nav.loc[:trough_date].idxmax()
        dd_days = int((pd.to_datetime(trough_date) - pd.to_datetime(peak_date)).days)

        metrics = pd.DataFrame([{
            "IC_mean": ic_mean,
            "IC_std": ic_std,
            "IC_IR": ic_ir,

            "avg_ret": avg_ret,
            "hedge_ret_total": hedge_total,
            "annual_ret": annual_ret,
            "MDD": mdd,
            "MAR": mar,
            "sharpe": sharpe,
            "win_ratio": win_ratio,
            "dd_days": dd_days,
            "dd_start_date": str(peak_date),
            "dd_end_date": str(trough_date),
        }])
    else:
        metrics = pd.DataFrame([{
            "IC_mean": ic_mean, "IC_std": ic_std, "IC_IR": ic_ir,

            "avg_ret": np.nan, "hedge_ret_total": np.nan, "annual_ret": np.nan,
            "MDD": np.nan, "MAR": np.nan, "sharpe": np.nan,
            "win_ratio": np.nan, "dd_days": np.nan,
            "dd_start_date": None, "dd_end_date": None
        }])

    # -------------------------
    # 7) plot
    # -------------------------
    if is_plot:
        mean_ret_by_group = group_ret.mean(skipna=True)
        plt.figure(figsize=(10, 4))
        plt.bar(mean_ret_by_group.index.astype(str), mean_ret_by_group.values, alpha=0.85)
        plt.title(f"{factor_col} - group mean return (daily fwd)")
        plt.xlabel("group")
        plt.ylabel("mean fwd return")
        plt.grid(True, axis="y", alpha=0.3)
        plt.show()

        plt.figure(figsize=(12, 5))
        for k in range(1, K + 1):
            lw = 2.2 if (k == 1 or k == K) else 1.0
            plt.plot(group_nav.index, group_nav[k].values, label=str(k), linewidth=lw, alpha=0.95)

        plt.title(f"{factor_col} - group net values (1..{K})")
        plt.xlabel("date")
        plt.ylabel("net value")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=5, fontsize=8)
        plt.show()

        mdd_plot = float(dd.min()) if len(dd) else -0.5
        mdd_plot = min(mdd_plot, -1e-6)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(hedge_nav.index, hedge_nav.values, label="hedge nav", linewidth=2.4)
        ax.set_xlabel("date")
        ax.set_ylabel("net value")
        ax.grid(True, alpha=0.3)

        ax_dd = ax.twinx()
        ax_dd.set_ylabel("drawdown")
        ax_dd.set_ylim(mdd_plot, 0.0)

        if dd_style == "twin":
            ax_dd.plot(dd.index, dd.values, linestyle="--", linewidth=1.6, alpha=0.9, label="drawdown")
        else:
            ax_dd.fill_between(dd.index, dd.values, 0.0, alpha=0.25, label="drawdown")

        plt.title(f"{factor_col} - hedge nav and drawdown")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_dd.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best")
        plt.show()

        plt.figure(figsize=(12, 4))
        plt.plot(ic_cum.index, ic_cum.values, linewidth=2.2)
        if ic_ir == ic_ir:
            plt.title(f"{factor_col} - cumulative IC ({ic_method}), IR={ic_ir:.3f}")
        else:
            plt.title(f"{factor_col} - cumulative IC ({ic_method})")
        plt.xlabel("date")
        plt.ylabel("cum IC")
        plt.grid(True, alpha=0.3)
        plt.show()

        # ✅ 新增：打印每年 IC（例如2018-2025），不影响其它任何输出
        print("===== Yearly IC (annualized IR) =====")
        print(metrics_yearly.loc[2018:2025, ["IC_mean", "IC_std", "IC_IR", "obs"]])

        print(metrics)

    return {
        "close": close,
        "factor": factor,
        "fwd_ret": fwd_ret,
        "ic_main": ic_main,
        "ic_cum": ic_cum,
        "ic_pearson": ic_pearson,
        "ic_spearman": ic_spearman,
        "ic_yearly": metrics_yearly,   # ✅ 年度IC表
        "group_ret": group_ret,
        "group_nav": group_nav,
        "hedge_ret": hedge_ret,
        "hedge_nav": hedge_nav,
        "dd": dd,
        "metrics": metrics,
    }
