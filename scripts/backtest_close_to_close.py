"""
Unified close-to-close backtest for ALL models using
the Backtesting.ipynb framework, then comparison chart.
Results saved to models_comparison/
"""
import os, copy, json, random, warnings
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')

BASE_DIR = 'e:/New_folder/Work/MyProject/capstone'
os.chdir(BASE_DIR)
OUT_DIR = 'models_comparison'
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ══════════════════════════════════════════════════════════
# SECTION 1: Backtesting Framework (from Backtesting.ipynb)
# ══════════════════════════════════════════════════════════

def prepare_close_bar_data(df, signal_col, price_col, spread_col,
                           day_col, stock_col, close_ts, close_te):
    df = df.copy().sort_values([day_col, "TimeStart", "TimeEnd", stock_col]).reset_index(drop=True)
    df_close = df[
        (df["TimeStart"] == close_ts) & (df["TimeEnd"] == close_te)
    ][[day_col, stock_col, signal_col, price_col, spread_col]].copy()
    df_close = df_close.sort_values([stock_col, day_col]).reset_index(drop=True)
    df_close["has_price_t"] = np.isfinite(df_close[price_col].astype(float))
    df_close["has_spread_t"] = np.isfinite(df_close[spread_col].astype(float))
    df_close["price_t1"] = df_close.groupby(stock_col)[price_col].shift(-1)
    df_close["spread_t1"] = df_close.groupby(stock_col)[spread_col].shift(-1)
    df_close["day_t1"] = df_close.groupby(stock_col)[day_col].shift(-1)
    all_days = np.sort(df_close[day_col].unique())
    next_day_map = pd.Series(all_days[1:], index=all_days[:-1]).to_dict()
    df_close["expected_day_t1"] = df_close[day_col].map(next_day_map)
    is_true_next_day = df_close["day_t1"].eq(df_close["expected_day_t1"])
    df_close["has_price_t1"] = is_true_next_day & np.isfinite(df_close["price_t1"].astype(float))
    df_close["has_spread_t1"] = is_true_next_day & np.isfinite(df_close["spread_t1"].astype(float))
    df_close["eligible"] = (
        df_close[signal_col].notna() & df_close["has_price_t"]
        & df_close["has_spread_t"] & df_close["has_price_t1"] & df_close["has_spread_t1"]
    )
    days = np.sort(df_close[day_col].unique())
    stocks = np.sort(df_close[stock_col].unique())
    price_mat = df_close.pivot(index=day_col, columns=stock_col, values=price_col).reindex(index=days, columns=stocks).astype("float64")
    spread_mat = df_close.pivot(index=day_col, columns=stock_col, values=spread_col).reindex(index=days, columns=stocks).astype("float64")
    signal_mat = df_close.pivot(index=day_col, columns=stock_col, values=signal_col).reindex(index=days, columns=stocks).astype("float64")
    eligible_mat = df_close.pivot(index=day_col, columns=stock_col, values="eligible").reindex(index=days, columns=stocks).fillna(False).astype(bool)
    return {"df_close": df_close, "price_mat": price_mat, "spread_mat": spread_mat,
            "signal_mat": signal_mat, "eligible_mat": eligible_mat}


def transform_signal_to_weight_topk(signal_mat, eligible_mat, top_k):
    idx = signal_mat.index.intersection(eligible_mat.index)
    cols = signal_mat.columns.intersection(eligible_mat.columns)
    signal_mat = signal_mat.loc[idx, cols].copy()
    eligible_mat = eligible_mat.loc[idx, cols].copy()
    signal_used_mat = signal_mat.where(eligible_mat, np.nan)
    rank_mat = signal_used_mat.rank(axis=1, method="first", ascending=False)
    selected_mat = (rank_mat <= top_k)
    n_selected = selected_mat.sum(axis=1)
    target_w_mat = selected_mat.div(n_selected.replace(0, np.nan), axis=0).fillna(0.0).astype("float64")
    return {"target_w_mat": target_w_mat, "signal_used_mat": signal_used_mat,
            "rank_mat": rank_mat, "selected_mat": selected_mat}


def run_close_to_close_backtest_from_weight(
    price_mat, spread_mat, target_w_mat, signal_used_mat=None,
    day_col="TradingDay", stock_col="SecuCode",
    init_cash=1_000_000.0, fee_rate=0.0003, lot_size=None,
):
    idx = price_mat.index.intersection(spread_mat.index).intersection(target_w_mat.index)
    cols = price_mat.columns.intersection(spread_mat.columns).intersection(target_w_mat.columns)
    price_mat = price_mat.loc[idx, cols].copy()
    spread_mat = spread_mat.loc[idx, cols].copy()
    target_w_mat = target_w_mat.loc[idx, cols].fillna(0.0).copy()
    if signal_used_mat is None:
        signal_used_mat = pd.DataFrame(np.nan, index=idx, columns=cols)
    else:
        signal_used_mat = signal_used_mat.loc[idx, cols].copy()
    days = price_mat.index.to_numpy()
    stocks = price_mat.columns.to_numpy()
    P = price_mat.to_numpy(dtype=np.float64)
    S = spread_mat.to_numpy(dtype=np.float64)
    W = target_w_mat.to_numpy(dtype=np.float64)
    T, N = P.shape
    shares = np.zeros(N, dtype=np.float64)
    cash = float(init_cash)
    nav_pre = np.zeros(T); nav_post = np.zeros(T)
    cash_arr = np.zeros(T); fee_arr = np.zeros(T)
    spread_cost_arr = np.zeros(T); turnover_arr = np.zeros(T)

    for t in range(T):
        p_mid = np.nan_to_num(P[t], nan=0.0)
        spr = np.nan_to_num(S[t], nan=0.0)
        w_tgt = np.nan_to_num(W[t], nan=0.0)
        nav_before = cash + np.dot(shares, p_mid)
        nav_pre[t] = nav_before
        tgt_value_mid = nav_before * w_tgt
        tgt_shares = np.where(p_mid > 0, tgt_value_mid / p_mid, 0.0)
        if lot_size is not None:
            tgt_shares = np.floor(tgt_shares / lot_size) * lot_size
        tgt_shares = np.maximum(tgt_shares, 0.0)
        delta = tgt_shares - shares
        buy_mask = delta > 1e-12; sell_mask = delta < -1e-12
        buy_px = p_mid + spr / 2.0; sell_px = p_mid - spr / 2.0
        sell_shares = np.where(sell_mask, -delta, 0.0)
        sell_notional = np.sum(sell_shares * sell_px)
        sell_fee = fee_rate * sell_notional
        cash_after_sell = cash + sell_notional - sell_fee
        buy_shares = np.where(buy_mask, delta, 0.0)
        buy_notional = np.sum(buy_shares * buy_px)
        buy_fee = fee_rate * buy_notional
        total_buy_needed = buy_notional + buy_fee
        if total_buy_needed > cash_after_sell + 1e-12:
            scale = max(0.0, min(1.0, cash_after_sell / total_buy_needed if total_buy_needed > 0 else 0.0))
            buy_shares = buy_shares * scale
            if lot_size is not None:
                buy_shares = np.floor(buy_shares / lot_size) * lot_size
            buy_notional = np.sum(buy_shares * buy_px)
            buy_fee = fee_rate * buy_notional
        cash = cash_after_sell - buy_notional - buy_fee
        final_delta = np.zeros(N)
        final_delta[sell_mask] = -sell_shares[sell_mask]
        final_delta[buy_mask] = buy_shares[buy_mask]
        shares = np.maximum(shares + final_delta, 0.0)
        nav_post[t] = cash + np.dot(shares, p_mid)
        cash_arr[t] = cash
        fee_arr[t] = sell_fee + buy_fee
        spread_cost_arr[t] = np.sum(buy_shares * (buy_px - p_mid)) + np.sum(sell_shares * (p_mid - sell_px))
        turnover_arr[t] = (sell_notional + buy_notional) / nav_before if nav_before > 1e-12 else 0.0

    holding_return_gross = np.full(T, np.nan)
    holding_return_net = np.full(T, np.nan)
    if T >= 2:
        holding_return_gross[:-1] = nav_pre[1:] / nav_post[:-1] - 1.0
        holding_return_net[:-1] = nav_post[1:] / nav_post[:-1] - 1.0

    equity_curve = pd.DataFrame({
        day_col: days, "nav_pre_close": nav_pre, "nav_post_close": nav_post,
        "holding_return_gross_cc": holding_return_gross,
        "holding_return_net_cc": holding_return_net,
        "turnover": turnover_arr, "fee_cost": fee_arr,
        "spread_cost": spread_cost_arr, "cash": cash_arr,
    })
    return {"equity_curve": equity_curve, "price_mat": price_mat}


# ══════════════════════════════════════════════════════════
# SECTION 2: MATD3 Signal Generation (lightweight)
# ══════════════════════════════════════════════════════════

def delta_to_gate(delta_raw):
    return 1.0 + 0.5 * torch.tanh(delta_raw)

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim=128, action_dim=1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim))
    def forward(self, x): return self.backbone(x)

def _safe_std(x):
    x = np.asarray(x, dtype=np.float32); return float(np.std(x)) if len(x) else 0.0
def _safe_mean(x):
    x = np.asarray(x, dtype=np.float32); return float(np.mean(x)) if len(x) else 0.0
def _safe_quantile(x, q):
    x = np.asarray(x, dtype=np.float32); return float(np.quantile(x, q)) if len(x) else 0.0
def _safe_skew(x):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < 3: return 0.0
    sd = x.std()
    if sd < 1e-12: return 0.0
    z = (x - x.mean()) / sd; return float((z**3).mean())
def _safe_kurtosis_excess(x):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < 4: return 0.0
    sd = x.std()
    if sd < 1e-12: return 0.0
    z = (x - x.mean()) / sd; return float((z**4).mean() - 3.0)
def _safe_corr(x, y):
    x = np.asarray(x, dtype=np.float32); y = np.asarray(y, dtype=np.float32)
    if len(x) == 0 or x.std() < 1e-12 or y.std() < 1e-12: return 0.0
    return float(np.corrcoef(x, y)[0, 1])
def _topk_mean(x, k=5):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0: return 0.0
    k = min(k, len(x)); return float(x[np.argpartition(x, -k)[-k:]].mean())
def _bottomk_mean(x, k=5):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0: return 0.0
    k = min(k, len(x)); return float(x[np.argpartition(x, k-1)[:k]].mean())
def _topk_share_nonneg(x, k=5):
    x = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    s = x.sum()
    if s <= 1e-12: return 0.0
    k = min(k, len(x)); return float(x[np.argpartition(x, -k)[-k:]].sum() / s)
def _hhi(w):
    w = np.asarray(w, dtype=np.float32); return float((w**2).sum()) if len(w) else 0.0
def _topk_weight_sum(w, k=5):
    w = np.asarray(w, dtype=np.float32)
    if len(w) == 0: return 0.0
    k = min(k, len(w)); return float(w[np.argpartition(w, -k)[-k:]].sum())

# Agent/shared cols for MATD3 state
AGENT_COLS = ['signal_pre_boost_cs_z','signal_hybrid_cs_z','signal_ssm_tf_cs_z',
              'signal_tf_cross_cs_z','signal_gat_tf_cs_z']
SHARED_COLS = ['sig_mean_5','sig_std_5','spread_over_mid','microprice_minus_mid_over_mid',
               'log_volume_sum_cs_rank','ret_mid_t1_lag1','ret_mid_t1_lag2',
               'mean_sentiment','abnormal_sentiment','sentiment_vol','message_volume']
N_AGENTS = 5; N_SHARED = 11; STATE_DIM = 55; TOPK_K = 5

@torch.no_grad()
def generate_matd3_close_signal(df_rl, model_dir):
    """Run MATD3 prediction on all bars, return close-bar per-stock final_scores."""
    # Load actors
    actors = []
    for i in range(N_AGENTS):
        a = Actor(STATE_DIM, 128, 1).to(DEVICE)
        a.load_state_dict(torch.load(os.path.join(model_dir, f'actor_agent_{i}.pt'), map_location=DEVICE))
        a.eval()
        actors.append(a)
    print(f'Loaded {N_AGENTS} actors from {model_dir}')

    # Prepare data
    df = df_rl.copy().sort_values(['TradingDay','TimeEnd','SecuCode']).reset_index(drop=True)
    df['ts_key'] = list(zip(df['TradingDay'].values, df['TimeEnd'].values))
    unique_ts = sorted(set(df['ts_key']))
    ts_groups = {k: g.sort_values('SecuCode').reset_index(drop=True) for k, g in df.groupby('ts_key', sort=True)}

    inv_map = {}; sell_map = {}; prev_day = None
    prev_gates = np.ones(N_AGENTS, dtype=np.float32)
    turnover_lag = 0.0
    results = []  # (TradingDay, TimeEnd, SecuCode, matd3_score)

    for ts_key in tqdm(unique_ts, desc='MATD3 gate scoring'):
        sub = ts_groups[ts_key]
        day, time_end = ts_key
        codes = sub['SecuCode'].values
        n = len(codes)

        # Retrieve portfolio state
        inv_w = np.array([inv_map.get(c, 0.0) for c in codes], dtype=np.float32)
        sell_w = np.array([sell_map.get(c, 0.0) for c in codes], dtype=np.float32)
        if prev_day is not None and day != prev_day:
            sell_w = inv_w.copy()

        # Build agent signal matrix (n_stock, n_agents)
        agent_mat = np.stack([sub[c].values.astype(np.float32) for c in AGENT_COLS], axis=1)
        # Build shared matrix (n_stock, N_SHARED)
        shared_mat = np.stack([sub[c].values.astype(np.float32) for c in SHARED_COLS], axis=1)
        sh_mean = np.nanmean(shared_mat, axis=0)
        sh_std = np.nanstd(shared_mat, axis=0)
        shared_global = np.concatenate([sh_mean, sh_std]).astype(np.float32)  # (2*N_SHARED,)

        ensemble_sig = agent_mat.mean(axis=1)

        # Build states and get gates
        cash = max(0.0, 1.0 - float(inv_w.sum()))
        invested = float(inv_w.sum())
        port_global = np.array([cash, invested, float(sell_w.sum()),
                                max(0.0, invested - float(sell_w.sum())),
                                _hhi(inv_w), float(inv_w.max()) if n else 0.0,
                                _topk_weight_sum(inv_w, TOPK_K), turnover_lag], dtype=np.float32)
        gates = np.zeros((N_AGENTS, 1), dtype=np.float32)
        for i in range(N_AGENTS):
            own = agent_mat[:, i]
            q10, q25, q50, q75, q90 = [_safe_quantile(own, q) for q in [.1,.25,.5,.75,.9]]
            own_summary = np.array([
                _safe_mean(own), _safe_std(own), q10, q25, q50, q75, q90,
                _topk_mean(own, TOPK_K), _bottomk_mean(own, TOPK_K),
                q90-q50, q50-q10, _safe_skew(own), _safe_kurtosis_excess(own),
                _topk_share_nonneg(own, 1), _topk_share_nonneg(own, TOPK_K)], dtype=np.float32)
            rel_summary = np.array([_safe_corr(own, ensemble_sig),
                                     _safe_mean(np.abs(own - ensemble_sig))], dtype=np.float32)
            inv_exp = float(np.dot(inv_w, own))
            sell_exp = float(np.dot(sell_w, own))
            if n > 0:
                k = min(TOPK_K, n)
                top_idx = np.argpartition(inv_w, -k)[-k:]
                top5_sig = float(own[top_idx].mean())
            else: top5_sig = 0.0
            port_agent = np.array([float(prev_gates[i]), _safe_mean(inv_w), _safe_std(inv_w),
                                    _safe_mean(sell_w), _safe_std(sell_w),
                                    inv_exp, sell_exp, top5_sig], dtype=np.float32)
            state_i = np.concatenate([own_summary, rel_summary, shared_global, port_global, port_agent])
            s_t = torch.tensor(state_i, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            delta_raw = actors[i](s_t)
            g = np.clip(delta_to_gate(delta_raw).squeeze(0).cpu().numpy(), 0.5, 1.5).astype(np.float32)
            gates[i] = g

        # Compute gate-weighted score per stock
        agent_scores = agent_mat * gates.reshape(1, -1)  # (n_stock, n_agents)
        final_score = agent_scores.mean(axis=1)  # (n_stock,)

        for j in range(n):
            results.append((day, time_end, codes[j], float(final_score[j])))

        # Update portfolio (simplified: use top-20 equal weight to track state)
        top_k = min(20, n)
        if top_k > 0:
            top_idx = np.argpartition(final_score, -top_k)[-top_k:]
            new_w = np.zeros(n, dtype=np.float32)
            new_w[top_idx] = 1.0 / top_k
        else:
            new_w = np.zeros(n, dtype=np.float32)

        inv_map = {codes[j]: float(new_w[j]) for j in range(n)}
        sell_map = {codes[j]: float(sell_w[j]) for j in range(n)}
        prev_day = day
        prev_gates = gates.reshape(-1).astype(np.float32)
        turnover_lag = float(np.abs(new_w - inv_w).sum())

    df_scores = pd.DataFrame(results, columns=['TradingDay','TimeEnd','SecuCode','matd3_score'])
    return df_scores


# ══════════════════════════════════════════════════════════
# SECTION 3: Metrics
# ══════════════════════════════════════════════════════════

def compute_metrics(eq, ann_factor=243):
    """Compute metrics from equity_curve DataFrame."""
    rets = eq['holding_return_net_cc'].dropna()
    n = len(rets)
    total_ret = (1 + rets).prod() - 1
    ann_ret = (1 + total_ret) ** (ann_factor / max(n, 1)) - 1
    ann_vol = rets.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / max(ann_vol, 1e-8)
    nav = (1 + rets).cumprod()
    dd = (nav - nav.cummax()) / nav.cummax().clip(lower=1e-8)
    max_dd = float(dd.min())
    calmar = ann_ret / max(abs(max_dd), 1e-8)
    neg = rets[rets < 0]
    sortino = ann_ret / max(neg.std() * np.sqrt(ann_factor) if len(neg) > 1 else 1e-8, 1e-8)
    win_rate = float((rets > 0).mean())
    avg_turnover = eq['turnover'].mean()
    return {'Total Return': total_ret, 'Ann. Return': ann_ret, 'Ann. Vol': ann_vol,
            'Sharpe': sharpe, 'Sortino': sortino, 'Calmar': calmar,
            'Max Drawdown': max_dd, 'Win Rate': win_rate, 'Avg Turnover': avg_turnover}


# ══════════════════════════════════════════════════════════
# SECTION 4: Run All Backtests
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    TOP_K = 5
    FEE_RATE = 0.0003
    CLOSE_TS = 1445
    CLOSE_TE = 1500

    # ── Load data ─────────────────────────────────────────
    print('Loading df_testsub_full.parquet ...')
    df_raw = pd.read_parquet('df_testsub_full.parquet')
    print(f'Shape: {df_raw.shape}')

    # Raw signal configs: (display_name, column_in_df)
    RAW_SIGNALS = [
        ('LightGBM',     'signal_pre_boost'),
        ('CNN+LSTM+TF',  'signal_hybrid'),
        ('SSM+TF',       'signal_ssm_tf'),
        ('Cross-Attn',   'signal_tf_cross'),
        ('GAT+TF',       'signal_gat_tf'),
    ]

    # ── EW Ensemble signal ────────────────────────────────
    raw_sig_cols = [s[1] for s in RAW_SIGNALS]
    df_raw['signal_ew_ensemble'] = df_raw[raw_sig_cols].mean(axis=1)
    RAW_SIGNALS.append(('EW Ensemble', 'signal_ew_ensemble'))

    # ── MATD3 signal (from trained model) ─────────────────
    print('\nGenerating MATD3 gate-weighted signals...')
    df_rl = pd.read_parquet('df_full_rl_clean.parquet')
    matd3_scores = generate_matd3_close_signal(df_rl, OUT_DIR)

    # Merge MATD3 scores back into df_raw
    df_raw = df_raw.merge(
        matd3_scores.rename(columns={'matd3_score': 'signal_matd3'}),
        on=['TradingDay', 'TimeEnd', 'SecuCode'], how='left'
    )
    df_raw['signal_matd3'] = df_raw['signal_matd3'].fillna(0.0)
    RAW_SIGNALS.append(('MATD3 (Composite)', 'signal_matd3'))

    # ── Run backtests ─────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'Running close-to-close backtests (top_k={TOP_K}, fee={FEE_RATE})')
    print(f'{"="*60}')

    all_eq = {}     # name -> equity_curve DataFrame
    all_metrics = {}

    for name, sig_col in RAW_SIGNALS:
        print(f'\n  [{name}] signal_col={sig_col}')
        prep = prepare_close_bar_data(
            df=df_raw, signal_col=sig_col, price_col='mid', spread_col='spread',
            day_col='TradingDay', stock_col='SecuCode',
            close_ts=CLOSE_TS, close_te=CLOSE_TE,
        )
        wobj = transform_signal_to_weight_topk(
            signal_mat=prep['signal_mat'], eligible_mat=prep['eligible_mat'], top_k=TOP_K,
        )
        res = run_close_to_close_backtest_from_weight(
            price_mat=prep['price_mat'], spread_mat=prep['spread_mat'],
            target_w_mat=wobj['target_w_mat'], signal_used_mat=wobj['signal_used_mat'],
            day_col='TradingDay', stock_col='SecuCode',
            init_cash=1_000_000.0, fee_rate=FEE_RATE, lot_size=None,
        )
        eq = res['equity_curve']
        gross_nav = (1 + eq['holding_return_gross_cc'].fillna(0)).cumprod()
        net_nav = (1 + eq['holding_return_net_cc'].fillna(0)).cumprod()
        print(f'    Gross NAV: {gross_nav.iloc[-1]:.4f}  Net NAV: {net_nav.iloc[-1]:.4f}')

        all_eq[name] = eq
        all_metrics[name] = compute_metrics(eq)

    # Also run zero-cost backtest for LightGBM (paper PnL)
    print('\n  [LightGBM Zero-Cost] ...')
    df_zero = df_raw.copy(); df_zero['spread'] = 0.0
    prep_z = prepare_close_bar_data(df=df_zero, signal_col='signal_pre_boost', price_col='mid',
                                     spread_col='spread', day_col='TradingDay', stock_col='SecuCode',
                                     close_ts=CLOSE_TS, close_te=CLOSE_TE)
    wobj_z = transform_signal_to_weight_topk(prep_z['signal_mat'], prep_z['eligible_mat'], TOP_K)
    res_z = run_close_to_close_backtest_from_weight(
        prep_z['price_mat'], prep_z['spread_mat'], wobj_z['target_w_mat'], wobj_z['signal_used_mat'],
        init_cash=1_000_000.0, fee_rate=0.0, lot_size=None)
    eq_z = res_z['equity_curve']
    gross_nav_z = (1 + eq_z['holding_return_gross_cc'].fillna(0)).cumprod()
    print(f'    Paper NAV: {gross_nav_z.iloc[-1]:.4f}')
    all_eq['LightGBM (Zero-Cost)'] = eq_z

    # ══════════════════════════════════════════════════════
    # SECTION 5: Metrics Table
    # ══════════════════════════════════════════════════════
    print(f'\n{"="*80}')
    print('PERFORMANCE METRICS (Close-to-Close, Net of Fee+Spread)')
    print(f'{"="*80}')
    df_m = pd.DataFrame(all_metrics).T
    df_m_fmt = df_m.copy()
    for c in ['Total Return', 'Ann. Return', 'Ann. Vol', 'Max Drawdown', 'Win Rate']:
        df_m_fmt[c] = df_m_fmt[c].map(lambda x: f'{x:.2%}')
    for c in ['Sharpe', 'Sortino', 'Calmar']:
        df_m_fmt[c] = df_m_fmt[c].map(lambda x: f'{x:.3f}')
    df_m_fmt['Avg Turnover'] = df_m_fmt['Avg Turnover'].map(lambda x: f'{x:.4f}')
    print(df_m_fmt.to_string())
    df_m.to_csv(os.path.join(OUT_DIR, 'metrics_comparison_full.csv'))

    # ══════════════════════════════════════════════════════
    # SECTION 6: PnL Comparison Chart
    # ══════════════════════════════════════════════════════
    print('\nGenerating comparison charts...')

    sig_colors = {'LightGBM': '#1f77b4', 'CNN+LSTM+TF': '#ff7f0e',
                  'SSM+TF': '#2ca02c', 'Cross-Attn': '#d62728', 'GAT+TF': '#9467bd',
                  'EW Ensemble': '#888888', 'MATD3 (Composite)': '#e31a1c',
                  'LightGBM (Zero-Cost)': '#1f77b4'}

    # ── Chart 1: NAV + Drawdown ───────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(16, 11), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'Close-to-Close Backtest: All Models Comparison\n(Top-{TOP_K} EW, Fee={FEE_RATE}, Init Cash=1M)',
                 fontsize=14, fontweight='bold')

    ax1 = axes[0]
    for name, eq in all_eq.items():
        rets = eq['holding_return_net_cc'].fillna(0) if 'Zero-Cost' not in name else eq['holding_return_gross_cc'].fillna(0)
        nav = (1 + rets).cumprod()
        days = pd.to_datetime(eq['TradingDay'].astype(str), format='%Y%m%d')
        lw = 3.0 if 'MATD3' in name else (2.0 if 'Zero-Cost' in name else 1.3)
        ls = '--' if 'Ensemble' in name or 'Zero-Cost' in name else '-'
        alpha = 0.95 if 'MATD3' in name else (0.6 if 'Zero-Cost' in name else 0.8)
        zorder = 10 if 'MATD3' in name else 5
        ax1.plot(days, nav.values, label=name, color=sig_colors.get(name, 'gray'),
                 linewidth=lw, linestyle=ls, alpha=alpha, zorder=zorder)

    ax1.axhline(1.0, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
    ax1.set_ylabel('NAV (Cumulative)', fontsize=12)
    ax1.set_title('Net Asset Value', fontsize=12)
    ax1.legend(fontsize=8, loc='best', ncol=2, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Drawdown panel
    ax2 = axes[1]
    for name, eq in all_eq.items():
        if 'Zero-Cost' in name: continue
        rets = eq['holding_return_net_cc'].fillna(0)
        nav = (1 + rets).cumprod()
        dd = (nav - nav.cummax()) / nav.cummax().clip(lower=1e-8)
        days = pd.to_datetime(eq['TradingDay'].astype(str), format='%Y%m%d')
        lw = 2.0 if 'MATD3' in name else 0.9
        ax2.plot(days, dd.values, label=name if 'MATD3' in name or 'Ensemble' in name else '',
                 color=sig_colors.get(name, 'gray'), linewidth=lw, alpha=0.7)
    ax2.set_ylabel('Drawdown', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.legend(fontsize=8, loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out1 = os.path.join(OUT_DIR, 'pnl_comparison_all_models.png')
    fig.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out1}')

    # ── Chart 2: Daily Return Bar Chart ───────────────────
    fig2, ax3 = plt.subplots(1, 1, figsize=(16, 6))
    fig2.suptitle('Daily Net Return: MATD3 vs LightGBM (Best Individual)', fontsize=13, fontweight='bold')

    eq_matd3 = all_eq.get('MATD3 (Composite)')
    eq_lgbm = all_eq.get('LightGBM')
    if eq_matd3 is not None and eq_lgbm is not None:
        days_dt = pd.to_datetime(eq_matd3['TradingDay'].astype(str), format='%Y%m%d')
        x = np.arange(len(days_dt))
        width = 0.4
        r_matd3 = eq_matd3['holding_return_net_cc'].fillna(0).values * 100
        r_lgbm = eq_lgbm['holding_return_net_cc'].fillna(0).values * 100
        ax3.bar(x - width/2, r_matd3, width, label='MATD3', color='#e31a1c', alpha=0.7)
        ax3.bar(x + width/2, r_lgbm, width, label='LightGBM', color='#1f77b4', alpha=0.7)
        ax3.axhline(0, color='black', linewidth=0.8)
        ax3.set_ylabel('Daily Net Return (%)')
        ax3.set_xlabel('Trading Day')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        step = max(1, len(x) // 20)
        ax3.set_xticks(x[::step])
        ax3.set_xticklabels([d.strftime('%Y-%m') for d in days_dt[::step]], rotation=45, ha='right', fontsize=7)
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, 'daily_returns_comparison.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved: {out2}')

    # ── Chart 3: Metrics Summary Bar Chart ────────────────
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
    fig3.suptitle('Key Metrics Comparison', fontsize=13, fontweight='bold')
    main_models = [n for n in all_metrics.keys()]
    x = np.arange(len(main_models))

    for ax_i, metric, title in zip(axes3,
        ['Sharpe', 'Max Drawdown', 'Total Return'],
        ['Sharpe Ratio', 'Max Drawdown', 'Total Return']):
        vals = [all_metrics[n][metric] for n in main_models]
        colors = ['#e31a1c' if 'MATD3' in n else '#1f77b4' for n in main_models]
        bars = ax_i.bar(x, vals, color=colors, alpha=0.8)
        ax_i.set_xticks(x)
        ax_i.set_xticklabels([n.replace(' (Composite)', '') for n in main_models],
                              rotation=30, ha='right', fontsize=7)
        ax_i.set_title(title)
        ax_i.axhline(0, color='black', linewidth=0.5)
        ax_i.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(bars, vals):
            fmt = f'{v:.2%}' if 'Return' in title or 'Drawdown' in title else f'{v:.2f}'
            ax_i.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                      fmt, ha='center', va='bottom', fontsize=7, fontweight='bold')
    plt.tight_layout()
    out3 = os.path.join(OUT_DIR, 'metrics_bar_chart.png')
    fig3.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f'Saved: {out3}')

    print(f'\n=== All results saved to {OUT_DIR}/ ===')
