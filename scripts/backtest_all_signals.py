"""
Backtest all individual signals + EW ensemble, generate PnL comparison chart.
Also serves as benchmark baseline for MATD3 comparison.
Outputs saved to matd3_composite_models/
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

BASE_DIR = 'e:/New_folder/Work/MyProject/capstone'
os.chdir(BASE_DIR)
OUT_DIR = 'matd3_composite_models'
os.makedirs(OUT_DIR, exist_ok=True)

ANN_FACTOR = 3888  # 16 bars/day * 243 days/year

# ── Config ────────────────────────────────────────────────
SIGNAL_COLS = [
    'signal_pre_boost_cs_z',
    'signal_hybrid_cs_z',
    'signal_ssm_tf_cs_z',
    'signal_tf_cross_cs_z',
    'signal_gat_tf_cs_z',
]
SIGNAL_NAMES = [
    'LightGBM (pre_boost)',
    'CNN+LSTM+TF (hybrid)',
    'SSM+TF (ssm_tf)',
    'Cross-Attn (tf_cross)',
    'GAT+TF (gat_tf)',
]
TOPK = 20
USE_HALF_SPREAD = True

# ── Load Data ─────────────────────────────────────────────
print('Loading df_full_rl_clean.parquet ...')
df = pd.read_parquet('df_full_rl_clean.parquet')
print(f'Shape: {df.shape}')
print(f'Split: {df["dataset_split"].value_counts().to_dict()}')

# ── Backtest Engine ───────────────────────────────────────
def backtest_signal(df, signal_col, topk=20, use_half_spread=True):
    """Walk-forward backtest: rebalance every bar, top-K equal weight, with spread costs."""
    nav = 1.0
    records = []
    prev_w = {}  # code -> weight

    groups = df.groupby(['TradingDay', 'TimeEnd'], sort=True)
    for (day, time_end), g in groups:
        codes   = g['SecuCode'].values
        signals = g[signal_col].values
        rets    = g['ret_mid_t1'].values.astype(np.float64)
        mids    = g['mid'].values.astype(np.float64)
        spreads = g['spread'].values.astype(np.float64)

        n = min(topk, len(signals))
        # Top-K by signal (handle NaN by filling -inf)
        sig_clean = np.nan_to_num(signals, nan=-np.inf)
        top_idx = np.argpartition(sig_clean, -n)[-n:]

        # Equal weight among top-K
        w = np.zeros(len(signals), dtype=np.float64)
        w[top_idx] = 1.0 / n

        # Spread cost
        prev_w_arr = np.array([prev_w.get(c, 0.0) for c in codes], dtype=np.float64)
        delta_w = w - prev_w_arr
        spread_frac = np.divide(spreads, np.maximum(mids, 1e-8),
                                out=np.zeros_like(spreads), where=mids > 0)
        half = 0.5 if use_half_spread else 1.0
        spread_cost = float(np.sum(np.abs(delta_w) * half * spread_frac))
        turnover = float(np.abs(delta_w).sum())

        gross_ret = float(np.dot(w, rets))
        net_ret   = gross_ret - spread_cost
        nav *= (1 + net_ret)

        records.append({
            'TradingDay': day, 'TimeEnd': time_end,
            'gross_return': gross_ret, 'net_return': net_ret,
            'spread_cost': spread_cost, 'turnover': turnover, 'nav': nav,
        })
        prev_w = {c: float(w[i]) for i, c in enumerate(codes)}

    return pd.DataFrame(records)


def backtest_ensemble(df, signal_cols, topk=20, use_half_spread=True):
    """Equal-weight ensemble: average all signals, then top-K."""
    df_tmp = df.copy()
    df_tmp['_ensemble_score'] = df_tmp[signal_cols].mean(axis=1)
    result = backtest_signal(df_tmp, '_ensemble_score', topk, use_half_spread)
    return result


# ── Performance Metrics ───────────────────────────────────
def compute_metrics(nav_series, ann_factor=ANN_FACTOR):
    rets      = nav_series.pct_change().dropna()
    total_ret = nav_series.iloc[-1] / nav_series.iloc[0] - 1
    n         = len(rets)
    ann_ret   = (1 + total_ret) ** (ann_factor / max(n, 1)) - 1
    ann_vol   = rets.std() * np.sqrt(ann_factor)
    sharpe    = ann_ret / max(ann_vol, 1e-8)
    rolling_max = nav_series.cummax()
    dd_ts     = (nav_series - rolling_max) / rolling_max.clip(lower=1e-8)
    max_dd    = float(dd_ts.min())
    calmar    = ann_ret / max(abs(max_dd), 1e-8)
    neg       = rets[rets < 0]
    sortino   = ann_ret / max(neg.std() * np.sqrt(ann_factor) if len(neg) > 1 else 1e-8, 1e-8)
    win_rate  = float((rets > 0).mean())
    return {
        'Total Return': total_ret,
        'Ann. Return':  ann_ret,
        'Ann. Vol':     ann_vol,
        'Sharpe':       sharpe,
        'Sortino':      sortino,
        'Calmar':       calmar,
        'Max Drawdown': max_dd,
        'Win Rate':     win_rate,
    }


# ── Run All Backtests ─────────────────────────────────────
print('\n=== Running Individual Signal Backtests ===')
results = {}

# Full period (train + test)
for sig_col, sig_name in zip(SIGNAL_COLS, SIGNAL_NAMES):
    t0 = time.time()
    bt = backtest_signal(df, sig_col, TOPK, USE_HALF_SPREAD)
    dt = time.time() - t0
    results[sig_name] = bt
    print(f'  {sig_name:30s}  NAV={bt["nav"].iloc[-1]:.4f}  ({dt:.1f}s)')

# Ensemble
t0 = time.time()
bt_ew = backtest_ensemble(df, SIGNAL_COLS, TOPK, USE_HALF_SPREAD)
results['EW Ensemble (5 signals)'] = bt_ew
dt = time.time() - t0
print(f'  {"EW Ensemble (5 signals)":30s}  NAV={bt_ew["nav"].iloc[-1]:.4f}  ({dt:.1f}s)')

# Find train/test boundary
train_days = set(df[df['dataset_split'] == 'train']['TradingDay'].unique())
# Get boundary index for the first result
first_bt = list(results.values())[0]
split_idx = None
for i, row in first_bt.iterrows():
    if row['TradingDay'] not in train_days:
        split_idx = i
        break

# ── Metrics Table ─────────────────────────────────────────
print('\n=== Full Period Metrics ===')
metrics_full = {}
for name, bt in results.items():
    metrics_full[name] = compute_metrics(bt['nav'])

df_metrics_full = pd.DataFrame(metrics_full).T
df_metrics_full_fmt = df_metrics_full.copy()
for c in ['Total Return', 'Ann. Return', 'Ann. Vol', 'Max Drawdown', 'Win Rate']:
    df_metrics_full_fmt[c] = df_metrics_full_fmt[c].map(lambda x: f'{x:.2%}')
for c in ['Sharpe', 'Sortino', 'Calmar']:
    df_metrics_full_fmt[c] = df_metrics_full_fmt[c].map(lambda x: f'{x:.3f}')
print(df_metrics_full_fmt.to_string())

# Test-only metrics
if split_idx is not None:
    print('\n=== Test Period Metrics ===')
    metrics_test = {}
    for name, bt in results.items():
        bt_test = bt.iloc[split_idx:].copy()
        bt_test_nav = (1 + bt_test['net_return']).cumprod()
        metrics_test[name] = compute_metrics(bt_test_nav)
    df_metrics_test = pd.DataFrame(metrics_test).T
    df_metrics_test_fmt = df_metrics_test.copy()
    for c in ['Total Return', 'Ann. Return', 'Ann. Vol', 'Max Drawdown', 'Win Rate']:
        df_metrics_test_fmt[c] = df_metrics_test_fmt[c].map(lambda x: f'{x:.2%}')
    for c in ['Sharpe', 'Sortino', 'Calmar']:
        df_metrics_test_fmt[c] = df_metrics_test_fmt[c].map(lambda x: f'{x:.3f}')
    print(df_metrics_test_fmt.to_string())

# ── Save metrics ──────────────────────────────────────────
df_metrics_full.to_csv(os.path.join(OUT_DIR, 'metrics_full_period.csv'))
if split_idx is not None:
    df_metrics_test.to_csv(os.path.join(OUT_DIR, 'metrics_test_period.csv'))

# ── Save NAV curves to parquet ────────────────────────────
for name, bt in results.items():
    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    bt.to_parquet(os.path.join(OUT_DIR, f'nav_{safe_name}.parquet'), index=False)
print(f'\nNAV curves saved to {OUT_DIR}/')

# ══════════════════════════════════════════════════════════
# PLOT 1: Full Period PnL Curves (Net NAV)
# ══════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle('Signal PnL Comparison — Top-20 EW Portfolio (Net of Spread)',
             fontsize=15, fontweight='bold')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2', '#000000']
ax = axes[0]

for idx, (name, bt) in enumerate(results.items()):
    lw = 2.5 if 'Ensemble' in name else 1.5
    ls = '-' if 'Ensemble' not in name else '--'
    ax.plot(bt['nav'].values, label=name, color=colors[idx % len(colors)],
            linewidth=lw, linestyle=ls, alpha=0.85)

if split_idx is not None:
    ax.axvline(split_idx, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='Train | Test')

ax.set_ylabel('NAV (Net)', fontsize=12)
ax.set_title('Cumulative PnL Curves — Full Period', fontsize=13)
ax.legend(fontsize=9, loc='upper left', ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, len(first_bt))

# Panel 2: Drawdown comparison (test period only)
ax2 = axes[1]
start = split_idx if split_idx is not None else 0
for idx, (name, bt) in enumerate(results.items()):
    bt_sub = bt.iloc[start:].copy()
    nav_sub = (1 + bt_sub['net_return']).cumprod()
    dd = (nav_sub - nav_sub.cummax()) / nav_sub.cummax().clip(lower=1e-8)
    ax2.plot(dd.values, label=name, color=colors[idx % len(colors)],
             linewidth=1.2, alpha=0.7)

ax2.set_ylabel('Drawdown', fontsize=12)
ax2.set_title('Test Period Drawdown', fontsize=13)
ax2.set_xlabel('Timestamp Index (test period)', fontsize=11)
ax2.legend(fontsize=8, loc='lower left', ncol=3)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'pnl_comparison_signals.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out_path}')

# ══════════════════════════════════════════════════════════
# PLOT 2: Test Period Only — Zoomed PnL
# ══════════════════════════════════════════════════════════
if split_idx is not None:
    fig2, ax3 = plt.subplots(1, 1, figsize=(14, 7))
    fig2.suptitle('Test Period PnL — Top-20 EW Portfolio (Net of Spread)',
                  fontsize=14, fontweight='bold')

    for idx, (name, bt) in enumerate(results.items()):
        bt_sub = bt.iloc[split_idx:].copy()
        nav_sub = (1 + bt_sub['net_return']).cumprod()
        lw = 2.5 if 'Ensemble' in name else 1.5
        ls = '-' if 'Ensemble' not in name else '--'
        ax3.plot(nav_sub.values, label=name, color=colors[idx % len(colors)],
                 linewidth=lw, linestyle=ls, alpha=0.85)

    ax3.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
    ax3.set_ylabel('NAV (Net)', fontsize=12)
    ax3.set_xlabel('Timestamp Index (test period)', fontsize=11)
    ax3.legend(fontsize=9, loc='best', ncol=2)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path2 = os.path.join(OUT_DIR, 'pnl_test_period_signals.png')
    fig2.savefig(out_path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved: {out_path2}')

# ══════════════════════════════════════════════════════════
# PLOT 3: Daily PnL Bar Chart (Test Period, aggregate by day)
# ══════════════════════════════════════════════════════════
if split_idx is not None:
    fig3, ax4 = plt.subplots(1, 1, figsize=(14, 5))
    fig3.suptitle('Daily Net Return — EW Ensemble vs Best Individual Signal (Test)',
                  fontsize=13, fontweight='bold')

    # EW Ensemble daily returns
    bt_ew_test = bt_ew.iloc[split_idx:].copy()
    daily_ew = bt_ew_test.groupby('TradingDay')['net_return'].sum()

    # Best individual signal (by Sharpe)
    best_name = df_metrics_test['Sharpe'].idxmax() if 'Sharpe' in df_metrics_test.columns else SIGNAL_NAMES[0]
    bt_best_test = results[best_name].iloc[split_idx:].copy()
    daily_best = bt_best_test.groupby('TradingDay')['net_return'].sum()

    x = np.arange(len(daily_ew))
    width = 0.4
    ax4.bar(x - width/2, daily_ew.values * 100, width, label='EW Ensemble', color='steelblue', alpha=0.7)
    ax4.bar(x + width/2, daily_best.values * 100, width, label=f'Best: {best_name}', color='darkorange', alpha=0.7)
    ax4.axhline(0, color='black', linewidth=0.8)
    ax4.set_ylabel('Daily Net Return (%)', fontsize=11)
    ax4.set_xlabel('Trading Day Index', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # Show every 10th day label
    day_labels = daily_ew.index.astype(str).values
    step = max(1, len(day_labels) // 15)
    ax4.set_xticks(x[::step])
    ax4.set_xticklabels(day_labels[::step], rotation=45, ha='right', fontsize=7)

    plt.tight_layout()
    out_path3 = os.path.join(OUT_DIR, 'daily_returns_comparison.png')
    fig3.savefig(out_path3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f'Saved: {out_path3}')

print('\n=== All signal backtests complete ===')
print(f'Results in: {OUT_DIR}/')
