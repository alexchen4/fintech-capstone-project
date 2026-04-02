"""
Create rl_full_pipeline_v5.ipynb from v4 with three key changes:
  1. EW top-k portfolio (removes v4 softmax + turnover cap -> fixes train/test mismatch)
  2. Calmar-focused reward: dd_lambda=0.15, remove vol penalty, turnover_lambda=0.25
  3. Updated comparison cells with monkey-patching for correct v2/v3 re-evaluation

Why v4 regressed vs v3:
  - Softmax portfolio in training != EW top-k in backtest -> train/test mismatch
  - Vol targeting constraint was too restrictive (vol_excess rarely > target)
  - turnover_lambda=0.5 was too aggressive, suppressing signal differentiation
"""
import json
import copy
import re

with open('notebooks/rl_full_pipeline_v4.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

nb = copy.deepcopy(nb)


def get_src(idx):
    return ''.join(nb['cells'][idx]['source'])


def set_src(idx, src):
    nb['cells'][idx]['source'] = [src]
    nb['cells'][idx]['outputs'] = []
    nb['cells'][idx]['execution_count'] = None


# ── Cell 1: Header ─────────────────────────────────────────────────────────────
src = get_src(0)
src = src.replace(
    'RL Full Pipeline v4: MATD3 + Turnover Penalty + Softmax Weights + Vol Targeting',
    'RL Full Pipeline v5: MATD3 + EW Portfolio + Calmar Reward (dd=0.15) + Turnover Penalty'
)
set_src(0, src)

# ── Cell 4: CFG parameters ──────────────────────────────────────────────────────
src = get_src(3)

cfg_changes = [
    ('dd_lambda:               float = 0.1    # Calmar: drawdown penalty weight',
     'dd_lambda:               float = 0.15   # v5: Calmar-focused (v3=0.1 < v5=0.15 < v2=0.3)'),
    ('vol_lambda:              float = 3.0   # Sharpe: variance penalty weight',
     'vol_lambda:              float = 0.0    # v5: removed -- dd_lambda handles risk'),
    ('vol_target_daily:        float = 0.00078  # v4: ~20% ann vol target per bar (20%/sqrt(252*16))',
     '# vol_target_daily: removed in v5 (was causing over-constraint)'),
    ('turnover_lambda:         float = 0.5   # v4: cost-aware training \u2014 penalise daily turnover',
     'turnover_lambda:         float = 0.25  # v5: moderate cost-aware training'),
    ('reward_scale:            float = 2000.0',
     'reward_scale:            float = 2500.0'),
    ('epochs:                      int   = 80',
     'epochs:                      int   = 100'),
    ('action_noise_std:            float = 0.10',
     'action_noise_std:            float = 0.12'),
    ("save_dir: str = 'matd3_v4_models'",
     "save_dir: str = 'matd3_v5_models'"),
]

for old, new in cfg_changes:
    if old in src:
        src = src.replace(old, new, 1)
    else:
        print(f'CFG WARNING not found: {old[:70]}')

set_src(3, src)

# ── Cell 10: Reward -- remove vol targeting, keep dd + turnover ─────────────────
src = get_src(9)

old_comment = ('# Composite reward = net_return - dd_lambda*drawdown - vol_lambda*rolling_var')
new_comment = ('# v5 reward = net_return - dd_lambda*drawdown - turnover_lambda*turnover\n'
               '# (no vol penalty -- drawdown directly captures risk for Calmar improvement)')
src = src.replace(old_comment, new_comment, 1)

old_penalty = (
    "    dd_penalty       = cfg.dd_lambda  * tracker.drawdown\n"
    "    # v4: volatility targeting \u2014 only penalise excess vol above target\n"
    "    rolling_vol      = float(np.std(tracker._buf)) if len(tracker._buf) >= 5 else 0.0\n"
    "    vol_excess       = max(0.0, rolling_vol - cfg.vol_target_daily)\n"
    "    var_penalty      = cfg.vol_lambda * vol_excess ** 2 * 1000.0\n"
    "    # v4: turnover penalty \u2014 discourage excessive rebalancing\n"
    "    turnover_penalty = cfg.turnover_lambda * turnover\n"
    "    composite        = net_return - dd_penalty - var_penalty - turnover_penalty"
)
new_penalty = (
    "    dd_penalty       = cfg.dd_lambda  * tracker.drawdown\n"
    "    # v5: no vol penalty -- dd_lambda directly targets Calmar via drawdown depth\n"
    "    turnover_penalty = cfg.turnover_lambda * turnover\n"
    "    composite        = net_return - dd_penalty - turnover_penalty"
)
if old_penalty in src:
    src = src.replace(old_penalty, new_penalty, 1)
else:
    print('REWARD WARNING: penalty block not found')

old_return = (
    "        'vol_excess':      vol_excess,\n"
    "        'turnover_penalty':turnover_penalty,\n"
)
new_return = (
    "        'turnover_penalty':turnover_penalty,\n"
)
src = src.replace(old_return, new_return, 1)

set_src(9, src)

# ── Cell 13: Portfolio -- remove softmax + turnover cap, restore EW top-k ───────
src = get_src(12)

old_fn = (
    'def build_portfolio_weights_from_agent_scores(\n'
    '    agent_scores: np.ndarray,\n'
    '    prev_w: np.ndarray,\n'
    '    long_only: bool = True,\n'
    '    max_weight_per_stock: float = 0.20,\n'
    '    topk: Optional[int] = None,\n'
    '    softmax_temperature: float = 3.0,    # v4: signal-proportional weighting\n'
    '    max_daily_turnover: float = 0.5,     # v4: hard cap on daily rebalancing\n'
    ') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:\n'
    '    """v4: Softmax portfolio weights + daily turnover cap."""\n'
    '    final_score = agent_scores.mean(axis=1).astype(np.float32)\n'
    '\n'
    '    if topk is not None and topk < len(final_score):\n'
    '        idx  = np.argpartition(final_score, -topk)[-topk:]\n'
    '        mask = np.zeros(len(final_score), dtype=bool); mask[idx] = True\n'
    '        in_scores = final_score.copy()\n'
    '        in_scores[~mask] = -1e9\n'
    '    else:\n'
    '        in_scores = final_score.copy()\n'
    '\n'
    '    if long_only:\n'
    '        # v4: softmax over top-k scores (signal-proportional weights)\n'
    '        valid = in_scores > -1e8\n'
    '        w = np.zeros(len(final_score), dtype=np.float32)\n'
    '        if valid.any():\n'
    '            s = in_scores[valid]\n'
    '            # Subtract max for numerical stability before exp\n'
    '            s = s - s.max()\n'
    '            e = np.exp(softmax_temperature * s)\n'
    '            e = e / e.sum()\n'
    '            w[valid] = e.astype(np.float32)\n'
    '        w = np.clip(w, 0.0, max_weight_per_stock).astype(np.float32)\n'
    '        s = w.sum()\n'
    '        if s > 1e-12:\n'
    '            w = (w / s).astype(np.float32)\n'
    '    else:\n'
    '        pos = np.maximum(in_scores, 0.0).astype(np.float32)\n'
    '        neg = np.maximum(-in_scores, 0.0).astype(np.float32)\n'
    '        if pos.sum() > 1e-12: pos = pos / pos.sum()\n'
    '        if neg.sum() > 1e-12: neg = neg / neg.sum()\n'
    '        w = (pos - neg).astype(np.float32)\n'
    '\n'
    '    # v4: daily turnover cap \u2014 smooth rebalancing\n'
    '    delta_raw = w - prev_w\n'
    '    total_chg = float(np.abs(delta_raw).sum())\n'
    '    if total_chg > max_daily_turnover and total_chg > 1e-8:\n'
    '        scale = max_daily_turnover / total_chg\n'
    '        w = (prev_w + delta_raw * scale).astype(np.float32)\n'
    '        w = np.clip(w, 0.0, max_weight_per_stock).astype(np.float32)\n'
    '        s = w.sum()\n'
    '        if s > 1e-12:\n'
    '            w = (w / s).astype(np.float32)\n'
    '\n'
    '    delta_w = (w - prev_w).astype(np.float32)\n'
    '    return final_score, w, np.maximum(delta_w, 0.0).astype(np.float32), np.maximum(-delta_w, 0.0).astype(np.float32)\n'
)

new_fn = (
    'def build_portfolio_weights_from_agent_scores(\n'
    '    agent_scores: np.ndarray,\n'
    '    prev_w: np.ndarray,\n'
    '    long_only: bool = True,\n'
    '    max_weight_per_stock: float = 0.20,\n'
    '    topk: Optional[int] = None,\n'
    ') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:\n'
    '    """v5: Equal-weight top-k portfolio -- consistent with close-to-close backtest.\n'
    '    Removes v4 softmax (caused train/test mismatch) and turnover cap.\n'
    '    """\n'
    '    final_score = agent_scores.mean(axis=1).astype(np.float32)\n'
    '\n'
    '    if topk is not None and topk < len(final_score):\n'
    '        # EW: select top-k by signal, assign equal weight to each\n'
    '        idx = np.argpartition(final_score, -topk)[-topk:]\n'
    '        w   = np.zeros(len(final_score), dtype=np.float32)\n'
    '        if topk > 0:\n'
    '            w[idx] = 1.0 / topk\n'
    '    else:\n'
    '        n = max(len(final_score), 1)\n'
    '        w = np.ones(len(final_score), dtype=np.float32) / n\n'
    '\n'
    '    if long_only:\n'
    '        w = np.clip(w, 0.0, max_weight_per_stock).astype(np.float32)\n'
    '        s = w.sum()\n'
    '        if s > 1e-12:\n'
    '            w = (w / s).astype(np.float32)\n'
    '    else:\n'
    '        pos = np.maximum(final_score, 0.0).astype(np.float32)\n'
    '        neg = np.maximum(-final_score, 0.0).astype(np.float32)\n'
    '        if pos.sum() > 1e-12: pos = pos / pos.sum()\n'
    '        if neg.sum() > 1e-12: neg = neg / neg.sum()\n'
    '        w = (pos - neg).astype(np.float32)\n'
    '\n'
    '    delta_w = (w - prev_w).astype(np.float32)\n'
    '    return final_score, w, np.maximum(delta_w, 0.0).astype(np.float32), np.maximum(-delta_w, 0.0).astype(np.float32)\n'
)

if old_fn in src:
    src = src.replace(old_fn, new_fn, 1)
    src = src.replace(
        '# build_portfolio_weights_from_agent_scores\n# execute_target_weight_t1',
        '# build_portfolio_weights_from_agent_scores (v5: EW top-k, no softmax)\n# execute_target_weight_t1'
    )
    src = src.replace(
        'print(\'build_portfolio_weights_from_agent_scores defined.',
        'print(\'v5 EW build_portfolio_weights_from_agent_scores defined.'
    )
else:
    print('PORTFOLIO WARNING: function not found')
    idx = src.find('def build_portfolio_weights')
    print(repr(src[idx:idx+200]))

set_src(12, src)

# ── Cells 17-20: Comparison with monkey-patching fix ────────────────────────────
# Read from the updated comparison script template
# (same as v4 but: references v5 instead of v4, includes v4 via softmax rollout)

cell17 = """\
# ============================================================
# STEP D: Close-to-Close Backtest -- v2 / v3 / v4 / v5 comparison
# ============================================================
%matplotlib inline
import os
OUT_DIR = 'models_comparison/matd3_v5'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Backtesting core functions ────────────────────────────────────────────

def prepare_close_bar_data(df, signal_col, price_col='mid', spread_col='spread',
                           day_col='TradingDay', stock_col='SecuCode',
                           close_ts=1445, close_te=1500):
    df = df.copy().sort_values([day_col, 'TimeStart', 'TimeEnd', stock_col]).reset_index(drop=True)
    df_close = df[(df['TimeStart']==close_ts)&(df['TimeEnd']==close_te)][
        [day_col,stock_col,signal_col,price_col,spread_col]].copy()
    df_close = df_close.sort_values([stock_col, day_col]).reset_index(drop=True)
    df_close['has_price_t']  = np.isfinite(df_close[price_col].astype(float))
    df_close['has_spread_t'] = np.isfinite(df_close[spread_col].astype(float))
    df_close['price_t1']  = df_close.groupby(stock_col)[price_col].shift(-1)
    df_close['spread_t1'] = df_close.groupby(stock_col)[spread_col].shift(-1)
    df_close['day_t1']    = df_close.groupby(stock_col)[day_col].shift(-1)
    all_days = np.sort(df_close[day_col].unique())
    ndm = pd.Series(all_days[1:], index=all_days[:-1]).to_dict()
    df_close['expected_day_t1'] = df_close[day_col].map(ndm)
    ok_next = df_close['day_t1'].eq(df_close['expected_day_t1'])
    df_close['has_price_t1']  = ok_next & np.isfinite(df_close['price_t1'].astype(float))
    df_close['has_spread_t1'] = ok_next & np.isfinite(df_close['spread_t1'].astype(float))
    df_close['eligible'] = (df_close[signal_col].notna() & df_close['has_price_t']
        & df_close['has_spread_t'] & df_close['has_price_t1'] & df_close['has_spread_t1'])
    days = np.sort(df_close[day_col].unique()); stocks = np.sort(df_close[stock_col].unique())
    pm = df_close.pivot(index=day_col, columns=stock_col, values=price_col).reindex(index=days, columns=stocks).astype('float64')
    sm = df_close.pivot(index=day_col, columns=stock_col, values=spread_col).reindex(index=days, columns=stocks).astype('float64')
    sg = df_close.pivot(index=day_col, columns=stock_col, values=signal_col).reindex(index=days, columns=stocks).astype('float64')
    el = df_close.pivot(index=day_col, columns=stock_col, values='eligible').reindex(index=days, columns=stocks).fillna(False).astype(bool)
    return {'price_mat': pm, 'spread_mat': sm, 'signal_mat': sg, 'eligible_mat': el}

def transform_signal_to_weight_topk(signal_mat, eligible_mat, top_k):
    idx = signal_mat.index.intersection(eligible_mat.index)
    cols = signal_mat.columns.intersection(eligible_mat.columns)
    su = signal_mat.loc[idx,cols].where(eligible_mat.loc[idx,cols], np.nan)
    rk = su.rank(axis=1, method='first', ascending=False)
    sel = (rk <= top_k); ns = sel.sum(axis=1)
    tw = sel.div(ns.replace(0, np.nan), axis=0).fillna(0.0).astype('float64')
    return {'target_w_mat': tw, 'signal_used_mat': su}

def run_c2c_backtest(price_mat, spread_mat, target_w_mat, init_cash=1e6, fee_rate=0.0003):
    idx = price_mat.index.intersection(spread_mat.index).intersection(target_w_mat.index)
    cols = price_mat.columns.intersection(spread_mat.columns).intersection(target_w_mat.columns)
    P = price_mat.loc[idx,cols].to_numpy(dtype=np.float64)
    S = spread_mat.loc[idx,cols].to_numpy(dtype=np.float64)
    W = target_w_mat.loc[idx,cols].fillna(0.0).to_numpy(dtype=np.float64)
    days = price_mat.loc[idx].index.to_numpy(); T,N = P.shape
    shares = np.zeros(N); cash = float(init_cash)
    nav_pre=np.zeros(T); nav_post=np.zeros(T); fee_a=np.zeros(T); sc_a=np.zeros(T); to_a=np.zeros(T)
    for t in range(T):
        p=np.nan_to_num(P[t]); s=np.nan_to_num(S[t]); w=np.nan_to_num(W[t])
        nb_=cash+np.dot(shares,p); nav_pre[t]=nb_
        ts_=np.where(p>0, nb_*w/p, 0.0); ts_=np.maximum(ts_,0.0)
        d_=ts_-shares; bm=d_>1e-12; sm_=d_<-1e-12
        bpx=p+s/2; spx=p-s/2
        ss_=np.where(sm_,-d_,0.0); sn=np.sum(ss_*spx); sf=fee_rate*sn; cas=cash+sn-sf
        bs_=np.where(bm,d_,0.0); bn=np.sum(bs_*bpx); bf=fee_rate*bn; tb=bn+bf
        if tb>cas+1e-12:
            sc_=max(0,min(1,cas/tb if tb>0 else 0)); bs_*=sc_; bn=np.sum(bs_*bpx); bf=fee_rate*bn
        fd=np.zeros(N); fd[sm_]=-ss_[sm_]; fd[bm]=bs_[bm]
        cash=cas-bn-bf; shares=np.maximum(shares+fd,0.0)
        nav_post[t]=cash+np.dot(shares,p); fee_a[t]=sf+bf
        sc_a[t]=np.sum(bs_*(bpx-p))+np.sum(ss_*(p-spx))
        to_a[t]=(sn+bn)/nb_ if nb_>1e-12 else 0
    rg=np.full(T,np.nan); rn=np.full(T,np.nan)
    if T>=2: rg[:-1]=nav_pre[1:]/nav_post[:-1]-1; rn[:-1]=nav_post[1:]/nav_post[:-1]-1
    return pd.DataFrame({'TradingDay':days,'nav_pre':nav_pre,'nav_post':nav_post,
        'ret_gross':rg,'ret_net':rn,'turnover':to_a,'fee':fee_a,'spread_cost':sc_a})

def bt_metrics(eq, ann=243):
    r=eq['ret_net'].dropna(); tr=(1+r).prod()-1; n=len(r)
    ar=(1+tr)**(ann/max(n,1))-1; av=r.std()*np.sqrt(ann); sh=ar/max(av,1e-8)
    nav=(1+r).cumprod(); dd=(nav-nav.cummax())/nav.cummax().clip(lower=1e-8); mdd=float(dd.min())
    cal=ar/max(abs(mdd),1e-8)
    neg=r[r<0]; sor=ar/max(neg.std()*np.sqrt(ann) if len(neg)>1 else 1e-8,1e-8)
    wr=float((r>0).mean()); at=eq['turnover'].mean()
    return {'Total Return':tr,'Ann. Return':ar,'Ann. Vol':av,'Sharpe':sh,
            'Sortino':sor,'Calmar':cal,'Max Drawdown':mdd,'Win Rate':wr,'Avg Turnover':at}

# ── Helper: load actor checkpoints into a fresh MATD3SharedCritic model ─────────

def load_model_for_inference(model_dir):
    \"""
    Load saved actors from model_dir into a fresh MATD3SharedCritic model.
    Returns the model, or None if checkpoints are not found.
    \"""
    ckpt0 = os.path.join(model_dir, 'actor_agent_0.pt')
    if not os.path.exists(ckpt0):
        print(f'  [{model_dir}] checkpoints not found, skipping')
        return None
    m = MATD3SharedCritic(n_agents=n_agents, state_dim=state_dim,
                          action_dim=action_dim, cfg=cfg)
    for i in range(n_agents):
        m.actors[i].load_state_dict(
            torch.load(os.path.join(model_dir, f'actor_agent_{i}.pt'), map_location=DEVICE)
        )
        m.actors[i].eval()
    print(f'  Loaded {n_agents} actors from {model_dir}')
    return m

# ── EW portfolio override: for v2/v3/v5 rollouts (no softmax, no cap) ──────────
# build_portfolio_weights_from_agent_scores is a global name resolved at call time.
# Reassigning it here affects all subsequent rollout_with_model calls.

def _ew_portfolio(agent_scores, prev_w, long_only=True, max_weight_per_stock=0.20, topk=None, **kwargs):
    \"""EW top-k portfolio -- matches v2/v3/v5 training (no softmax, no turnover cap).\"""
    final_score = agent_scores.mean(axis=1).astype(np.float32)
    if topk is not None and topk < len(final_score):
        idx = np.argpartition(final_score, -topk)[-topk:]
        w = np.zeros(len(final_score), dtype=np.float32)
        if topk > 0: w[idx] = 1.0 / topk
    else:
        n = max(len(final_score), 1)
        w = np.ones(len(final_score), dtype=np.float32) / n
    if long_only:
        w = np.clip(w, 0.0, max_weight_per_stock).astype(np.float32)
        s = w.sum()
        if s > 1e-12: w = (w / s).astype(np.float32)
    delta_w = (w - prev_w).astype(np.float32)
    return final_score, w, np.maximum(delta_w, 0.0).astype(np.float32), np.maximum(-delta_w, 0.0).astype(np.float32)

def _softmax_portfolio(agent_scores, prev_w, long_only=True, max_weight_per_stock=0.20,
                       topk=None, softmax_temperature=3.0, max_daily_turnover=0.5, **kwargs):
    \"""Softmax portfolio with turnover cap -- matches v4 training.\"""
    final_score = agent_scores.mean(axis=1).astype(np.float32)
    if topk is not None and topk < len(final_score):
        idx  = np.argpartition(final_score, -topk)[-topk:]
        mask = np.zeros(len(final_score), dtype=bool); mask[idx] = True
        in_s = final_score.copy(); in_s[~mask] = -1e9
    else:
        in_s = final_score.copy()
    w = np.zeros(len(final_score), dtype=np.float32)
    valid = in_s > -1e8
    if valid.any():
        s = in_s[valid] - in_s[valid].max()
        e = np.exp(softmax_temperature * s); e = e / e.sum()
        w[valid] = e.astype(np.float32)
    w = np.clip(w, 0.0, max_weight_per_stock).astype(np.float32)
    s = w.sum()
    if s > 1e-12: w = (w / s).astype(np.float32)
    delta_raw = w - prev_w; total_chg = float(np.abs(delta_raw).sum())
    if total_chg > max_daily_turnover and total_chg > 1e-8:
        scale = max_daily_turnover / total_chg
        w = (prev_w + delta_raw * scale).astype(np.float32)
        w = np.clip(w, 0.0, max_weight_per_stock).astype(np.float32)
        s = w.sum()
        if s > 1e-12: w = (w / s).astype(np.float32)
    delta_w = (w - prev_w).astype(np.float32)
    return final_score, w, np.maximum(delta_w, 0.0).astype(np.float32), np.maximum(-delta_w, 0.0).astype(np.float32)

# ── Load base data ───────────────────────────────────────────────────────────
print('Loading data...')
df_raw_full = pd.read_parquet('df_testsub_full.parquet')
df_full_rl  = pd.read_parquet('df_full_rl_clean.parquet')  # full bars for rollout

raw_signal_cols = ['signal_pre_boost','signal_hybrid','signal_ssm_tf','signal_tf_cross','signal_gat_tf']
df_raw_full['signal_ew_ensemble'] = df_raw_full[raw_signal_cols].mean(axis=1)

# v5 (current): extract final_score from df_full_pred (EW portfolio rollout, cell 16)
print('Extracting MATD3 v5 signal...')
v5_close = (df_full_pred[df_full_pred[cfg.time_col]==1500]
            [[cfg.stock_col, cfg.day_col, 'final_score']]
            .copy().rename(columns={'final_score':'signal_matd3_v5'}))
df_raw_full = df_raw_full.merge(v5_close, on=[cfg.stock_col,cfg.day_col], how='left')
df_raw_full['signal_matd3_v5'] = df_raw_full['signal_matd3_v5'].fillna(0.0)
print(f'  v5: {(df_raw_full["signal_matd3_v5"]!=0).sum()} non-zero rows')

# v3: rollout with EW portfolio (matches v3 training -- no softmax, no cap)
print('Running MATD3 v3 inference rollout (EW portfolio)...')
build_portfolio_weights_from_agent_scores = _ew_portfolio
v3_model = load_model_for_inference('matd3_v3_models')
has_v3 = False
if v3_model is not None:
    df_v3_pred, _, _ = rollout_with_model(df_full_rl, v3_model, cfg, train_mode=False)
    v3_close = (df_v3_pred[df_v3_pred[cfg.time_col]==1500]
                [[cfg.stock_col, cfg.day_col, 'final_score']]
                .rename(columns={'final_score':'signal_matd3_v3'}))
    df_raw_full = df_raw_full.merge(v3_close, on=[cfg.stock_col,cfg.day_col], how='left')
    df_raw_full['signal_matd3_v3'] = df_raw_full['signal_matd3_v3'].fillna(0.0)
    has_v3 = True
    print(f'  v3: {(df_raw_full["signal_matd3_v3"]!=0).sum()} non-zero rows')

# v2: rollout with EW portfolio (matches v2 training)
print('Running MATD3 v2 inference rollout (EW portfolio)...')
# build_portfolio_weights_from_agent_scores already = _ew_portfolio
v2_model = load_model_for_inference('matd3_v2_models')
has_v2 = False
if v2_model is not None:
    df_v2_pred, _, _ = rollout_with_model(df_full_rl, v2_model, cfg, train_mode=False)
    v2_close = (df_v2_pred[df_v2_pred[cfg.time_col]==1500]
                [[cfg.stock_col, cfg.day_col, 'final_score']]
                .rename(columns={'final_score':'signal_matd3_v2'}))
    df_raw_full = df_raw_full.merge(v2_close, on=[cfg.stock_col,cfg.day_col], how='left')
    df_raw_full['signal_matd3_v2'] = df_raw_full['signal_matd3_v2'].fillna(0.0)
    has_v2 = True
    print(f'  v2: {(df_raw_full["signal_matd3_v2"]!=0).sum()} non-zero rows')

# v4: rollout with SOFTMAX portfolio (matches v4 training -- softmax + turnover cap)
print('Running MATD3 v4 inference rollout (softmax portfolio)...')
build_portfolio_weights_from_agent_scores = _softmax_portfolio
v4_model = load_model_for_inference('matd3_v4_models')
has_v4 = False
if v4_model is not None:
    df_v4_pred, _, _ = rollout_with_model(df_full_rl, v4_model, cfg, train_mode=False)
    v4_close = (df_v4_pred[df_v4_pred[cfg.time_col]==1500]
                [[cfg.stock_col, cfg.day_col, 'final_score']]
                .rename(columns={'final_score':'signal_matd3_v4'}))
    df_raw_full = df_raw_full.merge(v4_close, on=[cfg.stock_col,cfg.day_col], how='left')
    df_raw_full['signal_matd3_v4'] = df_raw_full['signal_matd3_v4'].fillna(0.0)
    has_v4 = True
    print(f'  v4: {(df_raw_full["signal_matd3_v4"]!=0).sum()} non-zero rows')

# Restore v5 EW portfolio for any subsequent use
build_portfolio_weights_from_agent_scores = _ew_portfolio

# ── Build signal list and run backtests ─────────────────────────────────────
BT_TOPK = 5; BT_FEE = 0.0003
SIGNALS = [
    ('LightGBM',    'signal_pre_boost'),
    ('CNN+LSTM+TF', 'signal_hybrid'),
    ('SSM+TF',      'signal_ssm_tf'),
    ('Cross-Attn',  'signal_tf_cross'),
    ('GAT+TF',      'signal_gat_tf'),
    ('EW Ensemble', 'signal_ew_ensemble'),
]
if has_v2: SIGNALS.append(('MATD3 v2', 'signal_matd3_v2'))
if has_v3: SIGNALS.append(('MATD3 v3', 'signal_matd3_v3'))
if has_v4: SIGNALS.append(('MATD3 v4', 'signal_matd3_v4'))
SIGNALS.append(('MATD3 v5', 'signal_matd3_v5'))

all_eq = {}; all_metrics = {}
for name, sig_col in SIGNALS:
    prep = prepare_close_bar_data(df_raw_full, signal_col=sig_col)
    wobj = transform_signal_to_weight_topk(prep['signal_mat'], prep['eligible_mat'], BT_TOPK)
    eq   = run_c2c_backtest(prep['price_mat'], prep['spread_mat'], wobj['target_w_mat'], fee_rate=BT_FEE)
    net  = (1+eq['ret_net'].fillna(0)).cumprod().iloc[-1]
    print(f'  {name:20s}  Net NAV={net:.4f}')
    all_eq[name] = eq; all_metrics[name] = bt_metrics(eq)

print('\\n' + '='*80)
print(f'CLOSE-TO-CLOSE BACKTEST (Top-{BT_TOPK} EW, fee={BT_FEE})')
print('='*80)
df_m = pd.DataFrame(all_metrics).T
df_mf = df_m.copy()
for c in ['Total Return','Ann. Return','Ann. Vol','Max Drawdown','Win Rate']:
    df_mf[c] = df_mf[c].map(lambda x: f'{x:.2%}')
for c in ['Sharpe','Sortino','Calmar']:
    df_mf[c] = df_mf[c].map(lambda x: f'{x:.3f}')
df_mf['Avg Turnover'] = df_mf['Avg Turnover'].map(lambda x: f'{x:.4f}')
print(df_mf.to_string())
df_m.to_csv(os.path.join(OUT_DIR, 'metrics_comparison_full.csv'))
"""

cell18 = """\
# ============================================================
# Plot 1: PnL Comparison -- v2 / v3 / v4 / v5 highlighted
# ============================================================
%matplotlib inline
sig_colors = {
    'LightGBM':    '#1f77b4', 'CNN+LSTM+TF': '#ff7f0e',
    'SSM+TF':      '#2ca02c', 'Cross-Attn':  '#d62728',
    'GAT+TF':      '#9467bd', 'EW Ensemble': '#888888',
    'MATD3 v2':    '#ffb3de', 'MATD3 v3':    '#ff8c00',
    'MATD3 v4':    '#c49a11', 'MATD3 v5':    '#e31a1c',
}
sig_lw     = {'MATD3 v5':3.0, 'MATD3 v3':2.5, 'MATD3 v4':2.0, 'MATD3 v2':2.0, 'CNN+LSTM+TF':2.0}
sig_zorder = {'MATD3 v5':13, 'MATD3 v3':11, 'MATD3 v4':10, 'MATD3 v2':9, 'CNN+LSTM+TF':8}

fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios':[3,1]})
fig.suptitle(f'Close-to-Close Backtest: All Models (Top-{BT_TOPK} EW, Fee={BT_FEE})',
             fontsize=14, fontweight='bold')
ax1 = axes[0]
for name, eq in all_eq.items():
    nav  = (1+eq['ret_net'].fillna(0)).cumprod()
    days = pd.to_datetime(eq['TradingDay'].astype(str), format='%Y%m%d')
    lw   = sig_lw.get(name, 1.3)
    ls   = '--' if 'Ensemble' in name else '-'
    zo   = sig_zorder.get(name, 5)
    alpha = 0.95 if ('MATD3' in name or 'CNN' in name) else 0.55
    ax1.plot(days, nav.values, label=name, color=sig_colors.get(name,'gray'),
             linewidth=lw, linestyle=ls, alpha=alpha, zorder=zo)
ax1.axhline(1.0, color='gray', linestyle=':', linewidth=0.6)
ax1.set_ylabel('NAV (Net of Costs)', fontsize=11)
ax1.legend(fontsize=9, loc='upper left', ncol=2)
ax1.grid(True, alpha=0.3)
for name in ['MATD3 v2','MATD3 v3','MATD3 v4','MATD3 v5']:
    if name in all_eq:
        eq  = all_eq[name]
        nav = (1+eq['ret_net'].fillna(0)).cumprod()
        days= pd.to_datetime(eq['TradingDay'].astype(str), format='%Y%m%d')
        ax1.annotate(f'{name}: {nav.iloc[-1]:.2f}x',
                     xy=(days.iloc[-1], nav.iloc[-1]), xytext=(5,2),
                     textcoords='offset points', fontsize=8,
                     color=sig_colors.get(name,'black'), fontweight='bold')
ax2 = axes[1]
for name, eq in all_eq.items():
    nav  = (1+eq['ret_net'].fillna(0)).cumprod()
    dd   = (nav-nav.cummax())/nav.cummax().clip(lower=1e-8)
    days = pd.to_datetime(eq['TradingDay'].astype(str), format='%Y%m%d')
    lw   = sig_lw.get(name, 0.8)
    alpha = 0.9 if 'MATD3' in name else 0.4
    show = 'MATD3' in name or 'CNN' in name or 'Ensemble' in name
    ax2.plot(days, dd.values, color=sig_colors.get(name,'gray'),
             linewidth=lw, alpha=alpha, label=name if show else None)
ax2.set_ylabel('Drawdown'); ax2.set_xlabel('Date')
ax2.legend(fontsize=8, loc='lower left'); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'pnl_comparison_all_models.png'), dpi=150, bbox_inches='tight')
plt.show(); print('Saved: pnl_comparison_all_models.png')
"""

cell19 = """\
# ============================================================
# Plot 2: Metrics Bar Chart (4 panels) -- v2/v3/v4/v5 comparison
# ============================================================
%matplotlib inline
def bar_color(n):
    if n=='MATD3 v5':    return '#e31a1c'
    if n=='MATD3 v4':    return '#c49a11'
    if n=='MATD3 v3':    return '#ff8c00'
    if n=='MATD3 v2':    return '#ffb3de'
    if n=='CNN+LSTM+TF': return '#ff7f0e'
    return '#aec7e8'

names = list(all_metrics.keys()); x = np.arange(len(names))
metrics_to_plot = [
    ('Sharpe',       'Sharpe Ratio',       False),
    ('Total Return', 'Total Return',        True),
    ('Calmar',       'Calmar Ratio',        False),
    ('Max Drawdown', 'Max Drawdown',        True),
]
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle('Key Metrics Comparison -- MATD3 v2 / v3 / v4 / v5 vs Individual Signals',
             fontsize=13, fontweight='bold')
for ax, (metric, title, is_pct) in zip(axes, metrics_to_plot):
    vals   = [all_metrics[n][metric] for n in names]
    colors = [bar_color(n) for n in names]
    bars   = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=7)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, vals):
        fmt  = f'{v:.2%}' if is_pct else f'{v:.3f}'
        yoff = abs(v)*0.02 if v>=0 else -abs(v)*0.06
        ax.text(bar.get_x()+bar.get_width()/2, v+yoff, fmt,
                ha='center', va='bottom' if v>=0 else 'top',
                fontsize=6.5, fontweight='bold')
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor='#e31a1c', label='MATD3 v5 (current)'),
    Patch(facecolor='#c49a11', label='MATD3 v4'),
    Patch(facecolor='#ff8c00', label='MATD3 v3'),
    Patch(facecolor='#ffb3de', label='MATD3 v2'),
    Patch(facecolor='#ff7f0e', label='CNN+LSTM+TF'),
    Patch(facecolor='#aec7e8', label='Other signals'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=6,
           fontsize=8, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'metrics_bar_chart.png'), dpi=150, bbox_inches='tight')
plt.show(); print('Saved: metrics_bar_chart.png')
"""

cell20 = """\
# ============================================================
# Plot 3: Gate Weights -- v5 per-agent + v2/v3/v4/v5 mean comparison
# ============================================================
%matplotlib inline
agent_names = ['LightGBM','CNN+LSTM+TF','SSM+TF','Cross-Attn','GAT+TF']
gate_cols   = [f'gate_agent_{i}' for i in range(n_agents)]
gate_ts = (df_full_pred[[cfg.day_col,cfg.time_col]+gate_cols]
           .sort_values([cfg.day_col,cfg.time_col])
           .groupby([cfg.day_col,cfg.time_col],as_index=False).first())

# Panel A: v5 gate time series per agent
fig, axes = plt.subplots(n_agents, 1, figsize=(14, 3*n_agents), sharex=True)
fig.suptitle('MATD3 v5 Gate Weights per Agent [0, 3]', fontsize=14, fontweight='bold')
cc = plt.cm.tab10(np.linspace(0, 1, n_agents))
for i, (ax, gc, an, cl) in enumerate(zip(axes, gate_cols, agent_names, cc)):
    g = gate_ts[gc].fillna(1.0).values; mg = float(np.mean(g))
    ax.plot(g, color=cl, linewidth=0.8, alpha=0.85, label=an)
    ax.axhline(mg, color=cl, linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean={mg:.3f}')
    ax.axhline(1.0, color='black', linestyle=':', linewidth=0.6, alpha=0.4)
    ax.set_ylim(-0.05, 3.30); ax.set_ylabel('Gate')
    ax.set_title(f'Agent {i}: {an}', fontsize=9)
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)
axes[-1].set_xlabel('Timestamp Index')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'gate_weights_v5.png'), dpi=150, bbox_inches='tight')
plt.show(); print('Saved: gate_weights_v5.png')

# Gate summary table
print('\\n=== MATD3 v5 Gate Summary ===')
gs = {}
for gc, an in zip(gate_cols, agent_names):
    g = gate_ts[gc].dropna().values
    gs[an] = {'mean':round(np.mean(g),3),'std':round(np.std(g),3),
              'min':round(np.min(g),3),'max':round(np.max(g),3),
              '>1.0':f'{(g>1).mean():.1%}','near_0':f'{(g<0.05).mean():.1%}'}
print(pd.DataFrame(gs).T.to_string())

# Panel B: v2/v3/v4/v5 gate mean bar chart
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_title('Agent Gate Mean: MATD3 v2 vs v3 vs v4 vs v5', fontsize=12, fontweight='bold')
# v2/v3 gate means from published results (models_comparison/matd3_v3)
v2_means = [0.91, 2.00, 2.00, 1.87, 1.33]
v3_means = [0.91, 2.66, 2.07, 1.87, 1.33]
v4_means_path = 'models_comparison/matd3_v4/metrics_comparison_full.csv'
# v4 gate means: compute from v4 rollout if available, else use v3 as placeholder
if has_v4 and 'df_v4_pred' in dir():
    gt4 = (df_v4_pred[[cfg.day_col,cfg.time_col]+gate_cols]
           .groupby([cfg.day_col,cfg.time_col],as_index=False).first())
    v4_means = [float(gt4[gc].mean()) for gc in gate_cols]
else:
    v4_means = [1.0] * n_agents
v5_means = [float(gate_ts[gc].mean()) for gc in gate_cols]
bw = 0.20; xp = np.arange(n_agents)
ax.bar(xp-1.5*bw, v2_means, bw, label='v2  gate[0,2]',            color='#ffb3de', alpha=0.85)
ax.bar(xp-0.5*bw, v3_means, bw, label='v3  gate[0,3]',            color='#ff8c00', alpha=0.85)
ax.bar(xp+0.5*bw, v4_means, bw, label='v4  softmax+cap',          color='#c49a11', alpha=0.85)
ax.bar(xp+1.5*bw, v5_means, bw, label='v5  EW+Calmar+turnover',   color='#e31a1c', alpha=0.85)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='Neutral')
ax.set_xticks(xp); ax.set_xticklabels(agent_names, rotation=15, ha='right')
ax.set_ylabel('Mean Gate Weight'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
for j, v in enumerate(v5_means):
    ax.text(xp[j]+1.5*bw, v+0.03, f'{v:.2f}', ha='center', fontsize=8,
            fontweight='bold', color='#e31a1c')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'gate_version_comparison.png'), dpi=150, bbox_inches='tight')
plt.show(); print('Saved: gate_version_comparison.png')
"""

# Apply comparison cells
nb['cells'][17]['source'] = [cell17]
nb['cells'][17]['outputs'] = []
nb['cells'][17]['execution_count'] = None
nb['cells'][18]['source'] = [cell18]
nb['cells'][18]['outputs'] = []
nb['cells'][18]['execution_count'] = None
nb['cells'][19]['source'] = [cell19]
nb['cells'][19]['outputs'] = []
nb['cells'][19]['execution_count'] = None
nb['cells'][20]['source'] = [cell20]
nb['cells'][20]['outputs'] = []
nb['cells'][20]['execution_count'] = None

# ── Cell 22: Summary markdown ─────────────────────────────────────────────────
src22 = get_src(21)
src22 = src22.replace('## Step E: Summary (v4)', '## Step E: Summary (v5)')
src22 = src22.replace('matd3_v4_models', 'matd3_v5_models')
src22 = src22.replace('models_comparison/matd3_v4', 'models_comparison/matd3_v5')
set_src(21, src22)

# ── Save v5 notebook ──────────────────────────────────────────────────────────
with open('notebooks/rl_full_pipeline_v5.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Saved: notebooks/rl_full_pipeline_v5.ipynb')
print()
print('v5 changes vs v4:')
print('  Cell 4  CFG: dd_lambda=0.15, vol_lambda=0.0 (removed), turnover_lambda=0.25,')
print('               reward_scale=2500, epochs=100, noise=0.12, save_dir=matd3_v5_models')
print('  Cell 10 reward: removed vol_excess/var_penalty; reward = net_ret - dd - turnover')
print('  Cell 13 portfolio: removed softmax + turnover_cap; restored EW top-k')
print('  Cell 17-20: v2/v3/v4/v5 comparison with monkey-patching to match each version training')
print()
print('v4 regression root causes:')
print('  1. Softmax portfolio in training != EW in backtest -> train/test mismatch')
print('  2. vol_target_daily asymmetric penalty rarely active -> no effective vol constraint')
print('  3. turnover_lambda=0.5 too aggressive -> suppressed signal differentiation')
