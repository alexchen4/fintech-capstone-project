"""
Load pre-trained MATD3 model, run prediction, and generate final
PnL comparison chart: MATD3 vs all individual signals.
All outputs saved to models_comparison/
"""
import os, sys, copy, json, random, warnings
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')

BASE_DIR = 'e:/New_folder/Work/MyProject/capstone'
os.chdir(BASE_DIR)
OUT_DIR  = 'models_comparison'
MODEL_DIR = OUT_DIR  # checkpoints are in the same folder

ANN_FACTOR = 3888
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
seed_everything(42)

# ══════════════════════════════════════════════════════════
# CFG (must match training config)
# ══════════════════════════════════════════════════════════
@dataclass
class CFG:
    stock_col:  str = 'SecuCode'
    day_col:    str = 'TradingDay'
    time_col:   str = 'TimeEnd'
    price_col:  str = 'mid'
    spread_col: str = 'spread'
    ret_col:    str = 'ret_mid_t1'
    split_col:  str = 'dataset_split'
    agent_signal_cols = (
        'signal_pre_boost_cs_z','signal_hybrid_cs_z','signal_ssm_tf_cs_z',
        'signal_tf_cross_cs_z','signal_gat_tf_cs_z',
    )
    shared_state_cols = (
        'sig_mean_5','sig_std_5','spread_over_mid','microprice_minus_mid_over_mid',
        'log_volume_sum_cs_rank','ret_mid_t1_lag1','ret_mid_t1_lag2',
        'mean_sentiment','abnormal_sentiment','sentiment_vol','message_volume',
    )
    use_half_spread: bool = True
    long_only: bool = True
    topk: Optional[int] = 20
    max_weight_per_stock: float = 0.20
    dd_lambda: float = 0.5
    vol_lambda: float = 30.0
    vol_window: int = 50
    reward_scale: float = 1000.0
    hidden_dim: int = 128
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    gamma: float = 0.95
    tau: float = 0.005
    replay_capacity: int = 300000
    batch_size: int = 512
    warmup_steps: int = 2000
    epochs: int = 15
    action_noise_std: float = 0.05
    target_policy_noise_std: float = 0.02
    target_policy_noise_clip: float = 0.05
    train_every: int = 2
    updates_per_step: int = 2
    policy_delay: int = 2
    gate_reg_lambda: float = 0.01
    topk_state_k: int = 5
    save_dir: str = 'models_comparison'

cfg = CFG()
agent_cols  = list(cfg.agent_signal_cols)
shared_cols = list(cfg.shared_state_cols)
n_agents = len(agent_cols)
action_dim = 1
N_SHARED = len(shared_cols)
state_dim = 33 + 2 * N_SHARED  # 55

# ══════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════
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

def delta_to_gate(delta_raw):
    return 1.0 + 0.5 * torch.tanh(delta_raw)

# ══════════════════════════════════════════════════════════
# Networks
# ══════════════════════════════════════════════════════════
class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim=128, action_dim=1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    def forward(self, x): return self.backbone(x)

class SharedCritic(nn.Module):
    def __init__(self, n_agents, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        in_dim = n_agents * state_dim + n_agents * action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, states, actions):
        return self.net(torch.cat([states.reshape(states.size(0),-1),
                                    actions.reshape(actions.size(0),-1)], dim=1))

class MATD3SharedCritic:
    def __init__(self, n_agents, state_dim, action_dim, cfg):
        self.n_agents = n_agents; self.state_dim = state_dim
        self.action_dim = action_dim; self.cfg = cfg; self.update_step = 0
        self.actors = [Actor(state_dim, cfg.hidden_dim, action_dim).to(DEVICE) for _ in range(n_agents)]
        self.target_actors = [copy.deepcopy(a).to(DEVICE) for a in self.actors]
        self.critic1 = SharedCritic(n_agents, state_dim, action_dim, cfg.hidden_dim).to(DEVICE)
        self.critic2 = SharedCritic(n_agents, state_dim, action_dim, cfg.hidden_dim).to(DEVICE)
        self.target_critic1 = copy.deepcopy(self.critic1).to(DEVICE)
        self.target_critic2 = copy.deepcopy(self.critic2).to(DEVICE)
        self.actor_opts = [torch.optim.Adam(a.parameters(), lr=cfg.actor_lr) for a in self.actors]
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=cfg.critic_lr)

    @torch.no_grad()
    def act(self, states_np, noise_std=0.0):
        gates = []
        for i in range(self.n_agents):
            s = torch.tensor(states_np[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            delta_raw = self.actors[i](s)
            if noise_std > 0:
                delta_raw = delta_raw + noise_std * torch.randn_like(delta_raw)
            g = np.clip(delta_to_gate(delta_raw).squeeze(0).cpu().numpy(), 0.5, 1.5).astype(np.float32)
            gates.append(g)
        return np.stack(gates, axis=0)

# ══════════════════════════════════════════════════════════
# Data + State Builders
# ══════════════════════════════════════════════════════════
def map_weight_dict_to_codes(wm, codes):
    return np.asarray([wm.get(c, 0.0) for c in codes], dtype=np.float32)
def vec_to_map(codes, vec):
    return {c: float(w) for c, w in zip(codes, vec)}
def remap_vec(old_codes, vec, new_codes):
    return map_weight_dict_to_codes(vec_to_map(old_codes, vec), new_codes)

def prepare_dataframe(df_raw, cfg):
    need = [cfg.split_col, cfg.stock_col, cfg.day_col, cfg.time_col,
            cfg.price_col, cfg.spread_col, cfg.ret_col,
            *cfg.agent_signal_cols, *cfg.shared_state_cols]
    need = [c for c in need if c in df_raw.columns]
    df = df_raw[need].copy().replace([np.inf, -np.inf], np.nan)
    df = df.sort_values([cfg.stock_col, cfg.day_col, cfg.time_col]).reset_index(drop=True)
    fill_cols = list(cfg.agent_signal_cols) + list(cfg.shared_state_cols)
    fill_cols = [c for c in fill_cols if c in df.columns]
    df[fill_cols] = df.groupby(cfg.stock_col, sort=False)[fill_cols].ffill().bfill().fillna(0.0)
    df[cfg.spread_col] = df[cfg.spread_col].fillna(0.0).clip(lower=0.0)
    df[cfg.price_col] = df[cfg.price_col].replace(0, np.nan)
    df[cfg.price_col] = df.groupby(cfg.stock_col, sort=False)[cfg.price_col].ffill().bfill().fillna(1.0)
    df[cfg.ret_col] = df[cfg.ret_col].fillna(0.0)
    df = df.sort_values([cfg.day_col, cfg.time_col, cfg.stock_col]).reset_index(drop=True)
    df['ts_key'] = list(zip(df[cfg.day_col].values, df[cfg.time_col].values))
    unique_ts = df['ts_key'].drop_duplicates().tolist()
    ts_to_idx = {k: i for i, k in enumerate(unique_ts)}
    df['ts_idx'] = df['ts_key'].map(ts_to_idx).astype(int)
    snapshots = []
    for _, sub in df.groupby('ts_idx', sort=True):
        snapshots.append(sub.sort_values(cfg.stock_col).reset_index(drop=False))
    return df, snapshots

def build_snapshot_cache(df, snapshots, cfg):
    cache = []
    for sub in tqdm(snapshots, desc='Cache snapshots', leave=False):
        d = {}
        d['row_idx'] = sub['index'].values.astype(np.int64)
        d['codes'] = sub[cfg.stock_col].values
        d['n_stock'] = len(sub)
        d['day'] = sub[cfg.day_col].iloc[0]
        d['time'] = sub[cfg.time_col].iloc[0]
        if cfg.split_col in sub.columns:
            d['dataset_split'] = sub[cfg.split_col].iloc[0]
        d['mid'] = sub[cfg.price_col].values.astype(np.float32)
        d['spread'] = sub[cfg.spread_col].values.astype(np.float32)
        d['ret_next'] = sub[cfg.ret_col].values.astype(np.float32)
        d['agent_signal_mat'] = np.stack([sub[c].values.astype(np.float32) for c in agent_cols], axis=1)
        d['shared_mat'] = np.stack([sub[c].values.astype(np.float32) for c in shared_cols], axis=1)
        sh_mean = np.nanmean(d['shared_mat'], axis=0)
        sh_std = np.nanstd(d['shared_mat'], axis=0)
        d['shared_global_summary'] = np.concatenate([sh_mean, sh_std]).astype(np.float32)
        ensemble_sig = d['agent_signal_mat'].mean(axis=1)
        static_states = []
        for i in range(n_agents):
            own = d['agent_signal_mat'][:, i]
            q10, q25, q50, q75, q90 = [_safe_quantile(own, q) for q in [.1,.25,.5,.75,.9]]
            own_summary = np.array([
                _safe_mean(own), _safe_std(own), q10, q25, q50, q75, q90,
                _topk_mean(own, cfg.topk_state_k), _bottomk_mean(own, cfg.topk_state_k),
                q90-q50, q50-q10, _safe_skew(own), _safe_kurtosis_excess(own),
                _topk_share_nonneg(own, 1), _topk_share_nonneg(own, cfg.topk_state_k),
            ], dtype=np.float32)
            rel_summary = np.array([_safe_corr(own, ensemble_sig),
                                     _safe_mean(np.abs(own - ensemble_sig))], dtype=np.float32)
            static_states.append(np.concatenate([own_summary, rel_summary,
                                                  d['shared_global_summary']]).astype(np.float32))
        d['static_states'] = np.stack(static_states, axis=0)
        cache.append(d)
    return cache

def build_dynamic_agent_states(cur, inventory_w, sellable_w, prev_gates, turnover_lag, cfg):
    inventory_w = np.asarray(inventory_w, dtype=np.float32)
    sellable_w = np.asarray(sellable_w, dtype=np.float32)
    prev_gates = np.asarray(prev_gates, dtype=np.float32).reshape(-1)
    cash = max(0.0, 1.0 - float(inventory_w.sum()))
    invested = float(inventory_w.sum())
    sellable_total = float(sellable_w.sum())
    unsellable_total = max(0.0, invested - sellable_total)
    port_global = np.array([cash, invested, sellable_total, unsellable_total,
                             _hhi(inventory_w),
                             float(inventory_w.max()) if len(inventory_w) else 0.0,
                             _topk_weight_sum(inventory_w, cfg.topk_state_k),
                             float(turnover_lag)], dtype=np.float32)
    states = []
    for i in range(n_agents):
        own_sig = cur['agent_signal_mat'][:, i]
        inv_exp = float(np.dot(inventory_w, own_sig))
        sell_exp = float(np.dot(sellable_w, own_sig))
        if len(inventory_w) > 0:
            k = min(cfg.topk_state_k, len(inventory_w))
            top_idx = np.argpartition(inventory_w, -k)[-k:]
            top5_sig = float(own_sig[top_idx].mean())
        else: top5_sig = 0.0
        port_agent = np.array([
            float(prev_gates[i]) if i < len(prev_gates) else 1.0,
            _safe_mean(inventory_w), _safe_std(inventory_w),
            _safe_mean(sellable_w), _safe_std(sellable_w),
            inv_exp, sell_exp, top5_sig], dtype=np.float32)
        states.append(np.concatenate([cur['static_states'][i], port_global, port_agent]).astype(np.float32))
    return np.stack(states, axis=0)

# ══════════════════════════════════════════════════════════
# Portfolio Execution
# ══════════════════════════════════════════════════════════
def build_portfolio_weights(agent_scores, prev_w, long_only=True, max_w=0.20, topk=None):
    final_score = agent_scores.mean(axis=1).astype(np.float32)
    if topk and topk < len(final_score):
        idx = np.argpartition(final_score, -topk)[-topk:]
        mask = np.zeros(len(final_score), dtype=bool); mask[idx] = True
        score = np.where(mask, final_score, -1e9 if long_only else 0.0)
    else: score = final_score
    raw = np.maximum(score, 0.0).astype(np.float32)
    s = raw.sum()
    w = (raw / s).astype(np.float32) if s > 1e-12 else np.zeros_like(raw)
    w = np.clip(w, 0.0, max_w).astype(np.float32)
    s = w.sum()
    if s > 1e-12: w = (w / s).astype(np.float32)
    return final_score, w

def refresh_sellable(prev_day, cur_day, inv_w, sell_w):
    if prev_day is None or cur_day != prev_day:
        return np.asarray(inv_w, dtype=np.float32).copy()
    return np.asarray(sell_w, dtype=np.float32)

def execute_t1(target_w, inv_w, sell_w, max_w=0.20):
    target_w = np.clip(np.maximum(target_w, 0.0), 0.0, max_w).astype(np.float32)
    inv_w = np.asarray(inv_w, dtype=np.float32)
    sell_w = np.asarray(sell_w, dtype=np.float32)
    s = target_w.sum()
    if s > 1e-12: target_w = (target_w / s).astype(np.float32)
    desired_sell = np.maximum(inv_w - target_w, 0.0)
    desired_buy = np.maximum(target_w - inv_w, 0.0)
    sell_actual = np.minimum(desired_sell, sell_w)
    inv_after = inv_w - sell_actual
    sell_after = sell_w - sell_actual
    cash = max(0.0, 1.0 - float(inv_w.sum())) + float(sell_actual.sum())
    dbs = float(desired_buy.sum())
    buy_actual = (desired_buy * min(1.0, cash / dbs)).astype(np.float32) if dbs > 1e-12 and cash > 1e-12 else np.zeros_like(desired_buy)
    exec_w = np.clip(inv_after + buy_actual, 0.0, max_w).astype(np.float32)
    tot = float(exec_w.sum())
    if tot > 1.000001: exec_w = (exec_w / tot).astype(np.float32)
    return exec_w, np.maximum(exec_w - inv_w, 0.0), np.maximum(inv_w - exec_w, 0.0), exec_w.copy(), np.minimum(sell_after, exec_w)

# ══════════════════════════════════════════════════════════
# Load model + predict
# ══════════════════════════════════════════════════════════
def load_model(model_dir):
    model = MATD3SharedCritic(n_agents, state_dim, action_dim, cfg)
    for i in range(n_agents):
        model.actors[i].load_state_dict(
            torch.load(os.path.join(model_dir, f'actor_agent_{i}.pt'), map_location=DEVICE))
        tp = os.path.join(model_dir, f'target_actor_agent_{i}.pt')
        model.target_actors[i].load_state_dict(
            torch.load(tp, map_location=DEVICE) if os.path.exists(tp) else model.actors[i].state_dict())
    model.critic1.load_state_dict(torch.load(os.path.join(model_dir, 'shared_critic1.pt'), map_location=DEVICE))
    model.critic2.load_state_dict(torch.load(os.path.join(model_dir, 'shared_critic2.pt'), map_location=DEVICE))
    model.target_critic1.load_state_dict(torch.load(os.path.join(model_dir, 'target_shared_critic1.pt'), map_location=DEVICE))
    model.target_critic2.load_state_dict(torch.load(os.path.join(model_dir, 'target_shared_critic2.pt'), map_location=DEVICE))
    print(f'Model loaded from {model_dir}')
    return model

@torch.no_grad()
def predict_matd3(df_input, model, cfg):
    """Minimal predict-only rollout returning per-timestamp NAV records."""
    print('Preparing data...')
    df, snapshots = prepare_dataframe(df_input, cfg)
    print(f'  rows={len(df):,}, timestamps={len(snapshots):,}')
    print('Building cache...')
    cache = build_snapshot_cache(df, snapshots, cfg)

    records = []
    inv_map = {}; sell_map = {}; prev_day = None
    prev_gates = np.ones(n_agents, dtype=np.float32)
    turnover_lag = 0.0; nav = 1.0

    gate_history = {f'gate_{i}': [] for i in range(n_agents)}

    for t in tqdm(range(len(cache)), desc='MATD3 Predict'):
        cur = cache[t]
        cur_codes = cur['codes']; cur_day = cur['day']
        inv_w = map_weight_dict_to_codes(inv_map, cur_codes)
        sell_w = map_weight_dict_to_codes(sell_map, cur_codes)
        sell_w = refresh_sellable(prev_day, cur_day, inv_w, sell_w)

        states = build_dynamic_agent_states(cur, inv_w, sell_w, prev_gates, turnover_lag, cfg)
        gates = model.act(states, noise_std=0.0)

        agent_scores = (cur['agent_signal_mat'] * gates.reshape(1, -1)).astype(np.float32)
        _, target_w = build_portfolio_weights(agent_scores, inv_w, cfg.long_only, cfg.max_weight_per_stock, cfg.topk)
        exec_w, buy_w, sell_actual_w, inv_new, sell_new = execute_t1(target_w, inv_w, sell_w, cfg.max_weight_per_stock)

        # Reward calc
        delta_w = exec_w - inv_w
        spread_frac = np.divide(cur['spread'], np.maximum(cur['mid'], 1e-8),
                                out=np.zeros_like(cur['spread']), where=cur['mid'] > 0)
        gross_ret = float(np.dot(exec_w, cur['ret_next']))
        spread_cost = float(np.sum(np.abs(delta_w) * 0.5 * spread_frac))
        turnover = float(np.abs(delta_w).sum())
        net_ret = gross_ret - spread_cost
        nav *= (1 + net_ret)

        records.append({
            'TradingDay': cur_day, 'TimeEnd': cur['time'],
            'dataset_split': cur.get('dataset_split', 'unknown'),
            'gross_return': gross_ret, 'net_return': net_ret,
            'spread_cost': spread_cost, 'turnover': turnover, 'nav': nav,
        })
        for i in range(n_agents):
            gate_history[f'gate_{i}'].append(float(gates[i, 0]))

        inv_map = vec_to_map(cur_codes, inv_new)
        sell_map = vec_to_map(cur_codes, sell_new)
        prev_day = cur_day
        prev_gates = gates.reshape(-1).astype(np.float32)
        turnover_lag = turnover

    df_pred = pd.DataFrame(records)
    for k, v in gate_history.items():
        df_pred[k] = v
    return df_pred


# ══════════════════════════════════════════════════════════
# Performance Metrics
# ══════════════════════════════════════════════════════════
def compute_metrics(nav_series, ann_factor=ANN_FACTOR):
    rets = nav_series.pct_change().dropna()
    total_ret = nav_series.iloc[-1] / nav_series.iloc[0] - 1
    n = len(rets)
    ann_ret = (1 + total_ret) ** (ann_factor / max(n, 1)) - 1
    ann_vol = rets.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / max(ann_vol, 1e-8)
    dd_ts = (nav_series - nav_series.cummax()) / nav_series.cummax().clip(lower=1e-8)
    max_dd = float(dd_ts.min())
    calmar = ann_ret / max(abs(max_dd), 1e-8)
    neg = rets[rets < 0]
    sortino = ann_ret / max(neg.std() * np.sqrt(ann_factor) if len(neg) > 1 else 1e-8, 1e-8)
    win_rate = float((rets > 0).mean())
    return {'Total Return': total_ret, 'Ann. Return': ann_ret, 'Ann. Vol': ann_vol,
            'Sharpe': sharpe, 'Sortino': sortino, 'Calmar': calmar,
            'Max Drawdown': max_dd, 'Win Rate': win_rate}

# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == '__main__':
    # 1. Load data
    print('Loading df_full_rl_clean.parquet ...')
    df_full = pd.read_parquet('df_full_rl_clean.parquet')
    print(f'Shape: {df_full.shape}')

    # 2. Load model and predict
    print('\nLoading MATD3 model...')
    model = load_model(MODEL_DIR)

    print('\nRunning MATD3 prediction on FULL data...')
    matd3_pred = predict_matd3(df_full, model, cfg)
    matd3_pred.to_parquet(os.path.join(OUT_DIR, 'nav_MATD3_composite.parquet'), index=False)
    print(f'MATD3 final NAV: {matd3_pred["nav"].iloc[-1]:.4f}')

    # 3. Load individual signal NAV curves
    signal_files = {
        'LightGBM': 'nav_LightGBM_pre_boost.parquet',
        'CNN+LSTM+TF': 'nav_CNN+LSTM+TF_hybrid.parquet',
        'SSM+TF': 'nav_SSM+TF_ssm_tf.parquet',
        'Cross-Attn': 'nav_Cross-Attn_tf_cross.parquet',
        'GAT+TF': 'nav_GAT+TF_gat_tf.parquet',
        'EW Ensemble': 'nav_EW_Ensemble_5_signals.parquet',
    }

    all_navs = {}
    for name, fname in signal_files.items():
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.exists(fpath):
            bt = pd.read_parquet(fpath)
            all_navs[name] = bt
    all_navs['MATD3 (Composite)'] = matd3_pred

    # Find train/test boundary
    train_days = set(df_full[df_full['dataset_split'] == 'train']['TradingDay'].unique())
    first_nav = list(all_navs.values())[0]
    split_idx = None
    for i, row in first_nav.iterrows():
        if row['TradingDay'] not in train_days:
            split_idx = i; break

    # ══════════════════════════════════════════════════════
    # PLOT: Full PnL Comparison (all signals + MATD3)
    # ══════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 1, figsize=(18, 13), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle('PnL Comparison: MATD3 vs Individual Signals\n(Top-20 EW, Net of Spread)',
                 fontsize=15, fontweight='bold')

    # Colors: signals light, ensemble dashed, MATD3 thick red
    sig_colors = {'LightGBM': '#7fbadc', 'CNN+LSTM+TF': '#ffb366',
                  'SSM+TF': '#80d99a', 'Cross-Attn': '#e88a8a', 'GAT+TF': '#b8a3d9'}
    ew_color = '#888888'
    matd3_color = '#d62728'

    ax = axes[0]
    for name, bt in all_navs.items():
        if name == 'MATD3 (Composite)':
            ax.plot(bt['nav'].values, label=name, color=matd3_color, linewidth=3.0, alpha=0.95, zorder=10)
        elif name == 'EW Ensemble':
            ax.plot(bt['nav'].values, label=name, color=ew_color, linewidth=2.0,
                    linestyle='--', alpha=0.8, zorder=5)
        else:
            ax.plot(bt['nav'].values, label=name, color=sig_colors.get(name, 'gray'),
                    linewidth=1.2, alpha=0.7, zorder=3)

    if split_idx is not None:
        ax.axvline(split_idx, color='black', linestyle=':', linewidth=1.5, alpha=0.6, label='Train | Test')
    ax.axhline(1.0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_ylabel('NAV (Net)', fontsize=12)
    ax.set_title('Full Period Cumulative PnL', fontsize=13)
    ax.legend(fontsize=9, loc='best', ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Test period drawdown
    ax2 = axes[1]
    start = split_idx if split_idx is not None else 0
    for name, bt in all_navs.items():
        bt_sub = bt.iloc[start:].copy()
        nav_sub = (1 + bt_sub['net_return']).cumprod()
        dd = (nav_sub - nav_sub.cummax()) / nav_sub.cummax().clip(lower=1e-8)
        if name == 'MATD3 (Composite)':
            ax2.plot(dd.values, label=name, color=matd3_color, linewidth=2.0, zorder=10)
        elif name == 'EW Ensemble':
            ax2.plot(dd.values, label=name, color=ew_color, linewidth=1.5, linestyle='--', zorder=5)
        else:
            ax2.plot(dd.values, color=sig_colors.get(name, 'gray'), linewidth=0.8, alpha=0.5, zorder=3)
    ax2.set_ylabel('Drawdown', fontsize=12)
    ax2.set_xlabel('Timestamp Index (test period)', fontsize=11)
    ax2.set_title('Test Period Drawdown', fontsize=11)
    ax2.legend(fontsize=8, loc='lower left', ncol=4)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out1 = os.path.join(OUT_DIR, 'pnl_comparison_all_models.png')
    fig.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out1}')

    # ══════════════════════════════════════════════════════
    # PLOT: Test Period Only (zoomed)
    # ══════════════════════════════════════════════════════
    if split_idx is not None:
        fig2, ax3 = plt.subplots(1, 1, figsize=(16, 8))
        fig2.suptitle('Test Period PnL — MATD3 vs All Models (Net of Spread)',
                      fontsize=14, fontweight='bold')

        for name, bt in all_navs.items():
            bt_sub = bt.iloc[split_idx:].copy()
            nav_sub = (1 + bt_sub['net_return']).cumprod()
            if name == 'MATD3 (Composite)':
                ax3.plot(nav_sub.values, label=name, color=matd3_color, linewidth=3.0, zorder=10)
            elif name == 'EW Ensemble':
                ax3.plot(nav_sub.values, label=name, color=ew_color, linewidth=2.0,
                         linestyle='--', zorder=5)
            else:
                ax3.plot(nav_sub.values, label=name,
                         color=sig_colors.get(name, 'gray'), linewidth=1.2, alpha=0.7, zorder=3)

        ax3.axhline(1.0, color='gray', linestyle=':', linewidth=0.8)
        ax3.set_ylabel('NAV (Net)', fontsize=12)
        ax3.set_xlabel('Timestamp Index (test period)', fontsize=11)
        ax3.legend(fontsize=10, loc='best', ncol=2, framealpha=0.9)
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        out2 = os.path.join(OUT_DIR, 'pnl_test_all_models.png')
        fig2.savefig(out2, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f'Saved: {out2}')

    # ══════════════════════════════════════════════════════
    # PLOT: MATD3 Gate Weights Over Time
    # ══════════════════════════════════════════════════════
    gate_cols = [f'gate_{i}' for i in range(n_agents)]
    agent_names = ['LightGBM', 'CNN+LSTM+TF', 'SSM+TF', 'Cross-Attn', 'GAT+TF']

    fig3, axes3 = plt.subplots(n_agents, 1, figsize=(16, 3*n_agents), sharex=True)
    fig3.suptitle('MATD3 Agent Gate Weights Over Time', fontsize=14, fontweight='bold')

    ccolors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    for i, (ax_i, gc, aname, clr) in enumerate(zip(axes3, gate_cols, agent_names, ccolors)):
        g_vals = matd3_pred[gc].values
        ax_i.plot(g_vals, color=clr, linewidth=0.8, alpha=0.8, label=aname)
        mean_g = float(np.mean(g_vals))
        ax_i.axhline(mean_g, color=clr, linestyle='--', linewidth=1.5, alpha=0.7,
                      label=f'Mean={mean_g:.3f}')
        ax_i.axhline(1.0, color='black', linestyle=':', linewidth=0.6, alpha=0.4)
        if split_idx is not None:
            ax_i.axvline(split_idx, color='red', linestyle=':', linewidth=1, alpha=0.5)
        ax_i.set_ylim(0.45, 1.55)
        ax_i.set_ylabel('Gate')
        ax_i.set_title(f'Agent {i}: {aname}', fontsize=9)
        ax_i.legend(fontsize=7, loc='upper right')
        ax_i.grid(True, alpha=0.3)
    axes3[-1].set_xlabel('Timestamp Index')
    plt.tight_layout()
    out3 = os.path.join(OUT_DIR, 'gate_weights_matd3.png')
    fig3.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f'Saved: {out3}')

    # ══════════════════════════════════════════════════════
    # Metrics Comparison Table
    # ══════════════════════════════════════════════════════
    print('\n' + '='*80)
    print('FULL PERIOD METRICS COMPARISON')
    print('='*80)
    metrics_all = {}
    for name, bt in all_navs.items():
        metrics_all[name] = compute_metrics(bt['nav'])
    df_m = pd.DataFrame(metrics_all).T
    df_m_fmt = df_m.copy()
    for c in ['Total Return', 'Ann. Return', 'Ann. Vol', 'Max Drawdown', 'Win Rate']:
        df_m_fmt[c] = df_m_fmt[c].map(lambda x: f'{x:.2%}')
    for c in ['Sharpe', 'Sortino', 'Calmar']:
        df_m_fmt[c] = df_m_fmt[c].map(lambda x: f'{x:.3f}')
    print(df_m_fmt.to_string())
    df_m.to_csv(os.path.join(OUT_DIR, 'metrics_comparison_full.csv'))

    if split_idx is not None:
        print('\n' + '='*80)
        print('TEST PERIOD METRICS COMPARISON')
        print('='*80)
        metrics_test = {}
        for name, bt in all_navs.items():
            bt_sub = bt.iloc[split_idx:].copy()
            nav_sub = (1 + bt_sub['net_return']).cumprod()
            metrics_test[name] = compute_metrics(nav_sub)
        df_mt = pd.DataFrame(metrics_test).T
        df_mt_fmt = df_mt.copy()
        for c in ['Total Return', 'Ann. Return', 'Ann. Vol', 'Max Drawdown', 'Win Rate']:
            df_mt_fmt[c] = df_mt_fmt[c].map(lambda x: f'{x:.2%}')
        for c in ['Sharpe', 'Sortino', 'Calmar']:
            df_mt_fmt[c] = df_mt_fmt[c].map(lambda x: f'{x:.3f}')
        print(df_mt_fmt.to_string())
        df_mt.to_csv(os.path.join(OUT_DIR, 'metrics_comparison_test.csv'))

    # Gate summary
    print('\n' + '='*80)
    print('MATD3 GATE SUMMARY')
    print('='*80)
    gate_summary = {}
    for gc, aname in zip(gate_cols, agent_names):
        g = matd3_pred[gc].values
        gate_summary[aname] = {'mean': np.mean(g), 'std': np.std(g),
                                'min': np.min(g), 'max': np.max(g),
                                '>1.0': f'{(g > 1.0).mean():.1%}',
                                '<1.0': f'{(g < 1.0).mean():.1%}'}
    print(pd.DataFrame(gate_summary).T.to_string())

    print(f'\n=== All results saved to {OUT_DIR}/ ===')
