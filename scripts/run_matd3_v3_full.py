"""MATD3 v3 Full Pipeline: Train + Backtest (cells 2-20)"""
import matplotlib
matplotlib.use("Agg")
import os
os.chdir("e:/New_folder/Work/MyProject/capstone")

# ============================================================
# Cell 2
# ============================================================
# ============================================================
# STEP C: Core Imports
# ============================================================
import os, copy, json, random, sqlite3, warnings
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

seed_everything(42)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('DEVICE:', DEVICE)

ANN_FACTOR = 243    # 1 bar/day (close only) * 243 trading days/year
print('ANN_FACTOR:', ANN_FACTOR)


# ============================================================
# Cell 3
# ============================================================
# ============================================================
# CFG Dataclass
# agent_signal_cols and shared_state_cols are class-level
# tuples (NOT instance fields with field()) so they don't
# appear in asdict() — prevents JSON serialisation issues.
# ============================================================

@dataclass
class CFG:
    # ---------- basic columns ----------
    stock_col:  str = 'SecuCode'
    day_col:    str = 'TradingDay'
    time_col:   str = 'TimeEnd'
    price_col:  str = 'mid'
    spread_col: str = 'spread'
    ret_col:    str = 'ret_mid_t1'
    split_col:  str = 'dataset_split'

    # ---------- agents (class-level tuple) ----------
    agent_signal_cols = (
        'signal_pre_boost_cs_z',
        'signal_hybrid_cs_z',
        'signal_ssm_tf_cs_z',
        'signal_tf_cross_cs_z',
        'signal_gat_tf_cs_z',
    )

    # ---------- shared state features (includes sentiment) ----------
    shared_state_cols = (
        'sig_mean_5',
        'sig_std_5',
        'spread_over_mid',
        'microprice_minus_mid_over_mid',
        'log_volume_sum_cs_rank',
        'ret_mid_t1_lag1',
        'ret_mid_t1_lag2',
        'mean_sentiment',
        'abnormal_sentiment',
        'sentiment_vol',
        'message_volume',
    )

    # ---------- fee / cost ----------
    broker_commission_rate:       float = 0.0
    exchange_reg_transfer_rate:   float = 0.0
    sell_stamp_duty_rate:         float = 0.0
    use_half_spread:              bool  = True

    # ---------- portfolio ----------
    long_only:              bool          = True
    topk:                   Optional[int] = 5
    max_weight_per_stock:   float         = 0.30

    # ---------- composite reward ----------
    dd_lambda:               float = 0.1    # Calmar: drawdown penalty weight
    vol_lambda:              float = 3.0   # Sharpe: variance penalty weight
    vol_window:              int   = 50     # rolling window for variance
    risk_penalty_lambda:     float = 0.0
    turnover_penalty_lambda: float = 0.0
    reward_scale:            float = 2000.0

    # ---------- RL hyperparams ----------
    hidden_dim:                  int   = 128
    actor_lr:                    float = 1e-4
    critic_lr:                   float = 3e-4
    gamma:                       float = 0.95
    tau:                         float = 0.005

    replay_capacity:             int   = 300000
    batch_size:                  int   = 512
    warmup_steps:                int   = 2000
    epochs:                      int   = 50

    action_noise_std:            float = 0.10
    target_policy_noise_std:     float = 0.02
    target_policy_noise_clip:    float = 0.05

    train_every:                 int   = 2
    updates_per_step:            int   = 2
    policy_delay:                int   = 2
    gate_reg_lambda:             float = 0.001

    # ---------- state features ----------
    topk_state_k: int = 5

    # ---------- save ----------
    save_dir: str = 'matd3_v3_models'


cfg = CFG()

# Module-level aliases used throughout
agent_cols  = list(cfg.agent_signal_cols)
shared_cols = list(cfg.shared_state_cols)
n_agents    = len(agent_cols)
action_dim  = 1

print('agent_cols :', agent_cols)
print('shared_cols:', shared_cols)
print('n_agents   :', n_agents)


# ============================================================
# Cell 4
# ============================================================
# ============================================================
# Compute state_dim
# state_dim = 15 (own_summary)
#           +  2 (rel_summary)
#           + 2*N_SHARED (shared_global_summary)
#           +  8 (port_global)
#           +  8 (port_agent)
#           = 33 + 2*N_SHARED
# With N_SHARED=11: state_dim = 55
# ============================================================

OWN_STATIC_DIM    = 15
REL_STATIC_DIM    = 2
N_SHARED          = len(shared_cols)        # 11
SHARED_GLOBAL_DIM = 2 * N_SHARED           # 22
PORT_GLOBAL_DIM   = 8
PORT_AGENT_DIM    = 8

state_dim = (
    OWN_STATIC_DIM
    + REL_STATIC_DIM
    + SHARED_GLOBAL_DIM
    + PORT_GLOBAL_DIM
    + PORT_AGENT_DIM
)

print(f'N_SHARED      = {N_SHARED}')
print(f'state_dim     = 33 + 2*{N_SHARED} = {state_dim}')
assert state_dim == 33 + 2 * N_SHARED, f'Expected {33 + 2*N_SHARED}, got {state_dim}'


# ============================================================
# Cell 5
# ============================================================
# ============================================================
# Helper Statistics Functions + delta_to_gate
# ============================================================

def _safe_std(x):
    x = np.asarray(x, dtype=np.float32)
    return float(np.std(x)) if len(x) else 0.0

def _safe_mean(x):
    x = np.asarray(x, dtype=np.float32)
    return float(np.mean(x)) if len(x) else 0.0

def _safe_quantile(x, q):
    x = np.asarray(x, dtype=np.float32)
    return float(np.quantile(x, q)) if len(x) else 0.0

def _safe_skew(x):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < 3: return 0.0
    mu = x.mean(); sd = x.std()
    if sd < 1e-12: return 0.0
    z = (x - mu) / sd
    return float((z ** 3).mean())

def _safe_kurtosis_excess(x):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < 4: return 0.0
    mu = x.mean(); sd = x.std()
    if sd < 1e-12: return 0.0
    z = (x - mu) / sd
    return float((z ** 4).mean() - 3.0)

def _safe_corr(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if len(x) == 0 or len(y) == 0: return 0.0
    if x.std() < 1e-12 or y.std() < 1e-12: return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def _topk_mean(x, k=5):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0: return 0.0
    k = min(k, len(x))
    return float(x[np.argpartition(x, -k)[-k:]].mean())

def _bottomk_mean(x, k=5):
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0: return 0.0
    k = min(k, len(x))
    return float(x[np.argpartition(x, k-1)[:k]].mean())

def _topk_share_nonneg(x, k=5):
    x = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    s = x.sum()
    if s <= 1e-12: return 0.0
    k = min(k, len(x))
    return float(x[np.argpartition(x, -k)[-k:]].sum() / s)

def _hhi(w):
    w = np.asarray(w, dtype=np.float32)
    return float((w ** 2).sum()) if len(w) else 0.0

def _topk_weight_sum(w, k=5):
    w = np.asarray(w, dtype=np.float32)
    if len(w) == 0: return 0.0
    k = min(k, len(w))
    return float(w[np.argpartition(w, -k)[-k:]].sum())

def delta_to_gate(delta_raw):
    """Maps unconstrained delta_raw -> gate in [0, 3.0]. Wide range for aggressive differentiation."""
    return 1.5 + 1.5 * torch.tanh(delta_raw)

print('Helper functions defined.')


# ============================================================
# Cell 6
# ============================================================
# ============================================================
# Actor + SharedCritic Networks
# ============================================================

class Actor(nn.Module):
    """
    Returns pre-squash delta_raw.
    Actual gate = 1 + 0.5*tanh(delta_raw) in [0.5, 1.5].
    Architecture: Linear -> LayerNorm -> ReLU -> Linear -> ReLU -> Linear
    """
    def __init__(self, state_dim: int, hidden_dim: int = 128, action_dim: int = 1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SharedCritic(nn.Module):
    """
    Centralised critic: concatenates all agents' states + actions.
    in_dim = n_agents * state_dim + n_agents * action_dim
    """
    def __init__(self, n_agents: int, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        in_dim = n_agents * state_dim + n_agents * action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([
            states.reshape(states.size(0), -1),
            actions.reshape(actions.size(0), -1),
        ], dim=1)
        return self.net(x)


print('Actor and SharedCritic defined.')
_a = Actor(state_dim, 128, 1).to(DEVICE)
_s = SharedCritic(n_agents, state_dim, 1, 128).to(DEVICE)
_ds = torch.zeros(2, n_agents, state_dim, device=DEVICE)
_da = torch.zeros(2, n_agents, 1, device=DEVICE)
print('Actor output:', _a(torch.zeros(2, state_dim, device=DEVICE)).shape)
print('Critic output:', _s(_ds, _da).shape)
del _a, _s, _ds, _da


# ============================================================
# Cell 7
# ============================================================
# ============================================================
# MATD3SharedCritic: Multi-Agent TD3 with twin shared critics
# ============================================================

class MATD3SharedCritic:
    def __init__(self, n_agents: int, state_dim: int, action_dim: int, cfg: CFG):
        self.n_agents    = n_agents
        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.cfg         = cfg
        self.update_step = 0

        self.actors = [
            Actor(state_dim, cfg.hidden_dim, action_dim).to(DEVICE)
            for _ in range(n_agents)
        ]
        self.target_actors = [copy.deepcopy(a).to(DEVICE) for a in self.actors]

        self.critic1        = SharedCritic(n_agents, state_dim, action_dim, cfg.hidden_dim).to(DEVICE)
        self.critic2        = SharedCritic(n_agents, state_dim, action_dim, cfg.hidden_dim).to(DEVICE)
        self.target_critic1 = copy.deepcopy(self.critic1).to(DEVICE)
        self.target_critic2 = copy.deepcopy(self.critic2).to(DEVICE)

        self.actor_opts = [
            torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
            for actor in self.actors
        ]
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=cfg.critic_lr,
        )

    @torch.no_grad()
    def act(self, states_np: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        gates = []
        for i in range(self.n_agents):
            s = torch.tensor(states_np[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            delta_raw = self.actors[i](s)
            if noise_std > 0:
                delta_raw = delta_raw + noise_std * torch.randn_like(delta_raw)
            g = np.clip(delta_to_gate(delta_raw).squeeze(0).cpu().numpy(), 0.0, 3.0).astype(np.float32)
            gates.append(g)
        return np.stack(gates, axis=0)   # (n_agents, 1)

    def soft_update(self, online_net: nn.Module, target_net: nn.Module) -> None:
        tau = self.cfg.tau
        for p, tp in zip(online_net.parameters(), target_net.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    def update(self, replay: 'ReplayBuffer', batch_size: int = 256) -> dict:
        if len(replay) < batch_size:
            return {}

        self.update_step += 1
        states, actions, rewards, next_states, dones = replay.sample(batch_size)

        # ---- Critic update ----
        with torch.no_grad():
            next_actions = []
            for i in range(self.n_agents):
                d_next = self.target_actors[i](next_states[:, i, :])
                eps = torch.clamp(
                    torch.randn_like(d_next) * self.cfg.target_policy_noise_std,
                    -self.cfg.target_policy_noise_clip,
                     self.cfg.target_policy_noise_clip,
                )
                next_actions.append(torch.clamp(delta_to_gate(d_next + eps), 0.0, 3.0))
            next_actions = torch.stack(next_actions, dim=1)
            q_next = torch.min(
                self.target_critic1(next_states, next_actions),
                self.target_critic2(next_states, next_actions),
            )
            y = rewards + self.cfg.gamma * (1.0 - dones) * q_next

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        cl1 = F.mse_loss(q1, y)
        cl2 = F.mse_loss(q2, y)
        critic_loss = cl1 + cl2

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 1.0
        )
        self.critic_opt.step()

        logs = {
            'critic_loss':  float(critic_loss.item()),
            'critic1_loss': float(cl1.item()),
            'critic2_loss': float(cl2.item()),
        }

        # ---- Delayed actor update ----
        if self.update_step % self.cfg.policy_delay == 0:
            actor_losses = []
            for i in range(self.n_agents):
                curr_actions = actions.clone()
                prop_delta   = self.actors[i](states[:, i, :])
                prop_action  = delta_to_gate(prop_delta)
                curr_actions[:, i, :] = prop_action
                gate_reg   = ((prop_action - 1.0) ** 2).mean()
                actor_loss = (
                    -self.critic1(states, curr_actions).mean()
                    + self.cfg.gate_reg_lambda * gate_reg
                )
                self.actor_opts[i].zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1.0)
                self.actor_opts[i].step()
                actor_losses.append(float(actor_loss.item()))

            for i in range(self.n_agents):
                self.soft_update(self.actors[i], self.target_actors[i])
            self.soft_update(self.critic1, self.target_critic1)
            self.soft_update(self.critic2, self.target_critic2)

            for i, lv in enumerate(actor_losses):
                logs[f'actor_loss_{i}'] = lv

        return logs


print('MATD3SharedCritic defined.')


# ============================================================
# Cell 8
# ============================================================
# ============================================================
# ReplayBuffer
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity: int, n_agents: int, state_dim: int, action_dim: int):
        self.capacity   = capacity
        self.n_agents   = n_agents
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.buffer: list = []
        self.pos = 0

    def push(self, states, actions, reward, next_states, done):
        item = (
            np.asarray(states,      dtype=np.float32),
            np.asarray(actions,     dtype=np.float32),
            np.float32(reward),
            np.asarray(next_states, dtype=np.float32),
            np.float32(done),
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        idx   = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)
        states      = torch.tensor(np.stack(states),      dtype=torch.float32, device=DEVICE)
        actions     = torch.tensor(np.stack(actions),     dtype=torch.float32, device=DEVICE)
        rewards     = torch.tensor(np.array(rewards),     dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32, device=DEVICE)
        dones       = torch.tensor(np.array(dones),       dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


print('ReplayBuffer defined.')


# ============================================================
# Cell 9
# ============================================================
# ============================================================
# RewardTracker + calc_reward_composite
# Composite reward = net_return - dd_lambda*drawdown - vol_lambda*rolling_var
# ============================================================

class RewardTracker:
    """Tracks running NAV, peak NAV, and rolling return variance."""
    def __init__(self, vol_window: int = 50):
        self.vol_window = vol_window
        self._buf: List[float] = []
        self.nav: float = 1.0
        self.peak_nav: float = 1.0

    def reset(self) -> None:
        self._buf.clear()
        self.nav = 1.0
        self.peak_nav = 1.0

    def update(self, net_ret: float) -> None:
        self._buf.append(float(net_ret))
        if len(self._buf) > self.vol_window:
            self._buf.pop(0)
        self.nav *= (1.0 + net_ret)
        self.peak_nav = max(self.peak_nav, self.nav)

    @property
    def drawdown(self) -> float:
        return max(0.0, (self.peak_nav - self.nav) / max(self.peak_nav, 1e-8))

    @property
    def rolling_var(self) -> float:
        if len(self._buf) < 5:
            return 0.0
        return float(np.var(self._buf))


def calc_reward_composite(
    mid: np.ndarray,
    spread: np.ndarray,
    ret_next: np.ndarray,
    new_w: np.ndarray,
    prev_w: np.ndarray,
    tracker: RewardTracker,
    cfg: CFG,
) -> Tuple[float, dict]:
    """Composite reward = net_return - dd_penalty - var_penalty, scaled."""
    delta_w     = (new_w - prev_w).astype(np.float32)
    spread_frac = np.divide(
        spread,
        np.maximum(mid, 1e-8),
        out=np.zeros_like(spread, dtype=np.float32),
        where=mid > 0,
    )
    half_mult    = 0.5 if cfg.use_half_spread else 1.0
    gross_return = float(np.dot(new_w, ret_next))
    spread_cost  = float(np.sum(np.abs(delta_w) * half_mult * spread_frac))
    turnover     = float(np.abs(delta_w).sum())
    net_return   = gross_return - spread_cost

    tracker.update(net_return)

    dd_penalty  = cfg.dd_lambda  * tracker.drawdown
    var_penalty = cfg.vol_lambda * tracker.rolling_var
    composite   = net_return - dd_penalty - var_penalty

    return composite * cfg.reward_scale, {
        'gross_return':  gross_return,
        'spread_cost':   spread_cost,
        'turnover':      turnover,
        'net_return':    net_return,
        'drawdown':      tracker.drawdown,
        'nav':           tracker.nav,
        'reward':        composite,
        'reward_scaled': composite * cfg.reward_scale,
    }


print('RewardTracker and calc_reward_composite defined.')


# ============================================================
# Cell 10
# ============================================================
# ============================================================
# prepare_dataframe + build_snapshot_cache
# ============================================================

def map_weight_dict_to_codes(weight_map: dict, codes) -> np.ndarray:
    return np.asarray([weight_map.get(code, 0.0) for code in codes], dtype=np.float32)

def vec_to_map(codes, vec) -> dict:
    return {code: float(w) for code, w in zip(codes, vec)}

def remap_vec_between_codes(old_codes, vec, new_codes) -> np.ndarray:
    return map_weight_dict_to_codes(vec_to_map(old_codes, vec), new_codes)


def prepare_dataframe(df_raw: pd.DataFrame, cfg: CFG, show_progress: bool = True):
    pbar = tqdm(total=6, desc='Prepare dataframe', leave=False, disable=not show_progress)
    preserve_cols = [c for c in [cfg.split_col] if c in df_raw.columns]
    need_cols = preserve_cols + [
        cfg.stock_col, cfg.day_col, cfg.time_col,
        cfg.price_col, cfg.spread_col, cfg.ret_col,
        *cfg.agent_signal_cols,
        *cfg.shared_state_cols,
    ]
    missing = [c for c in need_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}')

    df = df_raw[need_cols].copy().replace([np.inf, -np.inf], np.nan)
    pbar.update(1); pbar.set_postfix_str(f'rows={len(df):,}')

    df = df.sort_values([cfg.stock_col, cfg.day_col, cfg.time_col]).reset_index(drop=True)
    pbar.update(1)

    fill_cols = list(cfg.agent_signal_cols) + list(cfg.shared_state_cols)
    df[fill_cols] = df.groupby(cfg.stock_col, sort=False)[fill_cols].ffill()
    df[fill_cols] = df.groupby(cfg.stock_col, sort=False)[fill_cols].bfill()
    df[fill_cols] = df[fill_cols].fillna(0.0)
    pbar.update(1)

    df[cfg.spread_col] = df[cfg.spread_col].fillna(0.0).clip(lower=0.0)
    df[cfg.price_col] = df[cfg.price_col].replace(0, np.nan)
    df[cfg.price_col] = df.groupby(cfg.stock_col, sort=False)[cfg.price_col].ffill()
    df[cfg.price_col] = df.groupby(cfg.stock_col, sort=False)[cfg.price_col].bfill()
    df[cfg.price_col] = df[cfg.price_col].fillna(1.0)
    df[cfg.ret_col] = df[cfg.ret_col].fillna(0.0)
    pbar.update(1)

    df = df.sort_values([cfg.day_col, cfg.time_col, cfg.stock_col]).reset_index(drop=True)
    pbar.update(1)

    df['ts_key'] = list(zip(df[cfg.day_col].values, df[cfg.time_col].values))
    unique_ts = df['ts_key'].drop_duplicates().tolist()
    ts_to_idx = {k: i for i, k in enumerate(unique_ts)}
    df['ts_idx'] = df['ts_key'].map(ts_to_idx).astype(int)

    snapshots = []
    it = df.groupby('ts_idx', sort=True)
    if show_progress:
        it = tqdm(it, total=df['ts_idx'].nunique(), desc='Build snapshots', leave=False)
    for _, sub in it:
        snapshots.append(sub.sort_values(cfg.stock_col).reset_index(drop=False))

    pbar.update(1); pbar.close()
    return df, snapshots


def build_snapshot_cache(df: pd.DataFrame, snapshots: list, cfg: CFG, show_progress: bool = True) -> list:
    cache = []
    it = tqdm(snapshots, total=len(snapshots), desc='Cache snapshots', leave=False) if show_progress else snapshots

    for sub in it:
        d = {}
        d['row_idx'] = sub['index'].values.astype(np.int64)
        d['codes']   = sub[cfg.stock_col].values
        d['n_stock'] = len(sub)
        d['day']     = sub[cfg.day_col].iloc[0]
        d['time']    = sub[cfg.time_col].iloc[0]

        if cfg.split_col in sub.columns:
            d['dataset_split'] = sub[cfg.split_col].iloc[0]

        d['mid']      = sub[cfg.price_col].values.astype(np.float32)
        d['spread']   = sub[cfg.spread_col].values.astype(np.float32)
        d['ret_next'] = sub[cfg.ret_col].values.astype(np.float32)

        # agent signal matrix: (n_stock, n_agents)
        d['agent_signal_mat'] = np.stack(
            [sub[c].values.astype(np.float32) for c in agent_cols], axis=1
        )

        # shared feature matrix: (n_stock, N_SHARED)
        d['shared_mat'] = np.stack(
            [sub[c].values.astype(np.float32) for c in shared_cols], axis=1
        )

        # shared_global_summary = [mean_per_col, std_per_col] over stocks -> 2*N_SHARED
        sh_mean = np.nanmean(d['shared_mat'], axis=0)  # (N_SHARED,)
        sh_std  = np.nanstd(d['shared_mat'],  axis=0)  # (N_SHARED,)
        d['shared_global_summary'] = np.concatenate([sh_mean, sh_std]).astype(np.float32)

        ensemble_sig = d['agent_signal_mat'].mean(axis=1)  # (n_stock,)

        static_states = []
        for i in range(n_agents):
            own  = d['agent_signal_mat'][:, i]
            q10  = _safe_quantile(own, 0.10)
            q25  = _safe_quantile(own, 0.25)
            q50  = _safe_quantile(own, 0.50)
            q75  = _safe_quantile(own, 0.75)
            q90  = _safe_quantile(own, 0.90)

            own_summary = np.array([
                _safe_mean(own), _safe_std(own),
                q10, q25, q50, q75, q90,
                _topk_mean(own,    cfg.topk_state_k),
                _bottomk_mean(own, cfg.topk_state_k),
                q90 - q50, q50 - q10,
                _safe_skew(own),
                _safe_kurtosis_excess(own),
                _topk_share_nonneg(own, 1),
                _topk_share_nonneg(own, cfg.topk_state_k),
            ], dtype=np.float32)   # 15 features

            rel_summary = np.array([
                _safe_corr(own, ensemble_sig),
                _safe_mean(np.abs(own - ensemble_sig)),
            ], dtype=np.float32)   # 2 features

            # static_state_i shape = 15+2+2*N_SHARED
            static_state_i = np.concatenate([
                own_summary,
                rel_summary,
                d['shared_global_summary'],
            ]).astype(np.float32)

            static_states.append(static_state_i)

        d['static_states'] = np.stack(static_states, axis=0)  # (n_agents, 15+2+2*N_SHARED)
        cache.append(d)

    return cache


print('prepare_dataframe and build_snapshot_cache defined.')


# ============================================================
# Cell 11
# ============================================================
# ============================================================
# build_dynamic_agent_states
# Full state = static_state_i + port_global + port_agent_i
# ============================================================

def build_dynamic_agent_states(
    cur: dict,
    inventory_w: np.ndarray,
    sellable_w: np.ndarray,
    prev_gates: np.ndarray,
    turnover_lag: float,
    cfg: CFG,
) -> np.ndarray:
    """Returns (n_agents, state_dim) array."""
    inventory_w = np.asarray(inventory_w, dtype=np.float32)
    sellable_w  = np.asarray(sellable_w,  dtype=np.float32)
    prev_gates  = np.asarray(prev_gates,  dtype=np.float32).reshape(-1)

    cash             = max(0.0, 1.0 - float(inventory_w.sum()))
    invested         = float(inventory_w.sum())
    sellable_total   = float(sellable_w.sum())
    unsellable_total = max(0.0, invested - sellable_total)

    # port_global: 8 features
    port_global = np.array([
        cash, invested, sellable_total, unsellable_total,
        _hhi(inventory_w),
        float(inventory_w.max()) if len(inventory_w) else 0.0,
        _topk_weight_sum(inventory_w, cfg.topk_state_k),
        float(turnover_lag),
    ], dtype=np.float32)

    states = []
    for i in range(n_agents):
        own_sig = cur['agent_signal_mat'][:, i]

        inv_exposure  = float(np.dot(inventory_w, own_sig))
        sell_exposure = float(np.dot(sellable_w,  own_sig))

        if len(inventory_w) > 0:
            k = min(cfg.topk_state_k, len(inventory_w))
            top_idx = np.argpartition(inventory_w, -k)[-k:]
            top5_hold_sig_mean = float(own_sig[top_idx].mean())
        else:
            top5_hold_sig_mean = 0.0

        # port_agent: 8 features
        port_agent = np.array([
            float(prev_gates[i]) if i < len(prev_gates) else 1.0,
            _safe_mean(inventory_w), _safe_std(inventory_w),
            _safe_mean(sellable_w),  _safe_std(sellable_w),
            inv_exposure, sell_exposure, top5_hold_sig_mean,
        ], dtype=np.float32)

        s = np.concatenate([
            cur['static_states'][i],  # 15+2+2*N_SHARED
            port_global,               # 8
            port_agent,                # 8
        ]).astype(np.float32)

        states.append(s)

    return np.stack(states, axis=0)   # (n_agents, state_dim)


print('build_dynamic_agent_states defined.')


# ============================================================
# Cell 12
# ============================================================
# ============================================================
# build_portfolio_weights_from_agent_scores
# execute_target_weight_t1
# refresh_sellable_on_new_day
# ============================================================

def build_portfolio_weights_from_agent_scores(
    agent_scores: np.ndarray,
    prev_w: np.ndarray,
    long_only: bool = True,
    max_weight_per_stock: float = 0.20,
    topk: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    final_score = agent_scores.mean(axis=1).astype(np.float32)

    if topk is not None and topk < len(final_score):
        idx  = np.argpartition(final_score, -topk)[-topk:]
        mask = np.zeros(len(final_score), dtype=bool); mask[idx] = True
        score = np.where(mask, final_score, -1e9 if long_only else 0.0)
    else:
        score = final_score

    if long_only:
        raw = np.maximum(score, 0.0).astype(np.float32)
        s   = raw.sum()
        w   = (raw / s).astype(np.float32) if s > 1e-12 else np.zeros_like(raw)
        w   = np.clip(w, 0.0, max_weight_per_stock).astype(np.float32)
        s   = w.sum()
        if s > 1e-12:
            w = (w / s).astype(np.float32)
    else:
        pos = np.maximum(score, 0.0).astype(np.float32)
        neg = np.maximum(-score, 0.0).astype(np.float32)
        if pos.sum() > 1e-12: pos = pos / pos.sum()
        if neg.sum() > 1e-12: neg = neg / neg.sum()
        w = (pos - neg).astype(np.float32)

    delta_w = (w - prev_w).astype(np.float32)
    return final_score, w, np.maximum(delta_w, 0.0).astype(np.float32), np.maximum(-delta_w, 0.0).astype(np.float32)


def refresh_sellable_on_new_day(prev_day, cur_day, inventory_w, sellable_w):
    if prev_day is None or cur_day != prev_day:
        return np.asarray(inventory_w, dtype=np.float32).copy()
    return np.asarray(sellable_w, dtype=np.float32)


def execute_target_weight_t1(
    target_w: np.ndarray,
    inventory_w: np.ndarray,
    sellable_w: np.ndarray,
    max_weight_per_stock: float = 0.20,
    long_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """T+1 execution with China A-share T+1 constraint."""
    target_w    = np.clip(np.maximum(target_w, 0.0) if long_only else target_w,
                          0.0, max_weight_per_stock).astype(np.float32)
    inventory_w = np.asarray(inventory_w, dtype=np.float32)
    sellable_w  = np.asarray(sellable_w,  dtype=np.float32)

    s = target_w.sum()
    if s > 1e-12:
        target_w = (target_w / s).astype(np.float32)

    desired_sell = np.maximum(inventory_w - target_w, 0.0).astype(np.float32)
    desired_buy  = np.maximum(target_w - inventory_w, 0.0).astype(np.float32)

    sell_w = np.minimum(desired_sell, sellable_w).astype(np.float32)

    inv_after_sell      = (inventory_w - sell_w).astype(np.float32)
    sellable_after_sell = (sellable_w  - sell_w).astype(np.float32)

    cash_available = max(0.0, 1.0 - float(inventory_w.sum())) + float(sell_w.sum())
    dbs = float(desired_buy.sum())
    if dbs <= 1e-12 or cash_available <= 1e-12:
        buy_w = np.zeros_like(desired_buy, dtype=np.float32)
    else:
        buy_w = (desired_buy * min(1.0, cash_available / dbs)).astype(np.float32)

    exec_w       = (inv_after_sell + buy_w).astype(np.float32)
    sellable_new = sellable_after_sell.astype(np.float32)

    exec_w = np.clip(exec_w, 0.0, max_weight_per_stock).astype(np.float32)
    tot = float(exec_w.sum())
    if tot > 1.000001:
        exec_w      = (exec_w / tot).astype(np.float32)
        inv_new     = exec_w.copy()
        sellable_new = np.minimum(sellable_new, inv_new).astype(np.float32)
        delta = (inv_new - inventory_w).astype(np.float32)
        buy_w  = np.maximum(delta,  0.0).astype(np.float32)
        sell_w = np.maximum(-delta, 0.0).astype(np.float32)
        exec_w = inv_new
    else:
        inv_new = exec_w.copy()

    cash_left = max(0.0, 1.0 - float(exec_w.sum()))
    return exec_w, buy_w, sell_w, inv_new, sellable_new, cash_left


print('Portfolio and execution helpers defined.')


# ============================================================
# Cell 13
# ============================================================
# ============================================================
# rollout_with_model
# Main rollout loop: train mode (with replay) or predict mode
# Uses composite reward; tracks nav_history in output
# ============================================================

def rollout_with_model(
    df_input: pd.DataFrame,
    model: MATD3SharedCritic,
    cfg: CFG,
    train_mode: bool = False,
    init_inventory_map: Optional[dict] = None,
    init_sellable_map:  Optional[dict] = None,
    init_prev_day=None,
    init_prev_gates: Optional[np.ndarray] = None,
    init_turnover_lag: float = 0.0,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], dict]:

    step_label = '3' if train_mode else '2'
    print(f'Step 1/{step_label}: prepare_dataframe ...')
    df, snapshots = prepare_dataframe(df_input, cfg, show_progress=True)
    print(f'Done. rows={len(df):,}, timestamps={len(snapshots):,}')

    if train_mode and len(snapshots) < 2:
        raise ValueError('Need at least 2 timestamps to train.')

    print(f'Step 2/{step_label}: build_snapshot_cache ...')
    cache = build_snapshot_cache(df, snapshots, cfg, show_progress=True)
    print(f'Done. cached timestamps={len(cache):,}')

    n_rows = len(df)

    # Allocate output buffers
    final_score_buf   = np.full(n_rows, np.nan, dtype=np.float32)
    target_weight_buf = np.full(n_rows, np.nan, dtype=np.float32)
    final_weight_buf  = np.full(n_rows, np.nan, dtype=np.float32)
    sellable_weight_buf = np.full(n_rows, np.nan, dtype=np.float32)
    reward_buf        = np.full(n_rows, np.nan, dtype=np.float32)
    reward_scaled_buf = np.full(n_rows, np.nan, dtype=np.float32)
    gross_return_buf  = np.full(n_rows, np.nan, dtype=np.float32)
    net_return_buf    = np.full(n_rows, np.nan, dtype=np.float32)
    spread_cost_buf   = np.full(n_rows, np.nan, dtype=np.float32)
    turnover_buf      = np.full(n_rows, np.nan, dtype=np.float32)
    nav_buf           = np.full(n_rows, np.nan, dtype=np.float32)
    drawdown_buf      = np.full(n_rows, np.nan, dtype=np.float32)
    cash_buf          = np.full(n_rows, np.nan, dtype=np.float32)
    invested_buf      = np.full(n_rows, np.nan, dtype=np.float32)
    gate_bufs         = {f'gate_agent_{i}': np.full(n_rows, np.nan, dtype=np.float32)
                         for i in range(n_agents)}

    # Portfolio state
    inventory_map = {} if init_inventory_map is None else dict(init_inventory_map)
    sellable_map  = {} if init_sellable_map  is None else dict(init_sellable_map)
    prev_day      = init_prev_day
    prev_gates    = (
        np.ones(n_agents, dtype=np.float32) if init_prev_gates is None
        else np.asarray(init_prev_gates, dtype=np.float32).reshape(-1)
    )
    turnover_lag  = float(init_turnover_lag)

    history = []
    best_mean_reward = -np.inf
    global_step = 0

    def _write_to_bufs(t_idx, cur, exec_w, sellable_w_cur, gates, info):
        row_idx = cur['row_idx']
        final_score_buf[row_idx]    = _final_score
        target_weight_buf[row_idx]  = _target_w
        final_weight_buf[row_idx]   = exec_w
        sellable_weight_buf[row_idx] = sellable_w_cur
        reward_buf[row_idx]         = info['reward']
        reward_scaled_buf[row_idx]  = info['reward_scaled']
        gross_return_buf[row_idx]   = info['gross_return']
        net_return_buf[row_idx]     = info['net_return']
        spread_cost_buf[row_idx]    = info['spread_cost']
        turnover_buf[row_idx]       = info['turnover']
        nav_buf[row_idx]            = info['nav']
        drawdown_buf[row_idx]       = info['drawdown']
        cash_buf[row_idx]           = max(0.0, 1.0 - float(exec_w.sum()))
        invested_buf[row_idx]       = float(exec_w.sum())
        for i in range(n_agents):
            gate_bufs[f'gate_agent_{i}'][row_idx] = float(gates[i, 0])

    if train_mode:
        print('Step 3/3: start training ...')
        replay = ReplayBuffer(cfg.replay_capacity, n_agents, state_dim, action_dim)
        # One RewardTracker per epoch (reset each epoch)
        epoch_pbar = tqdm(range(cfg.epochs), desc='Training Epoch', position=0)

        for epoch in epoch_pbar:
            epoch_rewards_raw = []
            epoch_rewards_scaled = []
            last_logs = {}
            tracker   = RewardTracker(vol_window=cfg.vol_window)

            # Reset portfolio state each epoch
            inventory_map_ep = {}
            sellable_map_ep  = {}
            prev_day_ep      = None
            prev_gates_ep    = np.ones(n_agents, dtype=np.float32)
            turnover_lag_ep  = 0.0

            ts_pbar = tqdm(range(len(cache) - 1), desc=f'Epoch {epoch+1}', position=1, leave=False)

            for t in ts_pbar:
                cur      = cache[t]
                nxt      = cache[t + 1]
                cur_day  = cur['day']
                nxt_day  = nxt['day']
                cur_codes = cur['codes']

                inventory_w_ep = map_weight_dict_to_codes(inventory_map_ep, cur_codes)
                sellable_w_ep  = map_weight_dict_to_codes(sellable_map_ep,  cur_codes)
                sellable_w_ep  = refresh_sellable_on_new_day(prev_day_ep, cur_day, inventory_w_ep, sellable_w_ep)

                states = build_dynamic_agent_states(cur, inventory_w_ep, sellable_w_ep, prev_gates_ep, turnover_lag_ep, cfg)
                gates  = model.act(states, noise_std=cfg.action_noise_std)

                agent_scores = (cur['agent_signal_mat'] * gates.reshape(1, -1)).astype(np.float32)
                _final_score, _target_w, _, _ = build_portfolio_weights_from_agent_scores(
                    agent_scores, inventory_w_ep, cfg.long_only, cfg.max_weight_per_stock, cfg.topk
                )
                exec_w, buy_w, sell_w, inv_new, sell_new_same, cash_left = execute_target_weight_t1(
                    _target_w, inventory_w_ep, sellable_w_ep, cfg.max_weight_per_stock, cfg.long_only
                )

                # ---- composite reward ----
                prev_exec_w = inventory_w_ep   # weight BEFORE execution
                reward_scaled, info = calc_reward_composite(
                    mid=cur['mid'], spread=cur['spread'], ret_next=cur['ret_next'],
                    new_w=exec_w, prev_w=prev_exec_w, tracker=tracker, cfg=cfg,
                )

                # Next state
                inv_next_aligned  = remap_vec_between_codes(cur_codes, inv_new,       nxt['codes'])
                sell_next_aligned = remap_vec_between_codes(cur_codes, sell_new_same, nxt['codes'])
                next_sellable     = refresh_sellable_on_new_day(cur_day, nxt_day, inv_next_aligned, sell_next_aligned)
                next_states       = build_dynamic_agent_states(nxt, inv_next_aligned, next_sellable,
                                                                gates.reshape(-1), info['turnover'], cfg)

                done = 1.0 if t == len(cache) - 2 else 0.0
                replay.push(states, gates, reward_scaled, next_states, done)

                if len(replay) >= max(cfg.batch_size, cfg.warmup_steps) and global_step % cfg.train_every == 0:
                    for _ in range(cfg.updates_per_step):
                        last_logs = model.update(replay, cfg.batch_size)

                _write_to_bufs(t, cur, exec_w, sellable_w_ep, gates, info)

                inventory_map_ep = vec_to_map(cur_codes, inv_new)
                sellable_map_ep  = vec_to_map(cur_codes, sell_new_same)
                prev_day_ep      = cur_day
                prev_gates_ep    = gates.reshape(-1).astype(np.float32)
                turnover_lag_ep  = info['turnover']

                epoch_rewards_raw.append(info['reward'])
                epoch_rewards_scaled.append(info['reward_scaled'])
                global_step += 1

                ts_pbar.set_postfix({
                    'net_ret': f"{info['net_return']:.5f}",
                    'nav':     f"{info['nav']:.4f}",
                    'dd':      f"{info['drawdown']:.4f}",
                    'critic':  f"{last_logs.get('critic_loss', float('nan')):.4f}" if last_logs else 'nan',
                })

            mean_r_raw    = float(np.mean(epoch_rewards_raw))    if epoch_rewards_raw    else float('nan')
            mean_r_scaled = float(np.mean(epoch_rewards_scaled)) if epoch_rewards_scaled else float('nan')

            rec = {'epoch': epoch + 1, 'mean_reward_raw': mean_r_raw, 'mean_reward_scaled': mean_r_scaled,
                   'final_nav': tracker.nav}
            rec.update(last_logs)
            history.append(rec)

            epoch_pbar.set_postfix({
                'mean_reward': f'{mean_r_raw:.6f}',
                'final_nav':   f'{tracker.nav:.4f}',
            })
            print(f'[Epoch {epoch+1}/{cfg.epochs}] mean_reward={mean_r_raw:.8f}, nav={tracker.nav:.4f}')

            if mean_r_raw > best_mean_reward:
                best_mean_reward = mean_r_raw
                save_matd3_model(model, cfg, cfg.save_dir)

        hist_df = pd.DataFrame(history)

    else:
        # Predict mode
        tracker   = RewardTracker(vol_window=cfg.vol_window)
        ts_pbar   = tqdm(range(len(cache)), desc='Predict timestamps')
        hist_df   = None

        for t in ts_pbar:
            cur       = cache[t]
            cur_day   = cur['day']
            cur_codes = cur['codes']

            inventory_w = map_weight_dict_to_codes(inventory_map, cur_codes)
            sellable_w  = map_weight_dict_to_codes(sellable_map,  cur_codes)
            sellable_w  = refresh_sellable_on_new_day(prev_day, cur_day, inventory_w, sellable_w)

            states = build_dynamic_agent_states(cur, inventory_w, sellable_w, prev_gates, turnover_lag, cfg)
            gates  = model.act(states, noise_std=0.0)

            agent_scores = (cur['agent_signal_mat'] * gates.reshape(1, -1)).astype(np.float32)
            _final_score, _target_w, _, _ = build_portfolio_weights_from_agent_scores(
                agent_scores, inventory_w, cfg.long_only, cfg.max_weight_per_stock, cfg.topk
            )
            exec_w, buy_w, sell_w, inv_new, sell_new_same, cash_left = execute_target_weight_t1(
                _target_w, inventory_w, sellable_w, cfg.max_weight_per_stock, cfg.long_only
            )

            prev_exec_w = inventory_w
            reward_scaled, info = calc_reward_composite(
                mid=cur['mid'], spread=cur['spread'], ret_next=cur['ret_next'],
                new_w=exec_w, prev_w=prev_exec_w, tracker=tracker, cfg=cfg,
            )

            _write_to_bufs(t, cur, exec_w, sellable_w, gates, info)

            inventory_map = vec_to_map(cur_codes, inv_new)
            sellable_map  = vec_to_map(cur_codes, sell_new_same)
            prev_day      = cur_day
            prev_gates    = gates.reshape(-1).astype(np.float32)
            turnover_lag  = info['turnover']

            ts_pbar.set_postfix({
                'net_ret': f"{info['net_return']:.5f}",
                'nav':     f"{info['nav']:.4f}",
            })

    # Build output dataframe
    df_out = df.copy()
    df_out['final_score']      = final_score_buf
    df_out['target_weight']    = target_weight_buf
    df_out['final_weight']     = final_weight_buf
    df_out['sellable_weight']  = sellable_weight_buf
    df_out['reward_ts']        = reward_buf
    df_out['reward_scaled_ts'] = reward_scaled_buf
    df_out['gross_return_ts']  = gross_return_buf
    df_out['net_return_ts']    = net_return_buf
    df_out['spread_cost_ts']   = spread_cost_buf
    df_out['turnover_ts']      = turnover_buf
    df_out['nav_ts']           = nav_buf
    df_out['drawdown_ts']      = drawdown_buf
    df_out['cash_ts']          = cash_buf
    df_out['invested_ts']      = invested_buf
    for k, v in gate_bufs.items():
        df_out[k] = v

    final_state = {
        'inventory_map': inventory_map,
        'sellable_map':  sellable_map,
        'prev_day':      prev_day,
        'prev_gates':    prev_gates,
        'turnover_lag':  turnover_lag,
    }

    return df_out, hist_df, final_state


print('rollout_with_model defined.')


# ============================================================
# Cell 14
# ============================================================
# ============================================================
# save_matd3_model / load_matd3_model
# train_matd3_on_frame / predict_matd3_weights
# build_nav_from_pred / run_matd3_on_one_frame
# ============================================================

def save_matd3_model(model: MATD3SharedCritic, cfg: CFG, save_dir: Optional[str] = None) -> None:
    if save_dir is None: save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
    for i, actor in enumerate(model.actors):
        torch.save(actor.state_dict(), os.path.join(save_dir, f'actor_agent_{i}.pt'))
    for i, ta in enumerate(model.target_actors):
        torch.save(ta.state_dict(), os.path.join(save_dir, f'target_actor_agent_{i}.pt'))
    torch.save(model.critic1.state_dict(),        os.path.join(save_dir, 'shared_critic1.pt'))
    torch.save(model.critic2.state_dict(),        os.path.join(save_dir, 'shared_critic2.pt'))
    torch.save(model.target_critic1.state_dict(), os.path.join(save_dir, 'target_shared_critic1.pt'))
    torch.save(model.target_critic2.state_dict(), os.path.join(save_dir, 'target_shared_critic2.pt'))
    print(f'Model saved to: {save_dir}')


def load_matd3_model(
    cfg: CFG, n_agents: int, state_dim: int, action_dim: int,
    save_dir: Optional[str] = None,
) -> MATD3SharedCritic:
    if save_dir is None: save_dir = cfg.save_dir
    model = MATD3SharedCritic(n_agents, state_dim, action_dim, cfg)
    for i in range(n_agents):
        model.actors[i].load_state_dict(
            torch.load(os.path.join(save_dir, f'actor_agent_{i}.pt'), map_location=DEVICE))
        ta_path = os.path.join(save_dir, f'target_actor_agent_{i}.pt')
        model.target_actors[i].load_state_dict(
            torch.load(ta_path, map_location=DEVICE) if os.path.exists(ta_path)
            else model.actors[i].state_dict())
    model.critic1.load_state_dict(
        torch.load(os.path.join(save_dir, 'shared_critic1.pt'), map_location=DEVICE))
    model.critic2.load_state_dict(
        torch.load(os.path.join(save_dir, 'shared_critic2.pt'), map_location=DEVICE))
    model.target_critic1.load_state_dict(
        torch.load(os.path.join(save_dir, 'target_shared_critic1.pt'), map_location=DEVICE))
    model.target_critic2.load_state_dict(
        torch.load(os.path.join(save_dir, 'target_shared_critic2.pt'), map_location=DEVICE))
    print(f'Model loaded from: {save_dir}')
    return model


def train_matd3_on_frame(
    df_train: pd.DataFrame, cfg: CFG
) -> Tuple[MATD3SharedCritic, pd.DataFrame, pd.DataFrame, dict]:
    model = MATD3SharedCritic(n_agents, state_dim, action_dim, cfg)
    df_out, hist_df, final_state = rollout_with_model(df_train, model, cfg, train_mode=True)
    return model, df_out, hist_df, final_state


@torch.no_grad()
def predict_matd3_weights(
    df_input: pd.DataFrame,
    model: MATD3SharedCritic,
    cfg: CFG,
    init_state: Optional[dict] = None,
) -> Tuple[pd.DataFrame, dict]:
    init_state = {} if init_state is None else init_state
    df_pred, _, final_state = rollout_with_model(
        df_input, model, cfg, train_mode=False,
        init_inventory_map=init_state.get('inventory_map'),
        init_sellable_map=init_state.get('sellable_map'),
        init_prev_day=init_state.get('prev_day'),
        init_prev_gates=init_state.get('prev_gates'),
        init_turnover_lag=init_state.get('turnover_lag', 0.0),
    )
    return df_pred, final_state


def build_nav_from_pred(df_pred: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    """Build per-timestamp NAV curve from rollout prediction dataframe."""
    cols = [cfg.day_col, cfg.time_col, cfg.split_col, 'gross_return_ts', 'net_return_ts',
            'spread_cost_ts', 'reward_ts', 'turnover_ts', 'nav_ts', 'drawdown_ts']
    cols = [c for c in cols if c in df_pred.columns]
    ts_curve = (
        df_pred[cols]
          .sort_values([cfg.day_col, cfg.time_col])
          .groupby([cfg.day_col, cfg.time_col], as_index=False)
          .first()
          .copy()
    )
    ts_curve['gross_nav'] = (1.0 + ts_curve['gross_return_ts'].fillna(0.0)).cumprod()
    ts_curve['net_nav']   = (1.0 + ts_curve['net_return_ts'].fillna(0.0)).cumprod()
    return ts_curve


def run_matd3_on_one_frame(df_full: pd.DataFrame, cfg: CFG) -> dict:
    """Full pipeline: train on train split, evaluate on test, return all results."""
    if cfg.split_col not in df_full.columns:
        raise ValueError(f'{cfg.split_col} not found in df_full')

    df_train = df_full[df_full[cfg.split_col] == 'train'].copy()
    df_test  = df_full[df_full[cfg.split_col] == 'test'].copy()
    print(f'Train rows: {len(df_train):,}  |  Test rows: {len(df_test):,}')

    # Train
    model, df_train_out, hist_df, train_final_state = train_matd3_on_frame(df_train, cfg)

    # Load best checkpoint
    loaded_model = load_matd3_model(cfg, n_agents, state_dim, action_dim, cfg.save_dir)

    # Full-period prediction (train + test, zero init)
    df_full_pred, _ = predict_matd3_weights(df_full, loaded_model, cfg, init_state=None)
    nav_full         = build_nav_from_pred(df_full_pred, cfg)

    # Test-only: zero initial state
    df_test_pred_zero, _ = predict_matd3_weights(df_test, loaded_model, cfg, init_state=None)
    nav_test_zero = build_nav_from_pred(df_test_pred_zero, cfg) if len(df_test_pred_zero) else pd.DataFrame()

    # Test-only: continuous from train final state
    df_test_pred_cont, _ = predict_matd3_weights(df_test, loaded_model, cfg, init_state=train_final_state)
    nav_test_cont = build_nav_from_pred(df_test_pred_cont, cfg) if len(df_test_pred_cont) else pd.DataFrame()

    return {
        'model':              loaded_model,
        'df_train_out':       df_train_out,
        'hist_df':            hist_df,
        'df_full_pred':       df_full_pred,
        'nav_full':           nav_full,
        'df_test_pred_zero':  df_test_pred_zero,
        'nav_test_zero':      nav_test_zero,
        'df_test_pred_cont':  df_test_pred_cont,
        'nav_test_cont':      nav_test_cont,
        'train_final_state':  train_final_state,
    }


print('All pipeline functions defined.')


# ============================================================
# Cell 15
# ============================================================
# ============================================================
# LOAD DATA + TRAIN
# ============================================================

# Make sure we are in capstone directory
os.chdir('e:/New_folder/Work/MyProject/capstone')

print('Loading df_full_rl_clean.parquet ...')
# Use close-bar-only data to match backtest rebalance frequency
import os
close_path = 'df_full_rl_clean_close.parquet'
if not os.path.exists(close_path):
    df_tmp = pd.read_parquet('df_full_rl_clean.parquet')
    df_tmp = df_tmp[df_tmp['TimeEnd'] == 1500].copy()
    df_tmp.to_parquet(close_path, index=False)
    del df_tmp
df_full = pd.read_parquet(close_path)
print('Shape:', df_full.shape)
print('Columns:', df_full.columns.tolist())
print('Split distribution:')
print(df_full['dataset_split'].value_counts())
print()

# Verify all required columns are present
required = (
    list(cfg.agent_signal_cols) +
    list(cfg.shared_state_cols) +
    [cfg.stock_col, cfg.day_col, cfg.time_col, cfg.price_col, cfg.spread_col, cfg.ret_col, cfg.split_col]
)
missing = [c for c in required if c not in df_full.columns]
if missing:
    print('ERROR: Missing columns:', missing)
else:
    print('All required columns present.')
    print()
    print('Running MATD3 pipeline ...')
    res = run_matd3_on_one_frame(df_full, cfg)
    print('\nPipeline complete!')


# ============================================================
# Cell 16
# ============================================================
# ============================================================
# Extract Results
# ============================================================

model          = res['model']
hist_df        = res['hist_df']
df_full_pred   = res['df_full_pred']
nav_full       = res['nav_full']
df_test_pred_zero = res['df_test_pred_zero']
nav_test_zero  = res['nav_test_zero']
df_test_pred_cont = res['df_test_pred_cont']
nav_test_cont  = res['nav_test_cont']

print('hist_df shape:', hist_df.shape if hist_df is not None else 'None')
print('df_full_pred shape:', df_full_pred.shape)
print('nav_full shape:', nav_full.shape)
print()

if hist_df is not None and len(hist_df):
    print('Training history summary:')
    print(hist_df[['epoch', 'mean_reward_raw', 'final_nav']].to_string(index=False))

print()
print('Full period NAV tail:')
print(nav_full[['gross_nav', 'net_nav']].tail())


# ============================================================
# Cell 17
# ============================================================
# ============================================================
# STEP D: Close-to-Close Backtest (Backtesting.ipynb framework)
# ============================================================
import os
OUT_DIR = 'models_comparison/matd3_v3'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Backtesting functions ─────────────────────────────────

def prepare_close_bar_data(df, signal_col, price_col='mid', spread_col='spread',
                           day_col='TradingDay', stock_col='SecuCode',
                           close_ts=1445, close_te=1500):
    df = df.copy().sort_values([day_col, 'TimeStart', 'TimeEnd', stock_col]).reset_index(drop=True)
    df_close = df[(df['TimeStart']==close_ts)&(df['TimeEnd']==close_te)][[day_col,stock_col,signal_col,price_col,spread_col]].copy()
    df_close = df_close.sort_values([stock_col, day_col]).reset_index(drop=True)
    df_close['has_price_t'] = np.isfinite(df_close[price_col].astype(float))
    df_close['has_spread_t'] = np.isfinite(df_close[spread_col].astype(float))
    df_close['price_t1'] = df_close.groupby(stock_col)[price_col].shift(-1)
    df_close['spread_t1'] = df_close.groupby(stock_col)[spread_col].shift(-1)
    df_close['day_t1'] = df_close.groupby(stock_col)[day_col].shift(-1)
    all_days = np.sort(df_close[day_col].unique())
    ndm = pd.Series(all_days[1:], index=all_days[:-1]).to_dict()
    df_close['expected_day_t1'] = df_close[day_col].map(ndm)
    ok_next = df_close['day_t1'].eq(df_close['expected_day_t1'])
    df_close['has_price_t1'] = ok_next & np.isfinite(df_close['price_t1'].astype(float))
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

# ── Generate MATD3 signal ─────────────────────────────────
print('Generating MATD3 v2 close-bar signals...')
loaded_model = load_matd3_model(cfg, n_agents, state_dim, action_dim, cfg.save_dir)

# Extract final_score from full-period rollout prediction
matd3_close = df_full_pred[df_full_pred[cfg.time_col] == 1500][
    [cfg.stock_col, cfg.day_col, 'final_score']
].copy().rename(columns={'final_score': 'signal_matd3'})

df_raw_full = pd.read_parquet('df_testsub_full.parquet')
df_raw_full = df_raw_full.merge(matd3_close, on=[cfg.stock_col, cfg.day_col], how='left')
df_raw_full['signal_matd3'] = df_raw_full['signal_matd3'].fillna(0.0)

# ── Run all backtests ─────────────────────────────────────
BT_TOPK = 5; BT_FEE = 0.0003
SIGNALS = [('LightGBM','signal_pre_boost'), ('CNN+LSTM+TF','signal_hybrid'),
           ('SSM+TF','signal_ssm_tf'), ('Cross-Attn','signal_tf_cross'),
           ('GAT+TF','signal_gat_tf')]
raw_cols = [s[1] for s in SIGNALS]
df_raw_full['signal_ew_ensemble'] = df_raw_full[raw_cols].mean(axis=1)
# Load MATD3 v2 model signal for comparison
import torch as _torch
class _ActorV2(nn.Module):
    def __init__(self, sd, hd=128, ad=1):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(sd,hd),nn.LayerNorm(hd),nn.ReLU(),nn.Linear(hd,hd),nn.ReLU(),nn.Linear(hd,ad))
    def forward(self, x): return self.backbone(x)

v2_dir = 'matd3_v2_models'
if os.path.exists(os.path.join(v2_dir, 'actor_agent_0.pt')):
    print('Loading MATD3 v2 model for comparison...')
    v2_actors = []
    for i in range(n_agents):
        a = _ActorV2(state_dim); a.load_state_dict(_torch.load(os.path.join(v2_dir, f'actor_agent_{i}.pt'), map_location='cpu')); a.eval()
        v2_actors.append(a)
    # v2 uses gate = 1.0 + 1.0 * tanh (range [0, 2])
    v2_results = []
    df_rl_v2 = pd.read_parquet('df_full_rl_clean_close.parquet')
    df_rl_v2 = df_rl_v2.sort_values(['TradingDay','TimeEnd','SecuCode']).reset_index(drop=True)
    v2_groups = {k: g.sort_values('SecuCode').reset_index(drop=True) for k, g in df_rl_v2.groupby(['TradingDay','TimeEnd'], sort=True)}
    _agent_cols = list(cfg.agent_signal_cols); _shared_cols = list(cfg.shared_state_cols)
    _inv2 = {}; _prev_day2 = None; _pg2 = np.ones(n_agents, dtype=np.float32); _tl2 = 0.0
    for (day, te), sub in sorted(v2_groups.items()):
        codes = sub['SecuCode'].values; n = len(codes)
        iw = np.array([_inv2.get(c,0.0) for c in codes], dtype=np.float32)
        if _prev_day2 is not None and day != _prev_day2: sw = iw.copy()
        else: sw = iw.copy()
        am = np.stack([sub[c].values.astype(np.float32) for c in _agent_cols], axis=1)
        sm_ = np.stack([sub[c].values.astype(np.float32) for c in _shared_cols], axis=1)
        sg = np.concatenate([np.nanmean(sm_,axis=0), np.nanstd(sm_,axis=0)]).astype(np.float32)
        ens = am.mean(axis=1)
        cash=max(0,1-iw.sum()); inv=iw.sum()
        pg = np.array([cash,inv,sw.sum(),max(0,inv-sw.sum()),float((iw**2).sum()),float(iw.max()) if n else 0,
                        float(iw[np.argpartition(iw,-min(5,n))[-min(5,n):]].sum()) if n else 0, _tl2], dtype=np.float32)
        gates2 = np.zeros((n_agents,1), dtype=np.float32)
        for i in range(n_agents):
            own=am[:,i]; q=[np.quantile(own,q) for q in [.1,.25,.5,.75,.9]]
            os_=np.array([own.mean(),own.std(),*q,
                float(own[np.argpartition(own,-min(5,n))[-min(5,n):]].mean()) if n else 0,
                float(own[np.argpartition(own,min(5,n)-1)[:min(5,n)]].mean()) if n else 0,
                q[4]-q[2],q[2]-q[0],0,0,0,0], dtype=np.float32)
            rs_=np.array([float(np.corrcoef(own,ens)[0,1]) if own.std()>1e-12 and ens.std()>1e-12 else 0,
                          float(np.abs(own-ens).mean())], dtype=np.float32)
            ie=float(np.dot(iw,own)); se=float(np.dot(sw,own))
            t5=float(own[np.argpartition(iw,-min(5,n))[-min(5,n):]].mean()) if n else 0
            pa=np.array([float(_pg2[i]),iw.mean(),iw.std(),sw.mean(),sw.std(),ie,se,t5], dtype=np.float32)
            st=np.concatenate([os_,rs_,sg,pg,pa])
            with _torch.no_grad():
                dr=v2_actors[i](_torch.tensor(st,dtype=_torch.float32).unsqueeze(0))
                g=np.clip((1.0+1.0*np.tanh(dr.squeeze().numpy())),0,2)
            gates2[i]=g
        fs = (am * gates2.reshape(1,-1)).mean(axis=1)
        for j in range(n): v2_results.append((day, te, codes[j], float(fs[j])))
        topk=min(5,n)
        if topk>0:
            ti=np.argpartition(fs,-topk)[-topk:]; nw=np.zeros(n,dtype=np.float32); nw[ti]=1.0/topk
        else: nw=np.zeros(n)
        _inv2={codes[j]:float(nw[j]) for j in range(n)}; _prev_day2=day; _pg2=gates2.reshape(-1); _tl2=float(np.abs(nw-iw).sum())
    v2_df = pd.DataFrame(v2_results, columns=['TradingDay','TimeEnd','SecuCode','signal_matd3_v2'])
    df_raw_full = df_raw_full.merge(v2_df, on=['TradingDay','TimeEnd','SecuCode'], how='left')
    df_raw_full['signal_matd3_v2'] = df_raw_full['signal_matd3_v2'].fillna(0.0)
    SIGNALS.append(('MATD3 v2', 'signal_matd3_v2'))
    print('  MATD3 v2 signal loaded')
else:
    print('MATD3 v2 checkpoint not found, skipping v2 comparison')

SIGNALS += [('EW Ensemble','signal_ew_ensemble'), ('MATD3 v3','signal_matd3')]

all_eq = {}; all_metrics = {}
for name, sig_col in SIGNALS:
    prep = prepare_close_bar_data(df_raw_full, signal_col=sig_col)
    wobj = transform_signal_to_weight_topk(prep['signal_mat'], prep['eligible_mat'], BT_TOPK)
    eq = run_c2c_backtest(prep['price_mat'], prep['spread_mat'], wobj['target_w_mat'], fee_rate=BT_FEE)
    net = (1+eq['ret_net'].fillna(0)).cumprod().iloc[-1]
    print(f'  {name:20s}  Net NAV={net:.4f}')
    all_eq[name] = eq; all_metrics[name] = bt_metrics(eq)

print('\n' + '='*80)
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


# ============================================================
# Cell 18
# ============================================================
# ============================================================
# Plot 1: PnL Comparison
# ============================================================
sig_colors = {'LightGBM':'#1f77b4','CNN+LSTM+TF':'#ff7f0e','SSM+TF':'#2ca02c',
              'Cross-Attn':'#d62728','GAT+TF':'#9467bd','EW Ensemble':'#888888',
              'MATD3 v2':'#ff69b4','MATD3 v3':'#e31a1c'}
fig, axes = plt.subplots(2, 1, figsize=(16, 11), gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle(f'Close-to-Close: All Models (Top-{BT_TOPK} EW, Fee={BT_FEE})', fontsize=14, fontweight='bold')
ax1 = axes[0]
for name, eq in all_eq.items():
    nav = (1+eq['ret_net'].fillna(0)).cumprod()
    days = pd.to_datetime(eq['TradingDay'].astype(str), format='%Y%m%d')
    lw = 3.0 if 'MATD3' in name else (2.0 if 'Ensemble' in name else 1.3)
    ls = '--' if 'Ensemble' in name else '-'
    ax1.plot(days, nav.values, label=name, color=sig_colors.get(name,'gray'), linewidth=lw, linestyle=ls, alpha=0.9, zorder=10 if 'MATD3' in name else 5)
ax1.axhline(1.0, color='gray', linestyle=':', linewidth=0.6); ax1.set_ylabel('NAV (Net)')
ax1.legend(fontsize=9, loc='best', ncol=2); ax1.grid(True, alpha=0.3)
ax2 = axes[1]
for name, eq in all_eq.items():
    nav = (1+eq['ret_net'].fillna(0)).cumprod()
    dd = (nav-nav.cummax())/nav.cummax().clip(lower=1e-8)
    days = pd.to_datetime(eq['TradingDay'].astype(str), format='%Y%m%d')
    lw = 2.0 if 'MATD3' in name else 0.9
    ax2.plot(days, dd.values, color=sig_colors.get(name,'gray'), linewidth=lw, alpha=0.7,
             label=name if 'MATD3' in name or 'Ensemble' in name else '')
ax2.set_ylabel('Drawdown'); ax2.set_xlabel('Date')
ax2.legend(fontsize=8, loc='lower left'); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'pnl_comparison_all_models.png'), dpi=150, bbox_inches='tight')
plt.show(); print('Saved: pnl_comparison_all_models.png')


# ============================================================
# Cell 19
# ============================================================
# ============================================================
# Plot 2: Key Metrics Bar Chart
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Key Metrics Comparison', fontsize=13, fontweight='bold')
names = list(all_metrics.keys()); x = np.arange(len(names))
for ax_i, metric, title in zip(axes, ['Sharpe','Max Drawdown','Total Return'], ['Sharpe Ratio','Max Drawdown','Total Return']):
    vals = [all_metrics[n][metric] for n in names]
    colors = ['#e31a1c' if 'MATD3' in n else '#1f77b4' for n in names]
    bars = ax_i.bar(x, vals, color=colors, alpha=0.8)
    ax_i.set_xticks(x)
    ax_i.set_xticklabels([n.replace(' v2','') for n in names], rotation=30, ha='right', fontsize=7)
    ax_i.set_title(title); ax_i.axhline(0, color='black', linewidth=0.5); ax_i.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, vals):
        fmt = f'{v:.2%}' if 'Return' in title or 'Drawdown' in title else f'{v:.2f}'
        ax_i.text(bar.get_x()+bar.get_width()/2, bar.get_height(), fmt, ha='center', va='bottom', fontsize=7, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'metrics_bar_chart.png'), dpi=150, bbox_inches='tight')
plt.show(); print('Saved: metrics_bar_chart.png')


# ============================================================
# Cell 20
# ============================================================
# ============================================================
# Plot 3: Gate Weights Analysis
# ============================================================
agent_names = ['LightGBM','CNN+LSTM+TF','SSM+TF','Cross-Attn','GAT+TF']
gate_cols = [f'gate_agent_{i}' for i in range(n_agents)]
gate_ts = (df_full_pred[[cfg.day_col, cfg.time_col]+gate_cols]
    .sort_values([cfg.day_col, cfg.time_col])
    .groupby([cfg.day_col, cfg.time_col], as_index=False).first())
fig, axes = plt.subplots(n_agents, 1, figsize=(14, 3*n_agents), sharex=True)
fig.suptitle('MATD3 v3 Gate Weights [0, 3]', fontsize=14, fontweight='bold')
cc = plt.cm.tab10(np.linspace(0, 1, n_agents))
for i, (ax, gc, an, cl) in enumerate(zip(axes, gate_cols, agent_names, cc)):
    g = gate_ts[gc].fillna(1.0).values
    ax.plot(g, color=cl, linewidth=0.8, alpha=0.85, label=an)
    mg = float(np.mean(g))
    ax.axhline(mg, color=cl, linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean={mg:.3f}')
    ax.axhline(1.0, color='black', linestyle=':', linewidth=0.6, alpha=0.4)
    ax.set_ylim(-0.05, 3.15); ax.set_ylabel('Gate')
    ax.set_title(f'Agent {i}: {an}', fontsize=9)
    ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)
axes[-1].set_xlabel('Timestamp Index')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'gate_weights_matd3.png'), dpi=150, bbox_inches='tight')
plt.show(); print('Saved: gate_weights_matd3.png')
print('\n=== Gate Summary ===')
gs = {}
for gc, an in zip(gate_cols, agent_names):
    g = gate_ts[gc].dropna().values
    gs[an] = {'mean':np.mean(g),'std':np.std(g),'min':np.min(g),'max':np.max(g),
              '>1.0':f'{(g>1).mean():.1%}','near_0':f'{(g<0.05).mean():.1%}'}
print(pd.DataFrame(gs).T.to_string())

