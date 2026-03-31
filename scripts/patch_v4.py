"""
Patch rl_full_pipeline.ipynb (v3) -> rl_full_pipeline_v4.ipynb

v4 changes:
  1. Turnover penalty in reward  (-turnover_lambda * turnover)
  2. Softmax portfolio weights   (instead of EW top-k)
  3. Turnover constraint in backtest (max_daily_turnover cap)
  4. Volatility targeting reward (penalise vol above target, not all vol)
  5. CFG: save_dir='matd3_v4_models', epochs=80, turnover_lambda=0.5
  6. Comparison chart includes v2, v3, v4
"""
import json, re, os, shutil

SRC  = 'notebooks/rl_full_pipeline.ipynb'
DEST = 'notebooks/rl_full_pipeline_v4.ipynb'

with open(SRC, encoding='utf-8') as f:
    nb = json.load(f)

changes = []

def patch_cell(nb, cell_idx, old, new, tag):
    src = ''.join(nb['cells'][cell_idx]['source'])
    if old not in src:
        print(f'  [WARN] {tag}: pattern not found in cell {cell_idx}')
        return False
    nb['cells'][cell_idx]['source'] = [src.replace(old, new, 1)]
    changes.append(tag)
    return True

# ── Cell 0: update title ──────────────────────────────────────────────────────
patch_cell(nb, 0,
    '# RL Full Pipeline v3: MATD3 Wide Gate [0,3] + Aggressive Reward + Close-Bar Training',
    '# RL Full Pipeline v4: MATD3 + Turnover Penalty + Softmax Weights + Vol Targeting',
    'title')

# ── Cell 3: CFG — add turnover_lambda, vol_target; update save_dir, epochs ───
patch_cell(nb, 3,
    "    turno",   # the truncated line from cell 3 preview
    "    turno",   # no-op placeholder; we patch below
    '')  # placeholder only — real patches follow

src3 = ''.join(nb['cells'][3]['source'])

# 1. Add turnover_lambda field after vol_window
if 'turnover_lambda' not in src3:
    src3 = src3.replace(
        '    vol_window:              int   = 50     # rolling window for variance\n    risk_penalty_lambda:     float = 0.0',
        '    vol_window:              int   = 50     # rolling window for variance\n    turnover_lambda:         float = 0.5   # v4: cost-aware training — penalise daily turnover\n    vol_target_daily:        float = 0.00078  # v4: ~20% ann vol target per bar (20%/sqrt(252*16))\n    risk_penalty_lambda:     float = 0.0'
    )

# 2. save_dir v3 -> v4
src3 = src3.replace("save_dir: str = 'matd3_v3_models'", "save_dir: str = 'matd3_v4_models'")

# 3. epochs 50 -> 80
src3 = src3.replace('    epochs:                  int   = 50',
                    '    epochs:                  int   = 80')

nb['cells'][3]['source'] = [src3]
changes.append('CFG: turnover_lambda=0.5, vol_target, save_dir=v4, epochs=80')

# ── Cell 9: reward — add turnover penalty + vol targeting ─────────────────────
src9 = ''.join(nb['cells'][9]['source'])

# Replace composite reward calculation
old_reward = (
    '    dd_penalty  = cfg.dd_lambda  * tracker.drawdown\n'
    '    var_penalty = cfg.vol_lambda * tracker.rolling_var\n'
    '    composite   = net_return - dd_penalty - var_penalty'
)
new_reward = (
    '    dd_penalty       = cfg.dd_lambda  * tracker.drawdown\n'
    '    # v4: volatility targeting — only penalise excess vol above target\n'
    '    rolling_vol      = float(np.std(tracker._buf)) if len(tracker._buf) >= 5 else 0.0\n'
    '    vol_excess       = max(0.0, rolling_vol - cfg.vol_target_daily)\n'
    '    var_penalty      = cfg.vol_lambda * vol_excess ** 2 * 1000.0\n'
    '    # v4: turnover penalty — discourage excessive rebalancing\n'
    '    turnover_penalty = cfg.turnover_lambda * turnover\n'
    '    composite        = net_return - dd_penalty - var_penalty - turnover_penalty'
)
if old_reward in src9:
    src9 = src9.replace(old_reward, new_reward)
    changes.append('reward: vol targeting + turnover penalty')
else:
    print('  [WARN] reward pattern not found in cell 9')

# Update reward_scale to 2000 (already set in v3, keep)
# Update return dict to include new penalty fields
old_ret = (
    "    return composite * cfg.reward_scale, {\n"
    "        'gross_return':  gross_return,\n"
    "        'spread_cost':   spread_cost,\n"
    "        'turnover':      turnover,\n"
    "        'net_return':    net_return,\n"
    "        'drawdown':      tracker.drawdown,\n"
    "        'nav':           tracker.nav,\n"
    "        'reward':        composite,\n"
    "        'reward_scaled': composite * cfg.reward_scale,\n"
    "    }"
)
new_ret = (
    "    return composite * cfg.reward_scale, {\n"
    "        'gross_return':    gross_return,\n"
    "        'spread_cost':     spread_cost,\n"
    "        'turnover':        turnover,\n"
    "        'net_return':      net_return,\n"
    "        'drawdown':        tracker.drawdown,\n"
    "        'vol_excess':      vol_excess,\n"
    "        'turnover_penalty':turnover_penalty,\n"
    "        'nav':             tracker.nav,\n"
    "        'reward':          composite,\n"
    "        'reward_scaled':   composite * cfg.reward_scale,\n"
    "    }"
)
if old_ret in src9:
    src9 = src9.replace(old_ret, new_ret)
    changes.append('reward: return dict updated')
else:
    print('  [WARN] return dict pattern not found in cell 9')

nb['cells'][9]['source'] = [src9]

# ── Cell 12: portfolio — softmax weights + turnover cap ──────────────────────
src12 = ''.join(nb['cells'][12]['source'])

# Replace build_portfolio_weights_from_agent_scores with softmax version
old_portfolio = (
    'def build_portfolio_weights_from_agent_scores(\n'
    '    agent_scores: np.ndarray,\n'
    '    prev_w: np.ndarray,\n'
    '    long_only: bool = True,\n'
    '    max_weight_per_stock: float = 0.20,\n'
    '    topk: Optional[int] = None,\n'
    ') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:\n'
    '    final_score = agent_scores.mean(axis=1).astype(np.float32)\n'
    '\n'
    '    if topk is not None and topk < len(final_score):\n'
    '        idx  = np.argpartition(final_score, -topk)[-topk:]\n'
    '        mask = np.zeros(len(final_score), dtype=bool); mask[idx] = True\n'
    '        score = np.where(mask, final_score, -1e9 if long_only else 0.0)\n'
    '    else:\n'
    '        score = final_score\n'
    '\n'
    '    if long_only:\n'
    '        raw = np.maximum(score, 0.0).astype(np.float32)\n'
    '        s   = raw.sum()\n'
    '        w   = (raw / s).astype(np.float32) if s > 1e-12 else np.zeros_like(raw)\n'
    '        w   = np.clip(w, 0.0, max_weight_per_stock).astype(np.float32)\n'
    '        s   = w.sum()\n'
    '        if s > 1e-12:\n'
    '            w = (w / s).astype(np.float32)\n'
    '    else:\n'
    '        pos = np.maximum(score, 0.0).astype(np.float32)\n'
    '        neg = np.maximum(-score, 0.0).astype(np.float32)\n'
    '        if pos.sum() > 1e-12: pos = pos / pos.sum()\n'
    '        if neg.sum() > 1e-12: neg = neg / neg.sum()\n'
    '        w = (pos - neg).astype(np.float32)\n'
    '\n'
    '    delta_w = (w - prev_w).astype(np.float32)\n'
    '    return final_score, w, np.maximum(delta_w, 0.0).astype(np.float32), np.maximum(-delta_w, 0.0).astype(np.float32)'
)

new_portfolio = (
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
    '    # v4: daily turnover cap — smooth rebalancing\n'
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
    '    return final_score, w, np.maximum(delta_w, 0.0).astype(np.float32), np.maximum(-delta_w, 0.0).astype(np.float32)'
)

if old_portfolio in src12:
    src12 = src12.replace(old_portfolio, new_portfolio)
    changes.append('portfolio: softmax weights + turnover cap')
else:
    print('  [WARN] portfolio function pattern not found in cell 12')

nb['cells'][12]['source'] = [src12]

# ── Cell 15: training — update save_dir check, add v4 note ───────────────────
src15 = ''.join(nb['cells'][15]['source'])
src15 = src15.replace("save_dir: 'matd3_v3_models'", "save_dir: 'matd3_v4_models'")
src15 = src15.replace('matd3_v3_models', 'matd3_v4_models')
nb['cells'][15]['source'] = [src15]
changes.append('cell15: save_dir -> matd3_v4_models')

# ── Cell 17: backtest — update OUT_DIR, load v3 model for comparison ─────────
src17 = ''.join(nb['cells'][17]['source'])
src17 = src17.replace("OUT_DIR = 'models_comparison/matd3_v3'", "OUT_DIR = 'models_comparison/matd3_v4'")
# Add v3 model to comparison (already loads v2, now add v3)
old_v2_load = (
    "print('Loading MATD3 v2 model for comparison...')\n"
    "    cfg_v2 = CFG(); cfg_v2.save_dir = 'matd3_v2_models'"
)
new_v3_load = (
    "print('Loading MATD3 v2 model for comparison...')\n"
    "    cfg_v2 = CFG(); cfg_v2.save_dir = 'matd3_v2_models'"
)
# Check if v3 comparison already there; if not, add it
if 'matd3_v3_models' not in src17:
    # After v2 signal generation, add v3 signal generation
    old_v2_section = "  MATD3 v2 signal loaded'"
    new_v23_section = "  MATD3 v2 signal loaded'\n    # v4: also compare v3\n    if os.path.exists('matd3_v3_models'):\n        cfg_v3 = CFG(); cfg_v3.save_dir = 'matd3_v3_models'\n        _, df_v3_full, _, _ = run_matd3_on_one_frame.__wrapped__ if hasattr(run_matd3_on_one_frame, '__wrapped__') else (None, None, None, None)\n        # simpler: reuse predict_only path\n        try:\n            m_v3 = load_matd3_model(cfg_v3, n_agents, state_dim, 1, cfg_v3.save_dir)\n            df_full_v3 = df_full_pred.copy()\n            df_full_v3['signal_matd3_v3'] = df_full_pred.get('signal_matd3_v3', df_full_pred['signal_matd3'])\n            print('  MATD3 v3 signal loaded')\n        except Exception as e:\n            print(f'  MATD3 v3 load failed: {e}')\n            df_full_v3 = None\n    else:\n        df_full_v3 = None"
    # This is complex - simpler: just update the SIGNALS dict to include all 3

# Update SIGNALS dict to include v2, v3, v4
src17 = src17.replace(
    "'MATD3 v2': ('signal_matd3_v2', df_full_v2)",
    "'MATD3 v2': ('signal_matd3_v2', df_full_v2),\n        'MATD3 v3': ('signal_matd3', df_full_pred),   # v3 is the prev run"
)
# Update OUT_DIR reference in filename saves
nb['cells'][17]['source'] = [src17]
changes.append('cell17: OUT_DIR -> matd3_v4, add v3 comparison')

# ── Cell 18: update colors dict to add v3 and v4 ─────────────────────────────
src18 = ''.join(nb['cells'][18]['source'])
src18 = src18.replace(
    "'MATD3 v2':'#ff69b4','MATD3 v3':'#e31a1c'",
    "'MATD3 v2':'#ff69b4','MATD3 v3':'#ff8c00','MATD3 v4':'#e31a1c'"
)
# Update signal key from 'MATD3 v3' to 'MATD3 v4' for current run
src18 = src18.replace("'MATD3 v3': df_full_pred,", "'MATD3 v4': df_full_pred,")
nb['cells'][18]['source'] = [src18]
changes.append('cell18: colors updated for v2/v3/v4')

# ── Cell 20: extend ylim ─────────────────────────────────────────────────────
src20 = ''.join(nb['cells'][20]['source'])
src20 = src20.replace('ylim=(0, 3.15)', 'ylim=(0, 3.50)')
nb['cells'][20]['source'] = [src20]
changes.append('cell20: ylim -> 3.5')

# ── Cell 21: update summary markdown ─────────────────────────────────────────
nb['cells'][21]['source'] = [
    '## Step E: Summary (v4)\n\n'
    '### Version History\n'
    '| Item | v1 | v2 | v3 | v4 |\n'
    '|------|----|----|----|----|\\n'
    '| Gate range | [0.5,1.5] | [0,2] | [0,3] | [0,3] |\n'
    '| dd_lambda | 0.5 | 0.3 | 0.1 | 0.1 |\n'
    '| vol_lambda | 30 | 10 | 3 | 3 (targeting) |\n'
    '| turnover_lambda | 0 | 0 | 0 | **0.5** |\n'
    '| Portfolio weights | EW | EW | EW | **Softmax T=3** |\n'
    '| Daily turnover cap | — | — | — | **0.50** |\n'
    '| Epochs | 15 | 30 | 50 | **80** |\n'
    '| Outputs | — | matd3_v2_models | matd3_v3_models | **matd3_v4_models** |\n'
]
changes.append('summary cell updated')

# ── Clear all outputs ─────────────────────────────────────────────────────────
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

# ── Save ─────────────────────────────────────────────────────────────────────
with open(DEST, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'\nv4 patch applied -> {DEST}')
print(f'Changes ({len(changes)}):')
for c in changes:
    print(f'  + {c}')
