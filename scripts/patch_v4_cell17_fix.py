"""
Fix v4 notebook cell 17: replace load_model_for_inference with monkey-patching approach
so v2/v3 use EW portfolio (matching their training) and v4 uses softmax (matching its training).
"""
import json

with open('notebooks/rl_full_pipeline_v4.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Replace the v3/v2 rollout section with monkey-patching approach
cell17_src = ''.join(nb['cells'][17]['source'])

# Find the section from "Helper" comment to end of signal list
# Replace everything from "load_model_for_inference" def through the signal list build

old_helper_to_signals = """\
# ── Helper: load actor checkpoints into a MATD3 model for inference ────────

def load_model_for_inference(model_dir):
    \"\"\"
    Load saved actors from model_dir into a fresh MATD3SharedCritic model.
    Uses the exact same architecture as training -- no state reconstruction differences.
    Returns the model, or None if checkpoints are not found.
    \"\"\"
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

# ── Load base data ───────────────────────────────────────────────────────────
print('Loading data...')
df_raw_full = pd.read_parquet('df_testsub_full.parquet')
df_full_rl  = pd.read_parquet('df_full_rl_clean.parquet')  # full bars for rollout

raw_signal_cols = ['signal_pre_boost','signal_hybrid','signal_ssm_tf','signal_tf_cross','signal_gat_tf']
df_raw_full['signal_ew_ensemble'] = df_raw_full[raw_signal_cols].mean(axis=1)

# v4: extract from df_full_pred (already in memory from Cell 16)
print('Extracting MATD3 v4 signal...')
v4_close = (df_full_pred[df_full_pred[cfg.time_col]==1500]
            [[cfg.stock_col, cfg.day_col, 'final_score']]
            .copy().rename(columns={'final_score':'signal_matd3_v4'}))
df_raw_full = df_raw_full.merge(v4_close, on=[cfg.stock_col,cfg.day_col], how='left')
df_raw_full['signal_matd3_v4'] = df_raw_full['signal_matd3_v4'].fillna(0.0)
print(f'  v4: {(df_raw_full["signal_matd3_v4"]!=0).sum()} non-zero rows')

# v3: full inference rollout using saved v3 actors via rollout_with_model()
# This uses the same prepare_dataframe + build_snapshot_cache code path as training,
# so the resulting signals will match the original v3 training run exactly.
print('Running MATD3 v3 inference rollout (same code path as training)...')
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

# v2: full inference rollout using saved v2 actors
print('Running MATD3 v2 inference rollout...')
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
    print(f'  v2: {(df_raw_full["signal_matd3_v2"]!=0).sum()} non-zero rows')"""

new_helper_to_signals = """\
# ── Helpers: portfolio overrides + model loader ──────────────────────────────
# build_portfolio_weights_from_agent_scores is a global name resolved at call time.
# Reassigning it here affects all subsequent rollout_with_model calls.

def _ew_portfolio(agent_scores, prev_w, long_only=True, max_weight_per_stock=0.20, topk=None, **kwargs):
    \"\"\"EW top-k portfolio -- matches v2/v3 training (no softmax, no turnover cap).\"\"\"
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

def load_model_for_inference(model_dir):
    \"\"\"Load saved actors from model_dir into a fresh MATD3SharedCritic model.\"\"\"
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

# ── Load base data ───────────────────────────────────────────────────────────
print('Loading data...')
df_raw_full = pd.read_parquet('df_testsub_full.parquet')
df_full_rl  = pd.read_parquet('df_full_rl_clean.parquet')  # full bars for rollout

raw_signal_cols = ['signal_pre_boost','signal_hybrid','signal_ssm_tf','signal_tf_cross','signal_gat_tf']
df_raw_full['signal_ew_ensemble'] = df_raw_full[raw_signal_cols].mean(axis=1)

# v4: extract from df_full_pred (already in memory from Cell 16 -- uses softmax internally)
print('Extracting MATD3 v4 signal...')
v4_close = (df_full_pred[df_full_pred[cfg.time_col]==1500]
            [[cfg.stock_col, cfg.day_col, 'final_score']]
            .copy().rename(columns={'final_score':'signal_matd3_v4'}))
df_raw_full = df_raw_full.merge(v4_close, on=[cfg.stock_col,cfg.day_col], how='left')
df_raw_full['signal_matd3_v4'] = df_raw_full['signal_matd3_v4'].fillna(0.0)
print(f'  v4: {(df_raw_full["signal_matd3_v4"]!=0).sum()} non-zero rows')

# v3: rollout with EW portfolio (matches v3 training -- no softmax, no turnover cap)
print('Running MATD3 v3 inference rollout (EW portfolio, matching v3 training)...')
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

# Restore original portfolio builder (softmax) for any subsequent use
build_portfolio_weights_from_agent_scores = _softmax_portfolio_orig = build_portfolio_weights_from_agent_scores"""

if old_helper_to_signals in cell17_src:
    cell17_src = cell17_src.replace(old_helper_to_signals, new_helper_to_signals, 1)
    nb['cells'][17]['source'] = [cell17_src]
    print('Cell 17 patched: v2/v3 now use EW portfolio rollout')
else:
    print('ERROR: Could not find helper section in cell 17')
    # Show what's there
    idx = cell17_src.find('load_model_for_inference')
    print(f'Found load_model_for_inference at {idx}')
    if idx >= 0:
        print(repr(cell17_src[idx:idx+200]))

with open('notebooks/rl_full_pipeline_v4.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Saved: notebooks/rl_full_pipeline_v4.ipynb')
