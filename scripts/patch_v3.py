"""Patch rl_full_pipeline.ipynb from v2 -> v3, preserving v2 in comparison."""
import json

with open('notebooks/rl_full_pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ═══════════════════════════════════════════════════════════
# Cell 0: Update title
# ═══════════════════════════════════════════════════════════
src0 = ''.join(nb['cells'][0]['source'])
src0 = src0.replace('v2: MATD3 with Wide Gate [0,2]', 'v3: MATD3 Wide Gate [0,3] + Aggressive Reward')
nb['cells'][0]['source'] = src0

# ═══════════════════════════════════════════════════════════
# Cell 3: CFG changes
#   - epochs 30 -> 50
#   - save_dir -> matd3_v3_models
#   - dd_lambda 0.3 -> 0.1 (less drawdown penalty = more aggressive)
#   - vol_lambda 10.0 -> 3.0 (less variance penalty = chase PnL harder)
#   - reward_scale 1000 -> 2000 (stronger gradient signal)
#   - action_noise 0.08 -> 0.10 (explore wider gate space)
# ═══════════════════════════════════════════════════════════
src3 = ''.join(nb['cells'][3]['source'])
src3 = src3.replace("epochs:                      int   = 30",
                     "epochs:                      int   = 50")
src3 = src3.replace("save_dir: str = 'matd3_v2_models'",
                     "save_dir: str = 'matd3_v3_models'")
src3 = src3.replace("dd_lambda:               float = 0.3",
                     "dd_lambda:               float = 0.1")
src3 = src3.replace("vol_lambda:              float = 10.0",
                     "vol_lambda:              float = 3.0")
src3 = src3.replace("reward_scale:            float = 1000.0",
                     "reward_scale:            float = 2000.0")
src3 = src3.replace("action_noise_std:            float = 0.08",
                     "action_noise_std:            float = 0.10")
nb['cells'][3]['source'] = src3

# ═══════════════════════════════════════════════════════════
# Cell 5: Gate range [0, 2] -> [0, 3]
# ═══════════════════════════════════════════════════════════
src5 = ''.join(nb['cells'][5]['source'])
src5 = src5.replace(
    '"""Maps unconstrained delta_raw -> gate in [0, 2.0]. Allows full signal suppression."""\n    return 1.0 + 1.0 * torch.tanh(delta_raw)',
    '"""Maps unconstrained delta_raw -> gate in [0, 3.0]. Wide range for aggressive differentiation."""\n    return 1.5 + 1.5 * torch.tanh(delta_raw)'
)
nb['cells'][5]['source'] = src5

# ═══════════════════════════════════════════════════════════
# Cell 7: Update gate clamp range to [0, 3.0]
# ═══════════════════════════════════════════════════════════
src7 = ''.join(nb['cells'][7]['source'])
src7 = src7.replace('np.clip(delta_to_gate(delta_raw).squeeze(0).cpu().numpy(), 0.0, 2.0)',
                     'np.clip(delta_to_gate(delta_raw).squeeze(0).cpu().numpy(), 0.0, 3.0)')
src7 = src7.replace("torch.clamp(delta_to_gate(d_next + eps), 0.0, 2.0)",
                     "torch.clamp(delta_to_gate(d_next + eps), 0.0, 3.0)")
nb['cells'][7]['source'] = src7

# ═══════════════════════════════════════════════════════════
# Cell 17: Update OUT_DIR + add v2 to comparison
# ═══════════════════════════════════════════════════════════
src17 = ''.join(nb['cells'][17]['source'])

# Change output dir
src17 = src17.replace("OUT_DIR = 'models_comparison/matd3'",
                       "OUT_DIR = 'models_comparison/matd3_v3'")

# Add v2 model signal to comparison: insert code after MATD3 v2 signal generation
# Find the line where we append MATD3 v2 to SIGNALS and add v2 loading after it
old_signals_line = "SIGNALS += [('EW Ensemble','signal_ew_ensemble'), ('MATD3 v2','signal_matd3')]"
new_signals_block = """# Load MATD3 v2 model signal for comparison
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

SIGNALS += [('EW Ensemble','signal_ew_ensemble'), ('MATD3 v3','signal_matd3')]"""

src17 = src17.replace(old_signals_line, new_signals_block)

# Rename 'MATD3 v2' -> 'MATD3 v3' for the current model's signal
src17 = src17.replace("('MATD3 v2','signal_matd3')", "('MATD3 v3','signal_matd3')")

nb['cells'][17]['source'] = src17

# ═══════════════════════════════════════════════════════════
# Cell 18: Update colors to include v2 and v3
# ═══════════════════════════════════════════════════════════
src18 = ''.join(nb['cells'][18]['source'])
src18 = src18.replace(
    "sig_colors = {'LightGBM':'#1f77b4','CNN+LSTM+TF':'#ff7f0e','SSM+TF':'#2ca02c',\n              'Cross-Attn':'#d62728','GAT+TF':'#9467bd','EW Ensemble':'#888888','MATD3 v2':'#e31a1c'}",
    "sig_colors = {'LightGBM':'#1f77b4','CNN+LSTM+TF':'#ff7f0e','SSM+TF':'#2ca02c',\n              'Cross-Attn':'#d62728','GAT+TF':'#9467bd','EW Ensemble':'#888888',\n              'MATD3 v2':'#ff69b4','MATD3 v3':'#e31a1c'}"
)
src18 = src18.replace("'MATD3' in name", "'MATD3' in name")  # keep as-is, both v2/v3 match
nb['cells'][18]['source'] = src18

# ═══════════════════════════════════════════════════════════
# Cell 20: Update gate y-axis for [0, 3]
# ═══════════════════════════════════════════════════════════
src20 = ''.join(nb['cells'][20]['source'])
src20 = src20.replace("ax.set_ylim(-0.05, 2.05)", "ax.set_ylim(-0.05, 3.15)")
src20 = src20.replace("'MATD3 v2 Gate Weights [0, 2]'", "'MATD3 v3 Gate Weights [0, 3]'")
nb['cells'][20]['source'] = src20

# ═══════════════════════════════════════════════════════════
# Cell 21: Update summary
# ═══════════════════════════════════════════════════════════
c21 = """## Step E: Summary (v3)

### Version History
| Item | v1 | v2 | v3 |
|------|----|----|----|
| Gate range | [0.5, 1.5] | [0, 2.0] | **[0, 3.0]** |
| dd_lambda | 0.5 | 0.3 | **0.1** |
| vol_lambda | 30.0 | 10.0 | **3.0** |
| reward_scale | 1000 | 1000 | **2000** |
| Epochs | 15 | 30 | **50** |
| Noise | 0.05 | 0.08 | **0.10** |

### v3 Design Philosophy
- **More aggressive PnL pursuit**: dd_lambda=0.1, vol_lambda=3.0 — minimal risk penalty
- **Wider gate [0, 3]**: CNN+LSTM+TF was capped at 2.0 in v2, now can go to 3.0
- **Stronger learning signal**: reward_scale=2000, 50 epochs

### Outputs
- `matd3_v3_models/` — v3 checkpoints
- `matd3_v2_models/` — v2 checkpoints (preserved)
- `models_comparison/matd3_v3/` — comparison charts with both v2 and v3
"""
nb['cells'][21]['source'] = c21
nb['cells'][21]['cell_type'] = 'markdown'

# ═══════════════════════════════════════════════════════════
with open('notebooks/rl_full_pipeline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('v3 patch applied:')
print('  Gate: [0, 2] -> [0, 3]')
print('  epochs: 30 -> 50')
print('  dd_lambda: 0.3 -> 0.1')
print('  vol_lambda: 10.0 -> 3.0')
print('  reward_scale: 1000 -> 2000')
print('  noise: 0.08 -> 0.10')
print('  save_dir: matd3_v3_models')
print('  OUT_DIR: models_comparison/matd3_v3')
print('  v2 model loaded for comparison in backtest')
print('  v2 notebook preserved as rl_full_pipeline_v2.ipynb')
