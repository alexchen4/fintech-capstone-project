"""MATD3 v3 Backtest + Comparison Script — cells 17-20"""
import os
os.chdir("e:/New_folder/Work/MyProject/capstone")

# ============================================================
# Cell 17
# ============================================================
# ============================================================
# STEP D: Close-to-Close Backtest (Backtesting.ipynb framework)
# ============================================================
import matplotlib
matplotlib.use('Agg')
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

