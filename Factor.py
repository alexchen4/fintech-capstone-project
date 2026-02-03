# =========================================================
# 001 价格动量（EMA 差）【技术】
# param: i=short, j=long
# =========================================================
def factor_001(df, i=4, j=12):
    ema_s = ewm_mean_by_st(df, df['ClosePrice'], i)
    ema_l = ewm_mean_by_st(df, df['ClosePrice'], j)
    return (ema_s - ema_l) / df['mid']


# =========================================================
# 002 RSI-like（成交价）
# param: i=window
# =========================================================
def factor_002(df, i=14):
    d = diff_by_st(df, df['ClosePrice'])
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)

    up_m = roll_mean_by_st(df, up, i)
    dn_m = roll_mean_by_st(df, dn, i).replace(0, np.nan)

    rs = up_m / dn_m
    return 1 - 1/(1+rs)


# =========================================================
# 003 VWAP 偏离动量
# param: i=window
# =========================================================
def factor_003(df, i=8):
    dev = (df['ClosePrice'] - df['VWAP']) / df['mid']
    return roll_mean_by_st(df, dev, i)


# =========================================================
# 004 区间突破强度（Donchian）
# param: i=window
# =========================================================
def factor_004(df, i=20):
    # rolling max/min：用 MultiIndex rolling
    high = _mi_series(df, df['HighPrice'])
    low  = _mi_series(df, df['LowPrice'])

    high_max = (
        high.groupby(level=[0,1], sort=False)
            .rolling(i, min_periods=5).max()
            .reset_index(level=[0,1], drop=True)
    )
    low_min = (
        low.groupby(level=[0,1], sort=False)
           .rolling(i, min_periods=5).min()
           .reset_index(level=[0,1], drop=True)
    )

    high_max = _mi_to_flat(high_max, df)
    low_min  = _mi_to_flat(low_min, df)

    denom = (high_max - low_min).replace(0, np.nan)
    return (df['ClosePrice'] - low_min) / denom


# =========================================================
# 005 成交量振荡器（短-长）
# param: i=short, j=long
# =========================================================
def factor_005(df, i=4, j=12):
    v_s = roll_mean_by_st(df, df['Volume_sum'], i)
    v_l = roll_mean_by_st(df, df['Volume_sum'], j).replace(0, np.nan)
    return (v_s - v_l) / v_l


# =========================================================
# 006 Amihud 非线性版
# param: i=window, k=power
# =========================================================
def factor_006(df, i=10, k=1.0):
    ret = pct_change_by_st(df, df['ClosePrice'])
    illiq = ret.abs() / df['Turnover_sum'].replace(0, np.nan)
    out = roll_mean_by_st(df, illiq, i)
    return out ** k


# =========================================================
# 007 深度不平衡（L1~Lj）
# param: j=depth
# =========================================================
def factor_007(df, j=4):
    bid_cols = [f'BidVolume{i}_last' for i in range(1, j+1)]
    ask_cols = [f'AskVolume{i}_last' for i in range(1, j+1)]
    bid = df[bid_cols].sum(axis=1)
    ask = df[ask_cols].sum(axis=1)
    return (bid - ask) / (bid + ask).replace(0, np.nan)


# =========================================================
# 008 深度加权压力（价格 × 深度）
# param: j=depth
# =========================================================
def factor_008(df, j=4):
    bid_p = sum(df[f'BidPrice{i}_last'] * df[f'BidVolume{i}_last'] for i in range(1, j+1))
    ask_p = sum(df[f'AskPrice{i}_last'] * df[f'AskVolume{i}_last'] for i in range(1, j+1))
    vol   = sum(df[f'BidVolume{i}_last'] + df[f'AskVolume{i}_last'] for i in range(1, j+1))
    mid_w = (bid_p + ask_p) / vol.replace(0, np.nan)
    return (mid_w - df['mid']) / df['mid']


# =========================================================
# 009 盘口集中度差
# param: j=depth
# =========================================================
def factor_009(df, j=4):
    bid_cols = [f'BidVolume{i}_last' for i in range(1, j+1)]
    ask_cols = [f'AskVolume{i}_last' for i in range(1, j+1)]
    bid = df[bid_cols].sum(axis=1).replace(0, np.nan)
    ask = df[ask_cols].sum(axis=1).replace(0, np.nan)
    return df['BidVolume1_last']/bid - df['AskVolume1_last']/ask


# =========================================================
# 010 订单簿“挤压”指标（order count）
# param: j=depth
# =========================================================
def factor_010(df, j=4):
    bid_cols = [f'BidOrder{i}_last' for i in range(1, j+1)]
    ask_cols = [f'AskOrder{i}_last' for i in range(1, j+1)]
    bid_o = df[bid_cols].sum(axis=1)
    ask_o = df[ask_cols].sum(axis=1)
    return (bid_o - ask_o) / (bid_o + ask_o).replace(0, np.nan)


# =========================================================
# 011 VWAP 反转强度
# param: i=window
# =========================================================
def factor_011(df, i=10):
    dev = (df['ClosePrice'] - df['VWAP']) / df['mid']
    return -roll_mean_by_st(df, dev, i)


# =========================================================
# 012 成交驱动方向性（signed volume）
# param: i=window
# =========================================================
def factor_012(df, i=8):
    sign = np.sign(df['ClosePrice'] - df['OpenPrice'])
    sv = sign * df['Volume_sum']
    return roll_mean_by_st(df, sv, i)


# =========================================================
# 013 价格冲击 proxy（Δmid / volume）
# param: i=window
# =========================================================
def factor_013(df, i=6):
    dmid = diff_by_st(df, df['mid']).abs()
    impact = dmid / df['Volume_sum'].replace(0, np.nan)
    return roll_mean_by_st(df, impact, i)


# =========================================================
# 014 区间波动 × 成交量
# param: i=window
# =========================================================
def factor_014(df, i=10):
    x = (df['HighPrice'] - df['LowPrice']) * df['Volume_sum']
    return roll_mean_by_st(df, x, i)


# =========================================================
# 015 Spread 压力
# param: i=window
# =========================================================
def factor_015(df, i=6):
    x = df['spread'] * df['Volume_sum']
    return roll_mean_by_st(df, x, i)


# =========================================================
# 016 成交加权趋势（VWAP-based）
# param: i=window
# =========================================================
def factor_016(df, i=10):
    ret = pct_change_by_st(df, df['VWAP'])
    wret = ret * df['Volume_sum']
    return roll_mean_by_st(df, wret, i)


# =========================================================
# 017 价量背离
# param: i=window
# =========================================================
def factor_017(df, i=10):
    r = pct_change_by_st(df, df['ClosePrice'])
    v = pct_change_by_st(df, df['Volume_sum'])
    return roll_mean_by_st(df, (r - v), i)


# =========================================================
# 018 VWAP 相对强弱（RS-like）
# param: i=window
# =========================================================
def factor_018(df, i=14):
    d = diff_by_st(df, df['VWAP'])
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    up_m = roll_mean_by_st(df, up, i)
    dn_m = roll_mean_by_st(df, dn, i).replace(0, np.nan)
    return up_m / dn_m


# =========================================================
# 019 深度 × spread 非线性
# param: j=depth, k=power
# =========================================================
def factor_019(df, j=4, k=1.0):
    depth_cols_b = [f'BidVolume{i}_last' for i in range(1, j+1)]
    depth_cols_a = [f'AskVolume{i}_last' for i in range(1, j+1)]
    depth = (df[depth_cols_b].sum(axis=1) + df[depth_cols_a].sum(axis=1))
    return (df['spread'] * depth) ** k


# =========================================================
# 020 成交强度异常
# param: i=window
# =========================================================
def factor_020(df, i=12):
    v = df['Volume_sum']
    mu = roll_mean_by_st(df, v, i)
    sd = roll_std_by_st(df, v, i).replace(0, np.nan)
    return (v - mu) / sd

# =========================
# 021-030: imbalance / pressure
# =========================
def factor_021(df, i=20):
    # depth imbalance (L1-4)
    bidD = _depth(df,"bid",4); askD = _depth(df,"ask",4)
    imb = _safe_div((bidD-askD), (bidD+askD))
    return _roll(df, imb, i, 5, "mean")

def factor_022(df, i=20):
    # L1 imbalance mean
    bv1 = _get(df,"BidVolume1_last"); av1=_get(df,"AskVolume1_last")
    imb1 = _safe_div((bv1-av1),(bv1+av1))
    return _roll(df, imb1, i, 5, "mean")

def factor_023(df, i=20):
    # imbalance volatility (std)
    bidD=_depth(df,"bid",4); askD=_depth(df,"ask",4)
    imb = _safe_div((bidD-askD),(bidD+askD))
    return _roll(df, imb, i, 5, "std")

def factor_024(df, i=20):
    # OFI proxy rolling sum (pressure)
    ofi = _ofi1_proxy(df)
    return _roll(df, ofi, i, 5, "sum")

def factor_025(df, i=20):
    # OFI normalized by depth
    ofi = _ofi1_proxy(df)
    dep = (_depth(df,"bid",1)+_depth(df,"ask",1)).replace(0,np.nan)
    x = _safe_div(ofi, dep)
    return _roll(df, x, i, 5, "mean")

def factor_026(df, i=20):
    # microprice - mid (signed pressure)
    mp = _microprice(df); mid=_mid(df)
    x = mp - mid
    return _roll(df, x, i, 5, "mean")

def factor_027(df, i=20):
    # microprice pressure normalized by spread
    mp=_microprice(df); mid=_mid(df); sp=_spread(df)
    x = _safe_div((mp-mid), sp.abs())
    return _roll(df, x, i, 5, "mean")

def factor_028(df, i=20):
    # weighted bid-ask price gap (WeightAsk-WeightBid) / mid
    wab = _wprice(df,"ask") - _wprice(df,"bid")
    mid=_mid(df)
    x = _safe_div(wab, mid.abs())
    return _roll(df, x, i, 5, "mean")

def factor_029(df, i=20):
    # bid depth share
    bidD=_depth(df,"bid",4); askD=_depth(df,"ask",4)
    x = _safe_div(bidD, (bidD+askD))
    return _roll(df, x, i, 5, "mean")

def factor_030(df, i=20):
    # ask depth share
    bidD=_depth(df,"bid",4); askD=_depth(df,"ask",4)
    x = _safe_div(askD, (bidD+askD))
    return _roll(df, x, i, 5, "mean")


# =========================
# 031-040: spread / book slope / convexity
# =========================
def factor_031(df, i=20):
    # effective relative spread: spread/mid
    sp=_spread(df); mid=_mid(df)
    x=_safe_div(sp, mid.abs())
    return _roll(df, x, i, 5, "mean")

def factor_032(df, i=20):
    # spread zscore
    sp=_spread(df)
    return _z(df, sp, i, 5)

def factor_033(df, i=20):
    # bid-side slope (P1-P4)/sumVol (approx)
    p1=_get(df,"BidPrice1_last"); p4=_get(df,"BidPrice4_last")
    v=_depth(df,"bid",4)
    x=_safe_div((p1-p4), v)
    return _roll(df, x, i, 5, "mean")

def factor_034(df, i=20):
    # ask-side slope (P4-P1)/sumVol
    p1=_get(df,"AskPrice1_last"); p4=_get(df,"AskPrice4_last")
    v=_depth(df,"ask",4)
    x=_safe_div((p4-p1), v)
    return _roll(df, x, i, 5, "mean")

def factor_035(df, i=20):
    # slope imbalance
    b = factor_033(df, i); a = factor_034(df, i)
    return _clip_inf(b - a)

def factor_036(df, i=20):
    # top-of-book tightness: (A1-B1)/A1
    b1=_get(df,"BidPrice1_last"); a1=_get(df,"AskPrice1_last")
    x=_safe_div((a1-b1), a1.abs())
    return _roll(df, x, i, 5, "mean")

def factor_037(df, i=20):
    # price ladder convexity bid: (P1-2P2+P3)
    p1=_get(df,"BidPrice1_last"); p2=_get(df,"BidPrice2_last"); p3=_get(df,"BidPrice3_last")
    x = p1 - 2*p2 + p3
    return _roll(df, x, i, 5, "mean")

def factor_038(df, i=20):
    # price ladder convexity ask: (P3-2P2+P1)
    p1=_get(df,"AskPrice1_last"); p2=_get(df,"AskPrice2_last"); p3=_get(df,"AskPrice3_last")
    x = p3 - 2*p2 + p1
    return _roll(df, x, i, 5, "mean")

def factor_039(df, i=20):
    # depth convexity bid: V1-2V2+V3
    v1=_get(df,"BidVolume1_last"); v2=_get(df,"BidVolume2_last"); v3=_get(df,"BidVolume3_last")
    x = v1 - 2*v2 + v3
    return _roll(df, x, i, 5, "mean")

def factor_040(df, i=20):
    # depth convexity ask: V3-2V2+V1
    v1=_get(df,"AskVolume1_last"); v2=_get(df,"AskVolume2_last"); v3=_get(df,"AskVolume3_last")
    x = v3 - 2*v2 + v1
    return _roll(df, x, i, 5, "mean")


# =========================
# 041-050: VWAP/mid deviation, return-efficiency, noise
# =========================
def factor_041(df, i=20):
    # VWAP - mid normalized by spread
    vwap=_get(df,"VWAP"); mid=_mid(df); sp=_spread(df)
    x=_safe_div((vwap-mid), sp.abs())
    return _roll(df, x, i, 5, "mean")

def factor_042(df, i=20):
    # close - mid normalized by spread
    close=_get(df,"ClosePrice"); mid=_mid(df); sp=_spread(df)
    x=_safe_div((close-mid), sp.abs())
    return _roll(df, x, i, 5, "mean")

def factor_043(df, i=20):
    # (close - vwap)/mid
    close=_get(df,"ClosePrice"); vwap=_get(df,"VWAP"); mid=_mid(df)
    x=_safe_div((close-vwap), mid.abs())
    return _roll(df, x, i, 5, "mean")

def factor_044(df, i=20):
    # mid return volatility
    r=_logret_mid(df)
    return _roll(df, r, i, 5, "std")

def factor_045(df, i=20):
    # absolute mid return mean (realized movement)
    r=_logret_mid(df).abs()
    return _roll(df, r, i, 5, "mean")

def factor_046(df, i=20):
    # "efficiency" proxy: |net change| / sum |changes|
    mid=_mid(df)
    gm=_g(df, mid)
    dm = gm.diff()
    num = _roll(df, dm, i, 5, "sum").abs()
    den = _roll(df, dm.abs(), i, 5, "sum")
    return _safe_div(num, den)

def factor_047(df, i=20):
    # variance ratio proxy: std(ret) / mean(|ret|)
    r=_logret_mid(df)
    std=_roll(df, r, i, 5, "std")
    mad=_roll(df, r.abs(), i, 5, "mean")
    return _safe_div(std, mad)

def factor_048(df, i=20):
    # noise-to-signal: spread / |mid change|
    sp=_spread(df).abs()
    dm=_g(df, _mid(df)).diff().abs()
    x=_safe_div(sp, dm)
    return _roll(df, x, i, 5, "mean")

def factor_049(df, i=20):
    # range normalized by vwap: (high-low)/vwap
    hi=_get(df,"HighPrice"); lo=_get(df,"LowPrice"); vwap=_get(df,"VWAP")
    x=_safe_div((hi-lo), vwap.abs())
    return _roll(df, x, i, 5, "mean")

def factor_050(df, i=20):
    # close location value (CLV) intraday rolling: (2C-H-L)/(H-L)
    c=_get(df,"ClosePrice"); h=_get(df,"HighPrice"); l=_get(df,"LowPrice")
    denom=(h-l).replace(0,np.nan)
    clv=(2*c - h - l)/denom
    return _roll(df, clv, i, 5, "mean")


# =========================
# 051-060: volume/turnover intensity, illiquidity, Kyle/impact proxies
# =========================
def factor_051(df, i=20):
    # volume intensity: Volume_sum / rolling mean
    v=_get(df,"Volume_sum")
    return _safe_div(v, _roll(df, v, i, 5, "mean"))

def factor_052(df, i=20):
    # turnover intensity: Turnover_sum / rolling mean
    to=_get(df,"Turnover_sum")
    return _safe_div(to, _roll(df, to, i, 5, "mean"))

def factor_053(df, i=20):
    # trades intensity: DealNum_sum / rolling mean
    dn=_get(df,"DealNum_sum")
    return _safe_div(dn, _roll(df, dn, i, 5, "mean"))

def factor_054(df, i=20):
    # avg trade size: Volume_sum / DealNum_sum
    v=_get(df,"Volume_sum"); dn=_get(df,"DealNum_sum")
    x=_safe_div(v, dn)
    return _roll(df, x, i, 5, "mean")

def factor_055(df, i=20):
    # avg trade notional: Turnover_sum / DealNum_sum
    to=_get(df,"Turnover_sum"); dn=_get(df,"DealNum_sum")
    x=_safe_div(to, dn)
    return _roll(df, x, i, 5, "mean")

def factor_056(df, i=20):
    # Amihud illiquidity rolling mean
    return _roll(df, _amihud(df), i, 5, "mean")

def factor_057(df, i=20):
    # Amihud zscore
    return _z(df, _amihud(df), i, 5)

def factor_058(df, i=20):
    # impact proxy: |ret| / volume
    r=_logret_mid(df).abs()
    v=_get(df,"Volume_sum").replace(0,np.nan)
    x=_safe_div(r, v)
    return _roll(df, x, i, 5, "mean")

def factor_059(df, i=20):
    # Kyle lambda proxy: cov(ret, signed_volume)/var(signed_volume)
    # signed_volume proxy via dTotalVolume * sign(ret)
    dv=_get(df,"dTotalVolume")
    r=_logret_mid(df)
    sv = dv * np.sign(r)
    m_r=_roll(df, r, i, 5, "mean")
    m_sv=_roll(df, sv, i, 5, "mean")
    cov = _roll(df, (r-m_r)*(sv-m_sv), i, 5, "mean")
    var = _roll(df, (sv-m_sv)**2, i, 5, "mean").replace(0,np.nan)
    return _safe_div(cov, var)

def factor_060(df, i=20):
    # orderflow pressure vs price change: corr(OFI, ret)
    ofi=_ofi1_proxy(df)
    r=_logret_mid(df)
    m1=_roll(df, ofi, i, 5, "mean"); m2=_roll(df, r, i, 5, "mean")
    cov=_roll(df, (ofi-m1)*(r-m2), i, 5, "mean")
    sd1=_roll(df, (ofi-m1)**2, i, 5, "mean")**0.5
    sd2=_roll(df, (r-m2)**2, i, 5, "mean")**0.5
    return _safe_div(cov, (sd1*sd2).replace(0,np.nan))


# =========================
# 061-070: composite signals (pressure × liquidity, mean-reversion/trend hybrids)
# =========================
def factor_061(df, i=20):
    # pressure * liquidity: (imbalance mean) * (1/spread)
    imb = factor_022(df, i)
    sp  = _roll(df, _spread(df).abs(), i, 5, "mean")
    return _safe_div(imb, sp)

def factor_062(df, i=20):
    # microprice pressure * volume intensity
    mp = _microprice(df); mid=_mid(df)
    press = _roll(df, (mp-mid), i, 5, "mean")
    vol_int = factor_051(df, i)
    return _clip_inf(press * vol_int)

def factor_063(df, i=20):
    # VWAP deviation * imbalance
    dev = factor_041(df, i)
    imb = factor_021(df, i)
    return _clip_inf(dev * imb)

def factor_064(df, i=20):
    # mean-reversion: -(mid - rolling mean(mid))/rolling std(mid)
    mid=_mid(df)
    z = _z(df, mid, i, 5)
    return -z

def factor_065(df, i=20):
    # trend: rolling sum of mid logret
    r=_logret_mid(df)
    return _roll(df, r, i, 5, "sum")

def factor_066(df, i=20):
    # trend adjusted by spread (cost-aware)
    tr = factor_065(df, i)
    sp = _roll(df, _spread(df).abs(), i, 5, "mean")
    return _safe_div(tr, sp)

def factor_067(df, i=20):
    # volatility-scaled OFI pressure
    ofi_sum = factor_024(df, i)
    vol = factor_044(df, i)
    return _safe_div(ofi_sum, vol)

def factor_068(df, i=20):
    # "stress" composite: spread zscore + amihud zscore + volume intensity zscore
    s1 = factor_032(df, i)
    s2 = factor_057(df, i)
    v  = _get(df,"Volume_sum")
    s3 = _z(df, v, i, 5)
    return _clip_inf(s1 + s2 - s3)

def factor_069(df, i=20):
    # orderbook skew: (V1+V2)/(V3+V4) bid - ask
    b12 = _get(df,"BidVolume1_last")+_get(df,"BidVolume2_last")
    b34 = _get(df,"BidVolume3_last")+_get(df,"BidVolume4_last")
    a12 = _get(df,"AskVolume1_last")+_get(df,"AskVolume2_last")
    a34 = _get(df,"AskVolume3_last")+_get(df,"AskVolume4_last")
    skew_b = _safe_div(b12, b34)
    skew_a = _safe_div(a12, a34)
    x = skew_b - skew_a
    return _roll(df, x, i, 5, "mean")

def factor_070(df, i=20):
    # execution pressure: (Close - microprice)/spread * turnover intensity
    close=_get(df,"ClosePrice"); mp=_microprice(df); sp=_spread(df)
    dev = _safe_div((close-mp), sp.abs())
    to_int = factor_052(df, i)
    return _clip_inf(dev * to_int)
