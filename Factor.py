import numpy as np
import pandas as pd

EPS = 1e-12          # 数值稳定
EPS_COUNT = 1.0      # 订单数/笔数这种“计数型”分母的平滑更合理用 1

# =========================
# 0) sort
# =========================
def prepare_sort(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["SecuCode", "TradingDay", "TimeEnd", "TimeStart"] if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True)

# =========================
# 1) basics (cross-day, group by SecuCode ONLY)
# =========================
def _get(df, c):
    return df[c].astype(float) if c in df.columns else pd.Series(np.nan, index=df.index, dtype="float64")

def _g(df, s: pd.Series):
    if "SecuCode" not in df.columns:
        raise ValueError("Need SecuCode in df.")
    return s.groupby(df["SecuCode"], sort=False)

def _roll(df, s, i: int, minp: int = 5, func: str = "mean"):
    r = getattr(_g(df, s).rolling(i, min_periods=minp), func)()
    return r.reset_index(level=0, drop=True)

def _clip_inf(s: pd.Series) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan)

def _safe_div(a: pd.Series, b: pd.Series, eps: float = EPS) -> pd.Series:
    # 科学：避免分母0导致 NaN；但仍保持数量级稳定
    return _clip_inf(a / (b + eps))

def _safe_div_zero(a: pd.Series, b: pd.Series) -> pd.Series:
    # 可解释：分母为0则输出0（用于imbalance这种“无信息=中性”）
    out = a / b.replace(0, np.nan)
    return _clip_inf(out).fillna(0.0)

def roll_mean_by_st(df: pd.DataFrame, s: pd.Series, w: int) -> pd.Series:
    out = _g(df, s).rolling(w, min_periods=max(2, w // 2)).mean()
    return out.reset_index(level=0, drop=True)

def roll_std_by_st(df: pd.DataFrame, s: pd.Series, w: int) -> pd.Series:
    out = _g(df, s).rolling(w, min_periods=max(2, w // 2)).std()
    return out.reset_index(level=0, drop=True)

def ewm_mean_by_st(df: pd.DataFrame, s: pd.Series, span: int) -> pd.Series:
    out = _g(df, s).ewm(span=span, adjust=False).mean()
    return out.reset_index(level=0, drop=True)

def diff_by_st(df: pd.DataFrame, s: pd.Series) -> pd.Series:
    return _g(df, s).diff()

def pct_change_by_st(df: pd.DataFrame, s: pd.Series) -> pd.Series:
    return _g(df, s).pct_change()

def _z(df, s, i: int, minp: int = 5):
    """
    科学 zscore：
    - warmup(minp未满) -> NaN（保留）
    - std=0 -> 0（表示常数窗口内，没有偏离）
    """
    m = _roll(df, s, i, minp, "mean")
    sd = _roll(df, s, i, minp, "std")
    z = (s - m) / (sd + EPS)
    z = z.mask(sd.isna(), np.nan)
    z = z.mask(sd == 0, 0.0)
    return z

# =========================
# 2) microstructure primitives
# =========================
def book_state(df):
    bid1, ask1 = _get(df, "BidPrice1_last"), _get(df, "AskPrice1_last")
    # 0=双边，1=仅bid，2=仅ask，3=都无
    return pd.Series(
        np.select(
            [(bid1>0)&(ask1>0), (bid1>0)&(ask1<=0), (bid1<=0)&(ask1>0)],
            [0, 1, 2],
            default=3,
        ),
        index=df.index,
        dtype="float64",
    )

def _mid(df):
    if "mid" in df.columns:
        return _get(df, "mid")
    bid1, ask1 = _get(df, "BidPrice1_last"), _get(df, "AskPrice1_last")
    m = np.where((bid1 > 0) & (ask1 > 0), 0.5 * (bid1 + ask1),
                 np.where(bid1 > 0, bid1, np.where(ask1 > 0, ask1, np.nan)))
    return pd.Series(m, index=df.index, dtype="float64")

def _spread_raw(df):
    bid1, ask1 = _get(df, "BidPrice1_last"), _get(df, "AskPrice1_last")
    return (ask1 - bid1).where((bid1 > 0) & (ask1 > 0))

def _spread(df):
    """
    科学处理单边盘口：
    - 双边：ask-bid
    - 单边：用每只股票最近一次有效spread做ffill（保持连续）
    """
    if "spread" in df.columns:
        sp = _get(df, "spread")
    else:
        sp = _spread_raw(df)
    return sp.groupby(df["SecuCode"], sort=False).ffill()

def _microprice(df):
    bid1, ask1 = _get(df, "BidPrice1_last"), _get(df, "AskPrice1_last")
    bv1, av1   = _get(df, "BidVolume1_last"), _get(df, "AskVolume1_last")
    denom = (bv1 + av1)
    mp = (ask1 * bv1 + bid1 * av1) / denom.replace(0, np.nan)
    # 双边且denom>0才算，否则退化为mid（避免NaN）
    return mp.where((bid1 > 0) & (ask1 > 0) & (denom > 0), _mid(df))

def _depth(df, side="bid", L=4):
    vols = []
    for lv in range(1, L + 1):
        c = f"{'Bid' if side == 'bid' else 'Ask'}Volume{lv}_last"
        vols.append(_get(df, c))
    return sum(vols)

def _wprice(df, side="bid"):
    c = "WeightBidPrice_last" if side == "bid" else "WeightAskPrice_last"
    return _get(df, c)

def _ofi1_proxy(df):
    bv1 = _get(df, "BidVolume1_last")
    av1 = _get(df, "AskVolume1_last")
    dbv = _g(df, bv1).diff()
    dav = _g(df, av1).diff()
    return (dbv - dav).astype(float)

def _logret_mid(df):
    m = _mid(df)
    gm = _g(df, m)
    # 若mid<=0会log出问题，这里防护一下
    m_pos = m.where(m > 0)
    prev = gm.shift(1).where(gm.shift(1) > 0)
    return np.log(m_pos) - np.log(prev)

def _amihud(df):
    r = _logret_mid(df).abs()
    to = _get(df, "Turnover_sum")
    v  = _get(df, "Volume_sum")
    denom = to.where(to > 0, v)
    return _safe_div(r, denom, eps=EPS)

# =========================
# 3) factors 001-070 (只改易爆missing的点)
# =========================

def factor_001(df, i=4, j=12):
    ema_s = ewm_mean_by_st(df, _get(df, "ClosePrice"), i)
    ema_l = ewm_mean_by_st(df, _get(df, "ClosePrice"), j)
    return _safe_div(ema_s - ema_l, _mid(df).abs(), eps=EPS)

def factor_002(df, i=14):
    d = diff_by_st(df, _get(df, "ClosePrice"))
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    up_m = roll_mean_by_st(df, up, i)
    dn_m = roll_mean_by_st(df, dn, i)
    rs = _safe_div(up_m, dn_m, eps=EPS)
    return 1 - 1/(1+rs)

def factor_003(df, i=8):
    dev = _safe_div(_get(df,"ClosePrice") - _get(df,"VWAP"), _mid(df).abs(), eps=EPS)
    return roll_mean_by_st(df, dev, i)

def factor_004(df, i=20):
    high_max = _roll(df, _get(df, "HighPrice"), i, minp=5, func="max")
    low_min  = _roll(df, _get(df, "LowPrice"),  i, minp=5, func="min")
    denom = (high_max - low_min)
    return _safe_div(_get(df, "ClosePrice") - low_min, denom, eps=EPS)

def factor_005(df, i=4, j=12):
    v_s = roll_mean_by_st(df, _get(df,"Volume_sum"), i)
    v_l = roll_mean_by_st(df, _get(df,"Volume_sum"), j)
    return _safe_div(v_s - v_l, v_l, eps=EPS)

def factor_006(df, i=10, k=1.0):
    ret = pct_change_by_st(df, _get(df,"ClosePrice"))
    illiq = _safe_div(ret.abs(), _get(df,"Turnover_sum"), eps=EPS)
    out = roll_mean_by_st(df, illiq, i)
    return out ** k

def factor_007(df, j=4):
    bid_cols = [f"BidVolume{t}_last" for t in range(1, j+1)]
    ask_cols = [f"AskVolume{t}_last" for t in range(1, j+1)]
    bid = df[bid_cols].astype(float).sum(axis=1)
    ask = df[ask_cols].astype(float).sum(axis=1)
    # imbalance：分母为0时中性=0
    return _safe_div_zero(bid - ask, bid + ask)

def factor_008(df, j=4):
    bid_p = sum(_get(df,f"BidPrice{t}_last") * _get(df,f"BidVolume{t}_last") for t in range(1, j+1))
    ask_p = sum(_get(df,f"AskPrice{t}_last") * _get(df,f"AskVolume{t}_last") for t in range(1, j+1))
    vol   = sum(_get(df,f"BidVolume{t}_last") + _get(df,f"AskVolume{t}_last") for t in range(1, j+1))
    mid_w = _safe_div(bid_p + ask_p, vol, eps=EPS)
    return _safe_div(mid_w - _mid(df), _mid(df).abs(), eps=EPS)

def factor_009(df, j=4):
    bid_cols = [f"BidVolume{t}_last" for t in range(1, j+1)]
    ask_cols = [f"AskVolume{t}_last" for t in range(1, j+1)]
    bid = df[bid_cols].astype(float).sum(axis=1)
    ask = df[ask_cols].astype(float).sum(axis=1)
    # 原来你把bid/ask总量=0直接NaN，这里改：总量=0时该项贡献为0
    a = _safe_div_zero(_get(df,"BidVolume1_last"), bid)
    b = _safe_div_zero(_get(df,"AskVolume1_last"), ask)
    return a - b

def factor_010(df, j=4):
    bid_cols = [f"BidOrder{t}_last" for t in range(1, j+1)]
    ask_cols = [f"AskOrder{t}_last" for t in range(1, j+1)]
    bid_o = df[bid_cols].astype(float).sum(axis=1)
    ask_o = df[ask_cols].astype(float).sum(axis=1)
    # 计数型：denom=0 说明无挂单/被扫空 -> 中性=0 + 平滑
    return (bid_o - ask_o) / (bid_o + ask_o + EPS_COUNT)

def factor_011(df, i=10):
    dev = _safe_div(_get(df,"ClosePrice") - _get(df,"VWAP"), _mid(df).abs(), eps=EPS)
    return -roll_mean_by_st(df, dev, i)

def factor_012(df, i=8):
    sign = np.sign(_get(df,"ClosePrice") - _get(df,"OpenPrice"))
    sv = sign * _get(df,"Volume_sum")
    return roll_mean_by_st(df, sv, i)

def factor_013(df, i=6):
    dmid = diff_by_st(df, _mid(df)).abs()
    impact = _safe_div(dmid, _get(df,"Volume_sum"), eps=EPS)
    return roll_mean_by_st(df, impact, i)

def factor_014(df, i=10):
    x = (_get(df,"HighPrice") - _get(df,"LowPrice")) * _get(df,"Volume_sum")
    return roll_mean_by_st(df, x, i)

def factor_015(df, i=6):
    x = _spread(df) * _get(df,"Volume_sum")
    return roll_mean_by_st(df, x, i)

def factor_016(df, i=10):
    ret = pct_change_by_st(df, _get(df,"VWAP"))
    wret = ret * _get(df,"Volume_sum")
    return roll_mean_by_st(df, wret, i)

def factor_017(df, i=10):
    r = pct_change_by_st(df, _get(df,"ClosePrice"))
    v = pct_change_by_st(df, _get(df,"Volume_sum"))
    return roll_mean_by_st(df, (r - v), i)

def factor_018(df, i=14):
    d = diff_by_st(df, _get(df,"VWAP"))
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    up_m = roll_mean_by_st(df, up, i)
    dn_m = roll_mean_by_st(df, dn, i)
    return _safe_div(up_m, dn_m, eps=EPS)

def factor_019(df, j=4, k=1.0):
    depth = _depth(df,"bid",j) + _depth(df,"ask",j)
    return (_spread(df) * depth) ** k

def factor_020(df, i=12):
    v = _get(df,"Volume_sum")
    mu = roll_mean_by_st(df, v, i)
    sd = roll_std_by_st(df, v, i)
    return (v - mu) / (sd + EPS)

# --- 021-030
def factor_021(df, i=20):
    bidD=_depth(df,"bid",4); askD=_depth(df,"ask",4)
    imb=_safe_div_zero(bidD-askD, bidD+askD)
    return _roll(df, imb, i, 5, "mean")

def factor_022(df, i=20):
    bv1=_get(df,"BidVolume1_last"); av1=_get(df,"AskVolume1_last")
    imb1=_safe_div_zero(bv1-av1, bv1+av1)
    return _roll(df, imb1, i, 5, "mean")

def factor_023(df, i=20):
    bidD=_depth(df,"bid",4); askD=_depth(df,"ask",4)
    imb=_safe_div_zero(bidD-askD, bidD+askD)
    return _roll(df, imb, i, 5, "std")

def factor_024(df, i=20):
    ofi=_ofi1_proxy(df)
    return _roll(df, ofi, i, 5, "sum")

def factor_025(df, i=20):
    ofi=_ofi1_proxy(df)
    dep=(_depth(df,"bid",1)+_depth(df,"ask",1))
    x=_safe_div(ofi, dep, eps=EPS)
    return _roll(df, x, i, 5, "mean")

def factor_026(df, i=20):
    x=_microprice(df) - _mid(df)
    return _roll(df, x, i, 5, "mean")

def factor_027(df, i=20):
    x=_safe_div(_microprice(df)-_mid(df), _spread(df).abs(), eps=EPS)
    return _roll(df, x, i, 5, "mean")

def factor_028(df, i=20):
    wab=_wprice(df,"ask") - _wprice(df,"bid")
    x=_safe_div(wab, _mid(df).abs(), eps=EPS)
    return _roll(df, x, i, 5, "mean")

def factor_029(df, i=20):
    bidD=_depth(df,"bid",4); askD=_depth(df,"ask",4)
    x=_safe_div_zero(bidD, bidD+askD)
    return _roll(df, x, i, 5, "mean")

def factor_030(df, i=20):
    bidD=_depth(df,"bid",4); askD=_depth(df,"ask",4)
    x=_safe_div_zero(askD, bidD+askD)
    return _roll(df, x, i, 5, "mean")

# --- 031-040
def factor_031(df, i=20):
    x=_safe_div(_spread(df), _mid(df).abs(), eps=EPS)
    return _roll(df, x, i, 5, "mean")

def factor_032(df, i=20):
    # zscore spread：std=0 -> 0
    return _z(df, _spread(df), i, 5)

def factor_033(df, i=20):
    p1=_get(df,"BidPrice1_last"); p4=_get(df,"BidPrice4_last")
    v=_depth(df,"bid",4)
    x=_safe_div(p1-p4, v, eps=EPS)
    return _roll(df, x, i, 5, "mean")

def factor_034(df, i=20):
    p1=_get(df,"AskPrice1_last"); p4=_get(df,"AskPrice4_last")
    v=_depth(df,"ask",4)
    x=_safe_div(p4-p1, v, eps=EPS)
    return _roll(df, x, i, 5, "mean")

def factor_035(df, i=20):
    return _clip_inf(factor_033(df,i) - factor_034(df,i))

def factor_036(df, i=20):
    b1=_get(df,"BidPrice1_last"); a1=_get(df,"AskPrice1_last")
    x=_safe_div(a1-b1, a1.abs(), eps=EPS)
    return _roll(df, x, i, 5, "mean")

def factor_037(df, i=20):
    p1=_get(df,"BidPrice1_last"); p2=_get(df,"BidPrice2_last"); p3=_get(df,"BidPrice3_last")
    x=p1-2*p2+p3
    return _roll(df, x, i, 5, "mean")

def factor_038(df, i=20):
    p1=_get(df,"AskPrice1_last"); p2=_get(df,"AskPrice2_last"); p3=_get(df,"AskPrice3_last")
    x=p3-2*p2+p1
    return _roll(df, x, i, 5, "mean")

def factor_039(df, i=20):
    v1=_get(df,"BidVolume1_last"); v2=_get(df,"BidVolume2_last"); v3=_get(df,"BidVolume3_last")
    x=v1-2*v2+v3
    return _roll(df, x, i, 5, "mean")

def factor_040(df, i=20):
    v1=_get(df,"AskVolume1_last"); v2=_get(df,"AskVolume2_last"); v3=_get(df,"AskVolume3_last")
    x=v3-2*v2+v1
    return _roll(df, x, i, 5, "mean")

# --- 041-050
def factor_041(df, i=20):
    x=_safe_div(_get(df,"VWAP")-_mid(df), _spread(df).abs(), eps=EPS)
    return _roll(df, x, i, 5, "mean")

def factor_042(df, i=20):
    x=_safe_div(_get(df,"ClosePrice")-_mid(df), _spread(df).abs(), eps=EPS)
    return _roll(df, x, i, 5, "mean")

def factor_043(df, i=20):
    x=_safe_div(_get(df,"ClosePrice")-_get(df,"VWAP"), _mid(df).abs(), eps=EPS)
    return _roll(df, x, i, 5, "mean")

def factor_044(df, i=20):
    return _roll(df, _logret_mid(df), i, 5, "std")

def factor_045(df, i=20):
    return _roll(df, _logret_mid(df).abs(), i, 5, "mean")

def factor_046(df, i=20):
    dm=_g(df,_mid(df)).diff()
    num=_roll(df, dm, i, 5, "sum").abs()
    den=_roll(df, dm.abs(), i, 5, "sum")
    # den=0 -> 说明窗口内完全没动，方向强度=0（中性）
    return _safe_div_zero(num, den)

def factor_047(df, i=20):
    r=_logret_mid(df)
    std=_roll(df, r, i, 5, "std")
    mad=_roll(df, r.abs(), i, 5, "mean")
    return _safe_div(std, mad, eps=EPS)

def factor_048(df, i=20):
    sp=_spread(df).abs()
    dm=_g(df,_mid(df)).diff().abs()
    x=_safe_div(sp, dm, eps=EPS)  # dm=0时不会NaN/inf爆炸
    return _roll(df, x, i, 5, "mean")

def factor_049(df, i=20):
    x=_safe_div(_get(df,"HighPrice")-_get(df,"LowPrice"), _get(df,"VWAP").abs(), eps=EPS)
    return _roll(df, x, i, 5, "mean")

def factor_050(df, i=20):
    c=_get(df,"ClosePrice"); h=_get(df,"HighPrice"); l=_get(df,"LowPrice")
    denom=(h-l)
    clv=(2*c-h-l) / (denom + EPS)
    return _roll(df, clv, i, 5, "mean")

# --- 051-060
def factor_051(df, i=20):
    v=_get(df,"Volume_sum")
    return _safe_div(v, _roll(df, v, i, 5, "mean"), eps=EPS)

def factor_052(df, i=20):
    to=_get(df,"Turnover_sum")
    return _safe_div(to, _roll(df, to, i, 5, "mean"), eps=EPS)

def factor_053(df, i=20):
    dn=_get(df,"DealNum_sum")
    return _safe_div(dn, _roll(df, dn, i, 5, "mean"), eps=EPS)

def factor_054(df, i=20):
    x=_safe_div(_get(df,"Volume_sum"), _get(df,"DealNum_sum"), eps=EPS)
    return _roll(df, x, i, 5, "mean")

def factor_055(df, i=20):
    x=_safe_div(_get(df,"Turnover_sum"), _get(df,"DealNum_sum"), eps=EPS)
    return _roll(df, x, i, 5, "mean")

def factor_056(df, i=20):
    return _roll(df, _amihud(df), i, 5, "mean")

def factor_057(df, i=20):
    # zscore amihud：std=0 -> 0
    return _z(df, _amihud(df), i, 5)

def factor_058(df, i=20):
    x=_safe_div(_logret_mid(df).abs(), _get(df,"Volume_sum"), eps=EPS)
    return _roll(df, x, i, 5, "mean")

def factor_059(df, i=20):
    dv=_get(df,"dTotalVolume")
    r=_logret_mid(df)
    sv=dv*np.sign(r)
    m_r=_roll(df, r, i, 5, "mean")
    m_sv=_roll(df, sv, i, 5, "mean")
    cov=_roll(df, (r-m_r)*(sv-m_sv), i, 5, "mean")
    var=_roll(df, (sv-m_sv)**2, i, 5, "mean")
    return _safe_div_zero(cov, var)

def factor_060(df, i=20):
    ofi=_ofi1_proxy(df)
    r=_logret_mid(df)
    m1=_roll(df, ofi, i, 5, "mean")
    m2=_roll(df, r, i, 5, "mean")
    cov=_roll(df, (ofi-m1)*(r-m2), i, 5, "mean")
    sd1=_roll(df, (ofi-m1)**2, i, 5, "mean")**0.5
    sd2=_roll(df, (r-m2)**2, i, 5, "mean")**0.5
    return _safe_div_zero(cov, sd1*sd2)

# --- 061-070
def factor_061(df, i=20):
    imb=factor_022(df,i)
    sp=_roll(df, _spread(df).abs(), i, 5, "mean")
    return _safe_div_zero(imb, sp)

def factor_062(df, i=20):
    press=_roll(df, _microprice(df)-_mid(df), i, 5, "mean")
    vol_int=factor_051(df,i)
    return _clip_inf(press*vol_int)

def factor_063(df, i=20):
    return _clip_inf(factor_041(df,i)*factor_021(df,i))

def factor_064(df, i=20):
    # zscore mid：std=0 -> 0
    return -_z(df, _mid(df), i, 5)

def factor_065(df, i=20):
    return _roll(df, _logret_mid(df), i, 5, "sum")

def factor_066(df, i=20):
    tr=factor_065(df,i)
    sp=_roll(df, _spread(df).abs(), i, 5, "mean")
    return _safe_div_zero(tr, sp)

def factor_067(df, i=20):
    return _safe_div_zero(factor_024(df,i), factor_044(df,i))

def factor_068(df, i=20):
    # f_032, f_057 改后 std=0->0，f_068 的 NaN 会显著下降
    s1=factor_032(df,i)
    s2=factor_057(df,i)
    v=_get(df,"Volume_sum")
    s3=_z(df, v, i, 5)
    return _clip_inf(s1+s2-s3)

def factor_069(df, i=20):
    b12=_get(df,"BidVolume1_last")+_get(df,"BidVolume2_last")
    b34=_get(df,"BidVolume3_last")+_get(df,"BidVolume4_last")
    a12=_get(df,"AskVolume1_last")+_get(df,"AskVolume2_last")
    a34=_get(df,"AskVolume3_last")+_get(df,"AskVolume4_last")
    x=_safe_div_zero(b12, b34) - _safe_div_zero(a12, a34)
    return _roll(df, x, i, 5, "mean")

def factor_070(df, i=20):
    dev=_safe_div(_get(df,"ClosePrice")-_microprice(df), _spread(df).abs(), eps=EPS)
    to_int=factor_052(df,i)
    return _clip_inf(dev*to_int)