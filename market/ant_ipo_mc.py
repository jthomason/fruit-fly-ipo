"""Anthropic, PBC (ANT) IPO Monte Carlo.

Three linked layers, all in $B unless noted:
  1. Fundamentals: quarterly revenue path from Q3 2026, cost structure converging to a long-run
     shape, a hard floor from non-cancellable compute commitments, DCF to an equity value per draw.
  2. Pricing: range midpoint anchored on reported investor talk, book demand moves the final price,
     first-day move driven by demand.
  3. Paths: 252 trading days after listing. Price = market's fundamental estimate x sentiment
     premium x supply effects. Earnings dates reveal the true fundamentals gradually; lock-up
     tranches add float; Nasdaq-100 fast entry at day 16; a possible price war.

Filing facts used (S-1 filed Oct 26, 2026): revenue Q1'26 4.731, Q2'26 11.555; Q2 compute and
infrastructure 5.558 (48.1% of revenue); Q2 personnel + S&M&P + other 5.023 (43.5%); cash and
short-term investments 131.651 at Jun 30, 2026; fully diluted shares ~16.937B (IPB 846.84M = 5%);
IPB 846.84M Class N shares; 91M RSUs granted after June; Series H $58.90 per Class A share
equivalent; lock-up schedule from Underwriting / Shares Eligible for Future Sale; counterparty
commitments from MD&A Contractual Obligations. Q3'26 revenue $16.8B is from Bloomberg reporting
(Oct 14, 2026); the filing leaves it blank.
"""
import json
import numpy as np

rng = np.random.default_rng(2026)
N = 20_000
tri = lambda a, c, b, n=N: rng.triangular(a, c, b, n)

# ---------------------------------------------------------------- constants from the filing
REV_Q3_26 = 16.8                       # Bloomberg, Oct 14 2026 (S-1 blank)
CASH_JUN26 = 131.651
FD_SHARES = 16.937 + 0.091 + 0.028     # FD at Jun 30 (already includes the 847M public benefit shares and the
                                       # 1.84B shares from Amazon notes) + post-June RSUs + Oct 2026 donation-match shares
OPTION_CASH = 1.797 * 2.89            # exercise proceeds on 1.80B options at $2.89 (FD counts them gross)
C0, O0 = 0.481, 0.435                  # Q2'26 compute share and other opex share of revenue
TAX = 0.21
YEARS = np.arange(2027, 2037)

# Non-cancellable commitment schedules ($B/yr, 2027-2036). Constructed from MD&A counterparty terms:
# Broadcom leases 161.2 over 5 yrs; Google 114.5 Apr'26-Jul'33; Amazon 110 May'26-Apr'36;
# Microsoft 44.9 Nov'26-May'33; Nscale 44.6 into 2033; Lambda 6.4 non-contingent; SpaceXAI only
# through its non-cancellable period (Q1 2027); ~150 of other fixed leases/commitments.
FRONT = np.array([66, 88, 98, 98, 91, 70, 48, 23, 21, 10], float)
# Back-loaded alternative honoring "substantially all of the remainder ... between 2031 and 2045".
BACK = np.array([58, 64, 66, 64, 52, 46, 40, 36, 34, 30], float)

def draw_params(n=N):
    P = dict(
        g4=tri(0.08, 0.22, 0.35, n),        # Q4'26 q/q growth (Q3 was +45%, Q2 +144%)
        d=tri(0.55, 0.75, 0.90, n),         # quarterly decay of the near-term excess growth
        G28=tri(0.15, 0.45, 0.90, n),       # underlying annual growth in 2028, fading to 6% by 2034
        L=np.clip(np.exp(rng.normal(np.log(800), 0.7, n)), 200, 4000),  # revenue ceiling, $B/yr
        c_inf=tri(0.30, 0.40, 0.52, n),     # long-run compute need, % of revenue (training + serving)
        o_inf=tri(0.20, 0.27, 0.35, n),     # long-run non-compute opex incl. SBC and CSP fees
        kappa=rng.uniform(0.85, 1.0, n),    # share of hard floor that truly binds (renegotiation, sublease)
        phi=rng.uniform(0, 1, n),           # weight on front-loaded commitment schedule
        wacc=tri(0.09, 0.105, 0.125, n),
        gT=rng.uniform(0.03, 0.045, n),
        war=rng.random(n) < 0.25,           # price war during 2027
        war_q=rng.integers(1, 5, n),        # quarter of 2027 it starts
        term=rng.uniform(4, 7, n),          # life used to capitalize rented compute (years)
        r_l=rng.uniform(0.07, 0.10, n),     # landlord's implied rate in the rent
        w_nwc=rng.uniform(0.08, 0.17, n),   # other operating capital, % of revenue (Q2'26: ~17%)
        bdraw=rng.random(n),                # Broadcom note draw fraction seed
    )
    return P

def medians(P):
    M = {k: np.full(1, np.median(v)) for k, v in P.items() if k not in ("war", "war_q")}
    M["war"] = np.zeros(1, bool); M["war_q"] = np.ones(1, int); M["bdraw"] = np.full(1, 0.5)
    return M

def fundamentals(P):
    n = len(P["g4"])
    # quarterly revenue Q4'26 .. Q4'36 (41 quarters)
    q = np.zeros((n, 41)); r = np.full(n, REV_Q3_26); g = P["g4"].copy()
    Lq = P["L"] / 4
    war = P["war"]; wq = P["war_q"]                     # war starts at quarter index 1..4 (2027)
    Lq = np.where(war, Lq * 0.7, Lq)
    for k in range(41):
        yr = 2026 + (k + 3) // 4                        # k=0 is Q4'26, k=1..4 is 2027
        frac = np.clip((yr - 2028) / 6, 0, 1)
        Gy = np.where(yr <= 2028, P["G28"], P["G28"] + (0.06 - P["G28"]) * frac)
        Gy = np.where(yr > 2034, 0.04, Gy)
        base_q = (1 + Gy) ** 0.25 - 1
        gk = base_q + np.maximum(P["g4"] - base_q, 0) * P["d"] ** k
        hit = war & (k >= wq) & (k < wq + 4)
        gk = np.where(hit, gk * 0.4, gk)
        gk = gk * np.clip(1 - r / Lq, -0.5, 1)
        r = r * (1 + gk)
        q[:, k] = r
    ann = np.stack([q[:, 1 + 4 * i: 5 + 4 * i].sum(1) for i in range(10)], 1)   # 2027..2036
    rev26 = 4.731 + 11.555 + REV_Q3_26 + q[:, 0]
    # cost shares converge linearly over 4 years
    t = np.arange(1, 11)
    w = np.clip(t / 4, 0, 1)
    o_inf = P["o_inf"] + np.where(war, 0.02, 0)[:, None].squeeze()
    c = C0 + (P["c_inf"][:, None] - C0) * w
    o = O0 + (o_inf[:, None] - O0) * w
    floor = P["kappa"][:, None] * (P["phi"][:, None] * FRONT + (1 - P["phi"][:, None]) * BACK)
    compute = np.maximum(c * ann, floor)
    ebit = ann * (1 - o) - compute
    # tax with NOL pool
    nol = np.full(n, 15.0); tax = np.zeros_like(ebit)
    for i in range(10):
        use = np.minimum(np.maximum(ebit[:, i], 0), nol)
        nol = nol - use + np.maximum(-ebit[:, i], 0)
        tax[:, i] = TAX * np.maximum(ebit[:, i] - use, 0)
    reinv = 0.03 * np.diff(np.concatenate([rev26[:, None], ann], 1), axis=1)
    fcf = ebit - tax - reinv
    disc = (1 + P["wacc"][:, None]) ** (-(t - 0.5 + 0.12))    # mid-year, valued mid-Nov 2026
    m_T = 1 - P["c_inf"] - o_inf
    tv_fcf = ann[:, -1] * (1 + P["gT"]) * m_T * (1 - TAX) - 0.03 * ann[:, -1] * P["gT"]
    tv = tv_fcf / (P["wacc"] - P["gT"]) * disc[:, -1]
    ev = (fcf * disc).sum(1) + tv
    # running cash for tenant-credit view (no new financing beyond IPO; IPO added later)
    cum_cash = CASH_JUN26 + np.cumsum(fcf, 1)
    # lease-adjusted ROIC: capitalize rented compute as if owned (annuity of the rent), add back the
    # interest embedded in the rent, tax at the statutory rate.
    AF = (1 - (1 + P["r_l"]) ** -P["term"]) / P["r_l"]
    K = compute * AF[:, None]
    IC = K + P["w_nwc"][:, None] * ann
    nopat = (ebit + P["r_l"][:, None] * K) * (1 - TAX)
    roic = nopat / IC
    ce = compute / ann
    mstar = P["wacc"][:, None] * (ce * AF[:, None] + P["w_nwc"][:, None]) / (1 - TAX) - P["r_l"][:, None] * ce * AF[:, None]
    return dict(ann=ann, rev26=rev26, ebit=ebit, fcf=fcf, ev=ev, floor=floor, compute=compute,
                margin=ebit / ann, cum_cash=cum_cash, binds=(floor > c * ann), q=q,
                roic=roic, IC=IC, mstar=mstar, AF=AF)

# ---------------------------------------------------------------- layer 1
P = draw_params()
F = fundamentals(P)
Pm = medians(P); Fm = fundamentals(Pm)
# Broadcom notes ($42B, only to fund Broadcom lease rent, "if needed"): 9% PIK, holder converts at 90% of
# 30-day VWAP. If revenue conditions fail: 13% PIK and 85%. Drawn mostly under stress. Value lost by
# holders = discount on the converted amount (PIK accrued ~2.5 yrs), draws centred ~2029.
def broadcom_cost(Px, Fx):
    stress = Px["war"] | Fx["binds"][:, 1]
    frac = np.where(stress, 0.5 + 0.5 * Px["bdraw"], 0.4 * Px["bdraw"])
    pik = np.where(Px["war"], 0.13, 0.09); k = np.where(Px["war"], 0.85, 0.90)
    return 42 * frac * (1 + pik) ** 2.5 * (1 / k - 1) * (1 + Px["wacc"]) ** -3.5, frac
BCOST, BFRAC = broadcom_cost(P, F)
EQ_PRE = F["ev"] + CASH_JUN26 + OPTION_CASH - BCOST     # pre-money equity value per draw
EQ_CONS = float(Fm["ev"][0] + CASH_JUN26 + OPTION_CASH - broadcom_cost(Pm, Fm)[0][0])

# ---------------------------------------------------------------- layer 2: pricing
M0 = tri(1.35, 1.60, 1.90)                       # range-midpoint post-money market cap, $T
D = np.exp(rng.normal(np.log(10), 0.5, N))       # book coverage (x)
z = np.clip(np.log(D / 10) / 0.5, -2, 2)
mc_ipo = np.clip(M0 * np.exp(0.08 * z), M0 * 0.85, M0 * 1.20) * 1000   # $B, post-money
deal = rng.uniform(70, 100, N)
psi = rng.uniform(0.55, 0.85, N)                 # primary share of the deal
primary = deal * psi
px_ipo = (mc_ipo - primary) / FD_SHARES          # $/share (post-money cap = p*S0 + primary)
shares_post = FD_SHARES + primary / px_ipo
pop = 0.10 + 0.06 * z + rng.normal(0, 0.10, N)
mc_d1 = mc_ipo * np.exp(pop)
eq_post = EQ_PRE + primary                       # DCF post-money equity, $B
cons_post = EQ_CONS + primary

# ---------------------------------------------------------------- layer 3: paths
T = 252
EARN = {70: 0.20, 122: 0.35, 186: 0.50, 250: 0.60}   # FY26, Q1'27, Q2'27, Q3'27 reports
# lock-up releases: fraction of fully diluted shares unlocked by day (assumed holder mix; filing
# leaves counts blank). Groups: employees/officers 17%, 180-day investors 45%, 365-day >0.5% 27%,
# founders 10%. Initial float = deal / market cap.
E, I, H, Fo = 0.17, 0.45, 0.27, 0.10
rel = np.zeros(T + 1)
def add(day, frac):
    if day <= T: rel[day] += frac
add(1, 0.10 * E); add(72, 0.35 * E); add(122, 0.55 * E)
for d_, f_ in [(27, .06), (48, .06), (62, .07), (76, .07), (72, .125), (85, .095), (99, .095), (113, .095)]:
    add(d_, f_ * I)
add(122, (1 - (.06 + .06 + .07 + .07 + .125 + .095 * 3)) * I)
for d_, f_ in [(27, .018), (48, .021), (72, .0375), (99, .0285), (122, .042), (156, .056), (170, .056),
               (184, .056), (198, .056), (186, .07), (213, .063), (227, .063), (241, .063)]:
    add(d_, f_ * H)
add(250, (1 - (.018 + .021 + .0375 + .0285 + .042 + .056 * 4 + .07 + .063 * 3)) * H)
add(122, 0.29 * Fo); add(250, 0.71 * Fo)

lam = tri(0.02, 0.05, 0.10)
sig = tri(0.40, 0.55, 0.75) / np.sqrt(252)
h = tri(0.5, 1.0, 2.0) * 252
theta = np.log(2) / h
s_inf = rng.normal(0.0, 0.15, N)                 # persistent premium or discount to the multiple rule
war_day = np.where(P["war"], 60 * (P["war_q"] - 1) + rng.integers(1, 60, N), 10_000)

# Market pricing rule: market cap = fair multiple x expected next-twelve-month (NTM) revenue.
# Fair EV/NTM multiple = (5 + 9 ln(1 + NTM growth)) x margin factor. At IPO consensus this gives
# roughly 14x, close to where the $1.5-2T talk sits; it compresses as growth slows.
hist = np.array([2.040, 4.731, 11.555, REV_Q3_26])          # Q4'25..Q3'26 actual
def qser(Fx):
    n = Fx["q"].shape[0]
    return np.concatenate([np.repeat(hist[None, :], n, 0), Fx["q"]], 1)   # idx 4 = Q4'26
QT = qser(F); QC = qser(Fm)
def ntm_ltm(Q, j):          # j = index of latest reported quarter in the combined series
    return Q[:, j + 1:j + 5].sum(1), Q[:, j - 3:j + 1].sum(1)
def margin_ntm(Fx, j):
    f = np.clip((j - 3) / 4, 0, 1)
    return Fx["margin"][:, 0] * (1 - f) + Fx["margin"][:, 1] * f
def fair_mult(g, m):
    return (5 + 9 * np.log1p(np.maximum(g, -0.5))) * np.clip(1 + 2 * (m - 0.10), 0.6, 1.4)

REPORT = {0: 3, 70: 4, 122: 5, 186: 6, 250: 7}        # day -> latest reported quarter index
WLEARN = {3: 0.0, 4: 0.45, 5: 0.65, 6: 0.80, 7: 0.90}  # weight on true path after each report
def market_value(j, wl, extra_w=0.0):
    wl = np.clip(wl + extra_w, 0, 1)
    nt, lt = ntm_ltm(QT, j); nc, _ = ntm_ltm(QC, j)
    ntm = np.exp((1 - wl) * np.log(nc) + wl * np.log(nt))
    m = (1 - wl) * margin_ntm(Fm, j) + wl * margin_ntm(F, j)
    g = ntm / lt - 1
    return fair_mult(g, m) * ntm, g, ntm

V0, g0, ntm0 = market_value(3, 0.0)
s = np.log(mc_d1) - np.log(V0)                   # day-one premium over the multiple rule
float0 = deal / mc_ipo
flt = float0.copy(); supply = np.zeros(N); idx = np.zeros(N)
path = np.zeros((N, T + 1)); path[:, 0] = np.log(mc_ipo); path[:, 1] = np.log(mc_d1)
j = 3; wl = 0.0; extra = np.zeros(N); V = V0
bdrag = np.log1p(-BCOST / mc_ipo)               # market prices the expected Broadcom discount
pbi = np.zeros(N); pbi_k = np.zeros(N)           # public benefit issuances: +2% of shares per +50% step
for t_ in range(2, T + 1):
    if t_ in REPORT:
        j = REPORT[t_]; wl = WLEARN[j]; s = s + rng.normal(0, 0.04, N)
        V, _, _ = market_value(j, wl, extra)
    if rel[t_] > 0:
        new = flt + rel[t_]
        supply = supply - lam * np.log(new / flt); flt = new
    if t_ == 16:
        idx = rng.normal(0.02, 0.03, N)
    hitwar = war_day == t_
    if hitwar.any():
        extra = np.where(hitwar, 0.25, extra)
        V, _, _ = market_value(j, wl, extra)
        s = s + np.where(hitwar, rng.normal(-0.08, 0.05, N), 0)
    s = s + theta * (s_inf - s) + sig * rng.standard_normal(N)
    lp = np.log(V) + s + supply + idx + bdrag + pbi
    step = np.floor(np.maximum(lp - np.log(mc_ipo), 0) / np.log(1.5))   # milestones crossed from IPO price
    newk = np.maximum(step - pbi_k, 0)
    pbi = pbi - newk * np.log(1.02); pbi_k = pbi_k + newk
    path[:, t_] = lp - newk * np.log(1.02)
mc_path = np.exp(path)                         # market cap, $B
px_path = mc_path / shares_post[:, None]

# ---------------------------------------------------------------- outputs
q = lambda x, p=(5, 25, 50, 75, 95): np.percentile(x, p).round(1).tolist()
below = lambda day: float((mc_path[:, day] < mc_ipo).mean())
out = dict(
    N=N,
    rev2026=q(F["rev26"]), rev2027=q(F["ann"][:, 0]), rev2028=q(F["ann"][:, 1]), rev2030=q(F["ann"][:, 3]),
    rev2027_cons=float(Fm["ann"][0, 0]), rev2028_cons=float(Fm["ann"][0, 1]),
    margin2028=q(F["margin"][:, 1] * 100), margin2030=q(F["margin"][:, 3] * 100),
    floor_binds_2028=float(F["binds"][:, 1].mean()), floor_binds_2029=float(F["binds"][:, 2].mean()),
    floor_2028=q(F["floor"][:, 1]),
    eq_pre=q(EQ_PRE), eq_cons=EQ_CONS,
    min_cash=q(F["cum_cash"].min(1) + primary),
    p_cash_below_5=float(((F["cum_cash"] + primary[:, None]).min(1) < 5).mean()),
    mc_ipo=q(mc_ipo), px_ipo=q(px_ipo), deal=q(deal), primary=q(primary), shares_post=q(shares_post),
    pop=q(pop * 100), p_break_day1=float((pop < 0).mean()), mc_d1=q(mc_d1),
    p_fund_above_ipo=float((eq_post >= mc_ipo).mean()),
    mc_d30=q(mc_path[:, 30]), mc_d122=q(mc_path[:, 122]), mc_d252=q(mc_path[:, 252]),
    px_d252=q(px_path[:, 252]),
    p_below_d30=below(30), p_below_d122=below(122), p_below_d252=below(252),
    ret_d252=q((mc_path[:, 252] / mc_ipo - 1) * 100),
    supply_total=q(-(lam * np.log((float0 + rel.sum()) / float0)) * 100),
)
# scenario buckets by fundamentals (price war separate), assigned before seeing the price path
rev28 = F["ann"][:, 1]; m30 = F["margin"][:, 3]; nowar = ~P["war"]
p75, p30 = np.percentile(rev28[nowar], [75, 30])
bull = nowar & (rev28 >= p75)
bear = nowar & (rev28 < p30)
base = nowar & ~bull & ~bear
warb = P["war"]
sc = {}
for name, m in [("Bull", bull), ("Base", base), ("Bear", bear), ("Price war", warb)]:
    sc[name] = dict(prob=float(m.mean()), rev2027=float(np.median(F["ann"][m, 0])), rev2028=float(np.median(rev28[m])),
                    rev2030=float(np.median(F["ann"][m, 3])), m2030=float(np.median(m30[m]) * 100),
                    eq=float(np.median(eq_post[m])), mc252=float(np.median(mc_path[m, 252])),
                    ret252=float(np.median(mc_path[m, 252] / mc_ipo[m] - 1) * 100),
                    p_below252=float((mc_path[m, 252] < mc_ipo[m]).mean()))
# reverse DCF: 2030 revenue at which the median draw's DCF equals the IPO market cap
ratio = eq_post / mc_ipo
order = np.argsort(F["ann"][:, 3]); r30s = F["ann"][order, 3]; rs = ratio[order]
bins = np.percentile(r30s, np.linspace(0, 100, 41))
mid, medr = [], []
for a, b in zip(bins[:-1], bins[1:]):
    m = (r30s >= a) & (r30s < b)
    if m.sum() > 50: mid.append(float(np.median(r30s[m]))); medr.append(float(np.median(rs[m])))
out["reverse_dcf_rev2030"] = float(np.interp(1.0, medr, mid))
out["reverse_curve"] = [mid, medr]
from scipy.stats import spearmanr
drv = {"2028 growth rate": P["G28"], "Revenue ceiling": np.log(P["L"]), "Long-run compute share": P["c_inf"],
       "Long-run other opex": P["o_inf"], "Cost of capital": P["wacc"], "Terminal growth": P["gT"],
       "Near-term growth": P["g4"], "Growth decay": P["d"], "Price war": P["war"].astype(float),
       "IPO valuation talk": M0, "Commitment floor binding": P["kappa"]}
out["tornado"] = {k: float(spearmanr(v, np.log(ratio))[0]) for k, v in drv.items()}
out["tornado_ret"] = {k: float(spearmanr(v, mc_path[:, 252] / mc_ipo)[0]) for k, v in drv.items()}
out["p_fund_above_ipo_by_wacc"] = {f"{lo:.3f}-{hi:.3f}": float((ratio[(P["wacc"] >= lo) & (P["wacc"] < hi)] >= 1).mean())
                                  for lo, hi in [(0.09, 0.10), (0.10, 0.11), (0.11, 0.125)]}
out["scenarios"] = sc
W = P["wacc"]
out["roic"] = {str(y): q(F["roic"][:, i] * 100) for y, i in [(2027, 0), (2028, 1), (2030, 3), (2033, 6)]}
out["p_roic_gt_wacc"] = {str(y): float((F["roic"][:, i] > W).mean()) for y, i in [(2027, 0), (2028, 1), (2030, 3), (2033, 6)]}
out["mstar"] = {str(y): q(F["mstar"][:, i] * 100) for y, i in [(2027, 0), (2028, 1), (2030, 3)]}
out["IC"] = {str(y): q(F["IC"][:, i]) for y, i in [(2027, 0), (2028, 1), (2030, 3)]}
out["roic_by_scen"] = {k: float(np.median(F["roic"][m_, 3]) * 100) for k, m_ in [("Bull", bull), ("Base", base), ("Bear", bear), ("Price war", warb)]}
out["broadcom_cost"] = q(BCOST); out["broadcom_frac"] = q(BFRAC)
out["pbi_issued_share"] = float((pbi_k > 0).mean()); out["pbi_k"] = q(pbi_k)
# value of operations at the IPO vs lease-adjusted invested capital today (Q2'26 run-rate)
AFq = (1 - (1 + P["r_l"]) ** -P["term"]) / P["r_l"]
IC_now = 5.558 * 4 * AFq + 7.9                   # rented compute capitalized + balance-sheet operating capital
ebit_q2 = 0.974 * 4
roic_now = (ebit_q2 + P["r_l"] * 5.558 * 4 * AFq) * (1 - TAX) / IC_now
ops_val = mc_ipo - CASH_JUN26 - primary - OPTION_CASH
out["IC_now"] = q(IC_now); out["roic_q2"] = q(roic_now * 100); out["ops_to_IC"] = q(ops_val / IC_now)
mstar_now = W * (0.481 * AFq + 7.9 / 46.2) / (1 - TAX) - P["r_l"] * 0.481 * AFq
out["mstar_q2"] = q(mstar_now * 100)

print(json.dumps(out, indent=1))
np.savez_compressed("market/ant_ipo_sim.npz", mc_path=mc_path[:4000], mc_ipo=mc_ipo, mc_d1=mc_d1,
                    eq_post=eq_post, pop=pop, rev=F["ann"], margin=F["margin"], floor=F["floor"],
                    compute=F["compute"], rel=rel, G28=P["G28"], war=P["war"], wacc=P["wacc"], primary=primary, deal=deal, roic=F["roic"], mstar=F["mstar"], IC=F["IC"], roic_now=roic_now, mstar_now=mstar_now, bcost=BCOST, px_ipo=px_ipo, cons_post=cons_post,
                    pct=np.percentile(mc_path, [5, 25, 50, 75, 95], axis=0))
json.dump(out, open("market/ant_ipo_results.json", "w"), indent=1)
