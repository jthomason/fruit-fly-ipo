"""A real fruit fly connectome trades a Monte Carlo simulation of Anthropic's IPO.

Brain:   FlyWire v630 whole-brain LIF model (Shiu et al. 2024, Nature), 127,400 neurons, ~14.7M connections.
Senses:  21 sugar gustatory receptor neurons (GRNs) and 21 bitter GRNs, IDs from the paper.
Actions: BUY  = MN9, the proboscis motor neuron (the paper's sugar -> feeding readout)
         SELL = DNge067 pair, the descending neuron the model drives hardest from bitter and never from sugar
Market:  ant_ipo_mc.py (20,000 paths, 252 trading days, FD-share-consistent pricing)

Phase 1 "due diligence": the fly reads the S-1's Risk Factors section (pages 27-120, 69,169 words), one word per
millisecond. Words in the Loughran-McDonald finance dictionary's Negative list stimulate bitter GRNs, Positive
words stimulate sugar GRNs. Spikes in MN9 during reading become its IPO allocation request (1 spike = 1 share).
Phase 2 "trading": each trading day is 100 ms of fly time. Up days stimulate sugar GRNs, down days stimulate bitter
GRNs, at rate = min(300 Hz, 100 Hz x |daily return| / daily vol). Each MN9 spike buys one share and each DNge067
spike sells one share, at that day's close. Long only, no margin, starting cash $10,000.
"""
import json, re, sys, time
import numpy as np, pandas as pd
from flybrain import FlyBrain, annotations, ROOT

S = json.load(open(ROOT / "sets630.json"))
MN9 = S["ids_mn9"]
DNGE067 = [720575940622537477, 720575940627827697]
SIG_D = 0.55 / np.sqrt(252)          # median daily vol in the market model
DAY_MS = 100.0
CASH0 = 10_000.0
WORD_MS = 1.0; CHUNK = 25            # 25 words per 25 ms block
K_WORD = 1500.0                       # Hz per unit word fraction -> each matching word = ~1.5 kicks per GRN


def load_market():
    z = np.load(ROOT / "market/ant_ipo_sim.npz")
    mc = z["mc_path"]; ipo = z["mc_ipo"][: len(mc)]; pxi = z["px_ipo"][: len(mc)]
    px = mc * (pxi / ipo)[:, None]    # px_ipo * shares_post = mc_ipo, so price = market cap / shares_post
    return px, pxi


def load_words():
    t = open(ROOT / "risk_factors.txt").read()
    words = re.findall(r"[A-Za-z][A-Za-z'’\-]*", t)
    lm = pd.read_csv(sys.modules["pysentiment2"].__path__[0] + "/static/LM.csv") if "pysentiment2" in sys.modules else None
    if lm is None:
        import pysentiment2, os
        lm = pd.read_csv(os.path.dirname(pysentiment2.__file__) + "/static/LM.csv")
    neg = set(lm.Word[lm.Negative > 0].str.lower()); pos = set(lm.Word[lm.Positive > 0].str.lower())
    wl = [w.lower().replace("’", "'") for w in words]
    tag = np.array([1 if w in neg else (2 if w in pos else 0) for w in wl], np.int8)
    return words, tag


class Fly:
    def __init__(self, silence=(), seed=0):
        self.B = B = FlyBrain(version=630, silence_ids=silence, seed=seed)
        self.sug = B.ix(S["neu_sugar"]); self.bit = B.ix(S["neu_bitter"])
        self.mn9 = B.ix(MN9); self.dn = B.ix(DNGE067)
        B.rec_mask[np.concatenate([self.sug, self.bit, self.mn9, self.dn])] = True

    def window(self, ms, r_sug, r_bit):
        drv = []
        if r_sug > 0: drv.append((self.sug, r_sug))
        if r_bit > 0: drv.append((self.bit, r_bit))
        c = self.B.run(ms, drv)
        return c

    def sparse(self, c):
        nz = np.nonzero(c)[0]
        return nz.astype(np.int32), c[nz].astype(np.uint16)


def read_s1(fly, words, tag, record=True, log=True):
    out = dict(chunks=[], buy=0, sell=0)
    n = len(words); t0 = time.time()
    for k in range(0, n, CHUNK):
        tg = tag[k:k + CHUNK]; L = len(tg)
        r_b = min(300.0, K_WORD * (tg == 1).mean()); r_s = min(300.0, K_WORD * (tg == 2).mean())
        c = fly.window(WORD_MS * L, r_s, r_b)
        b = int(c[fly.mn9].sum()); s = int(c[fly.dn].sum())
        out["buy"] += b; out["sell"] += s
        if record:
            out["chunks"].append(dict(k=k, neg=int((tg == 1).sum()), pos=int((tg == 2).sum()), b=b, s=s,
                                      sp=fly.sparse(c), tot=int(c.sum())))
        if log and (k // CHUNK) % 500 == 0:
            print(f"  read {k:6d}/{n} words  buy-spikes {out['buy']} sell-spikes {out['sell']}  {time.time()-t0:.0f}s", flush=True)
    return out


def trade(fly, px, alloc=0, record=False):
    """px: 253 prices, px[0] = IPO price. alloc: shares requested at the IPO price."""
    cash = CASH0; sh = 0
    a = int(min(alloc, cash // px[0])); cash -= a * px[0]; sh += a
    led = []
    for d in range(1, len(px)):
        r = px[d] / px[d - 1] - 1; zz = r / SIG_D
        r_s = min(300.0, 100.0 * max(zz, 0)); r_b = min(300.0, 100.0 * max(-zz, 0))
        c = fly.window(DAY_MS, r_s, r_b)
        b = int(c[fly.mn9].sum()); s = int(c[fly.dn].sum())
        net = b - s; q = 0
        if net > 0: q = int(min(net, cash // px[d]))
        elif net < 0: q = -int(min(-net, sh))
        cash -= q * px[d]; sh += q
        e = dict(d=d, px=float(px[d]), r=float(r), rs=r_s, rb=r_b, b=b, s=s, q=q, cash=cash, sh=sh, nav=cash + sh * px[d])
        if record: e["sp"] = fly.sparse(c); e["tot"] = int(c.sum())
        led.append(e)
    return dict(alloc=a, ledger=led, nav=led[-1]["nav"], bh=CASH0 // px[0] * px[-1] + CASH0 % px[0])


# ============================================================ v2: the fly is wired to the FUNDAMENTALS, not the price
# Everything below comes from ant_ipo_mc.py itself (market/ant_ipo_fund.npz, exported by market/export_fundamentals.py):
#   fair value  = the model's own market-pricing rule: fair EV/NTM multiple x expected NTM revenue, using only the
#                 fundamentals revealed so far (consensus blended toward the true path after each earnings report)
#   valuation   = log(market cap / fair value). Cheap -> sugar, rich -> bitter. Every day, 100 ms.
#   earnings    = days 70, 122, 186, 250: reported quarterly revenue vs the IPO consensus path. Beat -> sugar,
#                 miss -> bitter, for a 200 ms "earnings call".
#   price war   = the day the 2027 price war starts (25% of paths): 300 ms of bitter at 300 Hz.
# The fly never sees the stock price. It only trades at it.
EARN_DAYS = {70: 4, 122: 5, 186: 6, 250: 7}
V_SCALE = 0.15          # 15% rich/cheap -> 100 Hz
E_SCALE = 0.03          # 3% revenue beat/miss -> 100 Hz
CALL_MS = 200.0; WAR_MS = 300.0


def load_fund():
    f = np.load(ROOT / "market/ant_ipo_fund.npz"); z = np.load(ROOT / "market/ant_ipo_sim.npz")
    prem = np.log(z["mc_path"]) - f["LV"]
    surp = f["QT"][:, 4:8] / f["QC"][0, 4:8] - 1
    return dict(prem=prem, surp=surp, war_day=f["war_day"], war=f["war"], dcf=f["eq_post"] / f["mc_ipo"],
                fair_px=np.exp(f["LV"]) * (z["px_ipo"][: len(prem)] / z["mc_ipo"][: len(prem)])[:, None])


# ============================================================ v3: the fly's OWN walk-forward valuation
# v2 compared the price with the model's market value, but the model builds the price as market value x sentiment,
# so that gap carried no fundamental information. v3 values the company only from what has been reported so far:
#   NTM revenue = IPO consensus for the next 4 quarters, scaled by the latest reported quarter vs its consensus
#   margin      = IPO consensus, replaced by the true 2027 margin once 2027 quarters are reported (day 122 on)
#   multiple    = the model's own rule, (5 + 9 ln(1 + NTM growth)) x margin factor, on those reported inputs
#   minus the consensus Broadcom cost from the S-1
# The market, by contrast, also sees part of the true future revenue path, a sentiment premium and lock-up supply.
def fair_mult(g, m):
    return (5 + 9 * np.log1p(np.maximum(g, -0.5))) * np.clip(1 + 2 * (m - 0.10), 0.6, 1.4)


def fly_value(f):
    QT, QC, JJ = f["QT"], f["QC"], f["JJ"]; n, T1 = QT.shape[0], len(JJ)
    LVF = np.zeros((n, T1)); NTM = np.zeros((n, T1)); G = np.zeros((n, T1)); M = np.zeros((n, T1))
    for j in np.unique(JJ):
        r = QT[:, j] / QC[0, j]
        ntm = r * QC[0, j + 1:j + 5].sum(); ltm = QT[:, j - 3:j + 1].sum(1); g = ntm / ltm - 1
        fr = np.clip((j - 3) / 4, 0, 1)
        m27 = np.full(n, f["margin_cons"][0, 0]) if j < 5 else f["margin"][:, 0]
        m = m27 * (1 - fr) + f["margin_cons"][0, 1] * fr
        v = fair_mult(g, m) * ntm
        cols = JJ == j
        LVF[:, cols] = (np.log(v) + np.log1p(-f["bcost_cons"] / f["mc_ipo"]))[:, None]
        NTM[:, cols] = ntm[:, None]; G[:, cols] = g[:, None]; M[:, cols] = m[:, None]
    return LVF, NTM, G, M


def load_fund3():
    f = np.load(ROOT / "market/ant_ipo_fund.npz"); z = np.load(ROOT / "market/ant_ipo_sim.npz")
    LVF, NTM, G, M = fly_value(f)
    prem = np.log(z["mc_path"]) - LVF
    surp = f["QT"][:, 4:8] / f["QC"][0, 4:8] - 1
    sh = (z["mc_ipo"][: len(prem)] / z["px_ipo"][: len(prem)])[:, None]
    return dict(prem=prem, surp=surp, war_day=f["war_day"], war=f["war"], dcf=f["eq_post"] / f["mc_ipo"],
                fair_px=np.exp(LVF) / sh, mkt_px=np.exp(f["LV"]) / sh, ntm=NTM, g=G, m=M,
                info_gap=f["LV"] - LVF + np.log1p(-f["bcost_cons"] / f["mc_ipo"])[:, None])


def trade_fund(fly, px, prem, surp, war_day, alloc=0, record=False):
    cash = CASH0; sh = 0
    a = int(min(alloc, cash // px[0])); cash -= a * px[0]; sh += a
    led = []
    for d in range(1, len(px)):
        v = prem[d]
        r_s = min(300.0, 100.0 * max(-v, 0) / V_SCALE); r_b = min(300.0, 100.0 * max(v, 0) / V_SCALE)
        c = fly.window(DAY_MS, r_s, r_b).astype(np.int32)
        ev = None; es = 0.0
        if d in EARN_DAYS:
            es = float(surp[EARN_DAYS[d] - 4]); ev = "earnings"
            rr = min(300.0, 100.0 * abs(es) / E_SCALE)
            c = c + fly.window(CALL_MS, rr if es > 0 else 0, rr if es < 0 else 0)
        if d == war_day:
            ev = "war"; c = c + fly.window(WAR_MS, 0, 300.0)
        b = int(c[fly.mn9].sum()); s = int(c[fly.dn].sum())
        net = b - s; q = 0
        if net > 0: q = int(min(net, cash // px[d]))
        elif net < 0: q = -int(min(-net, sh))
        cash -= q * px[d]; sh += q
        e = dict(d=d, px=float(px[d]), prem=float(v), rs=r_s, rb=r_b, ev=ev, surp=es, b=b, s=s, q=q, cash=cash, sh=sh,
                 nav=cash + sh * px[d])
        if record: e["sp"] = fly.sparse(c); e["tot"] = int(c.sum())
        led.append(e)
    return dict(alloc=a, ledger=led, nav=led[-1]["nav"], bh=CASH0 // px[0] * px[-1] + CASH0 % px[0])
