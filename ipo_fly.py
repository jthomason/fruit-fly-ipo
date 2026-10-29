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
