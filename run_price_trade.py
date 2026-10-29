"""Phase 2: one trading year on the median Monte Carlo path, with full spike recording.
IPO allocation = max(0, buy spikes - sell spikes) from the reading phase (same rule as the daily trades)."""
import sys, pickle, time, numpy as np
from ipo_fly import *
from flybrain import annotations
tagname = sys.argv[1] if len(sys.argv) > 1 else "normal"
rd = pickle.load(open("results/read_normal.pkl", "rb"))["read"]
alloc = max(0, rd["buy"] - rd["sell"])
sil = ()
if tagname == "dopamine_off":
    a = annotations(); sil = a.root_id[a.top_nt == "dopamine"].values
fly = Fly(silence=sil, seed=11)
px, pxi = load_market(); r = px[:, -1] / px[:, 0] - 1
hero = int(np.argmin(np.abs(r - np.median(r))))
t0 = time.time(); tr = trade(fly, px[hero], alloc=alloc, record=True)
rec = fly.B.rec_buf[: fly.B.rec_n[0]].copy()
print(tagname, "path", hero, "alloc", tr["alloc"], "final value", round(tr["nav"], 2), "buy&hold", round(tr["bh"], 2), f"[{time.time()-t0:.0f}s]")
pickle.dump(dict(tag=tagname, hero=hero, px=px[hero], trade=tr, raster=rec,
                 idx=dict(sug=fly.sug, bit=fly.bit, mn9=fly.mn9, dn=fly.dn)), open(f"results/price_only/trade_{tagname}.pkl", "wb"))
