"""Fundamentals-wired fly on the median Monte Carlo path, with full spike recording. usage: python run_fund.py [normal|dopamine_off]"""
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
FD = load_fund()
t0 = time.time()
tr = trade_fund(fly, px[hero], FD["prem"][hero], FD["surp"][hero], FD["war_day"][hero], alloc=alloc, record=True)
rec = fly.B.rec_buf[: fly.B.rec_n[0]].copy()
print(tagname, "path", hero, "final", round(tr["nav"], 2), "buy&hold", round(tr["bh"], 2), f"[{time.time()-t0:.0f}s]", flush=True)
pickle.dump(dict(tag=tagname, hero=hero, px=px[hero], fair_px=FD["fair_px"][hero], prem=FD["prem"][hero], surp=FD["surp"][hero],
                 dcf=float(FD["dcf"][hero]), trade=tr, raster=rec, idx=dict(sug=fly.sug, bit=fly.bit, mn9=fly.mn9, dn=fly.dn)),
            open(f"results/fund_{tagname}.pkl", "wb"))
