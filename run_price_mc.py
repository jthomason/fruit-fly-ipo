"""Monte Carlo of the fly: stratified sample of IPO paths (evenly spaced quantiles of 1-year return).
usage: python run_mc.py SHARD NSHARDS NPATHS ALLOC"""
import sys, pickle, time, numpy as np
from ipo_fly import *
sh, ns, K, alloc = map(int, sys.argv[1:5])
px, pxi = load_market(); r = px[:, -1] / px[:, 0] - 1
o = np.argsort(r); pick = o[((np.arange(K) + 0.5) / K * len(r)).astype(int)]
fly = Fly(seed=100 + sh); out = []
for j, p in enumerate(pick):
    if j % ns != sh: continue
    fly.B.reset(); t0 = time.time()
    tr = trade(fly, px[p], alloc=alloc)
    L = tr["ledger"]
    out.append(dict(path=int(p), q=(j + 0.5) / K, ret=float(r[p]), nav=tr["nav"], bh=tr["bh"], alloc=tr["alloc"],
                    nav_curve=np.array([e["nav"] for e in L], np.float32), sh=np.array([e["sh"] for e in L], np.int32),
                    buys=int(sum(e["b"] for e in L)), sells=int(sum(e["s"] for e in L)),
                    trades=int(sum(1 for e in L if e["q"]))))
    print(f"path {p} q={(j+0.5)/K:.3f} ipo->1y {r[p]*100:+.1f}%  fly ${tr['nav']:,.0f}  buy&hold ${tr['bh']:,.0f}  {time.time()-t0:.0f}s", flush=True)
    pickle.dump(out, open(f"results/price_only/mc_shard{sh}.pkl", "wb"))
