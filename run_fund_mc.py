"""Fundamentals-wired fly on a stratified sample of IPO paths. usage: python run_fund_mc.py SHARD NSHARDS NPATHS"""
import sys, pickle, time, numpy as np
from ipo_fly import *
sh, ns, K = map(int, sys.argv[1:4])
px, pxi = load_market(); r = px[:, -1] / px[:, 0] - 1; FD = load_fund()
o = np.argsort(r); pick = o[((np.arange(K) + 0.5) / K * len(r)).astype(int)]
fly = Fly(seed=100 + sh); out = []
for j, p in enumerate(pick):
    if j % ns != sh: continue
    fly.B.reset(); t0 = time.time()
    tr = trade_fund(fly, px[p], FD["prem"][p], FD["surp"][p], FD["war_day"][p])
    L = tr["ledger"]
    out.append(dict(path=int(p), q=(j + 0.5) / K, ret=float(r[p]), nav=tr["nav"], bh=tr["bh"], war=bool(FD["war"][p]),
                    dcf=float(FD["dcf"][p]), nav_curve=np.array([e["nav"] for e in L], np.float32),
                    sh=np.array([e["sh"] for e in L], np.int32), buys=int(sum(e["b"] for e in L)),
                    sells=int(sum(e["s"] for e in L)), trades=int(sum(1 for e in L if e["q"]))))
    print(f"path {p} q={(j+0.5)/K:.3f} 1y {r[p]*100:+.1f}% war={int(FD['war'][p])} fly ${tr['nav']:,.0f} buy&hold ${tr['bh']:,.0f} {time.time()-t0:.0f}s", flush=True)
    pickle.dump(out, open(f"results/fund_mc_shard{sh}.pkl", "wb"))
