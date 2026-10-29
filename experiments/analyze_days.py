import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pickle, pandas as pd, json
z = np.load("results/full_raster.npz"); step, neu, win = z["step"], z["neu"], z["win"]
d = pickle.load(open("results/fund_normal.pkl", "rb")); idx = d["idx"]; L = d["trade"]["ledger"]
from ipo_fly import EARN_DAYS, load_fund3
FD = load_fund3(); wd = int(FD["war_day"][d["hero"]]); print("war day", wd, len(win))
# map windows to days
days = []; k = 0
for dd in range(1, 253):
    n = 1 + (dd in EARN_DAYS) + (dd == wd); days.append((dd, k, k + n)); k += n
assert k == len(win), (k, len(win))
DAYWIN = {dd: (win[a][0], win[b - 1][1]) for dd, a, b in days}
p = np.load("results/pos630_3d.npz", allow_pickle=True); sc = p["sc"]; ct = p["ct"]
def day_spikes(dd):
    a, b = DAYWIN[dd]; m = (step >= a) & (step < b); return step[m] - a, neu[m]
for dd in [60, 84, 87, 186]:
    s, n = day_spikes(dd); u = np.unique(n)
    print(dd, "spikes", len(s), "neurons", len(u), "len ms", (DAYWIN[dd][1] - DAYWIN[dd][0]) / 10, "ledger b/s", L[dd - 1]["b"], L[dd - 1]["s"])
    print("   superclass", pd.Series(sc[u]).value_counts().head(6).to_dict())
    first = pd.Series(s).groupby(n).min()
    for name in ["sug", "bit", "mn9", "dn"]:
        v = [x for x in idx[name] if x in first.index]; print("   ", name, len(v), "first ms", (first[v].min() / 10 if v else None))
json.dump({str(k): v for k, v in DAYWIN.items()}, open("results/daywin.json", "w"))
