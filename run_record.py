"""Re-run the median-path trading year with every neuron's spike times recorded (0.1 ms steps), for the brain animation.
Same seed and inputs as run_fund.py, so trades are identical; recording does not touch the dynamics."""
import sys, pickle, time, numpy as np
from ipo_fly import *
rd = pickle.load(open("results/read_normal.pkl", "rb"))["read"]
alloc = max(0, rd["buy"] - rd["sell"])
fly = Fly(seed=11); fly.B.rec_mask[:] = True
px, pxi = load_market(); r = px[:, -1] / px[:, 0] - 1
hero = int(np.argmin(np.abs(r - np.median(r))))
FD = load_fund3()
WIN = []; _w = fly.window
def window(ms, rs, rb):
    t0 = fly.B.t; c = _w(ms, rs, rb); WIN.append((t0, fly.B.t, rs, rb)); return c
fly.window = window
t0 = time.time()
tr = trade_fund(fly, px[hero], FD["prem"][hero], FD["surp"][hero], FD["war_day"][hero], alloc=alloc, record=False)
rec = fly.B.rec_buf[: fly.B.rec_n[0]]
print("final", round(tr["nav"], 2), "spikes", len(rec), f"[{time.time()-t0:.0f}s]")
np.savez_compressed("results/full_raster.npz", step=rec[:, 0].astype(np.int64), neu=rec[:, 1].astype(np.int32), win=np.array(WIN, np.float64))
