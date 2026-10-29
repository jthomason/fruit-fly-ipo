"""Does the fly's valuation signal carry a built-in edge? Correlate the day-100 premium (log price minus log fair value)
with the next 60 trading days' log return, across all 4,000 saved paths, for the fly's own walk-forward fair value
(v3, used for every result) and for the model's market value (v2, which the price is built from)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ipo_fly import load_fund, load_fund3
from flybrain import ROOT
mc = np.load(ROOT / "market/ant_ipo_sim.npz")["mc_path"]; fwd = np.log(mc[:, 160] / mc[:, 100])
for name, F in [("fly's own fair value (v3)", load_fund3()), ("model's market value (v2)", load_fund())]:
    print(f"{name:28s} corr = {np.corrcoef(F['prem'][:, 100], fwd)[0, 1]:+.3f}")
