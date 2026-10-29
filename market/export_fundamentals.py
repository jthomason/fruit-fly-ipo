"""Re-run ant_ipo_mc.py unchanged (same seed, same draws) while recording, every trading day, the market's
fundamental fair value (the model's own EV/NTM multiple rule on the fundamentals revealed so far).
Also saves the true and consensus quarterly revenue paths, the price-war day and the DCF value per path.
Writes market/ant_ipo_fund.npz (first 4,000 paths, matching ant_ipo_sim.npz)."""
import numpy as np
src = open("market/ant_ipo_mc.py").read()
src = src.replace("for t_ in range(2, T + 1):\n", "LV = np.zeros((N, T + 1)); LV[:, 0] = np.log(V0); LV[:, 1] = np.log(V0); JJ = np.zeros(T + 1, int); JJ[:2] = 3\nfor t_ in range(2, T + 1):\n", 1)
src = src.replace("    path[:, t_] = lp - newk * np.log(1.02)\n", "    path[:, t_] = lp - newk * np.log(1.02)\n    LV[:, t_] = np.log(V); JJ[t_] = j\n", 1)
src = src[: src.index("# ---------------------------------------------------------------- outputs")]
g = {}; exec(compile(src, "ant_ipo_mc.py", "exec"), g)
z = np.load("market/ant_ipo_sim.npz"); K = 4000
assert np.allclose(z["mc_path"], g["mc_path"][:K]), "paths differ from ant_ipo_sim.npz"
np.savez_compressed("market/ant_ipo_fund.npz", LV=g["LV"][:K], JJ=g["JJ"], QT=g["QT"][:K], QC=g["QC"][:K],
                    war_day=g["war_day"][:K], war=g["P"]["war"][:K], eq_post=g["eq_post"][:K], mc_ipo=g["mc_ipo"][:K],
                    margin=g["F"]["margin"][:K], binds=g["F"]["binds"][:K], rel=g["rel"])
print("ok: identical paths, fundamentals exported")
