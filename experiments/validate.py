"""Check the numba port against the paper's own Brian2 output (results/example/sugarR.parquet, 30 trials x 1 s)."""
import json, numpy as np, pandas as pd, sys; sys.path.insert(0, ".")
from flybrain import FlyBrain
S = json.load(open("sets630.json")); B = FlyBrain(version=630)
ref = pd.read_parquet("shiu/results/example/sugarR.parquet"); r = ref.flywire_id.value_counts() / 30
tot, mn9 = [], []
for tr in range(5):
    B.reset(); c = B.run(1000, [(B.ix(S["neu_sugar"]), 150)]); tot.append(c.sum()); mn9.append(c[B.ix([720575940660219265])][0])
print(f"total spikes/trial: ours {np.mean(tot):,.0f}  paper {len(ref)/30:,.0f}")
print(f"MN9 rate (Hz):      ours {np.mean(mn9):.0f}  paper {r.get(720575940660219265):.0f}")
