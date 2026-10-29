"""Which outputs respond to which taste? (How BUY = MN9 and SELL = DNge067 were chosen.) 3 trials x 1 s at 150 Hz."""
import json, numpy as np, sys; sys.path.insert(0, ".")
from flybrain import FlyBrain, annotations
S = json.load(open("sets630.json")); B = FlyBrain(version=630)
lab = annotations().drop_duplicates("root_id").set_index("root_id").reindex(B.ids)
res = {}
for name in ["neu_sugar", "neu_bitter", "neu_JON_CE", "neu_water"]:
    tot = np.zeros(B.N)
    for tr in range(3): B.reset(); tot += B.run(1000, [(B.ix(S[name]), 150)])
    res[name] = tot / 3
inp = set(B.ix(sum(S.values(), [])))
for name in ["neu_bitter", "neu_sugar"]:
    print("==", name, "-> top descending/motor neurons (Hz): [sugar, bitter]")
    k = 0
    for i in np.argsort(-res[name]):
        if i in inp or lab.iloc[i].super_class not in ("descending", "motor"): continue
        print(f"  {B.ids[i]} {lab.iloc[i].cell_type:10s} sugar {res['neu_sugar'][i]:5.0f}  bitter {res['neu_bitter'][i]:5.0f}"); k += 1
        if k == 8: break
print("MN9:", res["neu_sugar"][B.ix(S["ids_mn9"])], "Hz on sugar;", res["neu_bitter"][B.ix(S["ids_mn9"])], "on bitter")
