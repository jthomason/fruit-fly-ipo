"""Phase 1, due diligence: the fly reads the S-1 Risk Factors (one word per ms). Saves results/read_normal.pkl."""
import sys, pickle, time
from ipo_fly import *
from flybrain import annotations
tagname = sys.argv[1] if len(sys.argv) > 1 else "normal"
sil = ()
if tagname == "dopamine_off":
    a = annotations(); sil = a.root_id[a.top_nt == "dopamine"].values
fly = Fly(silence=sil, seed=7)
words, tag = load_words()
t0 = time.time(); rd = read_s1(fly, words, tag)
print(f"{tagname}: read {len(words):,} words ({(tag==1).sum():,} negative, {(tag==2).sum():,} positive) -> "
      f"{rd['buy']} buy spikes, {rd['sell']} sell spikes, IPO allocation request {max(0, rd['buy']-rd['sell'])} shares  [{time.time()-t0:.0f}s]")
pickle.dump(dict(tag=tagname, read=rd, idx=dict(sug=fly.sug, bit=fly.bit, mn9=fly.mn9, dn=fly.dn)),
            open(f"results/read_{tagname}.pkl", "wb"))
