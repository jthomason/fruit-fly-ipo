import numpy as np, pandas as pd, pickle
d = pickle.load(open("results/fund_normal.pkl", "rb")); L = d["trade"]["ledger"]
h = pickle.load(open("results/read_normal.pkl", "rb"))["read"]["chunks"]
z = np.load("results/full_raster.npz")
act = set(np.unique(z["neu"]).tolist())
for c in h: act |= set(c["sp"][0].tolist())
act = np.array(sorted(act)); print("ever-active neurons", len(act))
con = pd.read_parquet("shiu/2023_03_23_connectivity_630_final.parquet"); print(con.columns.tolist())
m = con.Presynaptic_Index.isin(act) & con.Postsynaptic_Index.isin(act)
e = con[m]; e = e[e["Connectivity"].abs() >= 3]
print("edges", len(e), "exc", (e["Excitatory"] > 0).sum())
np.savez("results/active_edges.npz", pre=e.Presynaptic_Index.values.astype(np.int32), post=e.Postsynaptic_Index.values.astype(np.int32),
         w=e["Excitatory x Connectivity"].values.astype(np.float32), act=act.astype(np.int32))
