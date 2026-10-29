import numpy as np, pandas as pd
a = pd.read_csv("ann/supplemental_files/Supplemental_file1_neuron_annotations.tsv", sep="\t", low_memory=False).drop_duplicates("root_id").set_index("root_id")
ids = pd.read_csv("shiu/2023_03_23_completeness_630_final.csv", index_col=0).index.values
l = a.reindex(ids)
np.savez("results/pos630_3d.npz", x=l.pos_x.values, y=l.pos_y.values, z=l.pos_z.values, sc=l.super_class.fillna("").values.astype(str),
         ct=l.cell_type.fillna("").values.astype(str), nt=l.top_nt.fillna("").values.astype(str))
print(np.nanpercentile(l.pos_z.values, [1, 50, 99]), np.nanpercentile(l.pos_x.values, [1, 50, 99]), np.nanpercentile(l.pos_y.values, [1, 50, 99]))
print(pd.Series(l.super_class).value_counts())
