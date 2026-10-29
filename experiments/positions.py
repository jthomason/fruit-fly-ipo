"""Neuron coordinates (FlyWire annotation 'pos' point, 4x4x40 nm voxels) for every v630 neuron that kept its root ID in v783."""
import numpy as np, pandas as pd
a = pd.read_csv("ann/supplemental_files/Supplemental_file1_neuron_annotations.tsv", sep="\t", low_memory=False).drop_duplicates("root_id").set_index("root_id")
ids = pd.read_csv("shiu/2023_03_23_completeness_630_final.csv", index_col=0).index.values
l = a.reindex(ids)
np.savez("results/pos630.npz", x=l.pos_x.values, y=l.pos_y.values, sc=l.super_class.fillna("").values.astype(str), ct=l.cell_type.fillna("").values.astype(str))
print(f"{np.isfinite(l.pos_x.values).mean():.1%} of v630 neurons have coordinates")
