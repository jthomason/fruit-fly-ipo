"""Whole-brain leaky integrate-and-fire model of the adult fruit fly, driven by the FlyWire connectome (v630 is used for every result here; v783 also loads).

This is a numba port of Shiu et al. 2024 (Nature, "A Drosophila computational brain model reveals sensorimotor
processing", github.com/philshiu/Drosophila_brain_model, MIT). Same equations and constants as their Brian2
model; we only re-implemented it so it runs fast enough on a 2-core cloud sandbox to trade an IPO in real time.

  dv/dt = (v_0 - v + g) / t_mbr        (unless refractory)
  dg/dt = -g / tau                      (unless refractory)
  spike: v > v_th -> v = v_rst, g = 0, refractory 2.2 ms
  synapse: g_post += w_syn * (signed synapse count), 1.8 ms delay
  sensory drive: Poisson kicks of w_syn * 250 into v at a given rate (that's how the paper "activates" neurons)

Neuron identity: FlyWire root IDs, with cell types from Schlegel et al. 2024 (flyconnectome/flywire_annotations).
"""
import numpy as np, pandas as pd
from numba import njit
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = dict(v_0=-52.0, v_rst=-52.0, v_th=-45.0, t_mbr=20.0, tau=5.0, t_rfc=2.2, t_dly=1.8, w_syn=0.275, f_poi=250,
         dt=0.1)   # mV, ms


@njit(cache=True, fastmath=True, boundscheck=False)
def _run(nsteps, t0, v, g, ref, rfc_zero, indptr, indices, data, ring, ring_n, D,
         inp_idx, inp_p, w_poi, ev, c_vg, eg, v0, vth, vrst, ref_steps, counts, rec_mask, rec_buf, rec_n, seed):
    np.random.seed(seed)
    N = v.shape[0]
    for k in range(nsteps):
        t = t0 + k
        s = t % D
        # 1) deliver spikes emitted D steps ago
        n = ring_n[s]
        for a in range(n):
            pre = ring[s, a]
            for e in range(indptr[pre], indptr[pre + 1]):
                g[indices[e]] += data[e]
        ring_n[s] = 0
        # 2) Poisson sensory drive
        for a in range(inp_idx.shape[0]):
            if np.random.random() < inp_p[a]:
                v[inp_idx[a]] += w_poi
        # 3) integrate (exact solution of the linear system over dt) + threshold
        m = 0
        for i in range(N):
            if ref[i] > 0:
                ref[i] -= 1
                continue
            gi = g[i]
            vi = v[i]
            if gi == 0.0 and vi == v0:
                continue
            vi = v0 + (vi - v0) * ev + gi * c_vg
            gi = gi * eg
            if abs(gi) < 1e-7:
                gi = 0.0
                if abs(vi - v0) < 1e-6:
                    vi = v0
            if vi > vth:
                vi = vrst
                gi = 0.0
                ref[i] = 0 if rfc_zero[i] else ref_steps
                ring[s, m] = i
                m += 1
                counts[i] += 1
                if rec_mask[i] and rec_n[0] < rec_buf.shape[0]:
                    rec_buf[rec_n[0], 0] = t
                    rec_buf[rec_n[0], 1] = i
                    rec_n[0] += 1
            v[i] = vi
            g[i] = gi
            continue
            if v[i] > vth:
                v[i] = vrst
                g[i] = 0.0
                ref[i] = 0 if rfc_zero[i] else ref_steps
                ring[s, m] = i
                m += 1
                counts[i] += 1
                if rec_mask[i] and rec_n[0] < rec_buf.shape[0]:
                    rec_buf[rec_n[0], 0] = t
                    rec_buf[rec_n[0], 1] = i
                    rec_n[0] += 1
        ring_n[s] = m
    return t0 + nsteps


class FlyBrain:
    def __init__(self, silence_ids=(), seed=0, version=783):
        fc, fp = {783: ("Completeness_783.csv", "Connectivity_783.parquet"),
                  630: ("2023_03_23_completeness_630_final.csv", "2023_03_23_connectivity_630_final.parquet")}[version]
        comp = pd.read_csv(ROOT / "shiu" / fc, index_col=0)
        self.ids = comp.index.values.astype(np.int64)
        self.N = N = len(self.ids)
        self.idx = {int(r): i for i, r in enumerate(self.ids)}
        con = pd.read_parquet(ROOT / "shiu" / fp)
        pre = con["Presynaptic_Index"].values.astype(np.int64)
        post = con["Postsynaptic_Index"].values.astype(np.int32)
        w = (con["Excitatory x Connectivity"].values * P["w_syn"]).astype(np.float32)
        if len(silence_ids):
            sil = np.zeros(N, bool); sil[[self.idx[int(r)] for r in silence_ids if int(r) in self.idx]] = True
            w[sil[pre]] = 0.0
        o = np.argsort(pre, kind="stable")
        self.indices = post[o]; self.data = w[o]
        self.indptr = np.zeros(N + 1, np.int64); np.add.at(self.indptr, pre + 1, 1); self.indptr = np.cumsum(self.indptr)
        self.n_syn = int(np.abs(con["Excitatory x Connectivity"].values).sum()); self.n_con = len(con)
        dt = P["dt"]; tm = P["t_mbr"]; tau = P["tau"]
        self.ev = np.float32(np.exp(-dt / tm)); self.eg = np.float32(np.exp(-dt / tau))
        self.c_vg = np.float32(tau / (tau - tm) * (np.exp(-dt / tau) - np.exp(-dt / tm)))
        self.D = int(round(P["t_dly"] / dt)); self.ref_steps = int(round(P["t_rfc"] / dt))
        self.w_poi = np.float32(P["w_syn"] * P["f_poi"])
        self.seed = seed
        self.reset()

    def reset(self):
        N = self.N
        self.v = np.full(N, P["v_0"], np.float32); self.g = np.zeros(N, np.float32)
        self.ref = np.zeros(N, np.int16); self.rfc_zero = np.zeros(N, np.bool_)
        self.ring = np.zeros((self.D, N), np.int32); self.ring_n = np.zeros(self.D, np.int64)
        self.t = 0
        self.rec_mask = np.zeros(N, np.bool_); self.rec_buf = np.zeros((2_000_000, 2), np.int64); self.rec_n = np.zeros(1, np.int64)

    def ix(self, root_ids):
        return np.array([self.idx[int(r)] for r in root_ids if int(r) in self.idx], np.int64)

    def run(self, ms, drive):
        """drive: list of (neuron_index_array, rate_Hz). Returns spike counts per neuron over the window."""
        idx_l, p_l = [], []
        for arr, rate in (drive.items() if isinstance(drive, dict) else drive):
            arr = np.asarray(arr, np.int64)
            idx_l.append(arr); p_l.append(np.full(len(arr), rate * P["dt"] / 1000.0))
            self.rfc_zero[arr] = True
        inp_idx = np.concatenate(idx_l) if idx_l else np.zeros(0, np.int64)
        inp_p = np.concatenate(p_l) if p_l else np.zeros(0)
        counts = np.zeros(self.N, np.int32)
        n = int(round(ms / P["dt"]))
        self.seed += 1
        self.t = _run(n, self.t, self.v, self.g, self.ref, self.rfc_zero, self.indptr, self.indices, self.data,
                      self.ring, self.ring_n, self.D, inp_idx, inp_p, self.w_poi, self.ev, self.c_vg, self.eg,
                      np.float32(P["v_0"]), np.float32(P["v_th"]), np.float32(P["v_rst"]), self.ref_steps,
                      counts, self.rec_mask, self.rec_buf, self.rec_n, self.seed)
        return counts


def annotations():
    a = pd.read_csv(ROOT / "ann/supplemental_files/Supplemental_file1_neuron_annotations.tsv", sep="\t", low_memory=False)
    return a
