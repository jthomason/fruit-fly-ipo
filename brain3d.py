"""3D brain renderer for the video. Real FlyWire neuron positions (pos_x, pos_y, pos_z), real spikes, real synapses.

- The point cloud (127,400 neurons) rotates slowly so the depth reads; each neuron is tinted by its FlyWire super class.
- Spiking neurons glow in a colour for their class (sugar GRN, bitter GRN, interneuron, descending, motor ...).
- Synapses between neurons that spike during the run (588 neurons, 13,861 connections with >= 3 synapses) light up
  when the presynaptic neuron fires. In slow-motion replays, pulses travel along them, in real spike order.
"""
import numpy as np, cv2, json

class Brain3D:
    def __init__(self, R, BW, BH, idx):
        p = np.load(R + "results/pos630_3d.npz", allow_pickle=True)
        X, Y, Z = p["x"].astype(float), p["y"].astype(float), p["z"].astype(float) * 10.0   # z voxels are 40 nm, x/y 4 nm
        for k in ("mn9", "dn", "sug", "bit"):            # fill the few missing coordinates by mirroring a partner
            v = idx[k]; miss = v[np.isnan(X[v])]; okv = v[~np.isnan(X[v])]
            for j, m in enumerate(miss):
                s = okv[j % len(okv)]; X[m] = 2 * 126000 - X[s]; Y[m] = Y[s]; Z[m] = Z[s]
        self.X, self.Y, self.Z = X, Y, Z; self.OK = ~np.isnan(X); self.N = len(X)
        self.BW, self.BH = BW, BH; self.idx = idx
        sc = p["sc"]; self.sc = sc; self.nt = p["nt"]
        tint = {"optic": (45, 70, 150), "central": (120, 80, 170), "sensory": (40, 140, 140), "visual_projection": (60, 95, 175),
                "descending": (170, 60, 110), "ascending": (150, 120, 60), "motor": (60, 160, 110), "sensory_ascending": (60, 140, 120),
                "visual_centrifugal": (90, 80, 170), "endocrine": (170, 120, 60)}
        self.tint = np.array([tint.get(s, (70, 80, 120)) for s in sc], np.float32) / 255
        spk = {"central": (255, 205, 90), "descending": (255, 90, 210), "motor": (90, 235, 255), "ascending": (255, 150, 60),
               "sensory": (255, 240, 200), "endocrine": (255, 150, 60)}
        self.spk = np.array([spk.get(s, (255, 230, 180)) for s in sc], np.float32) / 255
        self.spk[idx["sug"]] = np.array((60, 255, 130)) / 255; self.spk[idx["bit"]] = np.array((255, 70, 80)) / 255
        self.spk[idx["mn9"]] = np.array((120, 255, 200)) / 255; self.spk[idx["dn"]] = np.array((255, 120, 120)) / 255
        self.ACT = np.zeros(self.N, np.float32)
        e = np.load(R + "results/active_edges.npz")
        ok = self.OK[e["pre"]] & self.OK[e["post"]]
        self.epre, self.epost, self.ew = e["pre"][ok], e["post"][ok], e["w"][ok]
        o = np.argsort(self.epre, kind="stable"); self.epre, self.epost, self.ew = self.epre[o], self.epost[o], self.ew[o]
        self.eptr = {}
        for i, pr in enumerate(self.epre): self.eptr.setdefault(int(pr), []).append(i)
        self.dop = np.nonzero((self.nt == "dopamine") & self.OK)[0]
        a = e["act"]; a = a[self.OK[a]]
        self.SEZC = (float(np.median(X[a])), float(np.median(Y[a])), float(np.median(Z[a])))
        self.WC = (126500.0, 62000.0, float(np.nanmedian(Z)))
        self.pulses = []        # (pre, post, t_birth, strength)
        self.reveal = np.random.default_rng(0).random(self.N)
        self.ras = None

    # ---------------------------------------------------------------- views
    def view(self, kind, yaw, zoom_u=0.0):
        """kind blend: zoom_u 0 = whole brain, 1 = SEZ."""
        u = zoom_u
        c = [self.WC[i] + (self.SEZC[i] - self.WC[i]) * u for i in range(3)]
        s0 = self.BW / 200000; s1 = self.BW / 200000 * 2.7
        return dict(cx=c[0], cy=c[1], cz=c[2], s=s0 * (s1 / s0) ** u, yaw=yaw, zu=u)

    def proj(self, v, ii=None):
        X = self.X if ii is None else self.X[ii]; Y = self.Y if ii is None else self.Y[ii]; Z = self.Z if ii is None else self.Z[ii]
        ca, sa = np.cos(v["yaw"]), np.sin(v["yaw"])
        x = (X - v["cx"]) * ca + (Z - v["cz"]) * sa; z = -(X - v["cx"]) * sa + (Z - v["cz"]) * ca
        y = Y - v["cy"]
        pers = 1.0 / (1.0 + z * 2.2e-6)
        return x * v["s"] * pers + self.BW / 2, y * v["s"] * pers + self.BH / 2, z

    def _splat3(self, X, Y, cols, w):
        BW, BH = self.BW, self.BH
        m = (X >= 0) & (X < BW - 1) & (Y >= 0) & (Y < BH - 1)
        fi = Y[m].astype(np.int64) * BW + X[m].astype(np.int64)
        out = np.zeros((BH, BW, 3), np.float32)
        for c in range(3):
            out[..., c] = np.bincount(fi, weights=cols[m, c] * w[m], minlength=BW * BH).reshape(BH, BW)
        return out

    # ---------------------------------------------------------------- activity input
    def add_counts(self, sp, gain=1.0):
        i, c = sp; self.ACT[i] += c.astype(np.float32) * gain

    def load_raster(self, R):
        z = np.load(R + "results/full_raster.npz"); self.ras = (z["step"], z["neu"])
        self.daywin = {int(k): v for k, v in json.load(open(R + "results/daywin.json")).items()}

    def day_spikes(self, dd):
        a, b = self.daywin[dd]; st, ne = self.ras; m = (st >= a) & (st < b)
        return (st[m] - a) / 10.0, ne[m]          # ms, neuron

    def play(self, sp_ms, sp_n, ms_a, ms_b, t_now, gain=1.6, pulses=True):
        m = (sp_ms >= ms_a) & (sp_ms < ms_b)
        n = sp_n[m]
        np.add.at(self.ACT, n, gain)
        if pulses:
            for pr in np.unique(n):
                es = self.eptr.get(int(pr), [])
                if not es: continue
                es = sorted(es, key=lambda i: -self.ew[i])[:5]
                for i in es:
                    if self.ew[i] > 0: self.pulses.append((int(pr), int(self.epost[i]), t_now, 1.0))

    # ---------------------------------------------------------------- frame
    def frame(self, v, t, decay=0.6, reveal=1.0, dop_amt=0.0, edge_gain=1.0, pulse_life=0.45):
        BW, BH = self.BW, self.BH
        X, Y, Z = self.proj(v)
        ok = self.OK & (self.reveal < reveal)
        zr = np.clip((Z + 60000) / 120000, 0, 1)
        depth = (1.15 - 0.75 * zr).astype(np.float32)
        base = self._splat3(X[ok], Y[ok], self.tint[ok], depth[ok])
        sc = v["s"] / (BW / 200000)
        base = cv2.GaussianBlur(base, (0, 0), 0.7 + 0.35 * sc)
        lum = base.sum(2, keepdims=True)
        rgb = base / np.maximum(lum, 1e-6) * np.log1p(lum * 3.0 / sc ** 1.15) * 0.62
        # dopamine neurons
        if dop_amt > 0:
            dd = self.dop; Xd, Yd, _ = self.proj(v, dd)
            im = self._splat3(Xd, Yd, np.tile(np.array([[1.0, 0.25, 0.85]], np.float32), (len(dd), 1)), np.ones(len(dd), np.float32))
            rgb += np.clip(cv2.GaussianBlur(im, (0, 0), 1.2 * max(1, sc)) * 3.0, 0, 1) * dop_amt
        # synapses of active neurons
        act = np.minimum(self.ACT, 40)
        ov = np.zeros((BH, BW, 3), np.float32)
        hot = np.nonzero(act[self.epre] > 0.25)[0]
        if len(hot) and edge_gain > 0:
            inten = np.minimum(act[self.epre[hot]] / 4, 1) * np.minimum(np.abs(self.ew[hot]) / 12, 1)
            hot = hot[np.argsort(-inten)[:450]]; inten = np.minimum(act[self.epre[hot]] / 4, 1) * np.minimum(np.abs(self.ew[hot]) / 12, 1)
            px0, py0, _ = self.proj(v, self.epre[hot]); px1, py1, _ = self.proj(v, self.epost[hot])
            for k in range(len(hot)):
                c = self.spk[self.epre[hot[k]]] * float(inten[k]) * 0.30 * edge_gain
                if self.ew[hot[k]] < 0: c = np.array([0.35, 0.45, 1.0], np.float32) * float(inten[k]) * 0.22 * edge_gain
                cv2.line(ov, (int(px0[k] * 4), int(py0[k] * 4)), (int(px1[k] * 4), int(py1[k] * 4)), tuple(float(x) for x in c), 1, cv2.LINE_AA, 2)
        # travelling pulses
        if self.pulses:
            keep = []; P = []
            for pr, po, tb, s in self.pulses:
                f = (t - tb) / pulse_life
                if 0 <= f <= 1: keep.append((pr, po, tb, s)); P.append((pr, po, f))
                elif f < 0: keep.append((pr, po, tb, s))
            self.pulses = keep[-6000:]
            if P:
                pr = np.array([p[0] for p in P]); po = np.array([p[1] for p in P]); f = np.array([p[2] for p in P])
                a0, b0, _ = self.proj(v, pr); a1, b1, _ = self.proj(v, po)
                xs = a0 + (a1 - a0) * f; ys = b0 + (b1 - b0) * f
                im = self._splat3(xs, ys, self.spk[pr], (1 - f) * 0 + 1.0)
                pl = np.clip(cv2.GaussianBlur(im, (0, 0), 1.6) * 6, 0, 1.4) + np.clip(cv2.GaussianBlur(im, (0, 0), 5) * 30, 0, 1) * 0.5
            else: pl = 0
        else: pl = 0
        rgb += pl
        rgb += cv2.GaussianBlur(ov, (0, 0), 0.6) + cv2.GaussianBlur(ov, (0, 0), 3.0) * 0.5
        # spiking neurons
        nz = np.nonzero(act > 0.05)[0]
        if len(nz):
            Xa, Ya, _ = self.proj(v, nz)
            im = self._splat3(Xa, Ya, self.spk[nz], np.log1p(act[nz]) * 1.2)
            s2 = max(1.0, sc ** 0.8)
            core = cv2.GaussianBlur(im, (0, 0), 1.1 * s2); glow = cv2.GaussianBlur(im, (0, 0), 5.5 * s2)
            rgb += np.clip(core * 2.6 * s2, 0, 1.3) + np.clip(glow * 16 * s2, 0, 1) * 0.6
        self.ACT *= decay
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def screen(self, v, ii):
        X, Y, _ = self.proj(v, np.asarray(ii)); return float(np.mean(X)), float(np.mean(Y))
