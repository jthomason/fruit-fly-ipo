"""Render the real-data video (1920x1080, 30 fps). Every neuron flash, spike, trade and dollar is from results/*.pkl."""
import numpy as np, pickle, json, glob, math, subprocess, wave, sys, re, os
import cv2
from PIL import Image, ImageDraw, ImageFont

R = os.path.dirname(os.path.abspath(__file__)) + "/"
W, H, FPS = 1920, 1080, 30
# ------------------------------------------------------------------ data
hero = pickle.load(open(R + "results/read_normal.pkl", "rb"))
trd = pickle.load(open(R + "results/fund_normal.pkl", "rb"))
dop = pickle.load(open(R + "results/fund_dopamine_off.pkl", "rb"))
mc = sum([pickle.load(open(f, "rb")) for f in sorted(glob.glob(R + "results/fund_mc_shard*.pkl"))], [])
mc = sorted(mc, key=lambda d: d["q"])
chunks = hero["read"]["chunks"]; L = trd["trade"]["ledger"]; px = trd["px"]; LD = dop["trade"]["ledger"]
idx = trd["idx"]; raster = trd["raster"]
pos = np.load(R + "results/pos630.npz"); NX, NY = pos["x"].astype(float), pos["y"].astype(float)
for k in ("mn9", "dn", "sug", "bit"):   # fill the few missing coordinates by mirroring the partner neuron
    v = idx[k]; miss = v[np.isnan(NX[v])]; okv = v[~np.isnan(NX[v])]
    for j, m in enumerate(miss):
        src = okv[j % len(okv)]; NX[m] = 2 * 126000 - NX[src]; NY[m] = NY[src]
OK = ~np.isnan(NX)
import pandas as pd, pysentiment2, os
words = re.findall(r"[A-Za-z][A-Za-z'’\-]*", open(R + "risk_factors.txt").read())
lm = pd.read_csv(os.path.dirname(pysentiment2.__file__) + "/static/LM.csv")
NEG = set(lm.Word[lm.Negative > 0].str.lower()); POS = set(lm.Word[lm.Positive > 0].str.lower())
cum_neg = np.cumsum([c["neg"] for c in chunks]); cum_pos = np.cumsum([c["pos"] for c in chunks])
cum_b = np.cumsum([c["b"] for c in chunks]); cum_s = np.cumsum([c["s"] for c in chunks])
NAV = np.array([10000.0] + [e["nav"] for e in L]); NAVD = np.array([10000.0] + [e["nav"] for e in LD])
bh_sh = 10000 // px[0]; BH = bh_sh * px + (10000 - bh_sh * px[0])
NCON = 14_687_178
FAIR = trd["fair_px"]; SURP = trd["surp"]; EDAYS = [70, 122, 186, 250]

# ------------------------------------------------------------------ narration timing
VO = R + "vo/"
def wav_dur(p):
    with wave.open(p) as w: return w.getnframes() / w.getframerate()
SEG = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
LEAD, GAP, TAIL = 0.3, 0.22, 3.6
starts, durs = {}, {}; t = LEAD
for s in SEG:
    d = wav_dur(VO + f"p{s}.wav"); starts[s] = t; durs[s] = d; t += d + GAP
    if s in (3, 5, 9): t += 0.25
END = t - GAP; TOTAL = END + TAIL; NF = int(TOTAL * FPS)
json.dump(dict(starts=starts, durs=durs, total=TOTAL, end=END), open(R + "vo/timing.json", "w"))
def S0(s): return starts[s]
def S1(s): return starts[s] + durs[s]

CAP = {1: "I wired a real fruit fly brain to a Monte Carlo simulation of Anthropic's IPO. Then I let it trade.",
       2: "This is the FlyWire connectome. 127,000 real neurons, 15 million connections, simulated spike by spike.",
       3: "It never sees the stock price. It values the company only from the numbers reported so far: revenue, growth, margins. Below its fair value tastes like sugar, so it buys. Above it tastes bitter, so it sells. An earnings miss? Bitter.",
       4: "First, due diligence. I fed it all 94 pages of risk factors. 69,000 words, one every millisecond.",
       5: "Every negative word, like catastrophic, harmful or shutdown, hit its bitter neurons. 794 sell spikes, 404 buys. So it passed on the IPO.",
       6: "Its fair value was $84. The stock priced at $94 and ran to $131. Way too bitter. The fly sat in cash for 83 days.",
       7: "Day 84, the stock drops below fair value. Sugar. By day 93, it's all in, at an average of $66.",
       8: "Then the stock fell to $45. Four earnings reports, four misses. It never sold a share.",
       9: "After one year: buy and hold, $8,047. The fly: $11,517.",
       10: open(VO + "cap10.txt").read().strip(),
       11: "Then I turned off all its dopamine neurons. It made $36 less.",
       12: "Fruit flies live about 50 days. This one would have died in cash, waiting for a better price. The code is on GitHub. The fly is not a registered investment adviser."}
stt = json.load(open(VO + "stt_out.json"))
def groups(text, maxw=9):
    out = []
    for sent in re.split(r"(?<=[.!?:])\s+", text):
        ws = sent.split()
        while len(ws) > maxw:
            cut = maxw
            for j in range(min(maxw, len(ws) - 1), 3, -1):
                if ws[j - 1].endswith(","): cut = j; break
            out.append(" ".join(ws[:cut])); ws = ws[cut:]
        if ws: out.append(" ".join(ws))
    return out
CUES = []
for s in SEG:
    gs = groups(CAP[s]); ws = stt.get(f"p{s}.wav")
    t0 = ws[0][1] if ws else 0; t1 = ws[-1][2] if ws else durs[s]
    lens = np.array([len(g) + 6 for g in gs], float); edges = np.concatenate([[0], np.cumsum(lens) / lens.sum()])
    for k, g in enumerate(gs):
        CUES.append((S0(s) + t0 + edges[k] * (t1 - t0) - 0.05, S0(s) + t0 + edges[k + 1] * (t1 - t0) + 0.05, g))

# ------------------------------------------------------------------ fonts / colors
MB = "/usr/local/share/fonts/asi-document/Montserrat-Black.ttf"
g_ = glob.glob("/usr/local/share/fonts/**/Montserrat-Bold.ttf", recursive=True); MBOLD = g_[0] if g_ else MB
g_ = glob.glob("/usr/local/share/fonts/**/Montserrat-SemiBold.ttf", recursive=True); MSEMI = g_[0] if g_ else MBOLD
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"; MONOR = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
_fc = {}
def F(p, s):
    k = (p, s)
    if k not in _fc: _fc[k] = ImageFont.truetype(p, s)
    return _fc[k]
BG = (7, 9, 16); PANEL = (14, 18, 30); EDGE = (40, 48, 70); GOLD = (255, 196, 40); RED = (255, 64, 72)
GREEN = (60, 230, 130); CYAN = (70, 210, 255); WHITE = (240, 242, 248); GREY = (130, 140, 165); DIM = (70, 78, 100)
def ease(x): x = min(max(x, 0), 1); return x * x * (3 - 2 * x)
def lin(t, a, b): return min(max((t - a) / max(b - a, 1e-6), 0), 1)
def st(d, xy, txt, f, fill=WHITE, anchor="mm", sw=0, sf=(0, 0, 0)):
    d.text(xy, txt, font=f, fill=fill, anchor=anchor, stroke_width=sw, stroke_fill=sf)
def tw(txt, f): return f.getbbox(txt)[2] - f.getbbox(txt)[0]
def box(d, xy, fill=PANEL, outline=EDGE, r=14, w=2): d.rounded_rectangle(xy, r, fill=fill, outline=outline, width=w)

# ------------------------------------------------------------------ brain renderer
BXY = (24, 104, 1004, 654)            # left panel
BW, BH_ = BXY[2] - BXY[0], BXY[3] - BXY[1]
FULL = dict(cx=126500.0, cy=60000.0, s=BW / 196000)
SEZ = dict(cx=126000.0, cy=84000.0, s=BW / 196000 * 2.6)
def lerpv(a, b, u): return {k: a[k] + (b[k] - a[k]) * u for k in a}
def proj(view, ii=None):
    x = NX if ii is None else NX[ii]; y = NY if ii is None else NY[ii]
    return (x - view["cx"]) * view["s"] + BW / 2, (y - view["cy"]) * view["s"] + BH_ / 2
def splat(img, X, Y, w):
    m = (X >= 0) & (X < img.shape[1] - 1) & (Y >= 0) & (Y < img.shape[0] - 1)
    np.add.at(img, (Y[m].astype(int), X[m].astype(int)), w[m])
_basecache = {}
def base_img(view):
    key = tuple(round(view[k], 3) for k in ("cx", "cy", "s"))
    if key in _basecache: return _basecache[key]
    X, Y = proj(view); im = np.zeros((BH_, BW), np.float32); splat(im, X[OK], Y[OK], np.ones(OK.sum(), np.float32))
    sc = view["s"] / FULL["s"]
    im = cv2.GaussianBlur(im, (0, 0), 0.8 + 0.4 * sc); im = np.log1p(im * 3 / sc ** 2); im = im / max(np.percentile(im, 99.7), 1e-6)
    if len(_basecache) > 200: _basecache.clear()
    _basecache[key] = im; return im
ACT = np.zeros(len(NX), np.float32)
def add_counts(sp, gain=1.0):
    i, c = sp; ACT[i] += c.astype(np.float32) * gain
def brain_frame(view, decay=0.55, hl=None, hl_amt=0.0):
    """returns RGB uint8 BH_xBW image."""
    global ACT
    b = base_img(view)
    rgb = np.zeros((BH_, BW, 3), np.float32)
    rgb += b[..., None] * np.array([38, 52, 92], np.float32)[None, None] / 255 * 1.4
    act = np.minimum(ACT, 60)
    nz = np.nonzero(act > 0.05)[0]
    if len(nz):
        X, Y = proj(view, nz); im = np.zeros((BH_, BW), np.float32); splat(im, X, Y, np.log1p(act[nz]) * 1.2)
        sc = max(1.0, view["s"] / FULL["s"])
        core = cv2.GaussianBlur(im, (0, 0), 1.0 * sc); glow = cv2.GaussianBlur(im, (0, 0), 5.0 * sc)
        a = np.clip(core * 2.4 * sc, 0, 1.2); gl = np.clip(glow * 14 * sc, 0, 1)
        rgb += a[..., None] * np.array([1.0, 0.92, 0.62])[None, None] + gl[..., None] * np.array([1.0, 0.55, 0.15])[None, None] * 0.55
    if hl is not None and hl_amt > 0:
        for ii, col, rad in hl:
            X, Y = proj(view, ii); im = np.zeros((BH_, BW), np.float32); splat(im, X, Y, np.ones(len(ii), np.float32))
            gl = cv2.GaussianBlur(im, (0, 0), rad); gl = np.clip(gl / max(gl.max(), 1e-6), 0, 1) * hl_amt
            rgb += gl[..., None] * (np.array(col, np.float32) / 255)[None, None]
    ACT *= decay
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

# ------------------------------------------------------------------ timelines
TR0, TR1 = S0(6), S1(9) - 0.6          # trading playback window
KEY = [(S0(6), 0.0), (S1(6), 83.0), (S0(7) + 0.15 * durs[7], 84.0), (S1(7), 95.0), (S1(8), 250.0), (S0(9) + 1.0, 252.0)]
def day_at(t):
    if t <= KEY[0][0]: return 0.0
    for (a, da), (b, db) in zip(KEY[:-1], KEY[1:]):
        if t <= b: return da + (db - da) * ease(lin(t, a, b)) if db - da > 60 else da + (db - da) * lin(t, a, b)
    return 252.0
RD0, RD1 = S0(4) + 0.4, S0(5) + 0.58 * durs[5]  # reading playback window
def chunk_at(t): return int(np.clip(lin(t, RD0, RD1) * len(chunks), 0, len(chunks)))
from brain3d import Brain3D
B3 = Brain3D(R, BW, BH_, idx); B3.load_raster(R)
SMS = [(S0(3) + 0.55 * durs[3], S0(3) + 0.75 * durs[3], 86, "sugar"), (S0(3) + 0.75 * durs[3], S0(4) - 0.1, 22, "bitter"),
       (S0(6) + 0.62 * durs[6], S1(6), 60, "day"), (S0(7) + 0.02, S0(7) + 0.5 * durs[7], 84, "day")]
_dsp = {}
def dsp(dd):
    if dd not in _dsp: _dsp[dd] = B3.day_spikes(dd)
    return _dsp[dd]
LOOP = [dd for dd in range(84, 252) if L[dd - 1]["tot"] > 600]
SCN = {"optic": 59374, "central": 28690, "sensory": 8126, "visual projection": 6335, "descending": 1086, "motor": 85}

# ------------------------------------------------------------------ panels
def hud(d, t, phase, sub=""):
    d.rectangle((0, 0, W, 86), fill=(10, 13, 22)); d.line((0, 86, W, 86), fill=EDGE, width=2)
    st(d, (28, 30), "FRUIT FLY  ×  $ANT IPO", F(MB, 30), anchor="lm")
    st(d, (28, 64), f"FlyWire v630 · 127,400 neurons · {NCON/1e6:.1f}M connections · LIF model (Shiu et al., Nature 2024)", F(MONOR, 19), fill=GREY, anchor="lm")
    blink = (int(t * 2) % 2 == 0)
    d.ellipse((W - 250, 22, W - 234, 38), fill=RED if blink else (110, 30, 35))
    st(d, (W - 224, 30), "REAL SPIKES", F(MB, 24), anchor="lm")
    st(d, (W - 28, 64), phase, F(MONO, 20), fill=GOLD, anchor="rm")

def chart(d, t, x0, y0, x1, y1, day, show_nav=True):
    box(d, (x0, y0, x1, y1))
    st(d, (x0 + 22, y0 + 26), "$ANT share price vs the fly's fair value · path #%d (median 1-yr outcome)" % trd["hero"], F(MONOR, 18), fill=GREY, anchor="lm")
    cx0, cx1, cy0, cy1 = x0 + 80, x1 - 24, y0 + 56, y0 + int((y1 - y0) * (0.62 if show_nav else 0.9))
    lo, hi = 40, 140
    def Y(p): return cy1 - (p - lo) / (hi - lo) * (cy1 - cy0)
    def X(dd): return cx0 + dd / 252 * (cx1 - cx0)
    for p in (50, 75, 100, 125):
        d.line((cx0, Y(p), cx1, Y(p)), fill=(28, 34, 52), width=1); st(d, (cx0 - 10, Y(p)), f"${p}", F(MONOR, 17), fill=DIM, anchor="rm")
    d.line((cx0, Y(px[0]), cx1, Y(px[0])), fill=(120, 120, 140), width=1)
    st(d, (cx1, Y(px[0]) - 12), f"IPO ${px[0]:.2f}", F(MONO, 16), fill=GREY, anchor="rm")
    xx = X(50); col = (150, 120, 200)
    for yy in range(int(cy0), int(cy1), 10): d.line((xx, yy, xx, yy + 5), fill=tuple(c // 2 for c in col), width=2)
    st(d, (xx + 6, cy0 + 10), "day 50: avg fly lifespan", F(MONO, 15), fill=col, anchor="lm")
    for k, ed in enumerate(EDAYS):
        if day >= ed:
            xx = X(ed); cc = GREEN if SURP[k] > 0 else RED
            d.line((xx, cy1 - 6, xx, cy1 + 6), fill=cc, width=3)
            st(d, (xx, cy1 - 18), f"{['Q4\'26', 'Q1\'27', 'Q2\'27', 'Q3\'27'][k]} {SURP[k]*100:+.1f}%", F(MONO, 15), fill=cc)
    nf = int(min(day, 252))
    for i in range(0, nf, 2): d.line((X(i), Y(FAIR[i]), X(min(i + 1, 252)), Y(FAIR[min(i + 1, 252)])), fill=CYAN, width=3)
    st(d, (cx0 + 8, Y(FAIR[0]) + 16), f"fly's fair value ${int(FAIR[0])}", F(MONO, 16), fill=CYAN, anchor="lm")
    n = int(day)
    if n >= 1:
        pts = [(X(i), Y(px[i])) for i in range(n + 1)]
        frac = day - n
        if n < 252: pts.append((X(n + frac), Y(px[n] + (px[n + 1] - px[n]) * frac)))
        d.line(pts, fill=WHITE, width=3)
        for e in L[:n]:
            if e["q"] > 0:
                r_ = 4 + min(e["q"], 18) * 0.55; xx, yy = X(e["d"]), Y(e["px"]) + 14
                d.polygon([(xx, yy - r_), (xx - r_, yy + r_), (xx + r_, yy + r_)], fill=GREEN)
            elif e["q"] < 0:
                r_ = 4 + min(-e["q"], 18) * 0.55; xx, yy = X(e["d"]), Y(e["px"]) - 14
                d.polygon([(xx, yy + r_), (xx - r_, yy - r_), (xx + r_, yy - r_)], fill=RED)
        xx, yy = pts[-1]; d.ellipse((xx - 7, yy - 7, xx + 7, yy + 7), fill=GOLD)
    if show_nav:
        ny0, ny1 = cy1 + 46, y1 - 22
        st(d, (x0 + 22, cy1 + 24), "Portfolio value, starting with $10,000", F(MONOR, 18), fill=GREY, anchor="lm")
        lo2, hi2 = 4000, 14000
        def Y2(v): return ny1 - (v - lo2) / (hi2 - lo2) * (ny1 - ny0)
        for v in (5000, 10000):
            d.line((cx0, Y2(v), cx1, Y2(v)), fill=(28, 34, 52)); st(d, (cx0 - 10, Y2(v)), f"${v//1000}k", F(MONOR, 16), fill=DIM, anchor="rm")
        if n >= 1:
            d.line([(X(i), Y2(BH[i])) for i in range(n + 1)], fill=(120, 140, 200), width=2)
            d.line([(X(i), Y2(NAV[i])) for i in range(n + 1)], fill=GOLD, width=3)
            st(d, (cx1, cy1 + 24), f"buy & hold ${BH[n]:,.0f}", F(MONO, 18), fill=(140, 160, 220), anchor="rm")
            st(d, (cx1 - 250, cy1 + 24), f"fly ${NAV[n]:,.0f}", F(MONO, 18), fill=GOLD, anchor="rm")

def raster_panel(d, t, day, x0, y0, x1, y1):
    box(d, (x0, y0, x1, y1))
    rows = [("SUGAR GRNs", idx["sug"], GREEN), ("BITTER GRNs", idx["bit"], RED), ("MN9 → BUY", idx["mn9"], GREEN), ("DNge067 → SELL", idx["dn"], RED)]
    lab_w = 200; gx0, gx1 = x0 + lab_w, x1 - 16; gy = y0 + 16
    span = 8.0; dhi = day; dlo = dhi - span
    st(d, (x1 - 16, y1 - 14), f"spike raster · last {int(span)} trading days ({int(span*100)} ms of fly time)", F(MONOR, 14), fill=DIM, anchor="rd")
    heights = [62, 62, 22, 22]; yy = gy
    stepd = 1000.0
    tsel = (raster[:, 0] >= dlo * stepd) & (raster[:, 0] < dhi * stepd)
    rr = raster[tsel]
    for (lab, ii, col), hgt in zip(rows, heights):
        st(d, (x0 + 16, yy + hgt / 2), lab, F(MBOLD, 17), fill=col, anchor="lm")
        d.rectangle((gx0, yy, gx1, yy + hgt), fill=(10, 13, 22))
        m = np.isin(rr[:, 1], ii)
        if m.any():
            ti = rr[m, 0] / stepd; row = np.searchsorted(np.sort(ii), rr[m, 1]); nrow = len(ii)
            xs = gx0 + (ti - dlo) / span * (gx1 - gx0); ys = yy + (row + 0.5) / nrow * hgt
            for xv, yv in zip(xs, ys): d.line((xv, yv - max(1.2, hgt / nrow / 2), xv, yv + max(1.2, hgt / nrow / 2)), fill=col, width=2)
        yy += hgt + 10
    for k in range(int(math.ceil(dlo)), int(dhi) + 1):
        xv = gx0 + (k - dlo) / span * (gx1 - gx0)
        if gx0 <= xv <= gx1: d.line((xv, gy, xv, yy - 10), fill=(30, 36, 56), width=1)

def ticker_panel(d, t, ci, x0, y0, x1, y1):
    box(d, (x0, y0, x1, y1))
    st(d, (x0 + 22, y0 + 26), "ANTHROPIC, PBC · FORM S-1 · RISK FACTORS (pp. 27–120)", F(MONO, 18), fill=GREY, anchor="lm")
    wi = min(int(ci * 25), len(words))
    lines, cur, cw = [], [], 0; maxw = x1 - x0 - 50; f = F(MONOR, 22)
    k = max(0, wi - 170)
    while k < wi:
        w_ = words[k]; ww = tw(w_ + " ", f)
        if cw + ww > maxw: lines.append(cur); cur, cw = [], 0
        cur.append(w_); cw += ww; k += 1
    if cur: lines.append(cur)
    lines = lines[-15:]
    yy = y0 + 64
    for li, ln in enumerate(lines):
        xx = x0 + 24; fade = 0.35 + 0.65 * (li + 1) / len(lines)
        for w_ in ln:
            lw = w_.lower().replace("’", "'")
            col = RED if lw in NEG else (GREEN if lw in POS else tuple(int(c * fade) for c in (175, 182, 200)))
            ff = F(MONO, 22) if lw in NEG or lw in POS else f
            d.text((xx, yy), w_, font=ff, fill=col); xx += tw(w_ + " ", f)
        yy += 31
    d.line((x0 + 20, y1 - 70, x1 - 20, y1 - 70), fill=EDGE)
    c = max(ci - 1, 0)
    st(d, (x0 + 24, y1 - 38), f"words read {min(wi, len(words)):>6,}/{len(words):,}", F(MONO, 20), fill=WHITE, anchor="lm")
    st(d, (x1 - 24, y1 - 38), f"negative {cum_neg[c] if ci else 0:>5,}   positive {cum_pos[c] if ci else 0:>4,}", F(MONO, 20), fill=GREY, anchor="rm")

def counters(d, ci, x0, y0, x1, y1):
    box(d, (x0, y0, x1, y1))
    c = max(ci - 1, 0)
    sp_s = cum_s[c] if ci else 0; sp_b = cum_b[c] if ci else 0
    st(d, (x0 + (x1 - x0) * 0.27, y0 + 56), f"{sp_s:,}", F(MB, 66), fill=RED)
    st(d, (x0 + (x1 - x0) * 0.27, y0 + 112), "SELL SPIKES (DNge067)", F(MBOLD, 19), fill=RED)
    st(d, (x0 + (x1 - x0) * 0.73, y0 + 56), f"{sp_b:,}", F(MB, 66), fill=GREEN)
    st(d, (x0 + (x1 - x0) * 0.73, y0 + 112), "BUY SPIKES (MN9)", F(MBOLD, 19), fill=GREEN)
    st(d, ((x0 + x1) / 2, y1 - 22), "1 spike = 1 share · net spikes = order", F(MONOR, 16), fill=GREY)

def caption(d, t):
    for a, b, txt in CUES:
        if a <= t < b:
            f = F(MB, 40)
            ww = tw(txt, f)
            if ww > 1760: f = F(MB, int(40 * 1760 / ww))
            st(d, (W / 2, 1018), txt, f, fill=(255, 226, 90), sw=5)
            break

def stamp(img, t, t0, txt, col, sub=None, cx=W / 2, cy=H / 2 - 40, size=88):
    u = lin(t, t0, t0 + 0.25)
    if u <= 0: return
    sc = 1.6 - 0.6 * ease(u)
    f = F(MB, int(size * sc)); d = ImageDraw.Draw(img)
    w_ = tw(txt, f); h_ = int(size * sc * 1.25)
    d.rounded_rectangle((cx - w_ / 2 - 34, cy - h_ / 2 - 14, cx + w_ / 2 + 34, cy + h_ / 2 + (50 if sub else 14)), 18,
                        fill=(12, 14, 24), outline=col, width=5)
    st(d, (cx, cy), txt, f, fill=col)
    if sub: st(d, (cx, cy + h_ / 2 + 22), sub, F(MBOLD, 26), fill=WHITE)

# ------------------------------------------------------------------ MC + end panels
def mc_panel(d, t, x0, y0, x1, y1, u, ann=False, ann2=False):
    box(d, (x0, y0, x1, y1))
    st(d, (x0 + 22, y0 + 26), f"Fly vs buy-and-hold on {len(mc)} IPO paths (stratified across all 20,000)", F(MONOR, 18), fill=GREY, anchor="lm")
    cx0, cx1, cy0, cy1 = x0 + 90, x1 - 30, y0 + 70, y1 - 70
    lo, hi = 0, 30000
    def Y(v): return cy1 - (min(v, hi) - lo) / (hi - lo) * (cy1 - cy0)
    for v in (0, 10000, 20000, 30000):
        d.line((cx0, Y(v), cx1, Y(v)), fill=(28, 34, 52)); st(d, (cx0 - 10, Y(v)), f"${v//1000}k", F(MONOR, 16), fill=DIM, anchor="rm")
    n = len(mc); show = int(u * n + 0.999)
    bw = (cx1 - cx0) / n
    for k, m in enumerate(mc[:show]):
        xx = cx0 + (k + 0.5) * bw
        d.line((xx, Y(m["bh"]), xx, Y(m["nav"])), fill=(60, 66, 90), width=2)
        d.ellipse((xx - 5, Y(m["bh"]) - 5, xx + 5, Y(m["bh"]) + 5), fill=(120, 140, 200))
        c = GREEN if m["nav"] > m["bh"] else RED
        d.ellipse((xx - 7, Y(m["nav"]) - 7, xx + 7, Y(m["nav"]) + 7), fill=c)
    if ann:
        k = max(range(n), key=lambda j: mc[j]["nav"] - mc[j]["bh"]); m = mc[k]; xx = cx0 + (k + 0.5) * bw
        d.ellipse((xx - 13, Y(m["nav"]) - 13, xx + 13, Y(m["nav"]) + 13), outline=GOLD, width=3)
        st(d, (xx + 22, Y(m["nav"]) - 60), f"stock {m['ret']*100:+.0f}%", F(MBOLD, 24), fill=RED, anchor="lm")
        st(d, (xx + 22, Y(m["nav"]) - 30), f"fly {(m['nav']/10000-1)*100:+.0f}%", F(MBOLD, 24), fill=GREEN, anchor="lm")
    if ann2:
        k = max(range(n), key=lambda j: mc[j]["ret"]); m = mc[k]; xx = cx0 + (k + 0.5) * bw
        d.ellipse((xx - 13, Y(m["nav"]) - 13, xx + 13, Y(m["nav"]) + 13), outline=GOLD, width=3)
        st(d, (xx - 22, Y(m["bh"]) + 4), f"stock {m['ret']*100:+.0f}%", F(MBOLD, 24), fill=GREEN, anchor="rm")
        st(d, (xx - 22, Y(m["nav"]) + 40), f"fly {(m['nav']/10000-1)*100:+.0f}%", F(MBOLD, 24), fill=RED, anchor="rm")
    st(d, (cx0, cy1 + 28), "worst IPO path", F(MONOR, 16), fill=DIM, anchor="lm"); st(d, (cx1, cy1 + 28), "best IPO path", F(MONOR, 16), fill=DIM, anchor="rm")
    mx = (cx0 + cx1) / 2; yy = cy1 + 28
    d.ellipse((mx - 150, yy - 6, mx - 138, yy + 6), fill=GREEN); d.ellipse((mx - 132, yy - 6, mx - 120, yy + 6), fill=RED)
    st(d, (mx - 112, yy), "fly (beat / lost)", F(MONOR, 16), fill=GREY, anchor="lm")
    d.ellipse((mx + 95, yy - 5, mx + 105, yy + 5), fill=(120, 140, 200)); st(d, (mx + 113, yy), "buy & hold", F(MONOR, 16), fill=GREY, anchor="lm")

def dopamine_panel(d, t, x0, y0, x1, y1, u):
    box(d, (x0, y0, x1, y1))
    st(d, ((x0 + x1) / 2, y0 + 50), "DOPAMINE", F(MB, 44))
    on = u < 0.3
    sx, sy = (x0 + x1) / 2, y0 + 130
    d.rounded_rectangle((sx - 80, sy - 36, sx + 80, sy + 36), 36, fill=GREEN if on else (60, 60, 70))
    kx = sx + 44 if on else sx - 44; d.ellipse((kx - 30, sy - 30, kx + 30, sy + 30), fill=WHITE)
    st(d, ((x0 + x1) / 2, sy + 70), "ON" if on else "OFF · 5,670 dopaminergic neurons silenced", F(MBOLD, 24), fill=GREEN if on else RED)
    if u > 0.45:
        st(d, (x0 + (x1 - x0) * 0.28, y0 + 300), f"${NAV[-1]:,.0f}", F(MB, 58), fill=GOLD); st(d, (x0 + (x1 - x0) * 0.28, y0 + 350), "dopamine on", F(MBOLD, 22), fill=GREY)
        st(d, (x0 + (x1 - x0) * 0.72, y0 + 300), f"${NAVD[-1]:,.0f}", F(MB, 58), fill=RED); st(d, (x0 + (x1 - x0) * 0.72, y0 + 350), "dopamine off", F(MBOLD, 22), fill=GREY)
        st(d, ((x0 + x1) / 2, y0 + 420), f"buys {sum(e['b'] for e in L):,} → {sum(e['b'] for e in LD):,} spikes · sells {sum(e['s'] for e in L)} → {sum(e['s'] for e in LD)}", F(MONO, 20), fill=GREY)

# ------------------------------------------------------------------ frame
REPO = open(R + "repo_url.txt").read().strip() if os.path.exists(R + "repo_url.txt") else "github.com/…/fruitfly-ipo"
def frame(fi):
    t = fi / FPS
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    # ---------- brain (3D, real positions, real spikes, real synapses)
    yaw = 0.25 + 0.55 * math.sin(2 * math.pi * t / 26)
    day = day_at(t); ci = chunk_at(t); reveal = 1.0; dop = 0.0; decay = 0.6; smi = None
    for k_, (ta, tb, dd_, kind) in enumerate(SMS):
        if ta <= t < tb: smi = k_
    def loop_days(rate=6.0, gain=1.0):
        B3.add_counts(L[LOOP[int(t * rate) % len(LOOP)] - 1]["sp"], gain)
    if t < S0(2):
        zu = 0.12; loop_days()
    elif t < S0(3):
        zu = 0.0; reveal = ease(lin(t, S0(2), S0(2) + 3.0)) * 1.0001
        if t > S0(2) + 3.2: loop_days(gain=0.8)
    elif t < S0(4):
        zu = 0.12 + 0.68 * ease(lin(t, S0(3) - 0.3, S0(3) + 1.2))
        if smi is None: loop_days()
    elif t < S0(6):
        zu = 0.8
        prev = getattr(frame, "ci", 0)
        for c in chunks[prev:ci]: B3.add_counts(c["sp"], 0.6)
        frame.ci = ci
    elif t < S1(9) + 0.25:
        zu = 0.8 - 0.3 * ease(lin(t, S0(6), S0(6) + 1.2))
        prevd = getattr(frame, "dd", 0); nd = int(min(day, 252))
        if nd > prevd and smi is None:
            for e in L[prevd:nd]: B3.add_counts(e["sp"], 1.0 / max(1, (nd - prevd) ** 0.5))
        frame.dd = max(prevd, nd)
    else:
        zu = 0.5 - 0.4 * ease(lin(t, S1(9) + 0.25, S1(9) + 1.5)); loop_days(gain=0.8)
        if S0(11) <= t < S0(12): dop = 1.0 - ease(lin(t, S0(11) + 0.3 * durs[11], S0(11) + 0.42 * durs[11]))
    smtxt = None
    if smi is not None:
        ta, tb, dd_, kind = SMS[smi]; ms_, n_ = dsp(dd_); dl = 100.0
        su = ease(lin(t, ta, ta + 0.35)) * (1 - ease(lin(t, tb - 0.25, tb)))
        zu = zu + (0.85 - zu) * su
        m0 = lin(t - 1 / FPS, ta, tb) * dl if t - 1 / FPS >= ta else -1; m1 = lin(t, ta, tb) * dl
        if m0 < 0: B3.ACT[:] *= 0.2; B3.pulses = []
        B3.play(ms_, n_, max(m0, 0), m1, t); decay = 0.84
        sofar = n_[ms_ < m1]
        smtxt = (dd_, kind, m1, len(np.unique(sofar)), int(np.isin(sofar, idx["mn9"]).sum()), int(np.isin(sofar, idx["dn"]).sum()))
    bv = B3.view("w", yaw, zu)
    bimg = B3.frame(bv, t, decay=decay, reveal=reveal, dop_amt=dop)
    img.paste(Image.fromarray(bimg), (BXY[0], BXY[1])); d.rectangle(BXY, outline=EDGE, width=2)
    lab = "SUBESOPHAGEAL ZONE · taste + feeding · 3D" if zu > 0.4 else "WHOLE BRAIN · 3D · real neuron positions"
    st(d, (BXY[0] + 16, BXY[1] + 22), lab, F(MONO, 17), fill=GREY, anchor="lm")
    if zu > 0.3:
        for key_, txt_, col_ in (("mn9", "MN9 → BUY", GREEN), ("dn", "DNge067 → SELL", RED)):
            a_ = float(B3.ACT[idx[key_]].max()) / B3.ACT.dtype.type(1)
            if a_ > 0.3:
                sx, sy = B3.screen(bv, idx[key_]); sx += BXY[0]; sy += BXY[1]
                if BXY[0] + 30 < sx < BXY[2] - 30 and BXY[1] + 30 < sy < BXY[3] - 30:
                    rr = 16 + 6 * min(a_, 3)
                    d.ellipse((sx - rr, sy - rr, sx + rr, sy + rr), outline=col_, width=3)
                    st(d, (sx + rr + 8, sy - rr - 4), txt_, F(MBOLD, 22), fill=col_, anchor="lm", sw=3, sf=BG)
    if smtxt:
        dd_, kind, m1, nn, nb, ns = smtxt
        ta, tb = SMS[smi][0], SMS[smi][1]
        d.ellipse((BXY[2] - 236, BXY[1] + 48, BXY[2] - 222, BXY[1] + 62), fill=RED if int(t * 3) % 2 == 0 else (110, 30, 35))
        st(d, (BXY[2] - 212, BXY[1] + 56), f"SLOW MOTION ×{(tb - ta) / 0.1:.0f}", F(MONO, 18), fill=WHITE, anchor="lm")
        tl = {"sugar": "SUGAR DAY (cheap stock)", "bitter": "BITTER DAY (rich stock)", "day": f"REPLAY · TRADING DAY {dd_}"}[kind]
        st(d, (BXY[0] + 16, BXY[3] - 64), tl, F(MBOLD, 22), fill=GOLD, anchor="lm", sw=3, sf=BG)
        st(d, (BXY[0] + 16, BXY[3] - 30), f"{m1:5.1f} ms / 100 ms · {nn} neurons fired · BUY {nb} · SELL {ns}", F(MONO, 18), fill=WHITE, anchor="lm", sw=3, sf=BG)
    if S0(2) <= t < S0(3):
        yy = BXY[3] - 30; xx = BXY[0] + 16
        cols = {"optic": (80, 120, 230), "central": (170, 120, 230), "sensory": (70, 200, 200), "visual projection": (100, 140, 240), "descending": (240, 90, 160), "motor": (90, 220, 150)}
        for k_, v_ in SCN.items():
            if reveal < 0.3: break
            d.ellipse((xx, yy - 6, xx + 12, yy + 6), fill=cols[k_]); lbl = f"{k_} {v_:,}"
            st(d, (xx + 18, yy), lbl, F(MONOR, 15), fill=GREY, anchor="lm"); xx += 30 + tw(lbl, F(MONOR, 15))
    # ---------- scene content
    if t < S0(2):
        hud(d, t, "BOOTING")
        u = ease(lin(t, 0.2, 0.8))
        box(d, (1040, 104, 1896, 896))
        st(d, (1468, 300), "I WIRED A REAL", F(MB, 66), fill=WHITE); st(d, (1468, 380), "FRUIT FLY BRAIN", F(MB, 66), fill=WHITE)
        st(d, (1468, 470), "TO A MONTE CARLO OF", F(MB, 50), fill=GOLD); st(d, (1468, 535), "ANTHROPIC'S IPO", F(MB, 66), fill=GOLD)
        if t > S0(1) + 5.8: stamp(img, t, S0(1) + 5.8, "THEN I LET IT TRADE", GREEN, cx=1468, cy=700, size=46)
        box(d, (24, 670, 1004, 896)); st(d, (514, 760), "real connectome · real spikes", F(MBOLD, 30), fill=GREY)
        st(d, (514, 810), "simulated market · fake money", F(MBOLD, 24), fill=DIM)
    elif t < S0(3):
        hud(d, t, "CONNECTOME LOADED")
        box(d, (1040, 104, 1896, 896))
        u = lin(t, S0(2), S0(2) + 3.0)
        st(d, (1468, 250), f"{int(127400 * ease(u)):,}", F(MB, 110), fill=WHITE); st(d, (1468, 330), "NEURONS (FlyWire v630)", F(MBOLD, 30), fill=GREY)
        st(d, (1468, 470), f"{NCON * ease(lin(t, S0(2) + 0.6, S0(2) + 3.6)) / 1e6:.1f}M", F(MB, 110), fill=CYAN); st(d, (1468, 550), "CONNECTIONS", F(MBOLD, 30), fill=GREY)
        st(d, (1468, 680), "dv/dt = (v₀ − v + g) / τm", F(MONO, 30), fill=WHITE); st(d, (1468, 725), "leaky integrate-and-fire · 0.1 ms steps", F(MONOR, 22), fill=GREY)
        st(d, (1468, 800), "Shiu et al., Nature 2024 · MIT license", F(MONOR, 20), fill=DIM)
        box(d, (24, 670, 1004, 896)); st(d, (514, 760), "replaying real spike data", F(MBOLD, 28), fill=GREY)
        st(d, (514, 810), "from the trading run below", F(MONOR, 22), fill=DIM)
    elif t < S0(4):
        hud(d, t, "WIRING SENSES → MARKET")
        box(d, (1040, 104, 1896, 896))
        u1 = ease(lin(t, S0(3) + 0.55 * durs[3], S0(3) + 0.55 * durs[3] + 0.6)); u2 = ease(lin(t, S0(3) + 0.75 * durs[3], S0(3) + 0.75 * durs[3] + 0.6))
        def row(y, a, b, c, col, u):
            if u <= 0: return
            box(d, (1060, y - 60, 1420, y + 60), outline=col, w=3); st(d, (1240, y - 14), a, F(MB, 28), fill=col); st(d, (1240, y + 24), b, F(MONOR, 16), fill=GREY)
            d.line((1430, y, 1530, y), fill=col, width=5); d.polygon([(1530, y - 12), (1552, y), (1530, y + 12)], fill=col)
            box(d, (1566, y - 60, 1866, y + 60), outline=col, w=3); st(d, (1716, y - 14), c[0], F(MB, 30), fill=col); st(d, (1716, y + 24), c[1], F(MONOR, 16), fill=GREY)
        st(d, (1468, 170), "THE FLY NEVER SEES THE PRICE", F(MB, 36), fill=GOLD)
        row(330, "BELOW FAIR VALUE", "or earnings beat → 21 sugar GRNs", ("BUY", "MN9 feeding motor neuron"), GREEN, u1)
        row(560, "ABOVE FAIR VALUE", "or earnings miss → 21 bitter GRNs", ("SELL", "DNge067 descending neuron"), RED, u2)
        if t > S0(3) + 1.8 and u1 <= 0:
            st(d, (1468, 360), "ONLY REPORTED NUMBERS", F(MB, 44), fill=WHITE)
            st(d, (1468, 440), "revenue · growth · margins", F(MBOLD, 34), fill=CYAN)
            st(d, (1468, 510), "→ the fly's own fair value", F(MBOLD, 28), fill=GREY)
        if u1 > 0:
            st(d, (1468, 712), "fly's fair value = only what has been reported so far:", F(MONOR, 18), fill=GREY)
            st(d, (1468, 742), "revenue, growth, margin → the model's EV / NTM revenue rule", F(MONOR, 18), fill=GREY)
        if u2 > 0:
            st(d, (1468, 784), "100 Hz per 15% rich or cheap · 1 trading day = 100 ms", F(MONOR, 18), fill=GREY)
            st(d, (1468, 814), "earnings vs IPO consensus: 200 ms · price war: 300 ms bitter", F(MONOR, 18), fill=GREY)
        box(d, (24, 670, 1004, 896))
        st(d, (514, 730), "neurons picked from the published circuit:", F(MONOR, 20), fill=GREY)
        st(d, (514, 785), "sugar → MN9 (proboscis extension)", F(MBOLD, 26), fill=GREEN); st(d, (514, 835), "bitter → DNge067 (never fires for sugar)", F(MBOLD, 26), fill=RED)
    elif t < S0(6):
        hud(d, t, "DUE DILIGENCE: READING THE S-1")
        ticker_panel(d, t, ci, 1040, 104, 1896, 706)
        counters(d, ci, 1040, 720, 1896, 896)
        box(d, (24, 670, 1004, 896))
        st(d, (514, 730), "every LM-negative word = bitter stimulus", F(MBOLD, 26), fill=RED)
        st(d, (514, 780), "every LM-positive word = sugar stimulus", F(MBOLD, 26), fill=GREEN)
        st(d, (514, 835), "1 word per ms · Loughran-McDonald finance dictionary", F(MONOR, 19), fill=GREY)
        tw0 = S0(5)
        for k, (w_, at) in enumerate([("CATASTROPHIC", 1.7), ("HARMFUL", 2.7), ("SHUTDOWN", 3.6)]):
            if tw0 + at <= t < tw0 + 4.9:
                stamp(img, t, tw0 + at, w_, RED, cx=514, cy=230 + k * 140, size=54)
        if t >= S0(5) + 0.88 * durs[5]:
            stamp(img, t, S0(5) + 0.88 * durs[5], "IPO ALLOCATION: 0 SHARES", GOLD, sub=f"net spikes {hero['read']['buy'] - hero['read']['sell']:+,} · it passed", cx=W / 2, cy=430)
    elif t < S1(9) + 0.25:
        dd = int(min(day, 252))
        hud(d, t, f"TRADING DAY {max(dd,1):>3}/252")
        chart(d, t, 1040, 104, 1896, 896, day)
        raster_panel(d, t, max(day, 0.01), 24, 670, 1004, 896)
        e = L[max(dd, 1) - 1]
        st(d, (BXY[2] - 16, BXY[1] + 22), f"shares {e['sh']:>3}  cash ${e['cash']:,.0f}", F(MONO, 19), fill=WHITE, anchor="rm")
        if S0(6) + 0.68 * durs[6] < t < S1(6) + 0.2: stamp(img, t, S0(6) + 0.68 * durs[6], f"STOCK ${px.max():.0f} vs FAIR ${int(FAIR[0])} → CASH", RED, cx=514, cy=420, size=40)
        if S0(7) + 0.4 * durs[7] < t < S1(7) + 0.1: stamp(img, t, S0(7) + 0.4 * durs[7], "BELOW FAIR VALUE → ALL IN", GREEN, cx=514, cy=420, size=44)
        if S0(8) + 0.4 * durs[8] < t < S1(8) + 0.2: stamp(img, t, S0(8) + 0.4 * durs[8], "4 EARNINGS MISSES · 0 SHARES SOLD", RED, cx=514, cy=420, size=36)
        if t > S0(9) + 1.0:
            stamp(img, t, S0(9) + 1.0, f"FLY ${NAV[-1]:,.0f}  vs  BUY&HOLD ${BH[-1]:,.0f}", GOLD, sub="after 252 trading days", cx=W / 2, cy=420, size=52)
    elif t < S0(11):
        hud(d, t, "MONTE CARLO OF THE FLY")
        u = lin(t, S0(10) + 0.2, S0(10) + 2.8)
        mc_panel(d, t, 1040, 104, 1896, 896, u, ann=t > S0(10) + 4.24, ann2=t > S0(10) + 10.4)
        wins = sum(m["nav"] > m["bh"] for m in mc)
        box(d, (24, 670, 1004, 896))
        if t > S0(10) + 3.52:
            st(d, (514, 745), f"FLY WINS {wins}/{len(mc)}", F(MB, 60), fill=GOLD)
            dn = [m for m in mc if m["bh"] < 10000]; up = [m for m in mc if m["bh"] >= 10000]
            st(d, (514, 812), f"down years {sum(m['nav'] > m['bh'] for m in dn)}/{len(dn)} · up years {sum(m['nav'] > m['bh'] for m in up)}/{len(up)}", F(MONO, 24), fill=GREY)
            st(d, (514, 852), f"never traded (stayed in cash): {sum(m['trades'] == 0 for m in mc)}", F(MONO, 18), fill=DIM)
    elif t < S0(12):
        hud(d, t, "EXPERIMENT: DOPAMINE OFF")
        dopamine_panel(d, t, 1040, 104, 1896, 896, lin(t, S0(11), S1(11)))
        box(d, (24, 670, 1004, 896)); st(d, (514, 760), "silencing = all outgoing synapses", F(MBOLD, 26), fill=GREY); st(d, (514, 810), "of those neurons set to zero", F(MBOLD, 26), fill=GREY)
    else:
        hud(d, t, "END OF EXPERIMENT")
        box(d, (1040, 104, 1896, 896))
        if t < S0(12) + 0.45 * durs[12]:
            st(d, (1468, 220), "DAY 50", F(MB, 96), fill=(190, 150, 255)); st(d, (1468, 300), "average fruit fly lifespan", F(MBOLD, 28), fill=GREY)
            st(d, (1468, 430), f"fly ${NAV[50]:,.0f}", F(MB, 64), fill=GOLD); st(d, (1468, 500), f"{L[49]['sh']} shares · still waiting", F(MB, 40), fill=CYAN)
            st(d, (1468, 600), f"(buy & hold: ${BH[50]:,.0f})", F(MONO, 24), fill=GREY)
        else:
            st(d, (1468, 200), "REAL CONNECTOME.", F(MB, 54), fill=WHITE); st(d, (1468, 270), "REAL SPIKES.", F(MB, 54), fill=WHITE)
            st(d, (1468, 340), "FAKE MONEY.", F(MB, 54), fill=GOLD)
            st(d, (1468, 450), "code + data", F(MBOLD, 26), fill=GREY); st(d, (1468, 495), REPO, F(MONO, 26), fill=CYAN)
            st(d, (1468, 600), "Not financial advice. Not a registered investment adviser.", F(MONOR, 19), fill=GREY)
            st(d, (1468, 640), "Brain: FlyWire (Dorkenwald et al. 2024), model: Shiu et al. 2024", F(MONOR, 17), fill=DIM)
            st(d, (1468, 668), "Market: ANT IPO Monte Carlo (jamesthomason.com) · Text: S-1 filed Oct 26, 2026", F(MONOR, 17), fill=DIM)
            st(d, (1468, 696), "Sentiment: Loughran-McDonald dictionary", F(MONOR, 17), fill=DIM)
        box(d, (24, 670, 1004, 896)); st(d, (514, 760), "simulated fly never aged", F(MBOLD, 28), fill=GREY); st(d, (514, 810), f"it is still holding {L[-1]['sh']} shares", F(MBOLD, 28), fill=GOLD)
    caption(d, t)
    return img

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stills":
        ts = [float(x) for x in sys.argv[2:]]
        for x in ts:
            fi = int(x * FPS); B3.ACT[:] = 0; B3.pulses = []
            frame.ci = chunk_at(max(fi - 40, 0) / FPS); frame.dd = int(min(day_at(max(fi - 40, 0) / FPS), 252))
            for f_ in range(max(fi - 40, 0), fi): frame(f_)
            frame(fi).save(R + f"results/still_{x:05.1f}.png")
        sys.exit()
    out, f0, f1 = R + "results/video_silent.mp4", 0, NF
    if len(sys.argv) > 1 and sys.argv[1] == "part":
        f0, f1 = int(sys.argv[2]), min(int(sys.argv[3]), NF); out = R + f"results/vpart_{f0:05d}.mp4"
        if f0 > 0:
            w0 = max(f0 - 40, 0); frame.ci = chunk_at(w0 / FPS); frame.dd = int(min(day_at(w0 / FPS), 252))
            for f_ in range(w0, f0): frame(f_)
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        p.stdin.write(frame(fi).tobytes())
        if fi % 300 == 0: print(fi, NF, flush=True)
    p.stdin.close(); p.wait(); print("done", TOTAL)
