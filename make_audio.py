import numpy as np, wave, json, pickle, os
SR = 44100; R = os.path.dirname(os.path.abspath(__file__)) + "/"
tm = json.load(open(R + "vo/timing.json")); st = {int(k): v for k, v in tm["starts"].items()}; du = {int(k): v for k, v in tm["durs"].items()}
S0 = lambda s: st[s]; S1 = lambda s: st[s] + du[s]
N = int(tm["total"] * SR) + SR; voice = np.zeros(N); fx = np.zeros(N)
def rd(p):
    with wave.open(p) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float) / 32768
        if w.getnchannels() == 2: a = a.reshape(-1, 2).mean(1)
        assert w.getframerate() == SR, w.getframerate()
        return a
for s in st: a = rd(R + f"vo/p{s}.wav"); i = int(S0(s) * SR); voice[i:i + len(a)] += a[:N - i]
def add(sig, at, g=1.0):
    s = int(at * SR); e = min(N, s + len(sig))
    if 0 <= s < N: fx[s:e] += sig[:e - s] * g
def tone(f0, f1, dur, g=0.25, decay=4):
    tt = np.arange(int(dur * SR)) / SR; f = np.linspace(f0, f1, len(tt))
    return g * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * decay)
rng = np.random.default_rng(3); t = np.arange(N) / SR
beat = 60 / 124; kick = tone(120, 45, 0.25, 0.5, 18)
hat = rng.normal(0, 1, int(0.03 * SR)) * np.exp(-np.arange(int(0.03 * SR)) / SR * 150) * 0.05
b = 0.3
while b < tm["end"]: add(kick, b, 0.5); add(hat, b + beat / 2); b += beat
bass = 0.045 * np.sin(2 * np.pi * 55 * t) * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * t / beat))); bass[int(tm["end"] * SR):] = 0; fx += bass
# timeline identical to render_real.py
def ease(x): x = min(max(x, 0), 1); return x * x * (3 - 2 * x)
def lin(x, a, b): return min(max((x - a) / max(b - a, 1e-6), 0), 1)
KEY = [(S0(6), 0.0), (S1(6), 83.0), (S0(7) + 0.15 * du[7], 84.0), (S1(7), 95.0), (S1(8), 250.0), (S0(9) + 1.0, 252.0)]
def day_at(x):
    if x <= KEY[0][0]: return 0.0
    for (a, da), (b_, db) in zip(KEY[:-1], KEY[1:]):
        if x <= b_: return da + (db - da) * ease(lin(x, a, b_)) if db - da > 60 else da + (db - da) * lin(x, a, b_)
    return 252.0
L = pickle.load(open(R + "results/fund_normal.pkl", "rb"))["trade"]["ledger"]
ts = np.arange(S0(6), S0(9) + 1.2, 1 / 200); ds = np.array([day_at(x) for x in ts]); last = -1
for d in range(1, 253):
    k = np.searchsorted(ds, d)
    if k >= len(ts): break
    x = ts[k]
    if x - last < 0.07: continue
    q = L[d - 1]["q"]
    if L[d - 1]["ev"] == "earnings": add(tone(700, 350, 0.35, 0.18, 6), x)
    if q > 0: add(tone(1500, 1700, 0.06, 0.10, 30), x); last = x
    elif q < 0: add(tone(420, 330, 0.08, 0.12, 25), x); last = x
# reading: soft geiger clicks for sell spikes during reading window
ch = pickle.load(open(R + "results/read_normal.pkl", "rb"))["read"]["chunks"]
RD0, RD1 = S0(4) + 0.4, S0(5) + 0.62 * du[5]
click = rng.normal(0, 1, int(0.004 * SR)) * 0.10
for j, c in enumerate(ch):
    if c["s"] and rng.random() < 0.35: add(click, RD0 + (RD1 - RD0) * j / len(ch), 0.8)
add(tone(500, 1400, 0.4, 0.22), S0(2))
add(tone(880, 880, 0.5, 0.22, 6) + tone(1320, 1320, 0.5, 0.12, 6), S0(5) + 0.88 * du[5])     # allocation 0 stamp
add(tone(600, 250, 0.5, 0.25), S0(6) + 0.68 * du[6])
add(tone(1000, 1500, 0.3, 0.2), S0(7) + 0.43 * du[7])
add(tone(600, 250, 0.5, 0.25), S0(8) + 0.4 * du[8])
add(tone(523, 523, 0.3, 0.2) + tone(659, 659, 0.3, 0.15), S0(9) + 1.0)
add(tone(400, 150, 0.4, 0.3), S0(11))
add(tone(600, 300, 0.5, 0.2), S0(12) + 0.2); add(tone(450, 200, 0.8, 0.2), S1(12) + 0.3)
env = np.convolve(np.abs(voice), np.ones(4410) / 4410, mode="same"); duck = 1 - 0.55 * np.clip(env / 0.05, 0, 1)
mix = voice + fx * duck; mix = mix / max(1, np.abs(mix).max() / 0.95)
with wave.open(R + "results/mix.wav", "w") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())
print("ok", N / SR)
