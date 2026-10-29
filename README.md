# I wired a fruit fly brain to a Monte Carlo simulation of Anthropic's IPO

This time it is real. A whole-brain spiking model of an adult fruit fly, built from the FlyWire connectome
(127,400 neurons and 14.7 million connections), reads the Risk Factors section of Anthropic's S-1. It then trades
a simulated year of the stock, one day at a time. Every spike, trade and dollar in the video comes from this code.

The market is the Monte Carlo from
[Anthropic's IPO prices a lead like a moat](https://jamesthomason.com/anthropics-ipo-prices-a-lead-like-a-moat/)
(`market/ant_ipo_mc.py`, 20,000 paths over 252 trading days).

Video: [`media/fruit_fly_ipo_720p.mp4`](media/fruit_fly_ipo_720p.mp4)

## How it works

**The brain.** `flybrain.py` is a numba port of the leaky integrate-and-fire model from
Shiu et al., "A Drosophila computational brain model reveals sensorimotor processing", Nature (2024)
([code](https://github.com/philshiu/Drosophila_brain_model), MIT). It uses the same equations and constants:
v0 = -52 mV, threshold -45 mV, membrane time constant 20 ms, synaptic time constant 5 ms, 2.2 ms refractory period,
1.8 ms synaptic delay, 0.275 mV per synapse, and Poisson sensory kicks scaled by 250. The time step is 0.1 ms.
Synapse signs (excitatory or inhibitory) come from predicted neurotransmitters. There is no learning and no
plasticity; the wiring diagram is the whole fly.

The port was checked against the paper's own Brian2 output (`experiments/validate.py`). With sugar receptor input at
150 Hz, it gives 17,424 total spikes per second against the paper's 17,052. MN9 fires at 113 Hz against the paper's 93 Hz.

We use the **v630** connectome, which is the release the paper validated. With the newer v783 data, the same
parameters let local olfactory and APL neurons run away into saturation (764k spikes per second), so v783 is
loaded but not used.

**Senses.**
- 21 sugar gustatory receptor neurons (GRNs) and 21 bitter GRNs, with IDs taken from the paper's `figures.ipynb`.

**Actions.**
- BUY = **MN9**, the proboscis motor neuron. This is the paper's headline readout for sugar leading to feeding.
- SELL = **DNge067** (one neuron per side). `experiments/circuit_probe.py` showed it is the descending neuron that
  bitter input drives hardest (35 Hz). Sugar does not drive it at all (0 Hz).

Measured over 100 ms, the brain is structurally bullish (`experiments/dose_response.py`):

| Input rate | 100 Hz | 150 Hz | 200 Hz | 300 Hz |
|---|---|---|---|---|
| sugar → MN9 spikes | 10.0 | 15.4 | 17.8 | 19.2 |
| bitter → DNge067 spikes | 1.2 | 4.0 | 5.6 | 10.8 |

**Phase 1: due diligence (`run_read.py`).** The fly reads all 94 pages of Risk Factors (S-1 pages 27-120,
69,169 words) at one word per millisecond. Words in the Loughran-McDonald finance dictionary's Negative list
(2,774 hits) drive the bitter neurons. Positive words (625 hits) drive the sugar neurons. The rate is
1,500 Hz times the fraction of tagged words in each 25-word block, capped at 300 Hz. The IPO allocation is the
number of buy spikes minus sell spikes, with 1 spike equal to 1 share.

**Phase 2: trading (`run_trade.py`).** Each trading day lasts 100 ms of fly time. On up days the sugar neurons
are driven, and on down days the bitter neurons are driven. The rate is min(300 Hz, 100 Hz × |daily return| /
daily vol), where daily vol is 55%/√252. The net spike count (MN9 minus DNge067) is traded at that day's close.
The fly starts with $10,000 in cash. It can only hold long positions, with no margin and no shorting. The brain
state carries over from one day to the next.

**Phase 3: Monte Carlo of the fly (`run_mc.py`).** The fly trades 40 market paths. They are chosen at evenly
spaced quantiles of the 1-year return, from the 1st to the 99th percentile.

## Results

All results use v630 with fixed seeds: 7 for reading and 11 for trading.

**Reading the S-1.** It read 69,169 words and fired 794 sell spikes and 404 buy spikes. The brain produced
184,069 spikes in total, but only 523 neurons were ever active. Net -390 means an allocation of 0: **it passed on
the IPO.** "Catastrophic", "harmful" and "shutdown" all count as negative in Loughran-McDonald. "Existential" and
"blackmail" do not.

**The median year (path #307, -19.6%).** The IPO priced at $94.17. Day 1 closed at $97.71 (+3.8%). The stock
peaked at $131.31 on day 60, bottomed at $45.41 on day 145, and closed the year at $75.74.

- The fly bought 15 shares on day 1 and was all in by day 16.
- On its best day (day 54, +11.2%), MN9 fired 17 times. It had no cash left to buy with.
- On lock-up expiry (day 122, -9.0%), it fired 10 sell spikes and sold 10 shares.
- Over the year it fired 1,403 buy spikes and 290 sell spikes, which turned into 48 buy trades and 58 sell trades.
- **Final value: $7,529 for the fly vs $8,047 for buy and hold.** It ended the year holding 99 shares and $30 in cash.
- On day 50 (roughly a fruit fly's lifespan), the fly had $10,079 (+0.8%). Buy and hold had $10,908.

**Dopamine off.** We silenced all 5,670 dopaminergic neurons present in v630, using `top_nt == dopamine` from the
FlyWire annotations. The fly made 1,337 buys and 291 sells and finished at $7,461, which is $68 worse.

**Monte Carlo of the fly (40 paths).** The fly beat buy and hold on **11 of 40** paths. Its median final value was $7,949, against $8,039 for buy and hold. Its mean was $8,723, against $9,211. It ended with a profit on 13 paths, and buy and hold did on 14. Its best relative result came on a path where the stock fell 46%: the fly finished at $10,263 (+3%), while buy and hold finished at $5,420. On the best path (+175%), it trailed by about $4,000. It starts in cash and buys on up days, so it reaches full position after the stock has already moved, and bitter days shake shares loose. By construction it is a momentum trader with a bullish bias. That costs a little in most years and helps in some crashes.

Per-path numbers are in `results/mc_paths.csv`. Daily ledgers are in `results/ledger_*.csv`.

## Reproduce

```bash
./setup.sh                         # clones the model data and annotations, runs the market MC, downloads the S-1
python experiments/validate.py     # optional: compare against the paper's Brian2 output
python run_read.py                 # ~4 min
python run_trade.py                # ~2 min
python run_trade.py dopamine_off
python run_mc.py 0 2 40 0 & python run_mc.py 1 2 40 0   # ~35 min on 2 cores
python make_video.py               # needs narration wavs in vo/ (not included)
```

Speed is about 3 s of compute per second of fly time on a 2-core machine. Numba's Poisson draws depend on the
seed and the thread, so spike counts can differ by a few percent on other hardware.

## Caveats (honest ones)

- The trading rule, the stimulus scaling, and the choice of MN9 as BUY and DNge067 as SELL are our design
  choices. The connectome supplies the wiring, not the strategy.
- This is a point-neuron model with uniform parameters and no plasticity, neuromodulation dynamics, or body.
  "Dopamine off" means those neurons are silenced, not that their chemistry is modeled.
- Sentiment comes from a bag-of-words dictionary. The fly does not understand English.
- The market paths come from a model, not from history. The S-1 is real.
- Fake money. Not financial advice. The fly is not a registered investment adviser.

## Credits

- Connectome: FlyWire Consortium, Dorkenwald et al. and Schlegel et al., Nature (2024).
  [flywire_annotations](https://github.com/flyconnectome/flywire_annotations).
- Brain model: Shiu, Sterne, Spiller et al., Nature (2024).
  [Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) (MIT).
- Sentiment: Loughran and McDonald master dictionary, via `pysentiment2`.
- S-1: [Anthropic, PBC Form S-1, SEC EDGAR](https://www.sec.gov/Archives/edgar/data/2130357/000162828026067168/anthropicpbc-sx1.htm).
- Market model: `market/ant_ipo_mc.py` by James Thomason.

MIT License.
