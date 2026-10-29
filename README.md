# I wired a fruit fly brain to a Monte Carlo simulation of Anthropic's IPO

A whole-brain spiking model of an adult fruit fly, built from the FlyWire connectome (127,400 neurons and
14.7 million connections), reads the Risk Factors section of Anthropic's S-1. It is then wired to the
fundamentals of a Monte Carlo model of the company and trades a simulated year of the stock, one day at a time.
Every spike, trade and dollar in the video comes from this code.

The market is the Monte Carlo from
[Anthropic's IPO prices a lead like a moat](https://jamesthomason.com/anthropics-ipo-prices-a-lead-like-a-moat/)
(`market/ant_ipo_mc.py`, 20,000 paths over 252 trading days). It models revenue, margins, the compute floor and
the risk of a price war, and prices the stock with its own EV / NTM-revenue multiple rule.

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

**Phase 2: trading on fundamentals (`run_fund.py`).** The Monte Carlo generates walk-forward scenarios of the
company's fundamentals, and the fly decides what to do. It never sees the stock price; it only trades at it.
`market/export_fundamentals.py` re-runs `ant_ipo_mc.py` unchanged (same draws, same paths) and saves each path's true
and consensus quarterly revenue, margins and price-war day. Each day, the fly values the company only from what has
been reported so far (`fly_value` in `ipo_fly.py`):

- NTM revenue = the IPO consensus for the next four quarters, scaled by how the latest reported quarter came in
  against its consensus.
- Margin = the IPO consensus, replaced by the true 2027 margin once 2027 quarters are reported (day 122 on).
- Value = the model's own multiple rule, (5 + 9 ln(1 + NTM growth)) x margin factor, times that NTM revenue, minus
  the consensus Broadcom cost.

The market in the model knows more than the fly. Its price also reflects part of the true future revenue path, the
price war as soon as it starts, a sentiment premium and lock-up supply. Each trading day lasts 100 ms of fly time:

- Valuation gap, premium = log(price / fly's fair value). Below fair value drives the sugar neurons and above fair
  value drives the bitter neurons, at min(300 Hz, 100 Hz × |premium| / 0.15).
- Earnings days (days 70, 122, 186 and 250): an extra 200 ms call at min(300 Hz, 100 Hz × |surprise| / 3%).
  A beat is sugar and a miss is bitter.
- Price-war day: 300 ms of bitter at 300 Hz.

The net spike count (MN9 minus DNge067) is traded at that day's close, 1 spike = 1 share. The fly starts with
$10,000 in cash and the IPO allocation from Phase 1. It can only hold long positions, with no margin and no
shorting. The brain state carries over from one day to the next.

**Why the fly has its own fair value.** An earlier version (v2) compared the price with the model's market value.
But the model builds the price as market value × sentiment × supply, so that gap was only the sentiment and supply
terms, with no fundamental information, and its slow mean reversion gave a built-in edge.
`experiments/signal_check.py` measures it: across all 4,000 saved paths, the day-100 premium correlates −0.139 with the
next 60 days' return under v2, and +0.017 under the fly's own fair value. v2 results are kept in
`results/v2_market_value/`.

**Phase 3: Monte Carlo of the fly (`run_fund_mc.py`).** The fly trades 40 market paths. They are chosen at evenly
spaced quantiles of the 1-year return, from the 1st to the 99th percentile.

**Comparison: price-only fly (`run_price_trade.py`, `run_price_mc.py`).** The first version tasted the daily price
move instead: sugar on up days, bitter on down days, at min(300 Hz, 100 Hz × |daily return| / daily vol). Its results
are in `results/price_only/`.

## Results

All results use v630 with fixed seeds: 7 for reading and 11 for trading.

**Reading the S-1.** It read 69,169 words and fired 794 sell spikes and 404 buy spikes. The brain produced
184,069 spikes in total, but only 523 neurons were ever active. Net -390 means an allocation of 0: **it passed on
the IPO.** "Catastrophic", "harmful" and "shutdown" all count as negative in Loughran-McDonald. "Existential" and
"blackmail" do not.

**The median year (path #307, -19.6%).** The fly's fair value at the IPO was $84.47. The IPO priced at $94.17,
the stock peaked at $131.31 on day 60, bottomed at $45.41 on day 145, and closed the year at $75.74.

- Days 1-83: the stock was above fair value, so the fly tasted bitter and sat in cash. On day 60 it fired 11 sell
  spikes with nothing to sell.
- Day 84: the stock fell below fair value ($73.04). It bought 9 shares, and by day 93 it was all in: 152 shares at
  an average cost of $65.76.
- All four earnings reports missed the IPO consensus (-1.5%, -2.5%, -3.2%, -4.0%). Those bitter calls never outweighed
  the sugar from a cheap stock: 0 sell trades all year.
- Over the year it fired 2,782 buy spikes and 286 sell spikes, which turned into 10 buy trades and 0 sell trades.
- **Final value: $11,517 for the fly vs $8,047 for buy and hold.** It ended the year holding 152 shares and $4 in cash.
- On day 50 (roughly a fruit fly's lifespan), the fly had $10,000, all in cash. Buy and hold had $10,908.

**Dopamine off.** We silenced all 5,670 dopaminergic neurons present in v630, using `top_nt == dopamine` from the
FlyWire annotations. The fly finished at $11,481, which is $36 worse.

**Monte Carlo of the fly (40 paths).** The fly beat buy and hold on **30 of 40** paths. That sounds better than it is:
the stock falls on 26 of the 40 paths, and a long-only fly that starts in cash wins those by default. It beat buy and
hold on all 26, but still lost money on 20 of them (average $8,111 vs $6,027). On the 14 paths where the stock rose, it
won 4 (average $13,150 vs $15,123), because the stock rarely looked cheap. On 4 paths it never bought a share. Its
median final value was $9,932, against $8,039 for buy and hold, and its mean was $9,874, against $9,211. It ended with a
profit on 16 paths, and buy and hold did on 14. Price wars hurt: the fly's fair value only learns about a war from
later reported quarters, so it kept buying a stock the market had already marked down.

**Earlier versions, for comparison.**

| Version | Fly's signal | Median path | Wins (40 paths) | Fly mean | Buy-and-hold mean |
|---|---|---|---|---|---|
| Price only | daily price moves | $7,529 | 11 | $8,723 | $9,211 |
| v2 | price vs model's market value | $11,620 | 31 | $10,066 | $9,211 |
| **v3 (this one)** | price vs fly's own fair value | **$11,517** | **30** | **$9,874** | **$9,211** |

Tasting price moves, the brain chased momentum. Tasting value, it waits for a discount. The strategy comes from the
wiring, not the brain.

Per-path numbers are in `results/fund_mc_paths.csv`. Daily ledgers are in `results/fund_ledger_*.csv`. Summary
numbers are in `results/summary.json`.

## Reproduce

```bash
./setup.sh                         # clones the model data and annotations, runs the market MC, downloads the S-1
python experiments/validate.py     # optional: compare against the paper's Brian2 output
python experiments/signal_check.py # optional: predictive power of the valuation signal
python run_read.py                 # ~4 min
python run_fund.py                 # ~2 min, the fundamentals-wired fly on the median path
python run_fund.py dopamine_off
python run_fund_mc.py 0 2 40 & python run_fund_mc.py 1 2 40; wait   # ~30 min on 2 cores
python run_price_trade.py          # optional: the price-only comparison
python run_price_mc.py 0 2 40 0 & python run_price_mc.py 1 2 40 0; wait
python run_record.py               # ~1.5 min, re-runs the median-path year recording every spike (results/full_raster.npz)
python experiments/analyze_days.py # spike windows per trading day (results/daywin.json)
python experiments/make_edges.py   # synapses leaving every neuron that ever fired (results/active_edges.npz)
python make_video.py               # needs narration wavs in vo/ (not included)
python make_audio.py
```

Speed is about 3 s of compute per second of fly time on a 2-core machine. Numba's Poisson draws depend on the
seed and the thread, so spike counts can differ by a few percent on other hardware.

## The brain animation

The brain panel in the video is drawn from the simulation, not made up. Every dot is one of the 127,400 neurons
at its real FlyWire 3D position, tinted by superclass (optic, central, sensory, visual projection, descending,
motor). The neurons that light up are the ones that spiked in the recorded trading run (`run_record.py`: same seed
and inputs, all spikes kept, identical $11,516.91 result). The glowing lines are real connections (3 or more
synapses) between neurons that fire during the run, lit when the presynaptic neuron just fired, and the moving pulses follow their strongest excitatory outputs.
The slow-motion segments replay single 100 ms trading days spike by spike: a sugar day where MN9 fires
(buy), a bitter day where DNge067 fires (sell), day 60 (the $131 peak: sell spikes, but no shares to sell) and day 84 (the first
buy). The magenta glow in the dopamine scene marks the 5,670 dopaminergic neurons that the experiment silences.
The brain is sparse: on a typical day only 60 to 400 neurons fire, because the model has no spontaneous activity.

## Caveats (honest ones)

- The trading rule, the stimulus scaling, and the choice of MN9 as BUY and DNge067 as SELL are our design
  choices. The connectome supplies the wiring, not the strategy.
- The market is a model. The fly's fair value uses the model's own multiple rule on reported numbers, so the test is
  "if the market behaved like this Monte Carlo, what would a value-wired fly do", not a backtest.
- Most of the win rate comes from being long-only with a cash start in a model where the stock usually falls. The
  fly's valuation signal has no measurable predictive edge (`experiments/signal_check.py`).
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
