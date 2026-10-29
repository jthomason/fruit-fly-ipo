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

**Phase 2: trading on fundamentals (`run_fund.py`).** The fly never sees the stock price.
`market/export_fundamentals.py` re-runs `ant_ipo_mc.py` unchanged (same draws, same paths) and records, for every
path and day, the model's fair market value: its own EV / NTM-revenue rule applied to the fundamentals revealed so
far. It also records each quarter's true revenue against the IPO consensus, and the day a price war starts.
Each trading day lasts 100 ms of fly time:

- Valuation gap, premium = log(price / fair value). Below fair value drives the sugar neurons and above fair value
  drives the bitter neurons, at min(300 Hz, 100 Hz × |premium| / 0.15).
- Earnings days (days 70, 122, 186 and 250): an extra 200 ms call at min(300 Hz, 100 Hz × |surprise| / 3%).
  A beat is sugar and a miss is bitter.
- Price-war day: 300 ms of bitter at 300 Hz.

The net spike count (MN9 minus DNge067) is traded at that day's close, 1 spike = 1 share. The fly starts with
$10,000 in cash and the IPO allocation from Phase 1. It can only hold long positions, with no margin and no
shorting. The brain state carries over from one day to the next.

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

**The median year (path #307, -19.6%).** The model's fair value at the IPO was $84.51. The IPO priced at $94.17,
the stock peaked at $131.31 on day 60, bottomed at $45.41 on day 145, and closed the year at $75.74.

- Days 1-83: the stock was above fair value, so the fly tasted bitter and sat in cash. On day 60 it fired 11 sell
  spikes with nothing to sell.
- Day 84: the stock fell below fair value ($73.04). It bought 2 shares, and by day 94 it was all in: 153 shares at
  an average cost of about $65.
- All four earnings reports missed the IPO consensus (-1.5%, -2.5%, -3.2%, -4.0%). Those bitter calls never outweighed
  the sugar from a cheap stock: 0 sell trades all year.
- Over the year it fired 2,701 buy spikes and 283 sell spikes, which turned into 11 buy trades and 0 sell trades.
- **Final value: $11,620 for the fly vs $8,047 for buy and hold.** It ended the year holding 153 shares and $32 in cash.
- On day 50 (roughly a fruit fly's lifespan), the fly had $10,000, all in cash. Buy and hold had $10,908.

**Dopamine off.** We silenced all 5,670 dopaminergic neurons present in v630, using `top_nt == dopamine` from the
FlyWire annotations. The fly finished at $11,602, which is $18 worse.

**Monte Carlo of the fly (40 paths).** The fly beat buy and hold on **31 of 40** paths: all 26 paths where the stock
fell, and 5 of the 14 where it rose. It won on all 10 paths with a price war. On 4 paths the fly never bought a share and
stayed in cash all year, so its median final value is exactly $10,000, against $8,039 for buy
and hold. Its mean was $10,066, against $9,211. It ended with a profit on 17 paths, and buy and hold did on 14. Its best
relative result came on a path where the stock fell 52%: the fly finished at $12,695 (+27%), while buy and hold
finished at $4,765. On the best path (+175%), the fly made 1%, because the stock almost never looked cheap. It is a
value investor by construction: it waits, buys cheap, and misses rallies that never offer a discount.

**Price-only fly, for comparison.** Tasting daily price moves instead of fundamentals, the fly bought the IPO's early
rally and finished the median year at $7,529 (buy and hold $8,047). Across the same 40 paths it beat buy and hold on
11, with a median of $7,949 against $8,039. Wiring it to the fundamentals turned a momentum trader into a value
investor.

Per-path numbers are in `results/fund_mc_paths.csv`. Daily ledgers are in `results/fund_ledger_*.csv`. Summary
numbers are in `results/summary.json`.

## Reproduce

```bash
./setup.sh                         # clones the model data and annotations, runs the market MC, downloads the S-1
python experiments/validate.py     # optional: compare against the paper's Brian2 output
python run_read.py                 # ~4 min
python run_fund.py                 # ~2 min, the fundamentals-wired fly on the median path
python run_fund.py dopamine_off
python run_fund_mc.py 0 2 40 & python run_fund_mc.py 1 2 40; wait   # ~30 min on 2 cores
python run_price_trade.py          # optional: the price-only comparison
python run_price_mc.py 0 2 40 0 & python run_price_mc.py 1 2 40 0; wait
python make_video.py               # needs narration wavs in vo/ (not included)
python make_audio.py
```

Speed is about 3 s of compute per second of fly time on a 2-core machine. Numba's Poisson draws depend on the
seed and the thread, so spike counts can differ by a few percent on other hardware.

## Caveats (honest ones)

- The trading rule, the stimulus scaling, and the choice of MN9 as BUY and DNge067 as SELL are our design
  choices. The connectome supplies the wiring, not the strategy.
- "Fair value" is the Monte Carlo's own pricing rule, so the fly is trading the model against the model's own
  price paths. Its edge comes from that rule and from being
  long-only with a cash start, not from the fly predicting anything. In falling years, cash beats buy and hold.
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
