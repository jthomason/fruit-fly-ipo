#!/usr/bin/env bash
# Fetch the connectome + model data (Shiu et al. 2024, MIT), the FlyWire annotations, the market model output and the S-1 text.
set -e
pip install -r requirements.txt
[ -d shiu ] || git clone https://github.com/philshiu/Drosophila_brain_model.git shiu && git -C shiu checkout -q 91bdd1e7dcf193f3e7ca5a8933497fcef63b7960
[ -d ann ]  || git clone https://github.com/flyconnectome/flywire_annotations.git ann && git -C ann checkout -q a83b2776d60d5764cef36b927f5f9679c16c47a2
python market/ant_ipo_mc.py > /dev/null      # writes market/ant_ipo_sim.npz (20,000 paths, first 4,000 saved)
python market/export_fundamentals.py         # same draws, plus the daily fair value, revenue surprises and price-war day
python extract_s1.py                         # writes risk_factors.txt
python experiments/positions.py              # neuron coordinates for the video
