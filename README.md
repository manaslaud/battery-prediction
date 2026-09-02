# Battery SOH predictor

Estimates lithium-ion **state of health (SOH)** and **remaining useful life** from a few cell readings: cycle count, cumulative amp-hour throughput, and temperature.

The model is hybrid. A simple physics formula is the prior; a small neural net learns only the leftover error:

```text
SOH_hybrid = clip(SOH_physics + NN(features), 0, 1)
```

SOH is remaining discharge capacity divided by the cell’s first measured capacity (`1.0` = new, `0.8` = typical end of life).

## How the hybrid works

**Physics.** Normalized cycle count and throughput are combined with an Arrhenius-like temperature factor (reference 40 °C):

```text
degradation = (0.15 × cycle_norm + 0.10 × throughput_norm) × exp(0.01 × (T_K − 313.15))
SOH_physics = 1 − degradation
```

**Neural residual.** A tiny MLP (`3 → 16 → 8 → 1`) is trained so `physics + residual` matches measured SOH, with a penalty on large residuals so the net stays close to the formula.

**Remaining life.** From the current usage rate (Ah per cycle) and temperature, the hybrid curve is walked forward until it crosses 80% SOH (end of life), 50% (half-life), or an optional extra threshold. Usage is assumed constant.

Features come from the [Oxford Battery Degradation Dataset](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac) (Kokam pouch cells, ~40 °C aging). Per-cycle capacity is taken from the `C1dc` discharge phase.

On a random held-out split the hybrid is slightly more accurate than physics alone (MAE ≈ 0.017 vs 0.019). It interpolates well; it does not extrapolate well to unseen operating regimes.

## Layout

```text
data/          Feature CSV and trained hybrid_model.pt
src/           Training, CLI prediction, feature extraction
web/           Flask landing page and predictor
notebooks/     Research notebook (baselines and plots)
```

The web app is a form, not a training UI. Train from the command line; then open the predictor.

Install and run: see [SETUP.md](SETUP.md).
