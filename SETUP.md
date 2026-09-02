# Setup

Python 3.9 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

macOS often has no `python` command. Use `python3` to create the venv. After `source .venv/bin/activate`, `python` points at the venv (Flask and the rest are installed there).

If you skip activate, call the venv interpreter directly — system `python3` will fail with `No module named 'flask'`:

```bash
.venv/bin/python web/app.py
.venv/bin/python src/battery_soh_predictor.py --train
```

## Train

Creates `data/hybrid_model.pt` from `data/battery_features_clean.csv`:

```bash
python src/battery_soh_predictor.py --train
```

Skip this if `data/hybrid_model.pt` is already present.

## Web portal

With the venv activated:

```bash
python web/app.py
```

Or without activating:

```bash
.venv/bin/python web/app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) for the landing page, then **Predictor** (or `/predict`).

Enter cycle number, cumulative Ah, and temperature. End-of-life cycle/Ah default to medians from training. Remaining life (80% and 50% SOH) is on by default.

## Command-line predictor

Same model, interactive prompts:

```bash
python src/battery_soh_predictor.py
```

## Optional

Rebuild per-cycle features from a raw Oxford CSV (writes `data/battery_features_clean.csv`):

```bash
python src/features.py path/to/Oxford_orignal.csv
```

Dump the feature table to SQLite (`data/battery_degradation_clean.db`):

```bash
python src/convert_csv_to_db.py
```

Research plots and baseline comparison: open `notebooks/model.ipynb` with the same virtualenv (the notebook reads `../data/battery_features_clean.csv`).
