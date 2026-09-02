"""
Creates hybrid_model.pt from battery_features_clean.csv
and provides an interactive CLI for SOH prediction and RUL estimation
"""

import argparse
import sys
from pathlib import Path

import torch # type: ignore
import torch.nn as nn # type: ignore
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # type: ignore

np.random.seed(42)
torch.manual_seed(42)

_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = _ROOT / "data" / "battery_features_clean.csv"
MODEL_FILE = _ROOT / "data" / "hybrid_model.pt"
FEATURE_COLS = ["cycle_norm", "throughput_norm", "temperature_factor"]
T_REF = 313.15  # 40C reference, Kelvin
ALPHA, BETA = 0.15, 0.10


class ResidualNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 16), nn.Tanh(), nn.Dropout(0.15),
            nn.Linear(16, 8), nn.Tanh(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


def physics_soh(cycle_norm, throughput_norm, temperature_factor):
    degradation = (ALPHA * cycle_norm + BETA * throughput_norm) * temperature_factor
    return float(np.clip(1 - degradation, 0, 1))


def simulate_soh_curve(model, scaler, cycle_now, throughput_now, avg_temp,
                        max_cycle, max_throughput, horizon_cycles=None, step=1):
    
    if cycle_now > 0 and throughput_now > 0:
        rate = throughput_now / cycle_now
    else:
        rate = max_throughput / max_cycle

    if horizon_cycles is None:
        horizon_cycles = max(max_cycle * 3, cycle_now + 1)

    temp_K = avg_temp + 273.15
    temperature_factor = np.exp(0.01 * (temp_K - T_REF))

    cycles = np.arange(cycle_now, horizon_cycles + step, step, dtype=float)
    throughputs = rate * cycles
    cycle_norm = cycles / max_cycle
    throughput_norm = throughputs / max_throughput

    X = scaler.transform(np.column_stack([
        cycle_norm, throughput_norm, np.full_like(cycle_norm, temperature_factor)
    ]))
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        residual = model(X_t).numpy().flatten()

    phys = np.clip(1 - (ALPHA * cycle_norm + BETA * throughput_norm) * temperature_factor, 0, 1)
    hybrid = np.clip(phys + residual, 0, 1)
    return cycles, hybrid


def find_threshold_crossing(cycles, hybrid_soh, threshold):
    """First cycle at which hybrid_soh drops to or below `threshold`, or None."""
    below = np.where(hybrid_soh <= threshold)[0]
    if len(below) == 0:
        return None
    return float(cycles[below[0]])


def estimate_life_milestones(model, scaler, cycle_now, throughput_now, avg_temp,
                              max_cycle, max_throughput, current_soh, thresholds):
    
    horizon = max(max_cycle * 5, cycle_now + 1)
    cycles, hybrid = simulate_soh_curve(
        model, scaler, cycle_now, throughput_now, avg_temp,
        max_cycle, max_throughput, horizon_cycles=horizon,
    )
    results = {}
    for t in thresholds:
        if current_soh <= t:
            results[t] = {"cycle": cycle_now, "cycles_remaining": 0.0}
            continue
        crossing = find_threshold_crossing(cycles, hybrid, t)
        if crossing is None:
            results[t] = None
        else:
            results[t] = {"cycle": crossing, "cycles_remaining": crossing - cycle_now}
    return results


def load_and_prepare(path):
    df = pd.read_csv(path)
    counts = df.groupby("cell")["cycle"].count()
    usable = counts[counts >= 10].index.tolist()
    df = df[df["cell"].isin(usable)].copy().sort_values(["cell", "cycle"])

    df["cycle_norm"] = df["cycle"] / df.groupby("cell")["cycle"].transform("max")
    df["throughput_norm"] = df["cum_ah_throughput"] / df.groupby("cell")["cum_ah_throughput"].transform("max")
    df["temp_K"] = df["avg_temp"] + 273.15
    df["temperature_factor"] = np.exp(0.01 * (df["temp_K"] - T_REF))
    df["degradation"] = (ALPHA * df["cycle_norm"] + BETA * df["throughput_norm"]) * df["temperature_factor"]
    df["physics_soh"] = (1 - df["degradation"]).clip(0, 1)
    return df


def train_hybrid(Xtr, ytr, ptr, Xval, yval, pval, phys_weight=0.3, lr=0.01,
                  wd=1e-3, patience=150, max_epochs=3000):
    model = ResidualNN(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    best_val, best_state, ctr = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        residual = model(Xtr)
        pred = ptr + residual
        loss = loss_fn(pred, ytr) + phys_weight * torch.mean(residual ** 2)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(pval + model(Xval), yval).item()
        if vloss < best_val:
            best_val, best_state, ctr = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            ctr += 1
            if ctr > patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def do_train():
    if not DATA_FILE.exists():
        print(f"Could not find '{DATA_FILE}'.")
        print("Copy the CSV you used in the notebook next to this script, then re-run with --train.")
        sys.exit(1)

    df = load_and_prepare(DATA_FILE)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(train_df[FEATURE_COLS])
    Xtr = torch.tensor(scaler.transform(train_df[FEATURE_COLS]), dtype=torch.float32)
    ytr = torch.tensor(train_df["soh"].values, dtype=torch.float32).view(-1, 1)
    ptr = torch.tensor(train_df["physics_soh"].values, dtype=torch.float32).view(-1, 1)
    Xte = torch.tensor(scaler.transform(test_df[FEATURE_COLS]), dtype=torch.float32)
    yte = test_df["soh"].values
    pte = torch.tensor(test_df["physics_soh"].values, dtype=torch.float32).view(-1, 1)

    rng = np.random.RandomState(42)
    n = len(Xtr)
    idx = rng.permutation(n)
    val_n = max(10, int(0.15 * n))
    val_idx, tr_idx = idx[:val_n], idx[val_n:]
    Xtr_fit, ytr_fit, ptr_fit = Xtr[tr_idx], ytr[tr_idx], ptr[tr_idx]
    Xval, yval, pval = Xtr[val_idx], ytr[val_idx], ptr[val_idx]

    model = train_hybrid(Xtr_fit, ytr_fit, ptr_fit, Xval, yval, pval)

    with torch.no_grad():
        pred_hybrid = (pte + model(Xte)).numpy().flatten()
    mae = mean_absolute_error(yte, pred_hybrid)
    rmse = np.sqrt(mean_squared_error(yte, pred_hybrid))
    r2 = r2_score(yte, pred_hybrid)
    print(f"Hybrid PINN on held-out test set -> MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

    ref = df.groupby("cell")[["cycle", "cum_ah_throughput"]].max()

    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "ref_max_cycle": float(ref["cycle"].median()),
            "ref_max_throughput": float(ref["cum_ah_throughput"].median()),
        },
        MODEL_FILE,
    )
    print(f"Saved trained hybrid model to '{MODEL_FILE}'.")


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"No saved model found ('{MODEL_FILE}'). "
            "Run: python battery_soh_predictor.py --train first."
        )
    ckpt = torch.load(MODEL_FILE, weights_only=False)
    model = ResidualNN(input_size=len(FEATURE_COLS))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scaler = StandardScaler()
    scaler.mean_ = np.array(ckpt["scaler_mean"])
    scaler.scale_ = np.array(ckpt["scaler_scale"])
    scaler.n_features_in_ = len(FEATURE_COLS)
    return model, scaler, ckpt["ref_max_cycle"], ckpt["ref_max_throughput"]


def predict_soh(model, scaler, cycle, throughput, avg_temp, max_cycle, max_throughput):
    cycle_norm = cycle / max_cycle
    throughput_norm = throughput / max_throughput
    temp_K = avg_temp + 273.15
    temperature_factor = np.exp(0.01 * (temp_K - T_REF))

    phys_pred = physics_soh(cycle_norm, throughput_norm, temperature_factor)
    x = scaler.transform([[cycle_norm, throughput_norm, temperature_factor]])
    x_t = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        residual = model(x_t).item()
    hybrid_pred = float(np.clip(phys_pred + residual, 0, 1))
    return {
        "physics_soh": phys_pred,
        "residual": residual,
        "hybrid_soh": hybrid_pred,
    }


def _life_label(threshold):
    if abs(threshold - 0.80) < 1e-9:
        return "EOL (80%)"
    if abs(threshold - 0.50) < 1e-9:
        return "Half-life (50%)"
    return f"{threshold * 100:.0f}% SOH"


def predict_life(model, scaler, cycle, throughput, avg_temp,
                 max_cycle, max_throughput, current_soh, thresholds=(0.80, 0.50)):
    results = estimate_life_milestones(
        model, scaler, cycle, throughput, avg_temp,
        max_cycle, max_throughput, current_soh, thresholds,
    )
    horizon = max_cycle * 5
    rows = []
    for t in sorted(thresholds, reverse=True):
        r = results[t]
        row = {"threshold": t, "label": _life_label(t), "horizon": horizon}
        if r is None:
            row.update(status="not_reached", cycle=None, cycles_remaining=None)
        elif r["cycles_remaining"] <= 0:
            row.update(status="already_below", cycle=r["cycle"], cycles_remaining=0.0)
        else:
            row.update(status="ok", cycle=r["cycle"], cycles_remaining=r["cycles_remaining"])
        rows.append(row)
    return rows


def ask_float(prompt, default=None):
    while True:
        suffix = f" [{default}]: " if default is not None else ": "
        raw = input(prompt + suffix).strip()
        if raw.lower() in ("q", "quit", "exit"):
            print("Exiting.")
            sys.exit(0)
        if raw == "" and default is not None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a number (or 'q' to quit).")


def predict_one(model, scaler, ref_max_cycle, ref_max_throughput):
    print("\nEnter the battery's current readings ('q' to quit):")
    cycle = ask_float("  Cycle number")
    throughput = ask_float("  Cumulative Ah throughput")
    avg_temp = ask_float("  Average temperature (deg C)")
    max_cycle = ask_float("  Expected end-of-life cycle count", default=ref_max_cycle)
    max_throughput = ask_float("  Expected end-of-life Ah throughput", default=ref_max_throughput)

    pred = predict_soh(model, scaler, cycle, throughput, avg_temp, max_cycle, max_throughput)
    phys_pred, residual, hybrid_pred = pred["physics_soh"], pred["residual"], pred["hybrid_soh"]

    print("\n--- Prediction ---")
    print(f"  Physics-only SOH estimate : {phys_pred:.4f}")
    print(f"  NN residual correction    : {residual:+.4f}")
    print(f"  Hybrid PINN SOH estimate  : {hybrid_pred:.4f}  ({hybrid_pred * 100:.1f}% health)")
    print("-" * 30)

    want_rul = input("\nEstimate remaining useful life / half-life? [y/n]: ").strip().lower()
    if want_rul == "y":
        estimate_and_print_life(
            model, scaler, cycle, throughput, avg_temp,
            max_cycle, max_throughput, hybrid_pred,
        )


def estimate_and_print_life(model, scaler, cycle, throughput, avg_temp,
                             max_cycle, max_throughput, current_soh):
    
    thresholds = [0.80, 0.50]
    custom = input("  Custom threshold too? Enter SOH fraction (e.g. 0.7), or press Enter to skip: ").strip()
    if custom:
        try:
            c = float(custom)
            if 0 < c < 1 and c not in thresholds:
                thresholds.append(c)
        except ValueError:
            print("  (ignoring invalid threshold)")

    rows = predict_life(
        model, scaler, cycle, throughput, avg_temp,
        max_cycle, max_throughput, current_soh, thresholds,
    )

    print("\n--- Remaining Life Estimate ---")
    print("  Assumption: constant usage rate & temperature from current readings.")
    for row in rows:
        label = row["label"]
        if row["status"] == "not_reached":
            print(f"  {label:<14}: not reached within simulated horizon (>{row['horizon']:.0f} cycles)")
        elif row["status"] == "already_below":
            print(f"  {label:<14}: already at or below this threshold")
        else:
            print(f"  {label:<14}: cycle {row['cycle']:.0f}  (~{row['cycles_remaining']:.0f} cycles from now)")
    print("-" * 30)


def interactive_loop():
    try:
        model, scaler, ref_max_cycle, ref_max_throughput = load_model()
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)
    print("Loaded trained hybrid PINN model. Type 'q' at any prompt to quit.")
    while True:
        try:
            predict_one(model, scaler, ref_max_cycle, ref_max_throughput)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        again = input("\nPredict another reading? [y/n]: ").strip().lower()
        if again == "n":
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PIN-BatSOH hybrid model CLI")
    parser.add_argument("--train", action="store_true", help="Retrain the hybrid model and save it")
    args = parser.parse_args()

    if args.train:
        do_train()
    else:
        interactive_loop()