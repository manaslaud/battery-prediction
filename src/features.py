"""
Output: battery_features_clean.csv - small (~445 rows), safe to upload.
"""

import sys
from pathlib import Path

import pandas as pd 
import numpy as np

CHUNKSIZE = 1_000_000
MIN_DISCHARGE_SAMPLES = 5
_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = _ROOT / "data"  


def main(csv_path: str):
    disc_parts = []
    temp_parts = []

    usecols = ["cell", "cycle", "phase", "q", "T"]
    reader = pd.read_csv(csv_path, usecols=usecols, chunksize=CHUNKSIZE)

    for i, chunk in enumerate(reader):
        disc = chunk[chunk["phase"] == "C1dc"]
        if len(disc):
            g = (
                disc.groupby(["cell", "cycle"])["q"]
                .agg(["min", "max", "count"])
                .reset_index()
            )
            disc_parts.append(g)

        gt = (
            chunk.groupby(["cell", "cycle"])["T"]
            .agg(["sum", "count"])
            .reset_index()
        )
        temp_parts.append(gt)

        print(f"  processed chunk {i + 1} ({(i + 1) * CHUNKSIZE:,} rows)")

    print("Combining chunk-level aggregates...")

    disc_all = pd.concat(disc_parts, ignore_index=True)
    disc_final = (
        disc_all.groupby(["cell", "cycle"])
        .agg(q_min=("min", "min"), q_max=("max", "max"), n_samples=("count", "sum"))
        .reset_index()
    )
    disc_final["capacity"] = disc_final["q_max"] - disc_final["q_min"]

    before = len(disc_final)
    disc_final = disc_final[disc_final["n_samples"] >= MIN_DISCHARGE_SAMPLES]
    print(f"Dropped {before - len(disc_final)} low-confidence cycles "
          f"(< {MIN_DISCHARGE_SAMPLES} discharge samples)")

    temp_all = pd.concat(temp_parts, ignore_index=True)
    temp_final = (
        temp_all.groupby(["cell", "cycle"])
        .agg(t_sum=("sum", "sum"), t_count=("count", "sum"))
        .reset_index()
    )
    temp_final["avg_temp"] = temp_final["t_sum"] / temp_final["t_count"]

    features = disc_final.merge(temp_final[["cell", "cycle", "avg_temp"]],
                                 on=["cell", "cycle"], how="left")
    features = features.sort_values(["cell", "cycle"]).reset_index(drop=True)

    features["initial_capacity"] = features.groupby("cell")["capacity"].transform("first")
    features["soh"] = features["capacity"] / features["initial_capacity"]

    n_impossible = (features["soh"] > 1.02).sum()
    print(f"SOH values still > 1.02 after fix: {n_impossible} "
          f"(should be near 0 now vs. many before)")

    features["cum_ah_throughput"] = features.groupby("cell")["capacity"].cumsum()

    features["soh_smooth"] = (
        features.groupby("cell")["soh"]
        .transform(lambda x: x.rolling(window=3, center=True, min_periods=1).mean())
    )

    out_path = DATA_DIR / "battery_features_clean.csv"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}  shape={features.shape}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/features.py Oxford_orignal.csv")
        sys.exit(1)
    main(sys.argv[1])