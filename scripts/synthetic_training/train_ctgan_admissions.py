#!/usr/bin/env python3
"""
Train a CTGAN model on data/raw/ADMISSIONS.csv and save to samples/models/ctgan_mimic_admissions.pkl.
Also generates synthetic rows and saves them to samples/mimic/synthetic/ADMISSIONS_synth.csv.

Usage:
    python scripts/synthetic_training/train_ctgan_admissions.py \
        --epochs 50 --batch_size 500 --n_samples 1000
"""
import argparse
import os
import pickle
from pathlib import Path
import re
import pandas as pd
import numpy as np

def _import_ctgan():
    from ctgan import CTGAN
    return CTGAN

def _preprocess_for_ctgan(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    - Parse likely datetime columns to integer timestamps (seconds).
    - Impute NaNs in numeric columns (median).
    - Fill NaNs in object columns with 'Unknown'.
    - Return cleaned df and list of discrete (categorical) columns.
    """
    df = df.copy()
    dt_name_pattern = re.compile(r"(time|date)$", re.IGNORECASE)
    datetime_cols = [c for c in df.columns if dt_name_pattern.search(c)]

    for c in datetime_cols:
        dt = pd.to_datetime(df[c], errors="coerce", utc=False)
        ts = (dt.view("int64") // 1_000_000_000)
        ts = ts.where(~dt.isna(), np.nan)
        df[c] = ts

    name_hint_cats = {"ID", "CODE", "TYPE", "LOCATION", "INSURANCE", "LANGUAGE",
                      "RELIGION", "MARITAL", "ETHNICITY", "DIAGNOSIS", "FLAG"}
    likely_cat_by_name = []
    for c in df.columns:
        up = c.upper()
        if any(hint in up for hint in name_hint_cats):
            likely_cat_by_name.append(c)

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    discrete_columns = sorted(set(obj_cols + likely_cat_by_name))

    # Force those to object dtype
    for c in discrete_columns:
        if c in df.columns:
            df[c] = df[c].astype("object")

    # Handle missing vals
    num_cols = df.select_dtypes(include=["number", "floating", "integer"]).columns.tolist()
    num_cont_cols = [c for c in num_cols if c not in discrete_columns]
    for c in num_cont_cols:
        if df[c].isna().any():
            median = df[c].median()
            df[c] = df[c].fillna(median)

    # Fill missing w "Unknown"
    for c in discrete_columns:
        if df[c].isna().any():
            df[c] = df[c].fillna("Unknown")

    return df, discrete_columns

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw/ADMISSIONS.csv",
                   help="Path to ADMISSIONS.csv (default: data/raw/ADMISSIONS.csv)")
    p.add_argument("--out", default="samples/models/ctgan_mimic_admissions.pkl",
                   help="Output model path (default: samples/models/ctgan_mimic_admissions.pkl)")
    p.add_argument("--synth_out", default="samples/mimic/synthetic/ADMISSIONS_synth.csv",
                   help="Output CSV path for synthetic samples")
    p.add_argument("--n_samples", type=int, default=1000,
                   help="Number of synthetic rows to generate")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--log_every", type=int, default=10)
    args = p.parse_args()

    df = pd.read_csv(args.data)
    df_clean, discrete_columns = _preprocess_for_ctgan(df)

    CTGANSynthesizer = _import_ctgan()
    model = CTGANSynthesizer(epochs=args.epochs, batch_size=args.batch_size, verbose=True)
    model.fit(df_clean, discrete_columns=discrete_columns)

    # Save model
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(model, f)
    print(f"CTGAN model saved: {args.out}")

    if args.n_samples and args.n_samples > 0:
        synth_df = model.sample(args.n_samples)

        synth_dir = Path(args.synth_out).parent
        synth_dir.mkdir(parents=True, exist_ok=True)
        synth_df.to_csv(args.synth_out, index=False)
        print(f"Synthetic samples saved: {args.synth_out} (rows={len(synth_df)})")

if __name__ == "__main__":
    main()
