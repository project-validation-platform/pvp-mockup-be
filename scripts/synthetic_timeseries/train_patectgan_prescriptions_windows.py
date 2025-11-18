#!/usr/bin/env python3
"""
Train a DP PATECTGAN on PRESCRIPTIONS windows (time-series flattened to tabular),
then save the model bundle and (optionally) sample synthetic windows.

Usage (from repo root):
  python scripts/synthetic_timeseries/train_patectgan_prescriptions_windows.py \
    --data data/raw/PRESCRIPTIONS.csv \
    --topk 10 --window_len 14 --stride 3 \
    --epochs 50 --batch_size 64 --epsilon 1.0 --preprocessor_eps 0.1 \
    --n_samples 1000
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from snsynth.pytorch.nn import PATECTGAN

try:
    from _rx_ts_utils import (
        load_rx,
        pick_topk_drugs,
        build_daily_matrix,
        windowize,
        SUBJ_COL,
        HADM_COL,
    )
except ModuleNotFoundError:
    # allow running as module: python -m scripts.synthetic_timeseries.train_patectgan_prescriptions_windows
    from scripts.synthetic_timeseries._rx_ts_utils import (
        load_rx,
        pick_topk_drugs,
        build_daily_matrix,
        windowize,
        SUBJ_COL,
        HADM_COL,
    )


def _ensure_parents(path: str | Path) -> None:
    Path(os.path.dirname(str(path))).mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/PRESCRIPTIONS.csv")
    ap.add_argument("--topk", type=int, default=10, help="Top-K frequent drugs to model individually")
    ap.add_argument("--window_len", type=int, default=30, help="Days per window")
    ap.add_argument("--stride", type=int, default=7, help="Days to slide window")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epsilon", type=float, default=1.0, help="DP budget for training")
    ap.add_argument("--preprocessor_eps", type=float, default=0.1, help="DP budget for bounds inference")
    ap.add_argument("--n_samples", type=int, default=1000, help="Synthetic windows to sample after training (0 to skip)")
    ap.add_argument("--out", default="samples/models/timeseries/pategan_prescriptions.pkl")
    ap.add_argument("--synth_out", default="samples/mimic/synthetic/timeseries/prescriptions_windows_pategan.csv")
    ap.add_argument("--save_real_daily", default="samples/mimic/synthetic/timeseries/real_prescriptions_daily.csv",
                    help="Optional: save the real daily matrix for later viz (set '' to skip)")
    ap.add_argument("--save_top_drugs", default="samples/mimic/synthetic/timeseries/top_drugs.csv",
                    help="Optional: save the top-K drugs list (set '' to skip)")
    args = ap.parse_args()
    
    df = load_rx(args.data)  # uses exact column names from your CSV
    top = pick_topk_drugs(df, k=args.topk)
    if args.save_top_drugs:
        _ensure_parents(args.save_top_drugs)
        pd.DataFrame({"drug": top}).to_csv(args.save_top_drugs, index=False)

    # build a daily grid per (subject_id, hadm_id) with indicators for top drugs
    daily = build_daily_matrix(df, top_drugs=top)
    if args.save_real_daily:
        _ensure_parents(args.save_real_daily)
        daily.to_csv(args.save_real_daily, index=False)

    # windowize (flatten time windows -> one row per window)
    win_df, meta = windowize(daily, window_len=args.window_len, stride=args.stride)
    if win_df.empty:
        raise RuntimeError("No windows produced. Try reducing --window_len and/or --stride, or increase data cohort.")

    # Column typing for DP synth
    # For categorical data: IDs + month bin + later any 0/1 indicators we decide to treat as categorical
    # For continuous data: remaining truly numeric, varying columns
    categorical = [SUBJ_COL, HADM_COL, "start_month_bin"]
    drop_cols = ["start_date"]  # not used by the model

    numeric_cols = [c for c in win_df.columns if c not in categorical + drop_cols]

    # require refactoring to use the preprocessors 
    win_df[numeric_cols] = (
        win_df[numeric_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    const_cols = [c for c in numeric_cols if win_df[c].nunique(dropna=False) <= 1]
    if const_cols:
        print(f"[info] dropping {len(const_cols)} constant columns")
        win_df = win_df.drop(columns=const_cols)
        numeric_cols = [c for c in numeric_cols if c not in const_cols]

    # Move binary 0/1 into cat to avoid minmax on them
    bin_cols = []
    for c in list(numeric_cols):
        vals = set(win_df[c].unique())
        if vals <= {0.0, 1.0} or vals <= {0, 1}:
            bin_cols.append(c)

    if bin_cols:
        print(f"[info] treating {len(bin_cols)} binary columns as categorical")
        win_df[bin_cols] = win_df[bin_cols].astype(int).astype("object")
        categorical += bin_cols
        numeric_cols = [c for c in numeric_cols if c not in bin_cols]

    continuous = numeric_cols  # remaining real-valued columns

    print(f"[diag] windows={len(win_df)}  |categorical|={len(categorical)}  |continuous|={len(continuous)}")
    if len(continuous) == 0:
        print("[diag] no continuous columns (OK) — training with categorical-only features")

    # batch-size hygiene for PATECTGAN (must be even and <= dataset size)
    n = len(win_df)
    bs = min(args.batch_size, n)
    if bs % 2 != 0:
        bs -= 1
    if bs < 2:
        bs = 2
    args.batch_size = bs
    print(f"[info] using batch_size={args.batch_size} (n={n})")

    # fit DP model
    model = PATECTGAN(
        epochs=args.epochs,
        batch_size=args.batch_size,
        epsilon=args.epsilon,
        verbose=True
    )

    model.fit(
        win_df.drop(columns=drop_cols),
        categorical_columns=categorical,
        continuous_columns=continuous,
        ordinal_columns=[],
        preprocessor_eps=args.preprocessor_eps,
        nullable=False
    )

    # save model bundle (model + meta + top_drugs)
    _ensure_parents(args.out)
    with open(args.out, "wb") as f:
        pickle.dump({"model": model, "meta": meta, "top_drugs": top}, f)
    print(f"[✓] PATECTGAN (prescriptions windows) saved → {args.out}")

    # sample synthetic windows and save
    if args.n_samples and args.n_samples > 0:
        synth = model.sample(args.n_samples)
        _ensure_parents(args.synth_out)
        synth.to_csv(args.synth_out, index=False)
        print(f"Synthetic windows saved in {args.synth_out} (rows={len(synth)})")


if __name__ == "__main__":
    main()
