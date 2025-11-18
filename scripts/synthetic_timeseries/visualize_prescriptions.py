#!/usr/bin/env python3
"""
Visualize real vs synthetic PRESCRIPTIONS time-series.

Inputs:
  - real: data/raw/PRESCRIPTIONS.csv  (raw MIMIC-III style with exact headers)
  - synth windows: samples/mimic/synthetic/timeseries/prescriptions_windows_pategan.csv
Outputs:
  - Matplotlib figures: daily prevalence curves, window-sum histograms, co-occurrence heatmaps

Usage (from repo root):
  python scripts/synthetic_timeseries/visualize_prescriptions.py \
    --real data/raw/PRESCRIPTIONS.csv \
    --synth_windows samples/mimic/synthetic/timeseries/prescriptions_windows_pategan.csv \
    --topk 10 --window_len 14
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from _rx_ts_utils import (
        load_rx, pick_topk_drugs, build_daily_matrix, windowize,
        SUBJ_COL, HADM_COL, DRUG_COL
    )
except ModuleNotFoundError:
    from scripts.synthetic_timeseries._rx_ts_utils import (
        load_rx, pick_topk_drugs, build_daily_matrix, windowize,
        SUBJ_COL, HADM_COL, DRUG_COL
    )

def _features_from_windows_cols(win_df, window_len, base=["any_rx"]):
    """Infer feature base names (without _t{k})."""
    feat_names = set()
    for c in win_df.columns:
        if c.endswith(tuple([f"_t{i}" for i in range(window_len)])):
            base_name = c.rsplit("_t", 1)[0]
            feat_names.add(base_name)
    # ensure base ones are first
    ordered = []
    for b in base:
        if b in feat_names:
            ordered.append(b)
            feat_names.remove(b)
    return ordered + sorted(list(feat_names))

def _prevalence_from_windows(win_df, window_len, feat):
    """Average prevalence curve over position t=0..T-1 from synthetic windows."""
    cols = [f"{feat}_t{i}" for i in range(window_len) if f"{feat}_t{i}" in win_df.columns]
    if not cols:
        return None
    # assume binary/categorical encoded as 0/1
    arr = win_df[cols].to_numpy(dtype=float)
    return arr.mean(axis=0)  # shape (T,)

def _window_sum_distribution(win_df, window_len, feat):
    """Distribution of sum over the window for a binary feature."""
    cols = [f"{feat}_t{i}" for i in range(window_len) if f"{feat}_t{i}" in win_df.columns]
    if not cols:
        return None
    arr = win_df[cols].to_numpy(dtype=float)
    sums = arr.sum(axis=1)
    return sums

def _cooccurrence_from_windows(win_df, window_len, features, top_n=10):
    """Co-occurrence matrix across windows: any(t) for each feat, then pairwise co-occurrence."""
    feats = [f for f in features if any(f"{f}_t{i}" in win_df.columns for i in range(window_len))]
    if len(feats) > top_n:
        feats = feats[:top_n]

    bin_any = []
    for f in feats:
        cols = [f"{f}_t{i}" for i in range(window_len) if f"{f}_t{i}" in win_df.columns]
        mat = (win_df[cols] > 0.5).to_numpy(dtype=bool)
        any_f = mat.any(axis=1)  # any presence inside the window
        bin_any.append(any_f.astype(int))

    M = np.vstack(bin_any)  # shape: (F, Nwindows)
    # co-occurrence rate: P(A and B)
    co = (M @ M.T) / M.shape[1]
    return feats, co

def _cooccurrence_from_daily(daily_df, features, top_n=10):
    feats = [f for f in features if f in daily_df.columns]
    if len(feats) > top_n:
        feats = feats[:top_n]
    # per-admission indicator: any over whole stay
    grp = daily_df.groupby([SUBJ_COL, HADM_COL])
    bin_any = []
    for f in feats:
        any_f = (grp[f].max() > 0).astype(int)  # True if drug ever used in stay
        bin_any.append(any_f.values)

    M = np.vstack(bin_any)  # (F, Nadm)
    co = (M @ M.T) / M.shape[1]
    return feats, co

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="data/raw/PRESCRIPTIONS.csv")
    ap.add_argument("--synth_windows", required=True, help="CSV of synthetic windows (flattened)")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--window_len", type=int, default=14)
    ap.add_argument("--stride", type=int, default=3)  # only for making real windows if needed
    args = ap.parse_args()

    real_df = load_rx(args.real)
    top = pick_topk_drugs(real_df, k=args.topk)
    daily = build_daily_matrix(real_df, top_drugs=top)
    if daily.empty:
        raise RuntimeError("Daily matrix is empty. Check the real data.")

    win = pd.read_csv(args.synth_windows)
    if win.empty:
        raise RuntimeError("Synthetic windows file is empty.")

    # Figure: Daily prevalence curves (Real vs Synth) for any_rx and top-5 drugs
    plt.figure(figsize=(10, 5))
    real_any = daily.groupby("date")["any_rx"].mean()
    synth_any_curve = _prevalence_from_windows(win, args.window_len, "any_rx")
    plt.plot(real_any.index, real_any.values, label="Real any_rx (daily mean)", alpha=0.85)
    if synth_any_curve is not None:
        plt.plot(range(args.window_len), synth_any_curve, label="Synth any_rx (avg over window position)", alpha=0.85)
        
    plt.title("Any Prescription: Real (daily) vs Synthetic (window position)")
    plt.xlabel("Date (Real) / Window t (Synthetic)")
    plt.ylabel("Prevalence")
    plt.legend(); plt.tight_layout()

    # Per-drug prevalence (top 5)
    drug_cols = [f"rx_{d}" for d in top]
    for dcol in drug_cols[:5]:
        plt.figure(figsize=(10, 5))
        if dcol in daily.columns:
            r = daily.groupby("date")[dcol].mean()
            plt.plot(r.index, r.values, label=f"Real {dcol}", alpha=0.85)
        synth_curve = _prevalence_from_windows(win, args.window_len, dcol)
        if synth_curve is not None:
            plt.plot(range(args.window_len), synth_curve, label=f"Synth {dcol}", alpha=0.85)
        plt.title(f"Drug prevalence: {dcol}")
        plt.xlabel("Date (Real) / Window t (Synthetic)")
        plt.ylabel("Prevalence")
        plt.legend(); plt.tight_layout()

    # Figure: Distribution of window sums (how many days “on” in a 14-day window)
    feats_for_hist = ["any_rx"] + drug_cols[:3]
    for f in feats_for_hist:
        plt.figure(figsize=(8, 4))
        # Real: approximate rolling 14-day sum over admissions
        # (aligns with windows loosely; good for distribution-level comparison)
        if f in daily.columns:
            real_roll = (daily
                         .set_index("date")
                         .groupby([SUBJ_COL, HADM_COL])[f]
                         .rolling(args.window_len, min_periods=1).sum()
                         .reset_index(level=[SUBJ_COL, HADM_COL], drop=True))
            rr = real_roll.dropna().clip(0, args.window_len).to_numpy()
            if len(rr) > 0:
                plt.hist(rr, bins=range(0, args.window_len + 2), density=True, alpha=0.6, label="Real")
        sw = _window_sum_distribution(win, args.window_len, f)
        if sw is not None and len(sw) > 0:
            plt.hist(sw, bins=range(0, args.window_len + 2), density=True, alpha=0.6, label="Synth")
        plt.title(f"Window-sum distribution: {f} (# days active in window)")
        plt.xlabel("Days active in window"); plt.ylabel("Density")
        plt.legend(); plt.tight_layout()

    # Co-occurrence heatmaps (top-k drugs) — real vs synthetic
    # Synthetic co-occurrence across windows
    synth_feats_all = _features_from_windows_cols(win, args.window_len, base=["any_rx"])
    # keep only drug columns for coocc heatmaps
    synth_drug_feats = [f for f in synth_feats_all if f.startswith("rx_")]
    labs_s, mat_s = _cooccurrence_from_windows(win, args.window_len, synth_drug_feats, top_n=args.topk)

    # Real co-occurrence across admissions (ever used during stay)
    labs_r, mat_r = _cooccurrence_from_daily(daily, [f"rx_{d}" for d in top], top_n=args.topk)

    for title, labs, mat in [("Synthetic Co-occurrence (windows)", labs_s, mat_s),
                             ("Real Co-occurrence (by admission)", labs_r, mat_r)]:
        if labs and mat is not None:
            plt.figure(figsize=(0.6*len(labs)+3, 0.6*len(labs)+3))
            plt.imshow(mat, interpolation="nearest", aspect="auto")
            plt.title(title)
            plt.xticks(range(len(labs)), labs, rotation=90)
            plt.yticks(range(len(labs)), labs)
            plt.colorbar(label="P(A and B)")
            plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
