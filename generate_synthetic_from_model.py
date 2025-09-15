#!/usr/bin/env python3
"""
Generate synthetic samples from a saved CTGAN or PATEGAN model (.pkl).

Usage:
    python generate_synthetic_from_model.py \
        --model samples/models/ctgan_mimic_admissions.pkl \
        --out samples/mimic/synthetic/from_model.csv \
        --n_samples 500
"""
import argparse
import pickle
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pkl file (CTGAN or PATEGAN model)")
    parser.add_argument("--out", required=True, help="Path to output CSV for synthetic samples")
    parser.add_argument("--n_samples", type=int, default=500, help="Number of synthetic rows to generate")
    args = parser.parse_args()

    # Load model
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    # Generate samples
    df = model.sample(args.n_samples)

    # Save to CSV
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} synthetic rows to {args.out}")

if __name__ == "__main__":
    main()
