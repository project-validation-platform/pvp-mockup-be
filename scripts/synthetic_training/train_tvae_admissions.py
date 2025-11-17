#!/usr/bin/env python3
import argparse, os, pickle
from pathlib import Path
import pandas as pd

from ctgan import TVAE
from scripts.synthetic_training._preprocess_utils import preprocess_for_sdv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/ADMISSIONS.csv")
    ap.add_argument("--out", default="samples/models/tvae_mimic_admissions.pkl")
    ap.add_argument("--synth_out", default="samples/mimic/synthetic/ADMISSIONS_synth_tvae.csv")
    ap.add_argument("--n_samples", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=500)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df_clean, discrete_columns = preprocess_for_sdv(df)

    model = TVAE(epochs=args.epochs, batch_size=args.batch_size, verbose=True)
    model.fit(df_clean, discrete_columns=discrete_columns)

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f: pickle.dump(model, f)
    print(f"TVAE model saved: {args.out}")

    if args.n_samples > 0:
        synth = model.sample(args.n_samples)
        Path(os.path.dirname(args.synth_out)).mkdir(parents=True, exist_ok=True)
        synth.to_csv(args.synth_out, index=False)
        print(f"Synthetic saved: {args.synth_out} (rows={len(synth)})")

if __name__ == "__main__":
    main()
