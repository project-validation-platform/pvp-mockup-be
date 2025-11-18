#!/usr/bin/env python3
import argparse, os, pickle
from pathlib import Path
import pandas as pd
import numpy as np

from sdv.single_table import CopulaGANSynthesizer
from sdv.metadata import SingleTableMetadata

from scripts.synthetic_training._preprocess_utils import preprocess_for_sdv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/ADMISSIONS.csv")
    ap.add_argument("--out", default="samples/models/copulagan_mimic_admissions.pkl")
    ap.add_argument("--synth_out", default="samples/mimic/synthetic/ADMISSIONS_synth_copulagan.csv")
    ap.add_argument("--n_samples", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=500)
    args = ap.parse_args()

    df = pd.read_csv(args.data)

    df_clean, discrete_cols = preprocess_for_sdv(df)
    
    for c in df_clean.columns:
        if c not in discrete_cols:
            df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")
            # replace +/-inf -> NaN
            df_clean[c] = df_clean[c].replace([np.inf, -np.inf], np.nan)

            # If everything became NaN, treat as categorical instead
            if df_clean[c].notna().sum() == 0:
                discrete_cols.append(c)
                df_clean[c] = df_clean[c].astype("object").fillna("Unknown")
            else:
                # median-impute numerics
                med = df_clean[c].median()
                df_clean[c] = df_clean[c].fillna(med)

    for c in discrete_cols:
        df_clean[c] = df_clean[c].astype(str).replace({None: "Unknown", "nan": "Unknown"}).fillna("Unknown")

    # SDV synthesizers require metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_clean)

    model = CopulaGANSynthesizer(
        metadata,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True,
    )
    model.fit(df_clean)

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(model, f)
    print(f"CopulaGANSynthesizer model saved: {args.out}")

    if args.n_samples > 0:
        synth = model.sample(num_rows=args.n_samples)
        Path(os.path.dirname(args.synth_out)).mkdir(parents=True, exist_ok=True)
        synth.to_csv(args.synth_out, index=False)
        print(f"Synthetic saved: {args.synth_out} (rows={len(synth)})")

if __name__ == "__main__":
    main()
