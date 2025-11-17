#!/usr/bin/env python3
import argparse, os, pickle
from pathlib import Path
import pandas as pd

from snsynth.pytorch.nn import DPCTGAN
from ._preprocess_utils import preprocess_for_smartnoise

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/ADMISSIONS.csv")
    ap.add_argument("--out", default="samples/models/dpctgan_mimic_admissions.pkl")
    ap.add_argument("--synth_out", default="samples/mimic/synthetic/ADMISSIONS_synth_dpctgan.csv")
    ap.add_argument("--n_samples", type=int, default=1000)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=500)
    ap.add_argument("--epsilon", type=float, default=1.0, help="Total DP budget")
    ap.add_argument("--preprocessor_eps", type=float, default=0.1, help="DP budget for bounds")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df_clean, categorical_columns, continuous_columns = preprocess_for_smartnoise(df, bin_freq="M", low_card_threshold=10)

    model = DPCTGAN(epochs=args.epochs, batch_size=args.batch_size, epsilon=args.epsilon, verbose=True)
    
    if args.batch_size > len(df_clean):
        raise ValueError(f"Batch size ({args.batch_size}) must be <= number of samples ({len(df_clean)}).")

    model.fit(
        df_clean,
        categorical_columns=categorical_columns,
        continuous_columns=continuous_columns,
        ordinal_columns=[],
        preprocessor_eps=args.preprocessor_eps,
        nullable=False,
    )

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f: pickle.dump(model, f)
    print(f"DPCTGAN model saved: {args.out}")

    if args.n_samples > 0:
        synth = model.sample(args.n_samples)
        Path(os.path.dirname(args.synth_out)).mkdir(parents=True, exist_ok=True)
        synth.to_csv(args.synth_out, index=False)
        print(f"Synthetic saved: {args.synth_out} (rows={len(synth)})")

if __name__ == "__main__":
    main()
