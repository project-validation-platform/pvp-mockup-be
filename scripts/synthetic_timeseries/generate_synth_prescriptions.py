# scripts/synthetic_timeseries/generate_synth_prescriptions.py
#!/usr/bin/env python3
import argparse, os, pickle
from pathlib import Path
import pandas as pd
from scripts.synthetic_timeseries._rx_ts_utils import row_to_sequence

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to saved .pkl from training")
    ap.add_argument("--n_sequences", type=int, default=100)
    ap.add_argument("--out", default="samples/mimic/synthetic/timeseries/prescriptions_sequences.csv")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    meta = bundle["meta"]

    # Sample windows, then convert each window-row to a T-length sequence
    win_df = model.sample(args.n_sequences)
    seqs = []
    for _, r in win_df.iterrows():
        seq = row_to_sequence(r, meta)
        # carry ID columns if present
        for idc in ("SUBJECT_ID","HADM_ID"):
            if idc in win_df.columns:
                seq[idc] = r[idc]
        seqs.append(seq)
    out = pd.concat(seqs, ignore_index=True)

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Synthetic sequences saved in {args.out} (rows={len(out)})")

if __name__ == "__main__":
    main()
