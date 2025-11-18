#!/usr/bin/env python3
import argparse, os, pickle, re
from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd

SPARSE_DT_THRESHOLD = 0.20  # <20% non-null then bin to month & treat as categorical

def _import_patectgan():
    from snsynth.pytorch.nn import PATECTGAN
    return PATECTGAN
BIN_FREQ = "M"  # "M"=month buckets. Use "D" for day buckets if you want finer granularity.
def _preprocess_for_gan(df: pd.DataFrame):
    df = df.copy()

    # datetime binning
    dt_name_pattern = re.compile(r"(time|date)$", re.IGNORECASE)
    datetime_cols = [c for c in df.columns if dt_name_pattern.search(c)]
    binned_dt_as_cat = []
    for c in datetime_cols:
        dt = pd.to_datetime(df[c], errors="coerce", utc=False)
        binned = dt.dt.to_period("M").astype(str)
        df[c + "_mbin"] = binned
        binned_dt_as_cat.append(c + "_mbin")
        df.drop(columns=[c], inplace=True)

    # categorical detection
    name_hint_cats = {"ID","CODE","TYPE","LOCATION","INSURANCE","LANGUAGE",
                      "RELIGION","MARITAL","ETHNICITY","DIAGNOSIS","FLAG"}
    likely_cat_by_name = [c for c in df.columns if any(h in c.upper() for h in name_hint_cats)]
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols_all = df.select_dtypes(include=["number"]).columns.tolist()
    low_card_num_as_cat = []
    for c in numeric_cols_all:
        if c in obj_cols or c in likely_cat_by_name: continue
        nunq = df[c].nunique(dropna=True)
        if nunq <= 10:
            low_card_num_as_cat.append(c)

    categorical_columns = sorted(set(obj_cols + likely_cat_by_name + binned_dt_as_cat + low_card_num_as_cat))

    # force categorical dtype to str, replace None/NaN
    for c in categorical_columns:
        if c in df.columns:
            df[c] = df[c].astype(str)
            df[c] = df[c].replace({None: "Unknown", "nan": "Unknown"})
            df[c] = df[c].fillna("Unknown")

    # recompute continuous cols
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    continuous_columns = [c for c in numeric_cols if c not in categorical_columns]

    # impute numeric
    for c in list(continuous_columns):
        col = df[c].replace([np.inf, -np.inf], np.nan)
        if col.notna().sum() == 0:
            df.drop(columns=[c], inplace=True)
            continuous_columns.remove(c)
            continue
        med = col.median()
        if pd.isna(med): med = 0.0
        col = col.fillna(med)
        if col.min() == col.max():
            df.drop(columns=[c], inplace=True)
            continuous_columns.remove(c)
            continue
        df[c] = col

    print("categorical_columns ({}): {}".format(len(categorical_columns), categorical_columns))
    print("continuous_columns ({}): {}".format(len(continuous_columns), continuous_columns))
    return df, categorical_columns, continuous_columns



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/ADMISSIONS.csv")
    parser.add_argument("--out", default="samples/models/pategan_mimic_admissions.pkl")
    parser.add_argument("--synth_out", default="samples/mimic/synthetic/ADMISSIONS_synth_pategan.csv")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--epsilon", type=float, default=1.0, help="Total DP budget")
    parser.add_argument("--preprocessor-eps", type=float, default=None,
                        help="DP budget for bounds (default 10% of --epsilon)")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df_clean, categorical_columns, continuous_columns = _preprocess_for_gan(df)

    pre_eps = args.preprocessor_eps if args.preprocessor_eps is not None else max(1e-6, args.epsilon * 0.1)

    PATECTGAN = _import_patectgan()
    model = PATECTGAN(
        epochs=args.epochs,
        batch_size=args.batch_size,
        epsilon=args.epsilon,
        verbose=True,
    )

    model.fit(
        df_clean,
        categorical_columns=categorical_columns,
        ordinal_columns=[],
        continuous_columns=continuous_columns,
        preprocessor_eps=pre_eps,
        nullable=False,
    )

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(model, f)
    print(f"PATECTGAN model saved: {args.out}")

    if args.n_samples > 0:
        synth_df = model.sample(args.n_samples)
        Path(os.path.dirname(args.synth_out)).mkdir(parents=True, exist_ok=True)
        synth_df.to_csv(args.synth_out, index=False)
        print(f"Synthetic samples saved: {args.synth_out} (rows={len(synth_df)})")

if __name__ == "__main__":
    main()
