# scripts/synthetic_timeseries/_rx_ts_utils.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# require refactoring to make it flexible
SUBJ_COL = "subject_id"
HADM_COL = "hadm_id"
START_COL = "startdate"
END_COL = "enddate"
DRUG_COL = "drug"

def load_rx(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # sanity check
    required = [SUBJ_COL, START_COL]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}' in {path}")
    # parse datetimes
    df[START_COL] = pd.to_datetime(df[START_COL], errors="coerce")
    if END_COL in df.columns:
        df[END_COL] = pd.to_datetime(df[END_COL], errors="coerce")
    else:
        df[END_COL] = pd.NaT
    # drop rows with no startdate
    df = df.dropna(subset=[START_COL])
    # fill missing enddate as +1 day
    df.loc[df[END_COL].isna(), END_COL] = df[START_COL] + pd.Timedelta(days=1)
    # normalize drug string if present
    if DRUG_COL in df.columns:
        df[DRUG_COL] = df[DRUG_COL].astype(str).str.strip().str.lower()
    return df

def pick_topk_drugs(df: pd.DataFrame, k: int = 10) -> List[str]:
    if DRUG_COL not in df.columns:
        return []
    return df[DRUG_COL].value_counts().head(k).index.tolist()

def build_daily_matrix(
    df: pd.DataFrame,
    top_drugs: List[str],
) -> pd.DataFrame:
    rows = []
    for (s, h), grp in df.groupby([SUBJ_COL, HADM_COL], dropna=False):
        start_day = grp[START_COL].min().floor("D")
        end_day = grp[END_COL].max().ceil("D")
        if pd.isna(start_day) or pd.isna(end_day) or end_day < start_day:
            continue
        idx = pd.date_range(start_day, end_day, freq="D", inclusive="left")
        out = pd.DataFrame({SUBJ_COL: s, HADM_COL: h, "date": idx})
        out["any_rx"] = 0
        for d in top_drugs:
            out[f"rx_{d}"] = 0

        for _, r in grp.iterrows():
            sday = r[START_COL].floor("D")
            eday = r[END_COL].ceil("D")
            mask = (out["date"] >= sday) & (out["date"] < eday)
            out.loc[mask, "any_rx"] = 1
            dname = str(r.get(DRUG_COL, "")).strip().lower()
            if dname in top_drugs:
                out.loc[mask, f"rx_{dname}"] = 1

        rows.append(out)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def windowize(
    daily: pd.DataFrame,
    window_len: int = 30,
    stride: int = 7,
    features: List[str] = None,
) -> Tuple[pd.DataFrame, Dict]:
    if features is None:
        base = ["any_rx"]
        drug_cols = [c for c in daily.columns if c.startswith("rx_")]
        features = base + drug_cols

    records = []
    for (s, h), grp in daily.groupby([SUBJ_COL, HADM_COL], dropna=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        n = len(grp)
        for start in range(0, max(0, n - window_len + 1), stride):
            end = start + window_len
            if end > n:
                break
            slc = grp.iloc[start:end]
            row = {
                SUBJ_COL: s,
                HADM_COL: h,
                "start_date": slc["date"].iloc[0],
                "start_month_bin": slc["date"].iloc[0].to_period("M").strftime("%Y-%m"),
            }
            for f in features:
                vals = slc[f].to_numpy()
                for t in range(window_len):
                    row[f"{f}_t{t}"] = float(vals[t]) if np.isfinite(vals[t]) else 0.0
            records.append(row)

    win_df = pd.DataFrame(records)
    if not win_df.empty:
        win_df["start_month_bin"] = win_df["start_month_bin"].astype("object")

    meta = {
        "entity_cols": (SUBJ_COL, HADM_COL),
        "window_len": window_len,
        "stride": stride,
        "features": features,
    }
    return win_df, meta

def row_to_sequence(row: pd.Series, meta: Dict) -> pd.DataFrame:
    T = meta["window_len"]
    feats = meta["features"]
    dates = pd.date_range(pd.to_datetime(row["start_date"]), periods=T, freq="D")
    data = {"date": dates}
    for f in feats:
        data[f] = [row[f"{f}_t{t}"] for t in range(T)]
    data[SUBJ_COL] = row[SUBJ_COL]
    data[HADM_COL] = row[HADM_COL]
    return pd.DataFrame(data)
