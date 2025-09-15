# Minimal, battle-tested preprocessors you can reuse from any trainer

from __future__ import annotations
import re
from typing import List, Tuple
import numpy as np
import pandas as pd

# --- For SDV models (CTGAN/TVAE/CopulaGAN/GaussianCopula) ---
def preprocess_for_sdv(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Returns:
      df_clean: DataFrame with datetimes -> epoch seconds, no NaNs
      discrete_columns: list of categorical col names to pass to SDV models
    """
    df = df.copy()

    # 1) datetime -> epoch seconds (float); keep NaN for now
    dt_pat = re.compile(r"(time|date)$", re.IGNORECASE)
    dt_cols = [c for c in df.columns if dt_pat.search(c)]
    for c in dt_cols:
        dt = pd.to_datetime(df[c], errors="coerce", utc=False)
        ts = (dt.astype("int64") // 1_000_000_000).astype(float)
        ts[dt.isna()] = np.nan
        df[c] = ts

    # 2) likely categoricals by dtype + name hints
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    name_hints = {"ID","CODE","TYPE","LOCATION","INSURANCE","LANGUAGE","RELIGION","MARITAL","ETHNICITY","DIAGNOSIS","FLAG"}
    by_name = [c for c in df.columns if any(h in c.upper() for h in name_hints)]
    discrete_columns = sorted(set(obj_cols + by_name))
    for c in discrete_columns:
        df[c] = df[c].astype("object")

    # 3) impute
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cont_cols = [c for c in num_cols if c not in discrete_columns]
    for c in cont_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    for c in discrete_columns:
        if df[c].isna().any():
            df[c] = df[c].fillna("Unknown")

    return df, discrete_columns

# --- For SmartNoise DP models (DPCTGAN/PATECTGAN) ---
def preprocess_for_smartnoise(df: pd.DataFrame, bin_freq: str = "M", low_card_threshold: int = 10):
    """
    Robust preprocessing that avoids MinMax bounds issues:
      - Datetimes -> categorical period bins
      - Low-cardinality numerics -> categorical
      - Clean continuous numerics (finite, impute, drop-constant)
    Returns:
      df_clean, categorical_columns, continuous_columns
    """
    df = df.copy()

    # 1) bin all datetimes to period strings; drop original
    dt_pat = re.compile(r"(time|date)$", re.IGNORECASE)
    dt_cols = [c for c in df.columns if dt_pat.search(c)]
    binned_dt = []
    for c in dt_cols:
        dt = pd.to_datetime(df[c], errors="coerce", utc=False)
        b = dt.dt.to_period(bin_freq).astype(str).fillna("Unknown")
        newc = f"{c}_{bin_freq.lower()}bin"
        df[newc] = b
        binned_dt.append(newc)
    if dt_cols:
        df.drop(columns=dt_cols, inplace=True)

    # 2) categoricals by dtype + name + low-card numeric
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    name_hints = {"ID","CODE","TYPE","LOCATION","INSURANCE","LANGUAGE","RELIGION","MARITAL","ETHNICITY","DIAGNOSIS","FLAG"}
    by_name = [c for c in df.columns if any(h in c.upper() for h in name_hints)]
    low_card_num = []
    for c in df.select_dtypes(include=["number"]).columns:
        if df[c].nunique(dropna=True) <= low_card_threshold:
            low_card_num.append(c)

    categorical_columns = sorted(set(obj_cols + by_name + binned_dt + low_card_num))

    # normalize categorical values to strings w/o None
    for c in categorical_columns:
        if c in df.columns:
            df[c] = df[c].astype(str).replace({None: "Unknown", "nan": "Unknown"}).fillna("Unknown")

    # 3) continuous = the remaining numerics
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    continuous_columns = [c for c in numeric_cols if c not in categorical_columns]

    # 4) clean continuous
    for c in list(continuous_columns):
        s = df[c].replace([np.inf, -np.inf], np.nan)
        if s.notna().sum() == 0:
            df.drop(columns=[c], inplace=True)
            continuous_columns.remove(c)
            continue
        med = s.median()
        if pd.isna(med): med = 0.0
        s = s.fillna(med)
        if s.min() == s.max():
            df.drop(columns=[c], inplace=True)
            continuous_columns.remove(c)
            continue
        df[c] = s

    return df, categorical_columns, continuous_columns
