"""
Leakage Detector
================
Flags columns that are suspiciously correlated with the target variable.
Does NOT drop columns — only returns a list of suspects for the UI to show.
"""
from typing import List

import pandas as pd


def detect_potential_leakage(
    df: pd.DataFrame,
    target_col: str,
    threshold: float = 0.98,
) -> List[str]:
    """Detect columns that may cause data leakage.

    Checks:
      - Numeric columns: absolute Pearson correlation > threshold with target.
      - Categorical columns: value-for-value identical to the target column.

    Args:
        df:          DataFrame to inspect (should be raw_df).
        target_col:  Name of the target column.
        threshold:   Pearson |r| above which a numeric column is flagged (default 0.98).

    Returns:
        List of suspicious column names (never includes target_col itself).
    """
    if target_col not in df.columns:
        return []

    suspicious: List[str] = []
    target_series = df[target_col]

    # ── Numeric check ────────────────────────────────────────────────────────
    # Only run when the target itself is numeric (correlation is meaningful).
    numeric_cols = df.select_dtypes(include="number").columns
    if pd.api.types.is_numeric_dtype(target_series):
        for col in numeric_cols:
            if col == target_col:
                continue
            valid = df[[col, target_col]].dropna()
            if len(valid) < 2 or valid[col].nunique() < 2:
                continue
            corr = valid[col].corr(valid[target_col])
            if abs(corr) > threshold:
                suspicious.append(col)

    # ── Categorical check ────────────────────────────────────────────────────
    # Flag any object/category column whose values are 100% identical to target.
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if col == target_col:
            continue
        aligned = df[[col, target_col]].dropna()
        if len(aligned) == 0:
            continue
        if (aligned[col].astype(str) == aligned[target_col].astype(str)).all():
            suspicious.append(col)

    return suspicious
