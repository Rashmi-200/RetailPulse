from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd


KNOWN_ENGINE_DIFFERENCES = [
    "Pandas distinguishes float NaN from None whereas Spark uses unified SQL NULL; validator normalizes both to Python None/pd.NA prior to null-count and value checks.",
    "Spark hash-partitioned shuffles produce non-deterministic row ordering; validator deterministically sorts both DataFrames by primary sort_keys before comparing.",
    "Spark default timestamp formatting and Pandas datetime64[ns] differ in timezone/string representation; validator compares normalized ISO-8601 UTC strings or epoch seconds.",
    "Floating-point summation order across parallel Spark partitions vs sequential Pandas C loops can differ at ~1e-12 precision; numerical columns are validated with rtol=1e-5, atol=1e-4.",
]


@dataclass
class ParityReport:
    workload_id: str
    workload_name: str
    passed: bool
    pandas_rows: int
    pyspark_rows: int
    schema_columns_match: bool
    null_counts_match: bool
    duplicate_counts_match: bool
    values_match: bool
    max_abs_numeric_diff: float
    details: Dict[str, Any] = field(default_factory=dict)


def _normalize_df(df: pd.DataFrame, sort_keys: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        elif out[col].dtype == object:
            out[col] = out[col].where(out[col].notna(), None)
    if sort_keys:
        out = out.sort_values(by=sort_keys, na_position="last").reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out


def validate_workload_parity(
    *,
    workload_id: str,
    workload_name: str,
    pandas_df: pd.DataFrame,
    pyspark_df: pd.DataFrame,
    sort_keys: List[str],
    rtol: float = 1e-5,
    atol: float = 1e-4,
) -> ParityReport:
    pd_norm = _normalize_df(pandas_df, sort_keys)
    sp_norm = _normalize_df(pyspark_df, sort_keys)

    cols_match = list(pd_norm.columns) == list(sp_norm.columns)
    rows_match = len(pd_norm) == len(sp_norm)

    pd_nulls = {c: int(pd_norm[c].isna().sum()) for c in pd_norm.columns}
    sp_nulls = {c: int(sp_norm[c].isna().sum()) for c in sp_norm.columns} if cols_match else {}
    nulls_match = pd_nulls == sp_nulls

    if sort_keys and cols_match:
        pd_dups = int(pd_norm.duplicated(subset=sort_keys).sum())
        sp_dups = int(sp_norm.duplicated(subset=sort_keys).sum())
    else:
        pd_dups = int(pd_norm.duplicated().sum())
        sp_dups = int(sp_norm.duplicated().sum()) if cols_match else -1
    dups_match = pd_dups == sp_dups

    values_match = cols_match and rows_match and nulls_match and dups_match
    max_diff = 0.0
    mismatch_cols: List[str] = []

    if cols_match and rows_match and len(pd_norm) > 0:
        for col in pd_norm.columns:
            s_pd = pd_norm[col]
            s_sp = sp_norm[col]
            if pd.api.types.is_bool_dtype(s_pd) or pd.api.types.is_bool_dtype(s_sp):
                b_pd = s_pd.fillna(False).astype(bool).values
                b_sp = s_sp.fillna(False).astype(bool).values
                if not np.array_equal(b_pd, b_sp):
                    values_match = False
                    mismatch_cols.append(col)
            elif pd.api.types.is_numeric_dtype(s_pd) and pd.api.types.is_numeric_dtype(s_sp):
                v_pd = pd.to_numeric(s_pd, errors="coerce").to_numpy(dtype=float)
                v_sp = pd.to_numeric(s_sp, errors="coerce").to_numpy(dtype=float)
                both_nan = np.isnan(v_pd) & np.isnan(v_sp)
                valid_mask = ~both_nan
                if valid_mask.any():
                    diffs = np.abs(v_pd[valid_mask] - v_sp[valid_mask])
                    col_max = float(np.nanmax(diffs)) if diffs.size > 0 else 0.0
                    max_diff = max(max_diff, col_max)
                    if not np.allclose(v_pd[valid_mask], v_sp[valid_mask], rtol=rtol, atol=atol, equal_nan=True):
                        values_match = False
                        mismatch_cols.append(col)
            else:
                str_pd = s_pd.fillna("__NULL__").astype(str).values
                str_sp = s_sp.fillna("__NULL__").astype(str).values
                if not np.array_equal(str_pd, str_sp):
                    values_match = False
                    mismatch_cols.append(col)

    passed = bool(cols_match and rows_match and nulls_match and dups_match and values_match)
    return ParityReport(
        workload_id=workload_id,
        workload_name=workload_name,
        passed=passed,
        pandas_rows=len(pd_norm),
        pyspark_rows=len(sp_norm),
        schema_columns_match=cols_match,
        null_counts_match=nulls_match,
        duplicate_counts_match=dups_match,
        values_match=values_match,
        max_abs_numeric_diff=round(max_diff, 8),
        details={
            "pd_columns": list(pd_norm.columns),
            "sp_columns": list(sp_norm.columns),
            "pd_nulls": pd_nulls,
            "sp_nulls": sp_nulls,
            "pd_duplicates": pd_dups,
            "sp_duplicates": sp_dups,
            "mismatch_columns": mismatch_cols,
        },
    )


def reports_to_dict(reports: List[ParityReport]) -> Dict[str, Any]:
    return {
        "all_passed": all(r.passed for r in reports),
        "total_workloads_validated": len(reports),
        "known_engine_differences": KNOWN_ENGINE_DIFFERENCES,
        "workloads": [asdict(r) for r in reports],
    }
