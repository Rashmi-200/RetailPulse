import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.observability.tracker import utc_now
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ContractValidationReport:
    dataset_name: str
    total_records: int
    valid_records: int
    invalid_records: int
    duplicate_records: int
    overall_score: float
    dimension_scores: Dict[str, float]
    rule_results: List[Dict[str, Any]]
    profile_summary: Dict[str, Any]
    valid_df: pd.DataFrame
    invalid_df: pd.DataFrame


class DataContractEngine:
    """Profiles datasets, enforces YAML data contracts across 6 DQ dimensions, and splits valid vs invalid rows."""

    def __init__(self, engine: Engine, contracts_config: Dict[str, Any]):
        self.engine = engine
        self.contracts = contracts_config.get("contracts", {})

    def profile_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        profile: Dict[str, Any] = {"row_count": int(len(df)), "columns": {}}
        for col in df.columns:
            if col.startswith("_"):
                continue
            null_count = int(df[col].isna().sum())
            null_pct = round((null_count / len(df)) * 100, 2) if len(df) > 0 else 0.0
            col_info: Dict[str, Any] = {
                "dtype": str(df[col].dtype),
                "null_count": null_count,
                "null_pct": null_pct,
                "unique_count": int(df[col].nunique(dropna=True)),
            }
            if pd.api.types.is_numeric_dtype(df[col]) and not df[col].dropna().empty:
                col_info["min"] = float(df[col].min())
                col_info["max"] = float(df[col].max())
                col_info["mean"] = round(float(df[col].mean()), 4)
            profile["columns"][col] = col_info
        return profile

    def validate_and_split(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        batch_id: str,
        run_id: Optional[str] = None,
        layer: str = "silver",
        persist: bool = True,
    ) -> ContractValidationReport:
        contract = self.contracts.get(dataset_name, {})
        total_records = len(df)
        if total_records == 0:
            raise ValueError(f"Cannot validate empty dataset: {dataset_name}")

        work_df = df.copy()
        error_reasons = pd.Series([""] * total_records, index=work_df.index)
        rule_results: List[Dict[str, Any]] = []

        def flag_errors(mask: pd.Series, rule_name: str, dimension: str, detail: str):
            failed_count = int(mask.sum())
            pass_rate = round(((total_records - failed_count) / total_records) * 100, 2)
            rule_results.append(
                {
                    "rule": rule_name,
                    "dimension": dimension,
                    "failed_count": failed_count,
                    "pass_rate": pass_rate,
                    "status": "PASSED" if failed_count == 0 else "FAILED",
                }
            )
            if failed_count > 0:
                error_reasons.loc[mask] = error_reasons.loc[mask].apply(
                    lambda s: f"{s}; {detail}" if s else detail
                )

        # 1. Completeness: Required columns & non-null checks
        required_cols = contract.get("required_columns", [])
        missing_cols = [c for c in required_cols if c not in work_df.columns]
        if missing_cols:
            raise ValueError(f"Contract violation on {dataset_name}: missing required columns {missing_cols}")

        for col in contract.get("not_null", []):
            if col in work_df.columns:
                null_mask = work_df[col].isna() | (work_df[col].astype(str).str.strip() == "")
                flag_errors(null_mask, f"not_null:{col}", "Completeness", f"NULL_IN_{col.upper()}")

        # 2. Uniqueness: Primary key deduplication check
        pk_cols = contract.get("primary_key", [])
        duplicate_records = 0
        if pk_cols and all(c in work_df.columns for c in pk_cols):
            dup_mask = work_df.duplicated(subset=pk_cols, keep="first")
            duplicate_records = int(dup_mask.sum())
            flag_errors(dup_mask, f"unique_pk:{','.join(pk_cols)}", "Uniqueness", "DUPLICATE_PRIMARY_KEY")

        # 3. Accuracy & Validity: Numeric ranges & allowed categorical domains
        for col, bounds in contract.get("numeric_ranges", {}).items():
            if col in work_df.columns:
                numeric_series = pd.to_numeric(work_df[col], errors="coerce")
                type_fail_mask = work_df[col].notna() & numeric_series.isna()
                flag_errors(type_fail_mask, f"numeric_type:{col}", "Validity", f"INVALID_NUMERIC_TYPE_{col.upper()}")

                range_mask = pd.Series(False, index=work_df.index)
                if "min" in bounds:
                    range_mask = range_mask | (numeric_series.notna() & (numeric_series < bounds["min"]))
                if "max" in bounds:
                    range_mask = range_mask | (numeric_series.notna() & (numeric_series > bounds["max"]))
                flag_errors(range_mask, f"range:{col}", "Accuracy", f"OUT_OF_RANGE_{col.upper()}")

        for col, allowed in contract.get("allowed_values", {}).items():
            if col in work_df.columns:
                domain_mask = work_df[col].notna() & (~work_df[col].isin(allowed))
                flag_errors(domain_mask, f"allowed_values:{col}", "Validity", f"INVALID_DOMAIN_{col.upper()}")

        # Split valid vs invalid records
        invalid_mask = error_reasons != ""
        invalid_records = int(invalid_mask.sum())
        valid_records = total_records - invalid_records

        valid_df = work_df.loc[~invalid_mask].copy()
        invalid_df = work_df.loc[invalid_mask].copy()
        if not invalid_df.empty:
            invalid_df["_rejection_reason"] = error_reasons.loc[invalid_mask].values

        # Compute 6 DQ Dimension Scores (0 - 100)
        def avg_dim(dim_name: str, default: float = 100.0) -> float:
            rates = [r["pass_rate"] for r in rule_results if r["dimension"] == dim_name]
            return round(sum(rates) / len(rates), 2) if rates else default

        completeness = avg_dim("Completeness")
        uniqueness = avg_dim("Uniqueness")
        accuracy = avg_dim("Accuracy")
        validity = avg_dim("Validity")
        consistency = round((valid_records / total_records) * 100, 2)
        timeliness = 100.0 if "_ingested_at" in work_df.columns else 90.0

        dimension_scores = {
            "Completeness": completeness,
            "Accuracy": accuracy,
            "Consistency": consistency,
            "Validity": validity,
            "Uniqueness": uniqueness,
            "Timeliness": timeliness,
        }
        overall_score = round(sum(dimension_scores.values()) / len(dimension_scores), 2)
        profile_summary = self.profile_dataset(work_df)

        if persist:
            self._persist_quality_result(
                batch_id=batch_id,
                run_id=run_id,
                dataset_name=dataset_name,
                layer=layer,
                total_records=total_records,
                valid_records=valid_records,
                invalid_records=invalid_records,
                duplicate_records=duplicate_records,
                overall_score=overall_score,
                dimension_scores=dimension_scores,
                rule_results=rule_results,
                profile_summary=profile_summary,
            )

        logger.info(
            f"[DQ CONTRACT] {dataset_name}: Score={overall_score}/100 | "
            f"Valid={valid_records:,} | Invalid={invalid_records:,} | Duplicates={duplicate_records:,}"
        )

        return ContractValidationReport(
            dataset_name=dataset_name,
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=invalid_records,
            duplicate_records=duplicate_records,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            rule_results=rule_results,
            profile_summary=profile_summary,
            valid_df=valid_df,
            invalid_df=invalid_df,
        )

    def _persist_quality_result(
        self,
        batch_id: str,
        run_id: Optional[str],
        dataset_name: str,
        layer: str,
        total_records: int,
        valid_records: int,
        invalid_records: int,
        duplicate_records: int,
        overall_score: float,
        dimension_scores: Dict[str, float],
        rule_results: List[Dict[str, Any]],
        profile_summary: Dict[str, Any],
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO meta.quality_results (
                        batch_id, run_id, dataset_name, layer,
                        total_records, valid_records, invalid_records, duplicate_records,
                        overall_score, dimension_scores, rule_results, profile_summary, evaluated_at
                    ) VALUES (
                        :batch_id, :run_id, :dataset_name, :layer,
                        :total_records, :valid_records, :invalid_records, :duplicate_records,
                        :overall_score, CAST(:dimension_scores AS JSONB),
                        CAST(:rule_results AS JSONB), CAST(:profile_summary AS JSONB), :evaluated_at
                    )
                    """
                ),
                {
                    "batch_id": batch_id,
                    "run_id": run_id,
                    "dataset_name": dataset_name,
                    "layer": layer,
                    "total_records": total_records,
                    "valid_records": valid_records,
                    "invalid_records": invalid_records,
                    "duplicate_records": duplicate_records,
                    "overall_score": overall_score,
                    "dimension_scores": json.dumps(dimension_scores),
                    "rule_results": json.dumps(rule_results),
                    "profile_summary": json.dumps(profile_summary),
                    "evaluated_at": utc_now(),
                },
            )
