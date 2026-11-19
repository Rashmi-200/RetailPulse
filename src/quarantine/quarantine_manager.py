import json
from pathlib import Path
from typing import List, Optional
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.observability.tracker import ObservabilityTracker, generate_id, utc_now
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QuarantineManager:
    """Isolates invalid records into Parquet + PostgreSQL with error classification and audit tracking."""

    def __init__(self, engine: Engine, tracker: ObservabilityTracker, quarantine_dir: str = "data/quarantine"):
        self.engine = engine
        self.tracker = tracker
        self.quarantine_dir = Path(quarantine_dir)

    @staticmethod
    def classify_error(reason: str) -> tuple[str, str]:
        reason_upper = reason.upper()
        if "DUPLICATE_PRIMARY_KEY" in reason_upper:
            return "DUPLICATE_KEY", "DEDUPLICATE_OR_DROP"
        if "NULL_IN_" in reason_upper:
            return "MISSING_REQUIRED_FIELD", "MANUAL_SOURCE_CORRECTION"
        if "INVALID_NUMERIC_TYPE" in reason_upper:
            return "TYPE_MISMATCH", "CAST_OR_SOURCE_FIX"
        if "OUT_OF_RANGE" in reason_upper:
            return "RANGE_VIOLATION", "MANUAL_REVIEW_AND_CLIP"
        if "INVALID_DOMAIN" in reason_upper:
            return "DOMAIN_VIOLATION", "MAP_TO_STANDARD_DOMAIN"
        if "REFERENTIAL_INTEGRITY" in reason_upper:
            return "ORPHAN_FOREIGN_KEY", "WAIT_FOR_PARENT_BATCH_AND_REPLAY"
        return "CONTRACT_VIOLATION", "MANUAL_INSPECTION"

    def quarantine_records(
        self,
        invalid_df: pd.DataFrame,
        dataset_name: str,
        batch_id: str,
        pk_cols: List[str],
        run_id: Optional[str] = None,
    ) -> int:
        if invalid_df.empty:
            return 0

        count = len(invalid_df)
        out_dir = self.quarantine_dir / dataset_name / f"batch_id={batch_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = out_dir / "quarantined.parquet"
        invalid_df.to_parquet(parquet_path, index=False)

        now = utc_now()
        rows_to_insert = []
        for idx, row in invalid_df.iterrows():
            reason = str(row.get("_rejection_reason", "UNKNOWN_ERROR"))
            error_cat, rec_action = self.classify_error(reason)

            if pk_cols and all(c in invalid_df.columns for c in pk_cols):
                rec_key = "|".join(str(row[c]) for c in pk_cols)
            else:
                rec_key = f"row_{idx}"

            clean_dict = {
                k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else str(v) if not isinstance(v, (int, float, bool)) else v))
                for k, v in row.items()
                if not k.startswith("_")
            }

            rows_to_insert.append(
                {
                    "quarantine_id": generate_id("quar"),
                    "batch_id": batch_id,
                    "run_id": run_id,
                    "dataset_name": dataset_name,
                    "record_key": rec_key[:256],
                    "error_category": error_cat,
                    "rejection_reason": reason,
                    "raw_payload": json.dumps(clean_dict),
                    "recovery_status": "QUARANTINED",
                    "recommended_action": rec_action,
                    "quarantined_at": now,
                }
            )

        # Idempotent clean for same batch_id + dataset_name before inserting
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    DELETE FROM quarantine.quarantined_records
                    WHERE batch_id = :batch_id AND dataset_name = :dataset_name
                    """
                ),
                {"batch_id": batch_id, "dataset_name": dataset_name},
            )
            conn.execute(
                text(
                    """
                    INSERT INTO quarantine.quarantined_records (
                        quarantine_id, batch_id, run_id, dataset_name, record_key,
                        error_category, rejection_reason, raw_payload, recovery_status,
                        recommended_action, quarantined_at
                    ) VALUES (
                        :quarantine_id, :batch_id, :run_id, :dataset_name, :record_key,
                        :error_category, :rejection_reason, CAST(:raw_payload AS JSONB),
                        :recovery_status, :recommended_action, :quarantined_at
                    )
                    """
                ),
                rows_to_insert,
            )

        self.tracker.create_incident(
            incident_type="RECORDS_QUARANTINED",
            severity="MEDIUM" if count < 1000 else "HIGH",
            dataset_name=dataset_name,
            layer="silver",
            batch_id=batch_id,
            run_id=run_id,
            evidence_refs={"quarantined_count": count, "parquet_path": str(parquet_path)},
            recommended_action="Review quarantine.quarantined_records and trigger safe replay after fix.",
        )
        logger.warning(f"[QUARANTINED] {dataset_name}: {count:,} invalid records isolated (batch={batch_id})")
        return count
