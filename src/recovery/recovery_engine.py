import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from deltalake import DeltaTable, write_deltalake
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.contracts.contract_engine import DataContractEngine
from src.observability.tracker import ObservabilityTracker, generate_id, utc_now
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SafeRecoveryEngine:
    """Classifies failures, recommends policy-governed actions, and executes human-approved idempotent replays."""

    def __init__(
        self,
        engine: Engine,
        tracker: ObservabilityTracker,
        policies_cfg: Dict[str, Any],
        contracts_cfg: Dict[str, Any],
        silver_dir: str = "data/silver",
    ):
        self.engine = engine
        self.tracker = tracker
        self.policies = policies_cfg.get("policies", {})
        self.contracts_cfg = contracts_cfg
        self.contract_engine = DataContractEngine(engine, contracts_cfg)
        self.silver_dir = Path(silver_dir)

    def recommend_recovery(self, failure_code: str) -> Dict[str, Any]:
        policy = self.policies.get(
            failure_code.upper(),
            {
                "category": "UNKNOWN_INCIDENT",
                "auto_retry_allowed": False,
                "recommended_action": "MANUAL_INVESTIGATION_REQUIRED",
                "requires_human_approval": True,
            },
        )
        return {"failure_code": failure_code.upper(), **policy}

    def replay_corrected_records(
        self,
        dataset_name: str,
        batch_id: str,
        corrected_df: pd.DataFrame,
        approved_by: str,
        incident_id: Optional[str] = None,
        notes: str = "Operator-approved post-correction replay",
    ) -> Dict[str, Any]:
        """Validate corrected records against Data Contract and idempotently merge into Silver Delta."""
        if not approved_by or not approved_by.strip():
            raise PermissionError("Human approval ('approved_by') is mandatory for quarantine replay operations.")

        contract = self.contracts_cfg["contracts"].get(dataset_name, {})
        pk_cols: List[str] = contract.get("primary_key", [])

        # 1. Re-validate corrected batch before touching Silver
        report = self.contract_engine.validate_and_split(
            df=corrected_df,
            dataset_name=dataset_name,
            batch_id=batch_id,
            layer="silver_recovery",
            persist=True,
        )
        if report.invalid_records > 0:
            recovery_id = self._log_recovery(
                incident_id=incident_id,
                batch_id=batch_id,
                dataset_name=dataset_name,
                failure_category="DATA_CONTRACT_VIOLATION",
                recovery_action="REPLAY_REJECTED_POST_VALIDATION",
                approved_by=approved_by,
                records_reprocessed=0,
                status="FAILED_VALIDATION",
                notes=f"Post-recovery validation failed for {report.invalid_records} rows.",
            )
            raise ValueError(
                f"Recovery aborted (recovery_id={recovery_id}): {report.invalid_records} records still fail contract."
            )

        # 2. Idempotent upsert into Silver Delta (deduplicate by primary key so replay never creates duplicates)
        silver_delta = self.silver_dir / dataset_name / "_delta"
        existing_df = DeltaTable(str(silver_delta)).to_pandas()

        replay_df = report.valid_df.copy()
        replay_df["_silver_processed_at"] = utc_now().isoformat()

        # Align columns to existing Silver schema
        for col in existing_df.columns:
            if col not in replay_df.columns:
                replay_df[col] = None
        replay_df = replay_df[existing_df.columns]

        combined = pd.concat([existing_df, replay_df], ignore_index=True)
        if pk_cols:
            combined = combined.drop_duplicates(subset=pk_cols, keep="last").reset_index(drop=True)

        write_deltalake(str(silver_delta), combined, mode="overwrite", schema_mode="overwrite")

        # 3. Mark quarantined records as RECOVERED & resolve incident if linked
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE quarantine.quarantined_records
                    SET recovery_status = 'RECOVERED', resolved_at = :resolved_at
                    WHERE batch_id = :batch_id AND dataset_name = :dataset_name AND recovery_status = 'QUARANTINED'
                    """
                ),
                {"resolved_at": utc_now(), "batch_id": batch_id, "dataset_name": dataset_name},
            )
            if incident_id:
                conn.execute(
                    text(
                        """
                        UPDATE meta.incidents
                        SET status = 'RESOLVED', resolved_at = :resolved_at
                        WHERE incident_id = :incident_id
                        """
                    ),
                    {"resolved_at": utc_now(), "incident_id": incident_id},
                )

        recovery_id = self._log_recovery(
            incident_id=incident_id,
            batch_id=batch_id,
            dataset_name=dataset_name,
            failure_category="DATA_CONTRACT_VIOLATION",
            recovery_action="IDEMPOTENT_SILVER_REPLAY",
            approved_by=approved_by,
            records_reprocessed=len(replay_df),
            status="SUCCESS",
            notes=notes,
        )
        logger.info(
            f"[RECOVERY SUCCESS] {recovery_id}: replayed {len(replay_df)} validated rows into silver.{dataset_name} "
            f"(approved_by={approved_by})"
        )
        return {
            "recovery_id": recovery_id,
            "dataset_name": dataset_name,
            "replayed_rows": len(replay_df),
            "silver_total_rows": len(combined),
            "status": "SUCCESS",
        }

    def _log_recovery(
        self,
        incident_id: Optional[str],
        batch_id: str,
        dataset_name: str,
        failure_category: str,
        recovery_action: str,
        approved_by: str,
        records_reprocessed: int,
        status: str,
        notes: str,
    ) -> str:
        recovery_id = generate_id("rec")
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO meta.recovery_history (
                        recovery_id, incident_id, batch_id, dataset_name, failure_category,
                        recovery_action, approved_by, records_reprocessed, status, notes, executed_at
                    ) VALUES (
                        :recovery_id, :incident_id, :batch_id, :dataset_name, :failure_category,
                        :recovery_action, :approved_by, :records_reprocessed, :status, :notes, :executed_at
                    )
                    """
                ),
                {
                    "recovery_id": recovery_id,
                    "incident_id": incident_id,
                    "batch_id": batch_id,
                    "dataset_name": dataset_name,
                    "failure_category": failure_category,
                    "recovery_action": recovery_action,
                    "approved_by": approved_by,
                    "records_reprocessed": records_reprocessed,
                    "status": status,
                    "notes": notes,
                    "executed_at": utc_now(),
                },
            )
        return recovery_id
