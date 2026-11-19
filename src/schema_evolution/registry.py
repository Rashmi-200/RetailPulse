import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.observability.tracker import ObservabilityTracker, utc_now
from src.utils.logger import get_logger

logger = get_logger(__name__)

INTERNAL_META_COLS = {"_batch_id", "_ingested_at", "_source_system"}


@dataclass
class SchemaCheckResult:
    dataset_name: str
    layer: str
    version_number: int
    change_type: str
    is_breaking: bool
    diff_summary: Dict[str, Any]


class SchemaEvolutionRegistry:
    """Tracks dataset schema versions, detects drift, and blocks breaking changes."""

    def __init__(self, engine: Engine, tracker: ObservabilityTracker):
        self.engine = engine
        self.tracker = tracker

    @staticmethod
    def extract_schema_definition(df: pd.DataFrame) -> Dict[str, str]:
        return {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
            if col not in INTERNAL_META_COLS
        }

    @staticmethod
    def compute_schema_hash(schema_def: Dict[str, str]) -> str:
        canonical = json.dumps(schema_def, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _get_latest_version(self, dataset_name: str, layer: str) -> Optional[Dict[str, Any]]:
        query = text(
            """
            SELECT version_number, schema_hash, schema_definition
            FROM meta.schema_versions
            WHERE dataset_name = :dataset_name AND layer = :layer
            ORDER BY version_number DESC
            LIMIT 1;
            """
        )
        with self.engine.connect() as conn:
            row = conn.execute(query, {"dataset_name": dataset_name, "layer": layer}).fetchone()
        if not row:
            return None
        schema_def = row[2] if isinstance(row[2], dict) else json.loads(row[2])
        return {"version_number": row[0], "schema_hash": row[1], "schema_definition": schema_def}

    def check_and_register(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        layer: str,
        required_columns: List[str],
        batch_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> SchemaCheckResult:
        current_schema = self.extract_schema_definition(df)
        current_hash = self.compute_schema_hash(current_schema)
        latest = self._get_latest_version(dataset_name, layer)

        # Case 1: First time registering this schema
        if latest is None:
            self._insert_version(dataset_name, layer, 1, current_hash, current_schema, "initial", False, {})
            logger.info(f"[SCHEMA REGISTERED] {layer}.{dataset_name} v1 ({len(current_schema)} cols)")
            return SchemaCheckResult(dataset_name, layer, 1, "initial", False, {})

        # Case 2: Identical schema
        if latest["schema_hash"] == current_hash:
            return SchemaCheckResult(dataset_name, layer, latest["version_number"], "unchanged", False, {})

        # Case 3: Schema drift detected -> classify breaking vs non-breaking
        prev_schema: Dict[str, str] = latest["schema_definition"]
        prev_cols = set(prev_schema.keys())
        curr_cols = set(current_schema.keys())

        added_cols = sorted(list(curr_cols - prev_cols))
        removed_cols = sorted(list(prev_cols - curr_cols))
        removed_required = sorted([c for c in removed_cols if c in required_columns])

        type_changes = {}
        for col in prev_cols.intersection(curr_cols):
            if prev_schema[col] != current_schema[col]:
                # Allow benign int64 <-> float64 widening due to NaNs, flag incompatible shifts
                numeric_types = {"int64", "float64", "Int64", "Float64"}
                if not ({prev_schema[col], current_schema[col]}.issubset(numeric_types)):
                    type_changes[col] = {"from": prev_schema[col], "to": current_schema[col]}

        is_breaking = bool(removed_required or type_changes)
        change_type = "breaking_drift" if is_breaking else "non_breaking_evolution"
        diff_summary = {
            "added_columns": added_cols,
            "removed_columns": removed_cols,
            "removed_required_columns": removed_required,
            "type_changes": type_changes,
        }

        new_version = latest["version_number"] + 1
        self._insert_version(
            dataset_name, layer, new_version, current_hash, current_schema, change_type, is_breaking, diff_summary
        )

        severity = "CRITICAL" if is_breaking else "WARNING"
        self.tracker.create_incident(
            incident_type="SCHEMA_DRIFT",
            severity=severity,
            dataset_name=dataset_name,
            layer=layer,
            batch_id=batch_id,
            run_id=run_id,
            evidence_refs={"version": new_version, "diff": diff_summary},
            recommended_action=(
                "Block downstream Silver load and fix source schema."
                if is_breaking
                else "Review newly added optional columns and update data contract."
            ),
        )

        if is_breaking:
            raise ValueError(
                f"Breaking schema drift detected on {layer}.{dataset_name} (v{new_version}): {diff_summary}"
            )

        logger.warning(f"[SCHEMA EVOLUTION] Non-breaking drift on {layer}.{dataset_name} v{new_version}: {diff_summary}")
        return SchemaCheckResult(dataset_name, layer, new_version, change_type, False, diff_summary)

    def _insert_version(
        self,
        dataset_name: str,
        layer: str,
        version_number: int,
        schema_hash: str,
        schema_def: Dict[str, str],
        change_type: str,
        is_breaking: bool,
        diff_summary: Dict[str, Any],
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO meta.schema_versions (
                        dataset_name, layer, version_number, schema_hash,
                        schema_definition, change_type, is_breaking, diff_summary, registered_at
                    ) VALUES (
                        :dataset_name, :layer, :version_number, :schema_hash,
                        CAST(:schema_definition AS JSONB), :change_type, :is_breaking,
                        CAST(:diff_summary AS JSONB), :registered_at
                    )
                    """
                ),
                {
                    "dataset_name": dataset_name,
                    "layer": layer,
                    "version_number": version_number,
                    "schema_hash": schema_hash,
                    "schema_definition": json.dumps(schema_def),
                    "change_type": change_type,
                    "is_breaking": is_breaking,
                    "diff_summary": json.dumps(diff_summary),
                    "registered_at": utc_now(),
                },
            )
