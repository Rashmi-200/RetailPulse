import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.observability.tracker import ObservabilityTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SchemaEvolutionResult:
    dataset_name: str
    layer: str
    version_number: int
    schema_hash: str
    change_type: str
    is_breaking: bool
    diff_summary: Dict[str, Any]


class SchemaEvolutionRegistry:
    """Tracks canonical dataset schemas, detects breaking vs non-breaking drift, and records history."""

    def __init__(self, engine: Engine, tracker: ObservabilityTracker):
        self.engine = engine
        self.tracker = tracker

    @staticmethod
    def extract_canonical_schema(df: pd.DataFrame) -> Dict[str, str]:
        return {col: str(df[col].dtype) for col in sorted(df.columns)}

    @staticmethod
    def compute_schema_hash(schema_def: Dict[str, str]) -> str:
        canonical = json.dumps(schema_def, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _is_compatible_type_shift(old_dtype: str, new_dtype: str) -> bool:
        if old_dtype == new_dtype:
            return True
        numeric_widening = {("int32", "int64"), ("int64", "float64"), ("float32", "float64")}
        return (old_dtype, new_dtype) in numeric_widening

    def check_and_register(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        layer: str,
        required_columns: Optional[List[str]] = None,
        run_id: Optional[str] = None,
    ) -> SchemaEvolutionResult:
        required_columns = required_columns or []
        curr_schema = self.extract_canonical_schema(df)
        curr_hash = self.compute_schema_hash(curr_schema)

        with self.engine.connect() as conn:
            latest = conn.execute(
                text(
                    """
                    SELECT version_number, schema_hash, schema_definition
                    FROM meta.schema_versions
                    WHERE dataset_name = :dataset_name AND layer = :layer AND is_breaking = FALSE
                    ORDER BY version_number DESC
                    LIMIT 1;
                    """
                ),
                {"dataset_name": dataset_name, "layer": layer},
            ).mappings().fetchone()

            max_v = conn.execute(
                text(
                    """
                    SELECT COALESCE(MAX(version_number), 0)
                    FROM meta.schema_versions
                    WHERE dataset_name = :dataset_name AND layer = :layer;
                    """
                ),
                {"dataset_name": dataset_name, "layer": layer},
            ).scalar()

        if latest is None:
            self._insert_version(dataset_name, layer, 1, curr_hash, curr_schema, "initial", False, {"status": "baseline"})
            logger.info(f"[SCHEMA REGISTERED] {layer}.{dataset_name} v1 ({len(curr_schema)} cols)")
            return SchemaEvolutionResult(dataset_name, layer, 1, curr_hash, "initial", False, {"status": "baseline"})

        if latest["schema_hash"] == curr_hash:
            return SchemaEvolutionResult(
                dataset_name, layer, int(latest["version_number"]), curr_hash, "unchanged", False, {}
            )

        prev_schema: Dict[str, str] = latest["schema_definition"]
        added_cols = sorted(list(set(curr_schema.keys()) - set(prev_schema.keys())))
        removed_cols = sorted(list(set(prev_schema.keys()) - set(curr_schema.keys())))
        removed_required = [c for c in removed_cols if c in required_columns]

        type_changes: Dict[str, Dict[str, str]] = {}
        for col in set(curr_schema.keys()).intersection(set(prev_schema.keys())):
            if not self._is_compatible_type_shift(prev_schema[col], curr_schema[col]):
                type_changes[col] = {"from": prev_schema[col], "to": curr_schema[col]}

        is_breaking = bool(removed_cols or type_changes)
        change_type = "breaking_drift" if is_breaking else "non_breaking_evolution"
        next_version = int(max_v) + 1
        diff_summary = {
            "added_columns": added_cols,
            "removed_columns": removed_cols,
            "removed_required_columns": removed_required,
            "type_changes": type_changes,
        }

        self._insert_version(dataset_name, layer, next_version, curr_hash, curr_schema, change_type, is_breaking, diff_summary)

        severity = "CRITICAL" if is_breaking else "WARNING"
        self.tracker.create_incident(
            incident_type="SCHEMA_DRIFT",
            severity=severity,
            dataset_name=dataset_name,
            layer=layer,
            run_id=run_id,
            recommended_action=(
                f"Block promotion and inspect schema drift on {layer}.{dataset_name}: {json.dumps(diff_summary)}"
                if is_breaking
                else f"Review additive schema evolution v{next_version} on {layer}.{dataset_name}: {added_cols}"
            ),
        )

        if is_breaking:
            logger.error(f"[SCHEMA DRIFT BLOCKED] Breaking drift on {layer}.{dataset_name}: {diff_summary}")
            raise ValueError(f"Breaking schema drift detected on {layer}.{dataset_name}: {diff_summary}")

        logger.warning(f"[SCHEMA EVOLUTION] Non-breaking drift on {layer}.{dataset_name} v{next_version}: {diff_summary}")
        return SchemaEvolutionResult(dataset_name, layer, next_version, curr_hash, change_type, False, diff_summary)

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
                    "registered_at": datetime.now(timezone.utc),
                },
            )
