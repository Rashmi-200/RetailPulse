import json
from typing import Any, Dict, Optional
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.lineage.lineage_tracker import OpenLineageTracker


class RCAEvidenceCollector:
    """Correlates multi-source telemetry across meta.*, quarantine.*, and lineage for an incident."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.lineage = OpenLineageTracker(engine)

    def collect_for_incident(self, incident_id: str) -> Dict[str, Any]:
        with self.engine.connect() as conn:
            inc_row = conn.execute(
                text("SELECT * FROM meta.incidents WHERE incident_id = :id"),
                {"id": incident_id},
            ).mappings().fetchone()

            if not inc_row:
                raise ValueError(f"Incident not found: {incident_id}")

            incident = dict(inc_row)
            batch_id = incident.get("batch_id")
            run_id = incident.get("run_id")
            dataset_name = incident.get("dataset_name")
            layer = incident.get("layer")

            # 1. Task exceptions & execution metrics
            tasks = []
            if run_id:
                task_rows = conn.execute(
                    text(
                        """
                        SELECT task_run_id, task_name, layer, status, records_in, records_out,
                               exception_type, LEFT(stack_trace, 600) AS stack_trace_excerpt
                        FROM meta.task_runs
                        WHERE run_id = :run_id
                        """
                    ),
                    {"run_id": run_id},
                ).mappings().fetchall()
                tasks = [dict(r) for r in task_rows]

            # 2. Batch reconciliation metadata
            batch_info: Optional[Dict[str, Any]] = None
            if batch_id:
                b_row = conn.execute(
                    text(
                        """
                        SELECT batch_id, source_system, dataset_name, extraction_mode,
                               source_record_count, target_record_count, reconciliation_status
                        FROM meta.batch_metadata
                        WHERE batch_id = :batch_id
                        """
                    ),
                    {"batch_id": batch_id},
                ).mappings().fetchone()
                if b_row:
                    batch_info = dict(b_row)

            # 3. Data Quality validation results
            dq_results = []
            if batch_id or dataset_name:
                dq_rows = conn.execute(
                    text(
                        """
                        SELECT dq_result_id, batch_id, dataset_name, layer, total_records,
                               valid_records, invalid_records, duplicate_records,
                               overall_score, dimension_scores, rule_results
                        FROM meta.quality_results
                        WHERE (:batch_id IS NOT NULL AND batch_id = :batch_id)
                           OR (:dataset_name IS NOT NULL AND dataset_name = :dataset_name)
                        ORDER BY evaluated_at DESC
                        LIMIT 2
                        """
                    ),
                    {"batch_id": batch_id, "dataset_name": dataset_name},
                ).mappings().fetchall()
                dq_results = [dict(r) for r in dq_rows]

            # 4. Schema evolution history
            schema_changes = []
            if dataset_name:
                sc_rows = conn.execute(
                    text(
                        """
                        SELECT version_number, layer, change_type, is_breaking, diff_summary
                        FROM meta.schema_versions
                        WHERE dataset_name = :dataset_name
                        ORDER BY version_number DESC
                        LIMIT 3
                        """
                    ),
                    {"dataset_name": dataset_name},
                ).mappings().fetchall()
                schema_changes = [dict(r) for r in sc_rows]

            # 5. ML/Statistical anomaly alerts
            anomalies = []
            if batch_id or dataset_name:
                an_rows = conn.execute(
                    text(
                        """
                        SELECT anomaly_id, metric_name, detector_type, observed_value,
                               expected_range, severity, explanation
                        FROM meta.anomaly_alerts
                        WHERE (:batch_id IS NOT NULL AND batch_id = :batch_id)
                           OR (:dataset_name IS NOT NULL AND dataset_name = :dataset_name)
                        ORDER BY detected_at DESC
                        LIMIT 5
                        """
                    ),
                    {"batch_id": batch_id, "dataset_name": dataset_name},
                ).mappings().fetchall()
                anomalies = [dict(r) for r in an_rows]

            # 6. Quarantine sample records
            quarantine_samples = []
            if batch_id and dataset_name:
                q_rows = conn.execute(
                    text(
                        """
                        SELECT quarantine_id, record_key, error_category, rejection_reason, recommended_action
                        FROM quarantine.quarantined_records
                        WHERE batch_id = :batch_id AND dataset_name = :dataset_name
                        LIMIT 3
                        """
                    ),
                    {"batch_id": batch_id, "dataset_name": dataset_name},
                ).mappings().fetchall()
                quarantine_samples = [dict(r) for r in q_rows]

        # 7. OpenLineage upstream/downstream context
        qualified = f"{layer}.{dataset_name}" if (layer and dataset_name) else (dataset_name or "")
        lineage_context = self.lineage.get_lineage_for_dataset(qualified) if qualified else {}

        return {
            "incident_id": incident["incident_id"],
            "incident_type": incident["incident_type"],
            "severity": incident["severity"],
            "dataset_name": dataset_name,
            "layer": layer,
            "batch_id": batch_id,
            "run_id": run_id,
            "evidence_refs": incident.get("evidence_refs") or {},
            "task_executions": tasks,
            "batch_reconciliation": batch_info,
            "quality_reports": dq_results,
            "schema_history": schema_changes,
            "anomaly_alerts": anomalies,
            "quarantine_samples": quarantine_samples,
            "lineage": lineage_context,
        }
