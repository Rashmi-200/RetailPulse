import json
from typing import Any, Dict, List, Optional
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.observability.tracker import generate_id, utc_now
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OpenLineageTracker:
    """Emits OpenLineage-compatible RunEvents with dataset-level and column-level lineage."""

    def __init__(self, engine: Engine, namespace: str = "retailpulse.lakehouse"):
        self.engine = engine
        self.namespace = namespace

    def emit_event(
        self,
        event_type: str,
        run_id: str,
        job_name: str,
        inputs: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]],
        batch_id: Optional[str] = None,
        column_lineage: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        """Record an OpenLineage-compatible START/COMPLETE/FAIL event into meta.lineage_events."""
        event_id = generate_id("lin")
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO meta.lineage_events (
                        event_id, event_type, event_time, run_id, job_namespace,
                        job_name, batch_id, input_datasets, output_datasets, column_lineage
                    ) VALUES (
                        :event_id, :event_type, :event_time, :run_id, :job_namespace,
                        :job_name, :batch_id, CAST(:inputs AS JSONB),
                        CAST(:outputs AS JSONB), CAST(:column_lineage AS JSONB)
                    )
                    """
                ),
                {
                    "event_id": event_id,
                    "event_type": event_type.upper(),
                    "event_time": utc_now(),
                    "run_id": run_id,
                    "job_namespace": self.namespace,
                    "job_name": job_name,
                    "batch_id": batch_id,
                    "inputs": json.dumps(inputs),
                    "outputs": json.dumps(outputs),
                    "column_lineage": json.dumps(column_lineage or {}),
                },
            )
        logger.info(
            f"[LINEAGE {event_type.upper()}] {job_name}: "
            f"{[i['name'] for i in inputs]} -> {[o['name'] for o in outputs]}"
        )
        return event_id

    def get_lineage_for_dataset(self, qualified_dataset: str) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve upstream producers and downstream consumers for impact & root-cause analysis."""
        query = text(
            """
            SELECT event_id, event_type, event_time, run_id, job_name,
                   batch_id, input_datasets, output_datasets, column_lineage
            FROM meta.lineage_events
            WHERE event_type = 'COMPLETE'
            ORDER BY event_time DESC;
            """
        )
        upstream = []
        downstream = []
        with self.engine.connect() as conn:
            for row in conn.execute(query):
                inputs = row[6] if isinstance(row[6], list) else json.loads(row[6])
                outputs = row[7] if isinstance(row[7], list) else json.loads(row[7])
                col_lin = row[8] if isinstance(row[8], dict) else json.loads(row[8] or "{}")

                in_names = {i.get("name") for i in inputs}
                out_names = {o.get("name") for o in outputs}

                record = {
                    "job_name": row[4],
                    "run_id": row[3],
                    "batch_id": row[5],
                    "inputs": list(in_names),
                    "outputs": list(out_names),
                    "column_lineage": col_lin,
                }
                if qualified_dataset in out_names:
                    upstream.append(record)
                if qualified_dataset in in_names:
                    downstream.append(record)

        return {"dataset": qualified_dataset, "upstream_jobs": upstream, "downstream_jobs": downstream}
