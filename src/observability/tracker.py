import json
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from sqlalchemy import text
from sqlalchemy.engine import Engine
from src.utils.logger import get_logger

logger = get_logger(__name__)


def utc_now() -> datetime:
    """Return timezone-naive UTC timestamp for consistent PostgreSQL storage."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def generate_id(prefix: str) -> str:
    """Generate a traceable prefixed identifier (e.g., run_..., batch_..., inc_...)."""
    ts = utc_now().strftime("%Y%m%d%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{prefix}_{ts}_{short_uuid}"


class ObservabilityTracker:
    """Records pipeline runs, task executions, batch reconciliation, freshness, and incidents."""

    def __init__(self, engine: Engine):
        self.engine = engine

    @contextmanager
    def track_pipeline_run(self, pipeline_name: str, trigger_type: str = "manual"):
        run_id = generate_id("run")
        started_at = utc_now()
        start_perf = time.perf_counter()

        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO meta.pipeline_runs (run_id, pipeline_name, trigger_type, status, started_at)
                    VALUES (:run_id, :pipeline_name, :trigger_type, 'RUNNING', :started_at)
                    """
                ),
                {
                    "run_id": run_id,
                    "pipeline_name": pipeline_name,
                    "trigger_type": trigger_type,
                    "started_at": started_at,
                },
            )

        logger.info(f"[PIPELINE START] {pipeline_name} (run_id={run_id})")
        try:
            yield run_id
            duration = round(time.perf_counter() - start_perf, 4)
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE meta.pipeline_runs
                        SET status = 'SUCCESS', ended_at = :ended_at, duration_seconds = :duration
                        WHERE run_id = :run_id
                        """
                    ),
                    {"ended_at": utc_now(), "duration": duration, "run_id": run_id},
                )
            logger.info(f"[PIPELINE SUCCESS] {pipeline_name} completed in {duration}s (run_id={run_id})")
        except Exception as exc:
            duration = round(time.perf_counter() - start_perf, 4)
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE meta.pipeline_runs
                        SET status = 'FAILED', ended_at = :ended_at,
                            duration_seconds = :duration, error_summary = :error_summary
                        WHERE run_id = :run_id
                        """
                    ),
                    {
                        "ended_at": utc_now(),
                        "duration": duration,
                        "error_summary": f"{type(exc).__name__}: {exc}",
                        "run_id": run_id,
                    },
                )
            logger.error(f"[PIPELINE FAILED] {pipeline_name} after {duration}s: {exc}")
            raise

    @contextmanager
    def track_task_run(self, run_id: str, task_name: str, layer: str):
        task_run_id = generate_id("task")
        started_at = utc_now()
        start_perf = time.perf_counter()
        metrics: Dict[str, Any] = {"records_in": 0, "records_out": 0, "retry_count": 0}

        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO meta.task_runs (task_run_id, run_id, task_name, layer, status, started_at)
                    VALUES (:task_run_id, :run_id, :task_name, :layer, 'RUNNING', :started_at)
                    """
                ),
                {
                    "task_run_id": task_run_id,
                    "run_id": run_id,
                    "task_name": task_name,
                    "layer": layer,
                    "started_at": started_at,
                },
            )

        try:
            yield metrics
            duration = round(time.perf_counter() - start_perf, 4)
            throughput = round(metrics["records_out"] / duration, 2) if duration > 0 else 0.0
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE meta.task_runs
                        SET status = 'SUCCESS', ended_at = :ended_at, duration_seconds = :duration,
                            retry_count = :retry_count, records_in = :records_in,
                            records_out = :records_out, throughput_rows_per_sec = :throughput
                        WHERE task_run_id = :task_run_id
                        """
                    ),
                    {
                        "ended_at": utc_now(),
                        "duration": duration,
                        "retry_count": metrics.get("retry_count", 0),
                        "records_in": metrics.get("records_in", 0),
                        "records_out": metrics.get("records_out", 0),
                        "throughput": throughput,
                        "task_run_id": task_run_id,
                    },
                )
        except Exception as exc:
            duration = round(time.perf_counter() - start_perf, 4)
            tb = traceback.format_exc()
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE meta.task_runs
                        SET status = 'FAILED', ended_at = :ended_at, duration_seconds = :duration,
                            retry_count = :retry_count, exception_type = :exc_type, stack_trace = :tb
                        WHERE task_run_id = :task_run_id
                        """
                    ),
                    {
                        "ended_at": utc_now(),
                        "duration": duration,
                        "retry_count": metrics.get("retry_count", 0),
                        "exc_type": type(exc).__name__,
                        "tb": tb,
                        "task_run_id": task_run_id,
                    },
                )
            raise

    def record_batch_metadata(
        self,
        batch_id: str,
        run_id: str,
        source_system: str,
        dataset_name: str,
        extraction_mode: str,
        source_count: int,
        target_count: int,
        bronze_path: str,
        watermark_col: Optional[str] = None,
        watermark_start: Optional[str] = None,
        watermark_end: Optional[str] = None,
    ) -> str:
        recon_status = "MATCHED" if source_count == target_count else "MISMATCH"
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO meta.batch_metadata (
                        batch_id, run_id, source_system, dataset_name, extraction_mode,
                        watermark_column, watermark_start, watermark_end,
                        source_record_count, target_record_count, reconciliation_status,
                        bronze_path, ingested_at
                    ) VALUES (
                        :batch_id, :run_id, :source_system, :dataset_name, :extraction_mode,
                        :watermark_col, :watermark_start, :watermark_end,
                        :source_count, :target_count, :recon_status,
                        :bronze_path, :ingested_at
                    )
                    ON CONFLICT (batch_id) DO UPDATE SET
                        run_id = EXCLUDED.run_id,
                        source_system = EXCLUDED.source_system,
                        dataset_name = CASE
                            WHEN meta.batch_metadata.dataset_name = 'medallion_all' THEN EXCLUDED.dataset_name
                            WHEN meta.batch_metadata.dataset_name LIKE '%' || EXCLUDED.dataset_name || '%' THEN meta.batch_metadata.dataset_name
                            ELSE meta.batch_metadata.dataset_name || ',' || EXCLUDED.dataset_name
                        END,
                        extraction_mode = EXCLUDED.extraction_mode,
                        watermark_column = COALESCE(EXCLUDED.watermark_column, meta.batch_metadata.watermark_column),
                        watermark_start = COALESCE(EXCLUDED.watermark_start, meta.batch_metadata.watermark_start),
                        watermark_end = COALESCE(EXCLUDED.watermark_end, meta.batch_metadata.watermark_end),
                        source_record_count = CASE
                            WHEN meta.batch_metadata.dataset_name LIKE '%' || EXCLUDED.dataset_name || '%' THEN EXCLUDED.source_record_count
                            ELSE meta.batch_metadata.source_record_count + EXCLUDED.source_record_count
                        END,
                        target_record_count = CASE
                            WHEN meta.batch_metadata.dataset_name LIKE '%' || EXCLUDED.dataset_name || '%' THEN EXCLUDED.target_record_count
                            ELSE meta.batch_metadata.target_record_count + EXCLUDED.target_record_count
                        END,
                        reconciliation_status = EXCLUDED.reconciliation_status,
                        bronze_path = EXCLUDED.bronze_path,
                        ingested_at = EXCLUDED.ingested_at
                    """
                ),
                {
                    "batch_id": batch_id,
                    "run_id": run_id,
                    "source_system": source_system,
                    "dataset_name": dataset_name,
                    "extraction_mode": extraction_mode,
                    "watermark_col": watermark_col,
                    "watermark_start": watermark_start,
                    "watermark_end": watermark_end,
                    "source_count": source_count,
                    "target_count": target_count,
                    "recon_status": recon_status,
                    "bronze_path": bronze_path,
                    "ingested_at": utc_now(),
                },
            )
        return recon_status

    def upsert_dataset_metadata(
        self,
        layer: str,
        dataset_name: str,
        storage_format: str,
        storage_path: str,
        latest_batch_id: str,
        row_count: int,
    ) -> None:
        dataset_key = f"{layer}.{dataset_name}"
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO meta.dataset_metadata (
                        dataset_key, layer, dataset_name, storage_format,
                        storage_path, latest_batch_id, row_count, last_updated_at
                    ) VALUES (
                        :dataset_key, :layer, :dataset_name, :storage_format,
                        :storage_path, :latest_batch_id, :row_count, :last_updated_at
                    )
                    ON CONFLICT (dataset_key) DO UPDATE SET
                        storage_format = EXCLUDED.storage_format,
                        storage_path = EXCLUDED.storage_path,
                        latest_batch_id = EXCLUDED.latest_batch_id,
                        row_count = EXCLUDED.row_count,
                        last_updated_at = EXCLUDED.last_updated_at
                    """
                ),
                {
                    "dataset_key": dataset_key,
                    "layer": layer,
                    "dataset_name": dataset_name,
                    "storage_format": storage_format,
                    "storage_path": storage_path,
                    "latest_batch_id": latest_batch_id,
                    "row_count": row_count,
                    "last_updated_at": utc_now(),
                },
            )

    def create_incident(
        self,
        incident_type: str,
        severity: str,
        dataset_name: Optional[str] = None,
        layer: Optional[str] = None,
        batch_id: Optional[str] = None,
        run_id: Optional[str] = None,
        pipeline_name: Optional[str] = None,
        task_name: Optional[str] = None,
        evidence_refs: Optional[Dict[str, Any]] = None,
        recommended_action: Optional[str] = None,
    ) -> str:
        incident_id = generate_id("inc")
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO meta.incidents (
                        incident_id, batch_id, run_id, pipeline_name, task_name,
                        dataset_name, layer, incident_type, severity, status,
                        rca_status, evidence_refs, recommended_action, detected_at
                    ) VALUES (
                        :incident_id, :batch_id, :run_id, :pipeline_name, :task_name,
                        :dataset_name, :layer, :incident_type, :severity, 'OPEN',
                        'PENDING', CAST(:evidence_refs AS JSONB), :recommended_action, :detected_at
                    )
                    """
                ),
                {
                    "incident_id": incident_id,
                    "batch_id": batch_id,
                    "run_id": run_id,
                    "pipeline_name": pipeline_name,
                    "task_name": task_name,
                    "dataset_name": dataset_name,
                    "layer": layer,
                    "incident_type": incident_type,
                    "severity": severity,
                    "evidence_refs": json.dumps(evidence_refs or {}),
                    "recommended_action": recommended_action,
                    "detected_at": utc_now(),
                },
            )
        if run_id:
            dq_flag = "FAILED" if (severity == "CRITICAL" or incident_type == "DATA_CONTRACT_VIOLATION") else "WARNING"
            with self.engine.begin() as conn:
                conn.execute(
                    text("UPDATE meta.pipeline_runs SET dq_status = :dq WHERE run_id = :rid AND COALESCE(dq_status, 'PASSED') != 'FAILED'"),
                    {"dq": dq_flag, "rid": run_id},
                )
        logger.warning(f"[INCIDENT CREATED] {incident_id} | {incident_type} ({severity}) on {dataset_name}")
        return incident_id
