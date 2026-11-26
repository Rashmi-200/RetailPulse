import os
from contextlib import contextmanager
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.database.db_connection import get_engine, verify_connection
from src.observability.tracker import ObservabilityTracker, utc_now
from src.utils.config_loader import load_yaml_config
from src.utils.logger import get_logger

logger = get_logger("retailpulse.airflow_orchestration")

ORDERED_SILVER_DATASETS = [
    "customers",
    "sellers",
    "category_translation",
    "products",
    "geolocation",
    "orders",
    "order_items",
    "payments",
    "order_reviews",
]


class NonRetriablePipelineError(Exception):
    """Raised on permanent schema/contract violations that must not be retried."""


def derive_batch_id(dag_id: str, dag_run_id: str) -> str:
    """Deterministically derive one stable batch_id per logical DAG run (invariant across task retries)."""
    digest = hashlib.sha256(f"{dag_id}::{dag_run_id}".encode("utf-8")).hexdigest()[:16]
    return f"batch_{digest}"


def sanitize_run_id(dag_run_id: str) -> str:
    """Normalize Airflow dag_run_id into a <=64-char identifier for meta.pipeline_runs.run_id."""
    clean = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in dag_run_id)
    if len(clean) <= 64:
        return clean
    digest = hashlib.sha256(dag_run_id.encode("utf-8")).hexdigest()[:12]
    return f"{clean[:51]}_{digest}"


class RetailPulseOrchestrator:
    """Thin Airflow orchestration adapter delegating directly to existing RetailPulse services."""

    def __init__(
        self,
        engine: Optional[Engine] = None,
        pipeline_cfg: Optional[Dict[str, Any]] = None,
        contracts_cfg: Optional[Dict[str, Any]] = None,
        stage_hooks: Optional[Dict[str, Callable[..., Any]]] = None,
    ):
        project_root = Path(__file__).resolve().parents[2]
        if project_root.exists() and (project_root / "config").exists():
            os.chdir(project_root)
        self.engine = engine or get_engine()
        self.pipeline_cfg = pipeline_cfg or load_yaml_config(str(project_root / "config/pipeline_config.yaml"))
        self.contracts_cfg = contracts_cfg or load_yaml_config(str(project_root / "config/data_contracts.yaml"))
        self.tracker = ObservabilityTracker(self.engine)
        self.stage_hooks = stage_hooks or {}

    def _ensure_pipeline_run_row(self, run_id: str, dag_id: str, batch_id: str, trigger_type: str = "airflow") -> None:
        with self.engine.begin() as conn:
            conn.execute(text("ALTER TABLE meta.pipeline_runs ADD COLUMN IF NOT EXISTS dq_status VARCHAR(20) DEFAULT 'PENDING';"))
            conn.execute(
                text(
                    """
                    INSERT INTO meta.pipeline_runs (run_id, pipeline_name, trigger_type, status, dq_status, started_at)
                    VALUES (:rid, :pname, :ttype, 'RUNNING', 'PENDING', :started_at)
                    ON CONFLICT (run_id) DO UPDATE
                    SET status = 'RUNNING', ended_at = NULL, error_summary = NULL
                    """
                ),
                {
                    "rid": run_id,
                    "pname": dag_id,
                    "ttype": trigger_type,
                    "started_at": utc_now(),
                },
            )
            conn.execute(
                text(
                    """
                    INSERT INTO meta.batch_metadata (
                        batch_id, run_id, source_system, dataset_name, extraction_mode,
                        source_record_count, target_record_count, reconciliation_status, ingested_at
                    )
                    VALUES (:bid, :rid, 'airflow_orchestrator', 'medallion_all', 'incremental', 0, 0, 'RUNNING', :ts)
                    ON CONFLICT (batch_id) DO NOTHING
                    """
                ),
                {"bid": batch_id, "rid": run_id, "ts": utc_now()},
            )

    @contextmanager
    def _track_stage(self, run_id: str, task_name: str, layer: str, retry_count: int = 0):
        os.chdir(Path(__file__).resolve().parents[2])
        with self.tracker.track_task_run(run_id, task_name, layer=layer) as metrics:
            metrics["retry_count"] = retry_count
            yield metrics
        if retry_count > 0:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE meta.task_runs
                        SET retry_count = :rc
                        WHERE task_run_id = (
                            SELECT task_run_id FROM meta.task_runs
                            WHERE run_id = :rid AND task_name = :tname
                            ORDER BY started_at DESC, task_run_id DESC LIMIT 1
                        )
                        """
                    ),
                    {"rc": retry_count, "rid": run_id, "tname": task_name},
                )

    def _run_hook(self, stage_name: str, **kwargs: Any) -> Any:
        if stage_name in self.stage_hooks:
            return self.stage_hooks[stage_name](**kwargs)
        return None

    def task_ingest_sources(self, dag_id: str, dag_run_id: str, retry_count: int = 0) -> Dict[str, Any]:
        run_id = sanitize_run_id(dag_run_id)
        batch_id = derive_batch_id(dag_id, dag_run_id)
        verify_connection(self.engine)
        self._ensure_pipeline_run_row(run_id, dag_id, batch_id)

        with self._track_stage(run_id, "ingest_sources", layer="source", retry_count=retry_count) as metrics:
            hook_out = self._run_hook("ingest_sources", run_id=run_id, batch_id=batch_id, retry_count=retry_count)
            if hook_out is not None:
                metrics["records_in"] = int(hook_out.get("records_in", 0))
                metrics["records_out"] = int(hook_out.get("records_out", 0))
                return {"run_id": run_id, "batch_id": batch_id, **hook_out}

            from src.ingestion.bronze_ingestor import BronzeIngestor
            ingestor = BronzeIngestor(self.engine, self.pipeline_cfg, self.tracker)
            csv_sources = self.pipeline_cfg.get("sources", {}).get("csv_files", {})
            ingested = 0
            for ds_name, meta in csv_sources.items():
                src_path = Path(meta.get("path", ""))
                if src_path.exists():
                    ingestor.ingest_csv_source(run_id, ds_name, meta, batch_id=batch_id)
                    ingested += 1
            metrics["records_in"] = ingested
            metrics["records_out"] = ingested
            return {"run_id": run_id, "batch_id": batch_id, "sources_checked": len(csv_sources), "sources_ingested": ingested}

    def task_process_bronze(self, dag_id: str, dag_run_id: str, batch_id: str, retry_count: int = 0) -> Dict[str, Any]:
        from deltalake import DeltaTable

        run_id = sanitize_run_id(dag_run_id)
        verify_connection(self.engine)
        with self._track_stage(run_id, "process_bronze", layer="bronze", retry_count=retry_count) as metrics:
            hook_out = self._run_hook("process_bronze", run_id=run_id, batch_id=batch_id, retry_count=retry_count)
            if hook_out is not None:
                metrics["records_in"] = int(hook_out.get("records_in", 0))
                metrics["records_out"] = int(hook_out.get("records_out", 0))
                return {"run_id": run_id, "batch_id": batch_id, **hook_out}

            bronze_dir = Path(self.pipeline_cfg["paths"]["bronze_dir"])
            total_rows = 0
            tables_found = 0
            for ds_name in ORDERED_SILVER_DATASETS:
                delta_p = bronze_dir / ds_name / "_delta"
                if delta_p.exists():
                    df = DeltaTable(str(delta_p)).to_pandas()
                    total_rows += len(df)
                    tables_found += 1
            metrics["records_in"] = total_rows
            metrics["records_out"] = total_rows
            return {"run_id": run_id, "batch_id": batch_id, "bronze_tables": tables_found, "bronze_rows": total_rows}

    def task_transform_silver(self, dag_id: str, dag_run_id: str, batch_id: str, retry_count: int = 0) -> Dict[str, Any]:
        from deltalake import DeltaTable

        run_id = sanitize_run_id(dag_run_id)
        verify_connection(self.engine)
        with self._track_stage(run_id, "transform_silver", layer="silver", retry_count=retry_count) as metrics:
            hook_out = self._run_hook("transform_silver", run_id=run_id, batch_id=batch_id, retry_count=retry_count)
            if hook_out is not None:
                metrics["records_in"] = int(hook_out.get("records_in", 0))
                metrics["records_out"] = int(hook_out.get("records_out", 0))
                return {"run_id": run_id, "batch_id": batch_id, **hook_out}

            bronze_dir = Path(self.pipeline_cfg["paths"]["bronze_dir"])
            silver_dir = Path(self.pipeline_cfg["paths"]["silver_dir"])
            from src.lakehouse.silver_transformer import SilverTransformer
            transformer = SilverTransformer(self.engine, self.pipeline_cfg, self.contracts_cfg, self.tracker)
            valid_order_ids = None
            processed_datasets = 0
            total_silver_rows = 0
            for ds_name in ORDERED_SILVER_DATASETS:
                if (bronze_dir / ds_name / "_delta").exists():
                    silver_df = transformer.process_dataset(run_id, ds_name, valid_order_ids=valid_order_ids)
                    processed_datasets += 1
                    total_silver_rows += len(silver_df)
                    if ds_name == "orders" and "order_id" in silver_df.columns:
                        valid_order_ids = set(silver_df["order_id"].dropna().astype(str))
                elif (silver_dir / ds_name / "_delta").exists():
                    df_existing = DeltaTable(str(silver_dir / ds_name / "_delta")).to_pandas()
                    total_silver_rows += len(df_existing)
                    if ds_name == "orders" and "order_id" in df_existing.columns:
                        valid_order_ids = set(df_existing["order_id"].dropna().astype(str))

            metrics["records_in"] = total_silver_rows
            metrics["records_out"] = total_silver_rows
            return {"run_id": run_id, "batch_id": batch_id, "datasets_transformed": processed_datasets, "silver_rows": total_silver_rows}

    def task_validate_data_quality(self, dag_id: str, dag_run_id: str, batch_id: str, retry_count: int = 0) -> Dict[str, Any]:
        from deltalake import DeltaTable

        run_id = sanitize_run_id(dag_run_id)
        verify_connection(self.engine)
        with self._track_stage(run_id, "validate_data_quality", layer="silver_gate", retry_count=retry_count) as metrics:
            try:
                hook_out = self._run_hook("validate_data_quality", run_id=run_id, batch_id=batch_id, retry_count=retry_count)
                if hook_out is not None:
                    dq_status = hook_out.get("dq_status", "PASSED")
                    with self.engine.begin() as conn:
                        conn.execute(
                            text("UPDATE meta.pipeline_runs SET dq_status = :dqs WHERE run_id = :rid"),
                            {"dqs": dq_status, "rid": run_id},
                        )
                    if dq_status != "PASSED":
                        raise NonRetriablePipelineError(hook_out.get("reason", f"Data quality gate failed for batch {batch_id}"))
                    metrics["records_in"] = int(hook_out.get("records_in", 0))
                    metrics["records_out"] = int(hook_out.get("records_out", 0))
                    return {"run_id": run_id, "batch_id": batch_id, **hook_out}

                silver_dir = Path(self.pipeline_cfg["paths"]["silver_dir"])
                from src.contracts.contract_engine import DataContractEngine
                from src.schema_evolution.registry import SchemaEvolutionRegistry
                schema_reg = SchemaEvolutionRegistry(self.engine, self.tracker)
                contract_engine = DataContractEngine(self.engine, self.contracts_cfg)
                min_score = float(self.pipeline_cfg.get("quality", {}).get("min_overall_score", 85.0))

                total_checked = 0
                total_valid = 0
                scores = []
                for ds_name, contract in self.contracts_cfg.get("contracts", {}).items():
                    delta_p = silver_dir / ds_name / "_delta"
                    if not delta_p.exists():
                        continue
                    df = DeltaTable(str(delta_p)).to_pandas()
                    if df.empty:
                        continue
                    req_cols = contract.get("required_columns", [])
                    schema_reg.check_and_register(df, ds_name, "silver", required_columns=req_cols)
                    res = contract_engine.validate_and_split(df, ds_name, batch_id, layer="silver", run_id=run_id, persist=False)
                    total_checked += len(df)
                    total_valid += len(res.valid_df)
                    scores.append(res.overall_score)
                    if res.overall_score < min_score:
                        raise NonRetriablePipelineError(
                            f"DQ score {res.overall_score} below threshold {min_score} on {ds_name} (batch={batch_id})"
                        )

                with self.engine.begin() as conn:
                    conn.execute(
                        text("UPDATE meta.pipeline_runs SET dq_status = 'PASSED' WHERE run_id = :rid"),
                        {"rid": run_id},
                    )
                metrics["records_in"] = total_checked
                metrics["records_out"] = total_valid
                avg_score = round(sum(scores) / len(scores), 2) if scores else 100.0
                return {"run_id": run_id, "batch_id": batch_id, "dq_status": "PASSED", "avg_dq_score": avg_score}
            except ValueError as ve:
                with self.engine.begin() as conn:
                    conn.execute(text("UPDATE meta.pipeline_runs SET dq_status = 'FAILED' WHERE run_id = :rid"), {"rid": run_id})
                raise NonRetriablePipelineError(str(ve)) from ve
            except NonRetriablePipelineError:
                with self.engine.begin() as conn:
                    conn.execute(text("UPDATE meta.pipeline_runs SET dq_status = 'FAILED' WHERE run_id = :rid"), {"rid": run_id})
                raise

    def task_build_gold(self, dag_id: str, dag_run_id: str, batch_id: str, retry_count: int = 0) -> Dict[str, Any]:
        run_id = sanitize_run_id(dag_run_id)
        verify_connection(self.engine)
        with self._track_stage(run_id, "build_gold", layer="gold", retry_count=retry_count) as metrics:
            hook_out = self._run_hook("build_gold", run_id=run_id, batch_id=batch_id, retry_count=retry_count)
            if hook_out is not None:
                metrics["records_in"] = int(hook_out.get("records_in", 0))
                metrics["records_out"] = int(hook_out.get("records_out", 0))
                return {"run_id": run_id, "batch_id": batch_id, **hook_out}

            silver_dir = Path(self.pipeline_cfg["paths"]["silver_dir"])
            required_gold_inputs = ["customers", "sellers", "products", "orders", "order_items", "payments", "order_reviews"]
            if all((silver_dir / ds / "_delta").exists() for ds in required_gold_inputs):
                from src.lakehouse.gold_builder import GoldStarSchemaBuilder
                from src.lineage.lineage_tracker import OpenLineageTracker
                lineage = OpenLineageTracker(self.engine)
                builder = GoldStarSchemaBuilder(self.engine, self.pipeline_cfg, self.tracker, lineage)
                summary = builder.build_all(run_id)
                fact_rows = int(summary.get("FactOrders", 0)) if isinstance(summary, dict) else 0
                metrics["records_in"] = fact_rows
                metrics["records_out"] = fact_rows
                return {"run_id": run_id, "batch_id": batch_id, "gold_summary": summary}

            with self.engine.connect() as conn:
                fact_cnt = int(conn.execute(text("SELECT COUNT(*) FROM gold.factorders")).scalar() or 0)
            metrics["records_in"] = fact_cnt
            metrics["records_out"] = fact_cnt
            return {"run_id": run_id, "batch_id": batch_id, "gold_factorders_rows": fact_cnt}

    def task_detect_anomalies(self, dag_id: str, dag_run_id: str, batch_id: str, retry_count: int = 0) -> Dict[str, Any]:
        from deltalake import DeltaTable

        run_id = sanitize_run_id(dag_run_id)
        verify_connection(self.engine)
        with self._track_stage(run_id, "detect_anomalies", layer="analytics", retry_count=retry_count) as metrics:
            hook_out = self._run_hook("detect_anomalies", run_id=run_id, batch_id=batch_id, retry_count=retry_count)
            if hook_out is not None:
                metrics["records_in"] = int(hook_out.get("records_in", 0))
                metrics["records_out"] = int(hook_out.get("records_out", 0))
                return {"run_id": run_id, "batch_id": batch_id, **hook_out}

            silver_dir = Path(self.pipeline_cfg["paths"]["silver_dir"])
            orders_df = DeltaTable(str(silver_dir / "orders" / "_delta")).to_pandas()
            items_df = DeltaTable(str(silver_dir / "order_items" / "_delta")).to_pandas()
            with self.engine.begin() as conn:
                conn.execute(text("DELETE FROM meta.anomaly_alerts WHERE batch_id = :bid"), {"bid": batch_id})

            from src.anomaly_detection.detector import HybridAnomalyDetector
            detector = HybridAnomalyDetector(self.engine, self.tracker)
            alerts = detector.scan_daily_order_metrics(orders_df, items_df, batch_id=batch_id, run_id=run_id)
            metrics["records_in"] = len(orders_df)
            metrics["records_out"] = len(alerts)
            return {"run_id": run_id, "batch_id": batch_id, "alerts_generated": len(alerts)}

    def task_generate_forecasts(self, dag_id: str, dag_run_id: str, batch_id: str, retry_count: int = 0) -> Dict[str, Any]:
        run_id = sanitize_run_id(dag_run_id)
        verify_connection(self.engine)
        with self._track_stage(run_id, "generate_forecasts", layer="analytics", retry_count=retry_count) as metrics:
            hook_out = self._run_hook("generate_forecasts", run_id=run_id, batch_id=batch_id, retry_count=retry_count)
            if hook_out is not None:
                metrics["records_in"] = int(hook_out.get("records_in", 0))
                metrics["records_out"] = int(hook_out.get("records_out", 0))
                return {"run_id": run_id, "batch_id": batch_id, **hook_out}

            from src.analytics.demand_forecasting import RetailDemandForecaster
            forecaster = RetailDemandForecaster(self.engine)
            summary = forecaster.train_and_benchmark(target_col="daily_orders")
            metrics["records_in"] = int(summary.get("active_history_days", 0))
            metrics["records_out"] = int(summary.get("horizon_days", 7))
            return {"run_id": run_id, "batch_id": batch_id, "forecast_summary": summary}

    def task_finalize_pipeline(
        self,
        dag_id: str,
        dag_run_id: str,
        batch_id: str,
        upstream_states: Optional[Dict[str, str]] = None,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        run_id = sanitize_run_id(dag_run_id)
        verify_connection(self.engine)
        upstream_states = upstream_states or {}

        with self._track_stage(run_id, "finalize_pipeline", layer="observability", retry_count=retry_count) as metrics:
            with self.engine.connect() as conn:
                task_rows = conn.execute(
                    text("SELECT task_name, status, exception_type FROM meta.task_runs WHERE run_id = :rid AND task_name != 'finalize_pipeline'"),
                    {"rid": run_id},
                ).fetchall()
                dq_status = conn.execute(
                    text("SELECT dq_status FROM meta.pipeline_runs WHERE run_id = :rid"),
                    {"rid": run_id},
                ).scalar() or "UNKNOWN"

            failed_tasks = [r[0] for r in task_rows if r[1] == "FAILED"]
            for tname, st in upstream_states.items():
                if st in ("failed", "upstream_failed") and tname not in failed_tasks:
                    failed_tasks.append(tname)

            final_status = "FAILED" if (failed_tasks or dq_status == "FAILED") else "SUCCESS"
            err_summary = f"Failed upstream tasks: {failed_tasks}; dq_status={dq_status}" if final_status == "FAILED" else None

            with self.engine.begin() as conn:
                row_start = conn.execute(
                    text("SELECT started_at FROM meta.pipeline_runs WHERE run_id = :rid"),
                    {"rid": run_id},
                ).scalar()
                ended_at = utc_now()
                dur = (ended_at - row_start).total_seconds() if row_start else 0.0
                conn.execute(
                    text(
                        """
                        UPDATE meta.pipeline_runs
                        SET status = :st, ended_at = :ended, duration_seconds = :dur, error_summary = :err
                        WHERE run_id = :rid
                        """
                    ),
                    {"st": final_status, "ended": ended_at, "dur": dur, "err": err_summary, "rid": run_id},
                )
                conn.execute(
                    text(
                        """
                        UPDATE meta.batch_metadata
                        SET reconciliation_status = :st
                        WHERE batch_id = :bid
                        """
                    ),
                    {"st": final_status, "bid": batch_id},
                )
                if final_status == "FAILED":
                    inc_id = f"inc_{hashlib.sha256(f'{run_id}::final'.encode()).hexdigest()[:16]}"
                    conn.execute(
                        text(
                            """
                            INSERT INTO meta.incidents (
                                incident_id, batch_id, run_id, pipeline_name, task_name,
                                dataset_name, layer, incident_type, severity, status, detected_at
                            )
                            VALUES (
                                :iid, :bid, :rid, :pname, :tname,
                                'medallion_pipeline', 'orchestration', :itype, 'HIGH', 'OPEN', :ts
                            )
                            ON CONFLICT (incident_id) DO NOTHING
                            """
                        ),
                        {
                            "iid": inc_id,
                            "bid": batch_id,
                            "rid": run_id,
                            "pname": dag_id,
                            "tname": failed_tasks[0] if failed_tasks else "validate_data_quality",
                            "itype": "DATA_CONTRACT_VIOLATION" if dq_status == "FAILED" else "PIPELINE_TASK_FAILURE",
                            "ts": ended_at,
                        },
                    )

            metrics["records_in"] = len(task_rows)
            metrics["records_out"] = 1
            if final_status == "FAILED":
                raise RuntimeError(f"Pipeline finalized with status=FAILED ({err_summary})")
            return {"run_id": run_id, "batch_id": batch_id, "status": final_status, "dq_status": dq_status}
