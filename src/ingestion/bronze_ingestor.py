from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import requests
from deltalake import write_deltalake
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.ingestion.mock_api_server import app as mock_api_app
from src.ingestion.watermark_manager import WatermarkManager
from src.observability.tracker import ObservabilityTracker, generate_id, utc_now
from src.recovery.retry import retry_with_backoff
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BronzeIngestor:
    """Extracts data from CSV, PostgreSQL OLTP, and REST API into immutable Bronze Parquet & Delta tables."""

    def __init__(self, engine: Engine, config: dict, tracker: ObservabilityTracker):
        self.engine = engine
        self.config = config
        self.tracker = tracker
        self.watermarks = WatermarkManager(engine)
        self.raw_dir = Path(config["paths"]["raw_dir"])
        self.bronze_dir = Path(config["paths"]["bronze_dir"])
        self.api_client = TestClient(mock_api_app)

    def _stamp_bronze_metadata(self, df: pd.DataFrame, batch_id: str, source_system: str) -> pd.DataFrame:
        stamped = df.copy()
        stamped["_batch_id"] = batch_id
        stamped["_ingested_at"] = utc_now().isoformat()
        stamped["_source_system"] = source_system
        return stamped

    def _write_bronze(self, df: pd.DataFrame, dataset_name: str, batch_id: str, mode: str = "append") -> str:
        dataset_dir = self.bronze_dir / dataset_name
        batch_dir = dataset_dir / f"batch_id={batch_id}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        # 1. Immutable batch Parquet snapshot
        parquet_path = batch_dir / "part-000.parquet"
        df.to_parquet(parquet_path, index=False)

        # 2. Delta Lake table for ACID lakehouse queries & time travel
        delta_dir = dataset_dir / "_delta"
        write_deltalake(
            str(delta_dir),
            df,
            mode=mode,
            schema_mode="merge",
        )
        return str(parquet_path)

    def _reconcile_and_record(
        self,
        df: pd.DataFrame,
        run_id: str,
        batch_id: str,
        source_system: str,
        dataset_name: str,
        extraction_mode: str,
        bronze_path: str,
        watermark_col: Optional[str] = None,
        watermark_start: Optional[str] = None,
        watermark_end: Optional[str] = None,
    ) -> int:
        source_count = len(df)
        written_df = pd.read_parquet(bronze_path)
        target_count = len(written_df)

        recon_status = self.tracker.record_batch_metadata(
            batch_id=batch_id,
            run_id=run_id,
            source_system=source_system,
            dataset_name=dataset_name,
            extraction_mode=extraction_mode,
            source_count=source_count,
            target_count=target_count,
            bronze_path=bronze_path,
            watermark_col=watermark_col,
            watermark_start=watermark_start,
            watermark_end=watermark_end,
        )

        if recon_status != "MATCHED":
            self.tracker.create_incident(
                incident_type="RECONCILIATION_MISMATCH",
                severity="HIGH",
                dataset_name=dataset_name,
                layer="bronze",
                batch_id=batch_id,
                run_id=run_id,
                evidence_refs={"source_count": source_count, "target_count": target_count},
                recommended_action="Inspect Bronze disk write integrity and replay batch.",
            )
            raise RuntimeError(f"Reconciliation mismatch for {dataset_name}: {source_count} != {target_count}")

        delta_dir = self.bronze_dir / dataset_name / "_delta"
        self.tracker.upsert_dataset_metadata(
            layer="bronze",
            dataset_name=dataset_name,
            storage_format="delta+parquet",
            storage_path=str(delta_dir),
            latest_batch_id=batch_id,
            row_count=target_count,
        )
        return target_count

    def ingest_csv_source(self, run_id: str, dataset_name: str, meta: dict, cutoff_watermark: Optional[str] = None, batch_id: Optional[str] = None) -> int:
        with self.tracker.track_task_run(run_id, f"ingest_csv_{dataset_name}", layer="bronze") as metrics:
            csv_path = self.raw_dir / meta["file"]
            df = pd.read_csv(csv_path)
            watermark_col = meta.get("watermark_col")
            wm_start = None
            wm_end = None
            extraction_mode = "full"

            if watermark_col and watermark_col in df.columns:
                extraction_mode = "incremental"
                df = df.dropna(subset=[watermark_col]).copy()
                df[watermark_col] = df[watermark_col].astype(str)
                if cutoff_watermark:
                    df = df[df[watermark_col] <= cutoff_watermark]

                wm_start = self.watermarks.get_last_watermark("csv_files", dataset_name)
                if wm_start:
                    df = df[df[watermark_col] > wm_start]

                if df.empty:
                    logger.info(f"[SKIPPED - IDEMPOTENT] csv_files.{dataset_name}: no new rows after watermark {wm_start}")
                    return 0
                wm_end = str(df[watermark_col].max())

            if self.watermarks.is_batch_already_ingested("csv_files", dataset_name, wm_start, wm_end):
                logger.info(f"[SKIPPED - DUPLICATE WINDOW] csv_files.{dataset_name} ({wm_start} -> {wm_end})")
                return 0

            batch_id = batch_id or generate_id("batch")
            stamped_df = self._stamp_bronze_metadata(df, batch_id, "csv_files")
            write_mode = "overwrite" if extraction_mode == "full" else "append"
            bronze_path = self._write_bronze(stamped_df, dataset_name, batch_id, mode=write_mode)

            target_count = self._reconcile_and_record(
                stamped_df, run_id, batch_id, "csv_files", dataset_name,
                extraction_mode, bronze_path, watermark_col, wm_start, wm_end
            )
            metrics["records_in"] = len(df)
            metrics["records_out"] = target_count
            logger.info(f"[BRONZE CSV] {dataset_name}: {target_count:,} rows (mode={extraction_mode}, wm_end={wm_end})")
            return target_count

    @retry_with_backoff(max_retries=3, base_delay=0.5)
    def _query_postgres_table(self, schema: str, table_name: str) -> pd.DataFrame:
        with self.engine.connect() as conn:
            return pd.read_sql(text(f"SELECT * FROM {schema}.{table_name}"), conn)

    def ingest_postgres_source(self, run_id: str, table_name: str, schema: str, batch_id: Optional[str] = None) -> int:
        with self.tracker.track_task_run(run_id, f"ingest_pg_{table_name}", layer="bronze") as metrics:
            df = self._query_postgres_table(schema, table_name)
            batch_id = batch_id or generate_id("batch")
            stamped_df = self._stamp_bronze_metadata(df, batch_id, f"postgres.{schema}")
            bronze_path = self._write_bronze(stamped_df, table_name, batch_id, mode="overwrite")

            target_count = self._reconcile_and_record(
                stamped_df, run_id, batch_id, f"postgres.{schema}", table_name,
                "full_snapshot", bronze_path
            )
            metrics["records_in"] = len(df)
            metrics["records_out"] = target_count
            logger.info(f"[BRONZE PG] {schema}.{table_name}: {target_count:,} rows loaded.")
            return target_count

    @retry_with_backoff(max_retries=3, base_delay=0.5)
    def _call_payments_api(self, endpoint: str, offset: int, limit: int, simulate_timeout: bool = False) -> dict:
        response = self.api_client.get(
            endpoint, params={"offset": offset, "limit": limit, "simulate_timeout": simulate_timeout}
        )
        if response.status_code != 200:
            raise requests.HTTPError(f"API returned {response.status_code}: {response.text}")
        return response.json()

    def ingest_rest_api_source(
        self, run_id: str, dataset_name: str, meta: dict, simulate_timeout: bool = False, batch_id: Optional[str] = None
    ) -> int:
        with self.tracker.track_task_run(run_id, f"ingest_api_{dataset_name}", layer="bronze") as metrics:
            endpoint = meta["endpoint"]
            all_records = []
            offset = 0
            limit = 60000

            while True:
                payload = self._call_payments_api(endpoint, offset, limit, simulate_timeout=simulate_timeout)
                batch_records = payload["data"]
                all_records.extend(batch_records)
                if offset + len(batch_records) >= payload["total_records"] or not batch_records:
                    break
                offset += limit

            df = pd.DataFrame(all_records)
            batch_id = batch_id or generate_id("batch")
            stamped_df = self._stamp_bronze_metadata(df, batch_id, "rest_api")
            bronze_path = self._write_bronze(stamped_df, dataset_name, batch_id, mode="overwrite")

            target_count = self._reconcile_and_record(
                stamped_df, run_id, batch_id, "rest_api", dataset_name,
                "api_paginated", bronze_path
            )
            metrics["records_in"] = len(df)
            metrics["records_out"] = target_count
            logger.info(f"[BRONZE API] {dataset_name}: {target_count:,} rows fetched via {endpoint}.")
            return target_count
