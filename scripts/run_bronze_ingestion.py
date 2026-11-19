import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db_connection import get_engine, verify_connection
from src.ingestion.bronze_ingestor import BronzeIngestor
from src.observability.tracker import ObservabilityTracker
from src.utils.config_loader import load_yaml_config
from src.utils.logger import get_logger

logger = get_logger("bronze_pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RetailPulse Multi-Source Bronze Ingestion")
    parser.add_argument(
        "--cutoff",
        type=str,
        default=None,
        help="Optional timestamp cutoff (e.g., '2018-06-01 00:00:00') to test incremental watermark ingestion",
    )
    args = parser.parse_args()

    config = load_yaml_config()
    engine = get_engine()
    verify_connection(engine)

    tracker = ObservabilityTracker(engine)
    ingestor = BronzeIngestor(engine, config, tracker)

    with tracker.track_pipeline_run("bronze_multi_source_ingestion", trigger_type="cli") as run_id:
        # 1. Ingest CSV sources (with incremental watermark support)
        for ds_name, meta in config["sources"]["csv_files"].items():
            ingestor.ingest_csv_source(run_id, ds_name, meta, cutoff_watermark=args.cutoff)

        # 2. Ingest PostgreSQL OLTP sources
        pg_cfg = config["sources"]["postgres_oltp"]
        for tbl_name in pg_cfg["tables"].keys():
            ingestor.ingest_postgres_source(run_id, tbl_name, pg_cfg["schema"])

        # 3. Ingest Simulated REST API source (payments)
        for ds_name, meta in config["sources"]["rest_api"].items():
            ingestor.ingest_rest_api_source(run_id, ds_name, meta)

    logger.info("Bronze Multi-Source Ingestion completed successfully!")


if __name__ == "__main__":
    main()
