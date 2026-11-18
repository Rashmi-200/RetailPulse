import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db_connection import get_engine, verify_connection
from src.utils.config_loader import load_yaml_config
from src.utils.logger import get_logger

logger = get_logger("setup_sources")


def validate_raw_csvs(config: dict) -> None:
    """Verify that all expected Olist CSV files exist in data/raw."""
    raw_dir = Path(config["paths"]["raw_dir"])
    expected_files = []

    for section in ["csv_files", "postgres_oltp", "rest_api"]:
        items = config["sources"][section].get("tables", config["sources"][section])
        for _, meta in items.items():
            csv_name = meta.get("file") or meta.get("source_csv")
            if csv_name:
                expected_files.append(csv_name)

    missing = [f for f in expected_files if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing Olist CSV files in {raw_dir.resolve()}: {missing}\n"
            "Please download the Olist dataset from Kaggle and extract all 9 CSVs into data/raw/."
        )
    logger.info(f"All {len(expected_files)} raw Olist CSV files verified in {raw_dir}.")


def initialize_schemas(engine) -> None:
    """Create PostgreSQL schemas for simulated OLTP, metadata observability, and quarantine."""
    schemas = ["source_oltp", "meta", "quarantine"]
    with engine.begin() as conn:
        for schema in schemas:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema};"))
    logger.info(f"PostgreSQL schemas initialized: {schemas}")


def seed_oltp_tables(engine, config: dict) -> None:
    """Load master reference tables (customers, products, sellers, categories) into source_oltp."""
    raw_dir = Path(config["paths"]["raw_dir"])
    oltp_cfg = config["sources"]["postgres_oltp"]
    schema = oltp_cfg["schema"]

    for table_name, meta in oltp_cfg["tables"].items():
        csv_path = raw_dir / meta["source_csv"]
        df = pd.read_csv(csv_path)
        df.to_sql(
            name=table_name,
            con=engine,
            schema=schema,
            if_exists="replace",
            index=False,
            chunksize=5000,
        )
        logger.info(f"Seeded {schema}.{table_name}: {len(df):,} rows loaded.")


def main() -> None:
    logger.info("Starting RetailPulse 2.0 Multi-Source Environment Setup...")
    config = load_yaml_config()
    validate_raw_csvs(config)

    engine = get_engine()
    verify_connection(engine)
    initialize_schemas(engine)
    seed_oltp_tables(engine, config)
    logger.info("Milestone 2 source seeding completed successfully!")


if __name__ == "__main__":
    main()
