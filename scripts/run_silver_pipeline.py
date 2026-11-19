import sys
from pathlib import Path
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db_connection import get_engine, verify_connection
from src.lakehouse.silver_transformer import SilverTransformer
from src.observability.tracker import ObservabilityTracker
from src.utils.config_loader import load_yaml_config
from src.utils.logger import get_logger

logger = get_logger("silver_pipeline")


def ensure_quarantine_ddl(engine) -> None:
    ddl_path = PROJECT_ROOT / "sql" / "ddl" / "02_create_quarantine_tables.sql"
    with engine.begin() as conn:
        conn.execute(text(ddl_path.read_text(encoding="utf-8")))


def main() -> None:
    pipeline_cfg = load_yaml_config("config/pipeline_config.yaml")
    contracts_cfg = load_yaml_config("config/data_contracts.yaml")

    engine = get_engine()
    verify_connection(engine)
    ensure_quarantine_ddl(engine)

    tracker = ObservabilityTracker(engine)
    transformer = SilverTransformer(engine, pipeline_cfg, contracts_cfg, tracker)

    # Process parent datasets first so referential integrity can be enforced on child datasets
    ordered_datasets = [
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

    with tracker.track_pipeline_run("silver_transformation_and_quarantine", trigger_type="cli") as run_id:
        valid_order_ids = None
        for ds_name in ordered_datasets:
            silver_df = transformer.process_dataset(run_id, ds_name, valid_order_ids=valid_order_ids)
            if ds_name == "orders":
                valid_order_ids = set(silver_df["order_id"].unique())

    logger.info("Silver Layer & Smart Quarantine pipeline completed successfully!")


if __name__ == "__main__":
    main()
