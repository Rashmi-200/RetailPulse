import sys
from pathlib import Path
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db_connection import get_engine, verify_connection
from src.lakehouse.gold_builder import GoldStarSchemaBuilder
from src.lineage.lineage_tracker import OpenLineageTracker
from src.observability.tracker import ObservabilityTracker
from src.utils.config_loader import load_yaml_config
from src.utils.logger import get_logger

logger = get_logger("gold_pipeline")


def run_analytical_verification(engine) -> None:
    query = text(
        """
        SELECT
            is_late_delivery,
            COUNT(*) AS delivered_orders,
            ROUND(AVG(delivery_days)::numeric, 1) AS avg_delivery_days,
            ROUND(AVG(avg_review_score)::numeric, 2) AS avg_review_score
        FROM gold.factorders
        WHERE order_status = 'delivered' AND avg_review_score IS NOT NULL
        GROUP BY is_late_delivery;
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()
    logger.info("=== Gold Analytical Check: Late Delivery vs Customer Review Score ===")
    for r in rows:
        logger.info(f"  is_late_delivery={r[0]} | orders={r[1]:,} | avg_days={r[2]} | avg_review={r[3]}/5.0")


def main() -> None:
    config = load_yaml_config()
    engine = get_engine()
    verify_connection(engine)

    tracker = ObservabilityTracker(engine)
    lineage = OpenLineageTracker(engine)
    builder = GoldStarSchemaBuilder(engine, config, tracker, lineage)

    with tracker.track_pipeline_run("gold_dimensional_star_schema", trigger_type="cli") as run_id:
        summary = builder.build_all(run_id)
        logger.info(f"Gold Star Schema summary: {summary}")

    run_analytical_verification(engine)


if __name__ == "__main__":
    main()
