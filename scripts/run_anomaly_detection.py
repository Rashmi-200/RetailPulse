import sys
from pathlib import Path
from deltalake import DeltaTable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_detection.detector import HybridAnomalyDetector
from src.anomaly_detection.evaluator import evaluate_detectors_on_synthetic_scenarios
from src.database.db_connection import get_engine, verify_connection
from src.observability.tracker import ObservabilityTracker
from src.utils.config_loader import load_yaml_config
from src.utils.logger import get_logger

logger = get_logger("anomaly_pipeline")


def main() -> None:
    config = load_yaml_config()
    silver_dir = Path(config["paths"]["silver_dir"])

    engine = get_engine()
    verify_connection(engine)
    tracker = ObservabilityTracker(engine)
    detector = HybridAnomalyDetector(engine, tracker)

    with tracker.track_pipeline_run("ml_statistical_anomaly_detection", trigger_type="cli") as run_id:
        with tracker.track_task_run(run_id, "scan_silver_orders_anomalies", layer="silver") as metrics:
            orders_df = DeltaTable(str(silver_dir / "orders" / "_delta")).to_pandas()
            items_df = DeltaTable(str(silver_dir / "order_items" / "_delta")).to_pandas()
            batch_id = str(orders_df["_batch_id"].iloc[-1])

            alerts = detector.scan_daily_order_metrics(orders_df, items_df, batch_id=batch_id, run_id=run_id)
            metrics["records_in"] = len(orders_df)
            metrics["records_out"] = len(alerts)

    eval_df = evaluate_detectors_on_synthetic_scenarios(orders_df, items_df)
    logger.info("\n=== Anomaly Detector Benchmark (Standard Statistical vs. ML) ===\n" + eval_df.to_string(index=False))


if __name__ == "__main__":
    main()
