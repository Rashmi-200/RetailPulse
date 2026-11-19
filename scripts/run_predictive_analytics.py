import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.demand_forecasting import RetailDemandForecaster
from src.analytics.predictive_dq import PredictiveDQEngine
from src.database.db_connection import get_engine, verify_connection
from src.observability.tracker import ObservabilityTracker
from src.utils.logger import get_logger

logger = get_logger("predictive_analytics")


def main() -> None:
    engine = get_engine()
    verify_connection(engine)
    tracker = ObservabilityTracker(engine)

    with tracker.track_pipeline_run("predictive_dq_and_demand_forecasting", trigger_type="cli") as run_id:
        # 1. Predictive Data Quality Intelligence (Cold-Start Check + Training)
        with tracker.track_task_run(run_id, "train_predictive_dq", layer="analytics") as m1:
            pdq = PredictiveDQEngine(engine)
            cold_status = pdq.train_and_evaluate(allow_historical_simulation=False)
            logger.info(f"Cold-start guard check: {cold_status}")

            train_res = pdq.train_and_evaluate(allow_historical_simulation=True)
            risk_sample = pdq.predict_next_batch_risk(
                {
                    "prev_dq_score": 99.73,
                    "null_pct": 0.2,
                    "duplicate_pct": 0.82,
                    "schema_warning_count": 0,
                    "processing_duration_sec": 4.1,
                    "row_drop_pct": 0.82,
                }
            )
            logger.info(f"Current order_reviews batch degradation risk: {risk_sample}")
            m1["records_in"] = 120
            m1["records_out"] = 1

        # 2. Gold Layer Retail Demand Forecasting
        with tracker.track_task_run(run_id, "train_gold_demand_forecaster", layer="gold") as m2:
            forecaster = RetailDemandForecaster(engine)
            fc_res = forecaster.train_and_benchmark(target_col="daily_orders")
            logger.info("Next 7-Day Gold Order Volume Forecast:\n" + json.dumps(fc_res["next_7_days_forecast"], indent=2))
            m2["records_in"] = fc_res["test_window_days"]
            m2["records_out"] = 7

    logger.info("Milestone 12 Predictive DQ & Gold Demand Forecasting completed successfully!")


if __name__ == "__main__":
    main()
