import sys
from pathlib import Path
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db_connection import get_engine, verify_connection
from src.lineage.lineage_tracker import OpenLineageTracker
from src.observability.tracker import ObservabilityTracker
from src.utils.config_loader import load_yaml_config
from src.utils.logger import get_logger

logger = get_logger("lineage_sync")

COLUMN_LINEAGE_MAP = {
    "order_items": {
        "silver.order_items.total_item_value": [
            "bronze.order_items.price",
            "bronze.order_items.freight_value",
        ],
        "silver.order_items.shipping_limit_date": ["bronze.order_items.shipping_limit_date"],
    },
    "orders": {
        "silver.orders.order_purchase_timestamp": ["bronze.orders.order_purchase_timestamp"],
        "silver.orders.order_delivered_customer_date": ["bronze.orders.order_delivered_customer_date"],
    },
    "payments": {
        "silver.payments.payment_type": ["bronze.payments.payment_type"],
        "silver.payments.payment_value": ["bronze.payments.payment_value"],
    },
    "geolocation": {
        "silver.geolocation.geolocation_lat": ["bronze.geolocation.geolocation_lat (MEDIAN_BY_ZIP)"],
        "silver.geolocation.geolocation_lng": ["bronze.geolocation.geolocation_lng (MEDIAN_BY_ZIP)"],
    },
}


def main() -> None:
    config = load_yaml_config()
    engine = get_engine()
    verify_connection(engine)

    obs = ObservabilityTracker(engine)
    lineage = OpenLineageTracker(engine)

    # Map each dataset to its upstream source system
    source_map = {}
    for ds in config["sources"]["csv_files"].keys():
        source_map[ds] = f"csv_files.{ds}"
    for ds in config["sources"]["postgres_oltp"]["tables"].keys():
        source_map[ds] = f"postgres.source_oltp.{ds}"
    for ds in config["sources"]["rest_api"].keys():
        source_map[ds] = f"rest_api.{ds}"

    with obs.track_pipeline_run("openlineage_metadata_registration", trigger_type="cli") as run_id:
        with obs.track_task_run(run_id, "register_bronze_silver_lineage", layer="meta") as metrics:
            events_emitted = 0
            for ds_name, src_qualified in source_map.items():
                # Query latest batch_id for this dataset
                row = engine.connect().execute(
                    text(
                        """
                        SELECT latest_batch_id, row_count
                        FROM meta.dataset_metadata
                        WHERE dataset_key = :key
                        """
                    ),
                    {"key": f"silver.{ds_name}"},
                ).fetchone()

                batch_id = row[0] if row else "unknown_batch"
                silver_rows = row[1] if row else 0

                # 1. Source -> Bronze lineage event
                lineage.emit_event(
                    event_type="COMPLETE",
                    run_id=run_id,
                    job_name=f"ingest_{ds_name}_to_bronze",
                    inputs=[{"namespace": "retailpulse.sources", "name": src_qualified}],
                    outputs=[{"namespace": "retailpulse.lakehouse", "name": f"bronze.{ds_name}"}],
                    batch_id=batch_id,
                )
                events_emitted += 1

                # 2. Bronze -> Silver (+ Quarantine) lineage event with column lineage
                outputs = [
                    {
                        "namespace": "retailpulse.lakehouse",
                        "name": f"silver.{ds_name}",
                        "facets": {"rowCount": silver_rows},
                    }
                ]
                if ds_name == "order_reviews":
                    outputs.append({"namespace": "retailpulse.lakehouse", "name": f"quarantine.{ds_name}"})

                lineage.emit_event(
                    event_type="COMPLETE",
                    run_id=run_id,
                    job_name=f"transform_{ds_name}_to_silver",
                    inputs=[{"namespace": "retailpulse.lakehouse", "name": f"bronze.{ds_name}"}],
                    outputs=outputs,
                    batch_id=batch_id,
                    column_lineage=COLUMN_LINEAGE_MAP.get(ds_name, {}),
                )
                events_emitted += 1

            metrics["records_in"] = len(source_map)
            metrics["records_out"] = events_emitted

    # Verify lineage graph traversal for silver.order_items
    graph = lineage.get_lineage_for_dataset("silver.order_items")
    logger.info(f"Lineage verification for 'silver.order_items': {graph}")


if __name__ == "__main__":
    main()
