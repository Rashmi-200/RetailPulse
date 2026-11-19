import sys
from pathlib import Path
import pandas as pd
from deltalake import DeltaTable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db_connection import get_engine, verify_connection
from src.observability.tracker import ObservabilityTracker
from src.recovery.recovery_engine import SafeRecoveryEngine
from src.utils.config_loader import load_yaml_config
from src.utils.logger import get_logger

logger = get_logger("recovery_verification")


def main() -> None:
    policies_cfg = load_yaml_config("config/recovery_policies.yaml")
    contracts_cfg = load_yaml_config("config/data_contracts.yaml")
    engine = get_engine()
    verify_connection(engine)

    tracker = ObservabilityTracker(engine)
    recovery = SafeRecoveryEngine(engine, tracker, policies_cfg, contracts_cfg)

    # 1. Verify policy recommendations across failure classes
    for code in ["TRANSIENT_API_TIMEOUT", "SCHEMA_BREAKING", "MISSING_REQUIRED_FIELD"]:
        rec = recovery.recommend_recovery(code)
        logger.info(f"Policy Check [{code}] -> action={rec['recommended_action']}, human_approval={rec['requires_human_approval']}")

    # 2. Verify idempotent replay on silver.orders (replaying 2 existing orders must NOT inflate row count)
    silver_orders = DeltaTable("data/silver/orders/_delta").to_pandas()
    before_count = len(silver_orders)
    sample_patch = silver_orders.head(2).copy()
    batch_id = str(sample_patch["_batch_id"].iloc[0])

    res1 = recovery.replay_corrected_records(
        dataset_name="orders",
        batch_id=batch_id,
        corrected_df=sample_patch,
        approved_by="data_eng_lead",
        notes="Idempotency replay verification",
    )
    res2 = recovery.replay_corrected_records(
        dataset_name="orders",
        batch_id=batch_id,
        corrected_df=sample_patch,
        approved_by="data_eng_lead",
        notes="Second duplicate replay attempt (must remain idempotent)",
    )

    assert res1["silver_total_rows"] == before_count == res2["silver_total_rows"], "Idempotency check failed!"
    logger.info(f"Idempotent Replay Verified: before={before_count:,}, after_replay_1={res1['silver_total_rows']:,}, after_replay_2={res2['silver_total_rows']:,}")


if __name__ == "__main__":
    main()
