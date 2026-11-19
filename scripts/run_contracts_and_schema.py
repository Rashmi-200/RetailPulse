import sys
from pathlib import Path
from deltalake import DeltaTable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts.contract_engine import DataContractEngine
from src.database.db_connection import get_engine, verify_connection
from src.observability.tracker import ObservabilityTracker
from src.schema_evolution.registry import SchemaEvolutionRegistry
from src.utils.config_loader import load_yaml_config
from src.utils.logger import get_logger

logger = get_logger("schema_and_contracts")


def main() -> None:
    pipeline_cfg = load_yaml_config("config/pipeline_config.yaml")
    contracts_cfg = load_yaml_config("config/data_contracts.yaml")
    bronze_dir = Path(pipeline_cfg["paths"]["bronze_dir"])

    engine = get_engine()
    verify_connection(engine)
    tracker = ObservabilityTracker(engine)
    schema_registry = SchemaEvolutionRegistry(engine, tracker)
    contract_engine = DataContractEngine(engine, contracts_cfg)

    with tracker.track_pipeline_run("schema_and_contract_gate", trigger_type="cli") as run_id:
        for dataset_name, contract in contracts_cfg["contracts"].items():
            delta_path = bronze_dir / dataset_name / "_delta"
            if not delta_path.exists():
                logger.warning(f"Skipping {dataset_name}: Bronze Delta table not found.")
                continue

            with tracker.track_task_run(run_id, f"validate_{dataset_name}", layer="silver_gate") as metrics:
                df = DeltaTable(str(delta_path)).to_pandas()
                batch_id = str(df["_batch_id"].iloc[-1]) if "_batch_id" in df.columns else "unknown_batch"

                # 1. Register & check schema evolution
                schema_registry.check_and_register(
                    df=df,
                    dataset_name=dataset_name,
                    layer="bronze",
                    required_columns=contract.get("required_columns", []),
                    batch_id=batch_id,
                    run_id=run_id,
                )

                # 2. Enforce Data Contract & profile 6 DQ dimensions
                report = contract_engine.validate_and_split(
                    df=df,
                    dataset_name=dataset_name,
                    batch_id=batch_id,
                    run_id=run_id,
                    layer="silver",
                    persist=True,
                )
                metrics["records_in"] = report.total_records
                metrics["records_out"] = report.valid_records

    logger.info("Schema Registry & Data Contract evaluation completed across all datasets!")


if __name__ == "__main__":
    main()
