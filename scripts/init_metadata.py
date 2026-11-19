import sys
from pathlib import Path
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db_connection import get_engine, verify_connection
from src.observability.tracker import ObservabilityTracker
from src.utils.logger import get_logger

logger = get_logger("init_metadata")


def apply_ddl(engine, ddl_path: Path) -> None:
    sql_script = ddl_path.read_text(encoding="utf-8")
    with engine.begin() as conn:
        conn.execute(text(sql_script))
    logger.info(f"Applied DDL script: {ddl_path}")


def verify_meta_tables(engine) -> list[str]:
    query = text(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'meta'
        ORDER BY table_name;
        """
    )
    with engine.connect() as conn:
        tables = [row[0] for row in conn.execute(query)]
    return tables


def run_smoke_test(engine) -> None:
    tracker = ObservabilityTracker(engine)
    with tracker.track_pipeline_run("metadata_bootstrap_check", trigger_type="verification") as run_id:
        with tracker.track_task_run(run_id, "verify_observability_store", layer="meta") as metrics:
            tables = verify_meta_tables(engine)
            metrics["records_in"] = len(tables)
            metrics["records_out"] = len(tables)
    logger.info(f"Smoke test recorded cleanly under run_id={run_id}")


def main() -> None:
    engine = get_engine()
    verify_connection(engine)

    ddl_file = PROJECT_ROOT / "sql" / "ddl" / "01_create_meta_tables.sql"
    apply_ddl(engine, ddl_file)

    tables = verify_meta_tables(engine)
    logger.info(f"Verified {len(tables)} tables in 'meta' schema: {tables}")
    assert len(tables) == 10, f"Expected 10 meta tables, found {len(tables)}"

    run_smoke_test(engine)
    logger.info("Milestone 3 Observability & Metadata Foundation is ready!")


if __name__ == "__main__":
    main()
