import json
import sys
from pathlib import Path
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db_connection import get_engine, verify_connection
from src.rca.rca_analyzer import RootCauseAnalyzer
from src.utils.logger import get_logger

logger = get_logger("rca_diagnostics")


def main() -> None:
    engine = get_engine()
    verify_connection(engine)
    analyzer = RootCauseAnalyzer(engine)

    with engine.connect() as conn:
        pending = conn.execute(
            text("SELECT incident_id, incident_type, dataset_name FROM meta.incidents ORDER BY detected_at DESC")
        ).fetchall()

    if not pending:
        logger.info("No incidents found in meta.incidents.")
        return

    for inc_id, inc_type, ds in pending:
        report = analyzer.analyze_incident(inc_id)
        logger.info(f"\n=== RCA Report for {inc_id} ({inc_type} on {ds}) ===\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
