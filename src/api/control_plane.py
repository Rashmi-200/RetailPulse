from pathlib import Path
from typing import Optional
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from src.copilot.rag_copilot import HybridDataEngineeringCopilot
from src.database.db_connection import get_engine
from src.lineage.lineage_tracker import OpenLineageTracker
from src.rca.rca_analyzer import RootCauseAnalyzer
from src.recovery.recovery_engine import SafeRecoveryEngine
from src.utils.config_loader import load_yaml_config

app = FastAPI(
    title="RetailPulse AI — Data Engineering & Observability Control Plane",
    version="2.0.0",
    description="Backend-first REST API for a Multi-Component Data Engineering Platform built on a Three-Layer Medallion Lakehouse (Bronze, Silver, Gold).",
)

engine = get_engine()
copilot = HybridDataEngineeringCopilot(engine)
rca_analyzer = RootCauseAnalyzer(engine)
lineage_tracker = OpenLineageTracker(engine)


class CopilotQuery(BaseModel):
    question: str


@app.get("/api/v1/health")
def get_system_health():
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        runs = conn.execute(text("SELECT COUNT(*) FROM meta.pipeline_runs")).scalar()
        open_inc = conn.execute(text("SELECT COUNT(*) FROM meta.incidents WHERE status = 'OPEN'")).scalar()
        quar_cnt = conn.execute(text("SELECT COUNT(*) FROM quarantine.quarantined_records WHERE recovery_status = 'QUARANTINED'")).scalar()
    return {
        "status": "HEALTHY",
        "database": "connected",
        "total_pipeline_runs": int(runs or 0),
        "open_incidents": int(open_inc or 0),
        "active_quarantined_records": int(quar_cnt or 0),
    }


@app.get("/api/v1/runs")
def list_pipeline_runs(limit: int = 20):
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT run_id, pipeline_name, trigger_type, status, started_at, duration_seconds
                FROM meta.pipeline_runs
                ORDER BY started_at DESC
                LIMIT :limit;
                """
            ),
            {"limit": limit},
        ).mappings().fetchall()
    return {"runs": [dict(r) for r in rows]}


@app.get("/api/v1/quality")
def get_latest_quality_scores():
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT DISTINCT ON (dataset_name)
                    dataset_name, layer, total_records, valid_records, invalid_records,
                    duplicate_records, overall_score, dimension_scores, evaluated_at
                FROM meta.quality_results
                ORDER BY dataset_name, evaluated_at DESC;
                """
            )
        ).mappings().fetchall()
    return {"datasets": [dict(r) for r in rows]}


@app.get("/api/v1/incidents")
def list_incidents():
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT incident_id, incident_type, severity, dataset_name, layer,
                       status, rca_status, rca_report, recommended_action, detected_at
                FROM meta.incidents
                ORDER BY detected_at DESC;
                """
            )
        ).mappings().fetchall()
    return {"incidents": [dict(r) for r in rows]}


@app.post("/api/v1/incidents/{incident_id}/analyze")
def trigger_rca(incident_id: str):
    try:
        report = rca_analyzer.analyze_incident(incident_id)
        return {"incident_id": incident_id, "rca_report": report}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/v1/lineage/{layer}/{dataset_name}")
def get_lineage(layer: str, dataset_name: str):
    return lineage_tracker.get_lineage_for_dataset(f"{layer}.{dataset_name}")


@app.get("/api/v1/forecast")
def get_demand_forecast():
    model_path = Path("models/demand_forecaster.joblib")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Demand forecaster model artifact not found.")
    artifact = joblib.load(model_path)
    return {
        "best_model": artifact["model_name"],
        "target": artifact["target_col"],
        "benchmark": artifact["benchmark"],
        "next_7_days": artifact["next_7_days"],
    }


@app.post("/api/v1/copilot/ask")
def ask_copilot(payload: CopilotQuery):
    return copilot.ask(payload.question)
