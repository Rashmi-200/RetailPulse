from collections import defaultdict, deque
import hmac
import os
from pathlib import Path
import re
import time
from typing import Deque, Dict, Optional
import joblib
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.copilot.rag_copilot import HybridDataEngineeringCopilot
from src.database.db_connection import get_engine
from src.lineage.lineage_tracker import OpenLineageTracker
from src.rca.rca_analyzer import RootCauseAnalyzer
from src.utils.config_loader import is_insecure_secret, validate_runtime_environment
from src.utils.logger import get_logger

logger = get_logger("control_plane_api")

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_REQUEST_BUCKETS: Dict[str, Deque[float]] = defaultdict(deque)


def reset_rate_limiter() -> None:
    """Reset in-memory rate limiter buckets (useful for isolated tests)."""
    _REQUEST_BUCKETS.clear()


def _is_auth_enforced() -> bool:
    app_env = os.getenv("APP_ENV", "development").strip().lower()
    explicit = os.getenv("API_AUTH_REQUIRED", "").strip().lower()
    configured_key = os.getenv("RETAILPULSE_API_KEY", "").strip()
    if app_env in {"prod", "production"} or explicit in {"1", "true", "yes"}:
        return True
    return bool(configured_key)


def require_api_auth(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> None:
    """Enforce configurable API key / Bearer token authentication and per-client rate limiting."""
    # 1. Sliding-window per-client rate limiting
    try:
        max_per_min = int(os.getenv("API_RATE_LIMIT_PER_MIN", "120"))
    except ValueError:
        max_per_min = 120

    if max_per_min > 0:
        client_id = request.client.host if request.client else "unknown"
        now = time.monotonic()
        bucket = _REQUEST_BUCKETS[client_id]
        while bucket and (now - bucket[0]) > 60.0:
            bucket.popleft()
        if len(bucket) >= max_per_min:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please retry later.",
            )
        bucket.append(now)

    # 2. Authentication check (fail-closed in production or when RETAILPULSE_API_KEY is configured)
    if not _is_auth_enforced():
        return

    expected_key = os.getenv("RETAILPULSE_API_KEY", "").strip()
    if not expected_key or is_insecure_secret(expected_key):
        logger.error("API authentication is enabled but RETAILPULSE_API_KEY is missing or insecure.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not properly configured.",
        )

    supplied_token: Optional[str] = None
    if x_api_key and x_api_key.strip():
        supplied_token = x_api_key.strip()
    elif authorization and authorization.strip().lower().startswith("bearer "):
        supplied_token = authorization.strip()[7:].strip()

    if not supplied_token or not hmac.compare_digest(supplied_token, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _validate_identifier(value: str, field_name: str) -> str:
    if not _IDENTIFIER_PATTERN.fullmatch(value or ""):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid {field_name} format.",
        )
    return value


app = FastAPI(
    title="RetailPulse AI — Data Engineering & Observability Control Plane",
    version="2.0.0",
    description="Backend-first REST API for a Multi-Component Data Engineering Platform built on a Three-Layer Medallion Lakehouse (Bronze, Silver, Gold).",
)


@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    logger.error(f"Database error on {request.method} {request.url.path}: {exc.__class__.__name__}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Database service unavailable."},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled API error on {request.method} {request.url.path}: {exc.__class__.__name__}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )



@app.on_event("startup")
def validate_startup_configuration() -> None:
    """Enforce production secret & environment validation at API startup."""
    validate_runtime_environment(service="api")

engine = get_engine()
copilot = HybridDataEngineeringCopilot(engine)
rca_analyzer = RootCauseAnalyzer(engine)
lineage_tracker = OpenLineageTracker(engine)


class CopilotQuery(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


@app.get("/api/v1/health")
def get_system_health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            runs = conn.execute(text("SELECT COUNT(*) FROM meta.pipeline_runs")).scalar()
            open_inc = conn.execute(text("SELECT COUNT(*) FROM meta.incidents WHERE status = 'OPEN'")).scalar()
            quar_cnt = conn.execute(
                text("SELECT COUNT(*) FROM quarantine.quarantined_records WHERE recovery_status = 'QUARANTINED'")
            ).scalar()
        return {
            "status": "HEALTHY",
            "database": "connected",
            "total_pipeline_runs": int(runs or 0),
            "open_incidents": int(open_inc or 0),
            "active_quarantined_records": int(quar_cnt or 0),
        }
    except SQLAlchemyError as exc:
        logger.error(f"Health check database failure: {exc.__class__.__name__}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health check failed: database unreachable.",
        )


@app.get("/api/v1/runs", dependencies=[Depends(require_api_auth)])
def list_pipeline_runs(limit: int = Query(default=20, ge=1, le=500)):
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT run_id, pipeline_name, trigger_type, status AS execution_status, COALESCE(dq_status, 'PASSED') AS dq_status, started_at, duration_seconds
                FROM meta.pipeline_runs
                ORDER BY started_at DESC
                LIMIT :limit;
                """
            ),
            {"limit": limit},
        ).mappings().fetchall()
    return {"runs": [dict(r) for r in rows]}


@app.get("/api/v1/quality", dependencies=[Depends(require_api_auth)])
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


@app.get("/api/v1/incidents", dependencies=[Depends(require_api_auth)])
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


@app.post("/api/v1/incidents/{incident_id}/analyze", dependencies=[Depends(require_api_auth)])
def trigger_rca(incident_id: str):
    _validate_identifier(incident_id, "incident_id")
    try:
        report = rca_analyzer.analyze_incident(incident_id)
        return {"incident_id": incident_id, "rca_report": report}
    except ValueError:
        raise HTTPException(status_code=404, detail="Incident not found or analysis unavailable.")


@app.get("/api/v1/lineage/{layer}/{dataset_name}", dependencies=[Depends(require_api_auth)])
def get_lineage(layer: str, dataset_name: str):
    _validate_identifier(layer, "layer")
    _validate_identifier(dataset_name, "dataset_name")
    return lineage_tracker.get_lineage_for_dataset(f"{layer}.{dataset_name}")


@app.get("/api/v1/forecast", dependencies=[Depends(require_api_auth)])
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


@app.post("/api/v1/copilot/ask", dependencies=[Depends(require_api_auth)])
def ask_copilot(payload: CopilotQuery):
    try:
        return copilot.ask(payload.question)
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Query rejected by read-only security policy.",
        )
