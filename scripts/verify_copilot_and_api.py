import json
import sys
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.control_plane import app
from src.utils.logger import get_logger

logger = get_logger("control_plane_verify")


def main() -> None:
    client = TestClient(app)

    # 1. Verify /health
    health = client.get("/api/v1/health").json()
    logger.info(f"GET /api/v1/health -> {health}")

    # 2. Verify /forecast
    forecast = client.get("/api/v1/forecast").json()
    logger.info(f"GET /api/v1/forecast -> Best Model={forecast['best_model']}, Days={len(forecast['next_7_days'])}")

    # 3. Verify Hybrid SQL + FAISS RAG Copilot across realistic Data Engineering questions
    sample_questions = [
        "Which dataset has the lowest quality score?",
        "How many records were quarantined?",
        "What anomalies were detected?",
        "Explain the latest schema drift.",
        "Summarise the overall health of my data lakehouse.",
    ]
    for q in sample_questions:
        res = client.post("/api/v1/copilot/ask", json={"question": q}).json()
        logger.info(f"\n[COPILOT Q]: {q}\n[COPILOT A]: {res['answer']}\n[SOURCES]  : {res['rag_sources']}")


if __name__ == "__main__":
    main()
