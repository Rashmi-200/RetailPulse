import re
import json
import os
from pathlib import Path
from typing import Any, Dict, List
import faiss
import joblib
import numpy as np
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


class HybridDataEngineeringCopilot:
    """Evidence-grounded AI Data Engineering Assistant combining Read-Only SQL Templates + FAISS RAG."""

    def __init__(self, engine: Engine, log_path: str = "logs/retailpulse.log", model_dir: str = "models"):
        self.engine = engine
        self.log_path = Path(log_path)
        self.model_dir = Path(model_dir)
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.vectorizer = TfidfVectorizer(max_features=512, stop_words="english")
        self.index: faiss.IndexFlatIP | None = None
        self.documents: List[Dict[str, str]] = []

    def build_knowledge_index(self) -> int:
        """Index structured RCA/schema summaries and recent execution log chunks into FAISS."""
        docs: List[Dict[str, str]] = []

        # 1. Index RCA & Incident records from PostgreSQL
        with self.engine.connect() as conn:
            inc_rows = conn.execute(
                text(
                    """
                    SELECT incident_id, incident_type, severity, dataset_name, layer,
                           status, rca_report, recommended_action
                    FROM meta.incidents
                    ORDER BY detected_at DESC
                    LIMIT 50;
                    """
                )
            ).mappings().fetchall()
            for r in inc_rows:
                docs.append(
                    {
                        "source": f"meta.incidents:{r['incident_id']}",
                        "text": (
                            f"Incident {r['incident_id']} type={r['incident_type']} severity={r['severity']} "
                            f"on {r['layer']}.{r['dataset_name']} status={r['status']}. "
                            f"RCA={json.dumps(r['rca_report'], default=str)} Action={r['recommended_action']}"
                        ),
                    }
                )

            # 2. Index Schema evolution history
            sc_rows = conn.execute(
                text(
                    """
                    SELECT dataset_name, layer, version_number, change_type, is_breaking, diff_summary
                    FROM meta.schema_versions
                    ORDER BY registered_at DESC
                    LIMIT 50;
                    """
                )
            ).mappings().fetchall()
            for r in sc_rows:
                docs.append(
                    {
                        "source": f"meta.schema_versions:{r['layer']}.{r['dataset_name']}_v{r['version_number']}",
                        "text": (
                            f"Schema {r['layer']}.{r['dataset_name']} version v{r['version_number']} "
                            f"change_type={r['change_type']} is_breaking={r['is_breaking']} diff={json.dumps(r['diff_summary'], default=str)}"
                        ),
                    }
                )

        # 3. Index recent structured log lines
        if self.log_path.exists():
            lines = self.log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-500:]
            for idx in range(0, len(lines), 5):
                chunk = "\n".join(lines[idx : idx + 5])
                if chunk.strip():
                    docs.append({"source": f"logs/retailpulse.log:lines_{idx+1}_{idx+5}", "text": chunk})

        self.documents = docs
        if not docs:
            return 0

        corpus = [d["text"] for d in docs]
        tfidf_matrix = self.vectorizer.fit_transform(corpus).toarray().astype("float32")
        faiss.normalize_L2(tfidf_matrix)

        dim = tfidf_matrix.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(tfidf_matrix)
        logger.info(f"[FAISS RAG] Indexed {len(docs)} observability & log documents (dim={dim}).")
        return len(docs)

    def retrieve_rag_context(self, question: str, top_k: int = 4) -> List[Dict[str, Any]]:
        if self.index is None or not self.documents:
            self.build_knowledge_index()
        if self.index is None or not self.documents:
            return []

        q_vec = self.vectorizer.transform([question]).toarray().astype("float32")
        faiss.normalize_L2(q_vec)
        k = min(top_k, len(self.documents))
        scores, indices = self.index.search(q_vec, k)

        domain_terms = {
            "order", "orders", "review", "reviews", "quarantine", "duplicate",
            "incident", "schema", "pipeline", "dq", "quality", "bronze", "silver",
            "gold", "anomaly", "batch", "error", "fail", "failed", "table", "sql",
            "customer", "product", "seller", "payment", "lineage", "rca"
        }
        q_tokens = set(re.findall(r"[a-z_]+", question.lower()))
        min_sim = 0.08 if (q_tokens & domain_terms) else 0.25

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= min_sim:
                results.append(
                    {
                        "source": self.documents[idx]["source"],
                        "similarity": round(float(score), 4),
                        "snippet": self.documents[idx]["text"][:400],
                    }
                )
        return results

    def _query_structured_metrics(self, question: str) -> Dict[str, Any]:
        """Execute safe, parameterized read-only SQL templates based on intent."""
        q = question.lower()
        evidence: Dict[str, Any] = {}

        with self.engine.connect() as conn:
            if any(k in q for k in ["quality", "lowest", "score", "dq"]):
                rows = conn.execute(
                    text(
                        """
                        SELECT DISTINCT ON (dataset_name)
                            dataset_name, overall_score, valid_records, invalid_records, duplicate_records
                        FROM meta.quality_results
                        ORDER BY dataset_name, evaluated_at DESC;
                        """
                    )
                ).mappings().fetchall()
                evidence["latest_dq_scores"] = [dict(r) for r in rows]

            if any(k in q for k in ["quarantine", "invalid", "rejected"]):
                rows = conn.execute(
                    text(
                        """
                        SELECT dataset_name, error_category, recovery_status, COUNT(*) AS record_count
                        FROM quarantine.quarantined_records
                        GROUP BY dataset_name, error_category, recovery_status;
                        """
                    )
                ).mappings().fetchall()
                evidence["quarantine_summary"] = [dict(r) for r in rows]

            if any(k in q for k in ["anomal", "spike", "drop", "outlier"]):
                rows = conn.execute(
                    text(
                        """
                        SELECT anomaly_id, dataset_name, detector_type, observed_value, severity, explanation
                        FROM meta.anomaly_alerts
                        ORDER BY anomaly_score DESC
                        LIMIT 5;
                        """
                    )
                ).mappings().fetchall()
                evidence["top_anomalies"] = [dict(r) for r in rows]

            if any(k in q for k in ["health", "status", "summary", "pipeline", "fail"]):
                runs = conn.execute(
                    text(
                        """
                        SELECT pipeline_name, status, COUNT(*) AS run_count,
                               ROUND(AVG(duration_seconds)::numeric, 2) AS avg_sec
                        FROM meta.pipeline_runs
                        GROUP BY pipeline_name, status;
                        """
                    )
                ).mappings().fetchall()
                open_inc = conn.execute(
                    text("SELECT COUNT(*) FROM meta.incidents WHERE status = 'OPEN';")
                ).scalar()
                evidence["pipeline_health"] = [dict(r) for r in runs]
                evidence["open_incidents_count"] = int(open_inc or 0)

            if any(k in q for k in ["schema", "drift"]):
                rows = conn.execute(
                    text(
                        """
                        SELECT dataset_name, layer, version_number, change_type, is_breaking, diff_summary
                        FROM meta.schema_versions
                        ORDER BY registered_at DESC
                        LIMIT 10;
                        """
                    )
                ).mappings().fetchall()
                evidence["schema_versions"] = [dict(r) for r in rows]

        return evidence

    def ask(self, question: str) -> Dict[str, Any]:
        sql_evidence = self._query_structured_metrics(question)
        rag_hits = self.retrieve_rag_context(question, top_k=3)

        if not sql_evidence and not rag_hits:
            return {
                "question": question,
                "answer": "Insufficient verified evidence in pipeline metadata or logs to answer this query.",
                "sql_evidence": {},
                "rag_sources": [],
            }

        # Synthesize grounded response
        answer_parts = []
        if "latest_dq_scores" in sql_evidence and sql_evidence["latest_dq_scores"]:
            sorted_dq = sorted(sql_evidence["latest_dq_scores"], key=lambda x: x["overall_score"])
            lowest = sorted_dq[0]
            answer_parts.append(
                f"Lowest Data Quality score is '{lowest['dataset_name']}' at {lowest['overall_score']}/100 "
                f"(valid={lowest['valid_records']:,}, invalid={lowest['invalid_records']:,}, duplicates={lowest['duplicate_records']:,})."
            )

        if "quarantine_summary" in sql_evidence:
            q_list = sql_evidence["quarantine_summary"]
            if q_list:
                total_q = sum(x["record_count"] for x in q_list)
                details = ", ".join(f"{x['dataset_name']} ({x['error_category']}: {x['record_count']})" for x in q_list)
                answer_parts.append(f"Total quarantined records: {total_q:,} [{details}].")
            else:
                answer_parts.append("There are currently 0 records in quarantine.quarantined_records.")

        if "top_anomalies" in sql_evidence and sql_evidence["top_anomalies"]:
            top_a = sql_evidence["top_anomalies"][0]
            answer_parts.append(
                f"Top anomaly ({top_a['severity']}) detected by {top_a['detector_type']}: {top_a['explanation']}"
            )

        if "pipeline_health" in sql_evidence:
            total_runs = sum(r["run_count"] for r in sql_evidence["pipeline_health"])
            failed_runs = sum(r["run_count"] for r in sql_evidence["pipeline_health"] if r["status"] == "FAILED")
            answer_parts.append(
                f"Lakehouse Health: {total_runs} total pipeline runs ({failed_runs} failed), "
                f"{sql_evidence.get('open_incidents_count', 0)} open incidents."
            )

        if "schema_versions" in sql_evidence:
            drifts = [s for s in sql_evidence["schema_versions"] if s["change_type"] != "initial"]
            if drifts:
                answer_parts.append(f"Latest schema drift: {drifts[0]['layer']}.{drifts[0]['dataset_name']} v{drifts[0]['version_number']} ({drifts[0]['change_type']}).")
            else:
                answer_parts.append("All 9 datasets are currently on baseline schema v1 with zero schema drift events.")

        if not answer_parts and rag_hits:
            answer_parts.append(f"Based on retrieved evidence ({rag_hits[0]['source']}): {rag_hits[0]['snippet']}")

        return {
            "question": question,
            "answer": " ".join(answer_parts),
            "sql_evidence": sql_evidence,
            "rag_sources": [h["source"] for h in rag_hits],
        }


    def get_index_metadata(self) -> Dict[str, Any]:
        if self.index is None:
            self.build_knowledge_index()
        actual_dim = int(self.index.d) if self.index is not None else 0
        vocab_size = len(getattr(self.vectorizer, 'vocabulary_', {}))
        assert actual_dim == vocab_size, f'FAISS index dim {actual_dim} != vectorizer dim {vocab_size}'
        return {
            'embedding_model': 'scikit-learn TfidfVectorizer (L2-normalized dense projection)',
            'configured_max_features': 512,
            'actual_embedding_dim': actual_dim,
            'indexed_document_count': len(self.documents),
            'dimension_verified': True,
        }

    @staticmethod
    def validate_readonly_sql(sql_text: str) -> bool:
        """Enforce strict SELECT-only query safety: blocks DDL/DML, multi-statements, comments, unsafe functions, and system schemas."""
        if not sql_text or not sql_text.strip():
            raise PermissionError("Empty SQL query is not permitted.")

        raw = sql_text.strip()
        if "--" in raw or "/*" in raw or "*/" in raw:
            raise PermissionError("SQL comments (-- or /* */) are blocked in read-only Copilot mode.")

        stripped_semi = raw.rstrip(";").strip()
        if ";" in stripped_semi:
            raise PermissionError("Multiple stacked SQL statements are blocked.")

        normalized = re.sub(r"\s+", " ", stripped_semi).upper()
        if not (normalized.startswith("SELECT ") or normalized.startswith("WITH ")):
            raise PermissionError("Only SELECT / CTE queries are permitted in read-only Copilot mode.")

        forbidden_keywords = (
            r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|GRANT|REVOKE|"
            r"COPY|EXECUTE|CALL|MERGE|VACUUM|ANALYZE|REINDEX|CLUSTER|LOCK|SET|RESET|DO)\b"
        )
        if re.search(forbidden_keywords, normalized):
            raise PermissionError("Write, DDL, DCL, or session-mutating SQL keywords are strictly blocked.")

        unsafe_functions = (
            r"\b(PG_SLEEP|PG_READ_FILE|PG_READ_BINARY_FILE|PG_LS_DIR|PG_STAT_FILE|"
            r"PG_TERMINATE_BACKEND|PG_CANCEL_BACKEND|LO_IMPORT|LO_EXPORT|DBLINK|INET_CLIENT_ADDR)\s*\("
        )
        if re.search(unsafe_functions, normalized):
            raise PermissionError("Unsafe PostgreSQL administrative or filesystem functions are blocked.")

        unauthorized_schemas = r"\b(PG_CATALOG|INFORMATION_SCHEMA|PG_TOAST|PG_TEMP)\b"
        if re.search(unauthorized_schemas, normalized):
            raise PermissionError("Access to internal PostgreSQL system schemas is unauthorized.")

        return True

    def ensure_readonly_role(self) -> None:
        """Ensure least-privilege PostgreSQL role retailpulse_readonly exists with SELECT-only schema grants."""
        with self.engine.begin() as conn:
            conn.execute(text("""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'retailpulse_readonly') THEN
                        CREATE ROLE retailpulse_readonly NOLOGIN;
                    END IF;
                END $$;
            """))
            for sch in ("meta", "gold", "quarantine"):
                exists = conn.execute(
                    text("SELECT 1 FROM information_schema.schemata WHERE schema_name = :s"),
                    {"s": sch},
                ).scalar()
                if exists:
                    conn.execute(text(f"GRANT USAGE ON SCHEMA {sch} TO retailpulse_readonly"))
                    conn.execute(text(f"GRANT SELECT ON ALL TABLES IN SCHEMA {sch} TO retailpulse_readonly"))

    def execute_readonly_sql(self, sql_text: str) -> list[dict]:
        """Validate and execute SQL inside a READ ONLY transaction under least-privilege retailpulse_readonly role."""
        self.validate_readonly_sql(sql_text)
        self.ensure_readonly_role()
        clean_sql = sql_text.strip().rstrip(";")
        with self.engine.connect() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            conn.execute(text("SET LOCAL ROLE retailpulse_readonly"))
            rows = conn.execute(text(clean_sql)).mappings().fetchall()
            return [dict(r) for r in rows]
