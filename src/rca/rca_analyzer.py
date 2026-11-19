import json
import os
from typing import Any, Dict, Optional
import requests
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.rca.evidence_collector import RCAEvidenceCollector
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

SYSTEM_RCA_PROMPT = """You are a Principal Data Engineer performing Root Cause Analysis (RCA).
Strictly analyze ONLY the JSON evidence packet provided.
Rules:
1. NEVER invent causes, columns, or error messages not present in the evidence.
2. Cite exact identifiers (batch_id, run_id, task_run_id, quarantine_id, rule names, metric values).
3. If evidence is insufficient to confirm a root cause, explicitly state 'Insufficient evidence to confirm root cause'.
4. Return ONLY valid JSON with keys:
   - what_happened
   - why_it_happened
   - affected_assets
   - supporting_evidence (list of strings with exact IDs)
   - recommended_action
   - confidence_level ('CONFIRMED_BY_EVIDENCE' | 'CONTRIBUTING_FACTORS_IDENTIFIED' | 'INSUFFICIENT_EVIDENCE')
"""


class RootCauseAnalyzer:
    """Produces evidence-grounded 5-point RCA reports using LLM API with deterministic fallback."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.collector = RCAEvidenceCollector(engine)
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    def _call_llm_if_configured(self, evidence: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        payload_str = json.dumps(evidence, default=str, indent=2)

        if self.gemini_key:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_key}"
                body = {
                    "contents": [{"parts": [{"text": f"{SYSTEM_RCA_PROMPT}\n\nEVIDENCE:\n{payload_str}"}]}],
                    "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
                }
                resp = requests.post(url, json=body, timeout=20)
                if resp.status_code == 200:
                    text_out = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                    return json.loads(text_out)
            except Exception as exc:
                logger.warning(f"Gemini API unavailable, falling back to deterministic evidence synthesizer: {exc}")

        if self.openai_key:
            try:
                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"},
                    json={
                        "model": "gpt-4o-mini",
                        "temperature": 0.1,
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": SYSTEM_RCA_PROMPT},
                            {"role": "user", "content": payload_str},
                        ],
                    },
                    timeout=20,
                )
                if resp.status_code == 200:
                    return json.loads(resp.json()["choices"][0]["message"]["content"])
            except Exception as exc:
                logger.warning(f"OpenAI API unavailable, falling back to deterministic evidence synthesizer: {exc}")

        return None

    def _deterministic_evidence_synthesis(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        """Strictly derives the 5-point RCA report directly from verified database telemetry."""
        inc_type = ev["incident_type"]
        ds = ev.get("dataset_name") or "unknown_dataset"
        layer = ev.get("layer") or "unknown_layer"
        batch_id = ev.get("batch_id") or "N/A"
        run_id = ev.get("run_id") or "N/A"

        citations = [f"incident_id={ev['incident_id']}", f"run_id={run_id}", f"batch_id={batch_id}"]
        failed_tasks = [t for t in ev.get("task_executions", []) if t.get("status") == "FAILED"]

        if failed_tasks:
            t = failed_tasks[0]
            citations.append(f"task_run_id={t['task_run_id']}")
            what = f"Task '{t['task_name']}' failed in the {layer} layer while processing dataset '{ds}'."
            why = f"Exception {t.get('exception_type')}: {str(t.get('stack_trace_excerpt', '')).splitlines()[-1] if t.get('stack_trace_excerpt') else 'See stack trace'}."
            action = "Inspect failed task exception, fix upstream schema/code defect, and execute policy-approved replay."
            confidence = "CONFIRMED_BY_EVIDENCE"

        elif inc_type == "RECORDS_QUARANTINED" and ev.get("quality_reports"):
            dq = ev["quality_reports"][0]
            failed_rules = [r for r in dq.get("rule_results", []) if r.get("status") == "FAILED"]
            q_samples = ev.get("quarantine_samples", [])
            if q_samples:
                citations.append(f"sample_quarantine_id={q_samples[0]['quarantine_id']}")
            citations.append(f"dq_result_id={dq['dq_result_id']}")
            rule_desc = ", ".join(f"{r['rule']} (failed={r['failed_count']})" for r in failed_rules) or "contract rule failure"
            what = (
                f"{dq['invalid_records']} out of {dq['total_records']} records in '{ds}' failed Data Contract validation "
                f"(DQ Score={dq['overall_score']}/100) and were routed to quarantine."
            )
            why = f"Contract validation detected {rule_desc}. Sample error category: {q_samples[0]['error_category'] if q_samples else 'CONTRACT_VIOLATION'}."
            action = (
                f"Review quarantined rows in quarantine.quarantined_records for batch '{batch_id}' "
                f"and execute '{q_samples[0]['recommended_action'] if q_samples else 'MANUAL_INSPECTION'}'."
            )
            confidence = "CONFIRMED_BY_EVIDENCE"

        elif inc_type == "SCHEMA_DRIFT" and ev.get("schema_history"):
            latest_sc = ev["schema_history"][0]
            citations.append(f"schema_version=v{latest_sc['version_number']}")
            what = f"Schema drift ({latest_sc['change_type']}) detected on '{layer}.{ds}' (v{latest_sc['version_number']})."
            why = f"Schema diff detected: {json.dumps(latest_sc.get('diff_summary', {}))}."
            action = (
                "Pipeline blocked due to breaking schema change. Coordinate with source owner or update data contract."
                if latest_sc.get("is_breaking")
                else "Non-breaking schema evolution recorded. Verify downstream Silver/Gold mappings."
            )
            confidence = "CONFIRMED_BY_EVIDENCE"

        elif inc_type == "STATISTICAL_ML_ANOMALY" and ev.get("anomaly_alerts"):
            top_alerts = ev["anomaly_alerts"]
            citations.extend([f"anomaly_id={a['anomaly_id']}" for a in top_alerts[:3]])
            what = f"{len(top_alerts)} statistical/ML anomalies detected on '{ds}' metrics ({top_alerts[0]['metric_name']})."
            why = f"Detectors ({top_alerts[0]['detector_type']}) flagged values outside {top_alerts[0]['expected_range']}. Top finding: {top_alerts[0]['explanation']}"
            action = "Valid business records were preserved in Silver. Verify whether spike/drop corresponds to a legitimate business event or partial source feed."
            confidence = "CONFIRMED_BY_EVIDENCE"

        else:
            what = f"Incident '{inc_type}' recorded on '{layer}.{ds}'."
            why = "Insufficient evidence in task_runs, quality_results, or schema_versions to confirm a specific root cause."
            action = "Inspect raw source files and application logs manually."
            confidence = "INSUFFICIENT_EVIDENCE"

        upstream = [u["job_name"] for u in ev.get("lineage", {}).get("upstream_jobs", [])]
        return {
            "what_happened": what,
            "why_it_happened": why,
            "affected_assets": {
                "dataset": ds,
                "layer": layer,
                "batch_id": batch_id,
                "upstream_jobs": upstream,
            },
            "supporting_evidence": citations,
            "recommended_action": action,
            "confidence_level": confidence,
        }

    def analyze_incident(self, incident_id: str) -> Dict[str, Any]:
        evidence = self.collector.collect_for_incident(incident_id)
        report = self._call_llm_if_configured(evidence)
        generation_mode = "LLM_API"
        if report is None:
            report = self._deterministic_evidence_synthesis(evidence)
            generation_mode = "EVIDENCE_SYNTHESIZER"

        report["analysis_engine"] = generation_mode

        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE meta.incidents
                    SET rca_status = 'COMPLETED',
                        rca_report = CAST(:rca_report AS JSONB),
                        recommended_action = :rec_action
                    WHERE incident_id = :incident_id
                    """
                ),
                {
                    "rca_report": json.dumps(report),
                    "rec_action": report.get("recommended_action", ""),
                    "incident_id": incident_id,
                },
            )

        logger.info(f"[RCA COMPLETED] {incident_id} via {generation_mode} | Confidence={report.get('confidence_level')}")
        return report
