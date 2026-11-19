import json
from datetime import datetime, timezone
from typing import Any, Dict, List
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.rca.evidence_collector import RCAEvidenceCollector
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RootCauseAnalyzer:
    """Evidence-grounded diagnostic analyzer correlating 7 pipeline telemetry sources without fabricating unsupported claims."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.collector = RCAEvidenceCollector(engine)

    def analyze_incident(self, incident_id: str) -> Dict[str, Any]:
        bundle = self.collector.collect_for_incident(incident_id)
        inc_type = bundle["incident_type"]
        dataset = bundle["dataset_name"]
        layer = bundle["layer"]

        evidence_refs: List[Dict[str, Any]] = [
            {"type": "incident", "id": incident_id, "incident_type": inc_type, "severity": bundle["severity"]}
        ]
        checks_evaluated = 4
        checks_with_evidence = 0

        # 1. Task execution evidence
        failed_tasks = [t for t in bundle.get("task_executions", []) if t.get("status") == "FAILED"]
        if failed_tasks:
            checks_with_evidence += 1
            for t in failed_tasks:
                evidence_refs.append(
                    {"type": "task_run", "id": t["task_run_id"], "task_name": t["task_name"], "exception": t.get("exception_type")}
                )

        # 2. Data Quality & Quarantine evidence
        dq_list = bundle.get("quality_reports", [])
        q_samples = bundle.get("quarantine_samples", [])
        has_dq_defect = bool(dq_list and int(dq_list[0].get("invalid_records", 0)) > 0) or bool(q_samples)
        if has_dq_defect:
            checks_with_evidence += 1
            dq0 = dq_list[0] if dq_list else {}
            evidence_refs.append(
                {
                    "type": "quality_result",
                    "id": dq0.get("dq_result_id"),
                    "overall_score": float(dq0.get("overall_score", 0.0)),
                    "invalid_records": int(dq0.get("invalid_records", len(q_samples))),
                    "sample_quarantine_ids": [q["quarantine_id"] for q in q_samples],
                }
            )

        # 3. Schema drift evidence
        drifts = [s for s in bundle.get("schema_history", []) if s.get("change_type") != "initial"]
        if drifts:
            checks_with_evidence += 1
            evidence_refs.append(
                {
                    "type": "schema_version",
                    "version": drifts[0]["version_number"],
                    "change_type": drifts[0]["change_type"],
                    "diff": drifts[0]["diff_summary"],
                }
            )

        # 4. Statistical / ML anomaly evidence
        anomalies = bundle.get("anomaly_alerts", [])
        if anomalies:
            checks_with_evidence += 1
            evidence_refs.append(
                {
                    "type": "anomaly_alert",
                    "id": anomalies[0]["anomaly_id"],
                    "detector": anomalies[0]["detector_type"],
                    "explanation": anomalies[0]["explanation"],
                }
            )

        completeness_score = round(checks_with_evidence / checks_evaluated, 2)
        downstream = bundle.get("lineage", {}).get("downstream_jobs", [])

        # Insufficient-evidence guardrail
        if checks_with_evidence == 0:
            report = {
                "status": "INSUFFICIENT_EVIDENCE",
                "incident_id": incident_id,
                "root_cause": "Insufficient verified telemetry evidence to isolate root cause.",
                "evidence": evidence_refs,
                "evidence_completeness_score": 0.0,
                "downstream_impact": downstream,
                "recommended_action": "Inspect raw upstream source logs and re-run diagnostic telemetry collection.",
                "requires_human_approval": True,
                "analyzed_at": datetime.now(timezone.utc).isoformat(),
            }
            self._persist_rca(incident_id, report)
            return report

        # Deterministic evidence-backed classification
        if inc_type in ("DATA_CONTRACT_VIOLATION", "RECORDS_QUARANTINED") or has_dq_defect:
            inv_cnt = int(dq_list[0]["invalid_records"]) if dq_list else len(q_samples)
            cats = sorted({q["error_category"] for q in q_samples}) if q_samples else ["CONTRACT_VIOLATION"]
            root_cause = (
                f"Contract rule failures on {layer}.{dataset}: {inv_cnt} invalid/duplicate records "
                f"isolated to quarantine ({cats})."
            )
            action = f"Review quarantined rows for {dataset} and execute policy-approved idempotent replay."
        elif inc_type == "SCHEMA_DRIFT" or drifts:
            root_cause = f"Upstream schema change detected on {layer}.{dataset}: {drifts[0]['diff_summary']}."
            action = f"Verify upstream producer contract for {dataset} before promoting schema version."
        elif inc_type == "STATISTICAL_ML_ANOMALY" or anomalies:
            root_cause = f"Operational KPI deviation detected on {layer}.{dataset}: {anomalies[0]['explanation']}."
            action = "Validate whether KPI shift reflects a genuine business event (e.g., promotion) or upstream ingestion lag."
        else:
            root_cause = f"Pipeline execution failure in task '{failed_tasks[0]['task_name']}': {failed_tasks[0].get('exception_type')}."
            action = f"Resolve task error in '{failed_tasks[0]['task_name']}' and trigger safe batch replay."

        report = {
            "status": "DIAGNOSED",
            "incident_id": incident_id,
            "root_cause": root_cause,
            "evidence": evidence_refs,
            "evidence_completeness_score": completeness_score,
            "downstream_impact": downstream,
            "recommended_action": action,
            "requires_human_approval": True,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._persist_rca(incident_id, report)
        logger.info(f"[RCA COMPLETE] {incident_id} | completeness={completeness_score} | cause={root_cause}")
        return report

    def _persist_rca(self, incident_id: str, report: Dict[str, Any]) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE meta.incidents
                    SET rca_status = :status,
                        rca_report = CAST(:report AS JSONB),
                        recommended_action = :action
                    WHERE incident_id = :incident_id;
                    """
                ),
                {
                    "status": report["status"],
                    "report": json.dumps(report, default=str),
                    "action": report["recommended_action"],
                    "incident_id": incident_id,
                },
            )
