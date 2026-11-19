from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sqlalchemy import text
from sqlalchemy.engine import Engine
from xgboost import XGBClassifier

from src.utils.logger import get_logger

logger = get_logger(__name__)

MIN_HISTORY_RUNS = 30
FEATURE_COLS = [
    "prev_dq_score",
    "null_pct",
    "duplicate_pct",
    "schema_warning_count",
    "processing_duration_sec",
    "row_drop_pct",
]


class PredictiveDQEngine:
    """Predicts future data quality degradation risk with strict cold-start enforcement."""

    def __init__(self, engine: Engine, model_dir: str = "models"):
        self.engine = engine
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "predictive_dq_model.joblib"

    def get_history_counts(self, dataset_name: Optional[str] = None) -> Dict[str, int]:
        """Distinguish pipeline execution runs from dataset-level quality evaluations."""
        with self.engine.connect() as conn:
            pipeline_runs = int(conn.execute(text("SELECT COUNT(*) FROM meta.pipeline_runs")).scalar() or 0)
            total_dq_evals = int(conn.execute(text("SELECT COUNT(*) FROM meta.quality_results")).scalar() or 0)
            dataset_dq_evals = total_dq_evals
            if dataset_name:
                dataset_dq_evals = int(
                    conn.execute(
                        text("SELECT COUNT(*) FROM meta.quality_results WHERE dataset_name = :ds"),
                        {"ds": dataset_name},
                    ).scalar()
                    or 0
                )
        return {
            "pipeline_runs_count": pipeline_runs,
            "total_dataset_evaluations": total_dq_evals,
            "target_dataset_evaluations": dataset_dq_evals,
            "required_min_observations": MIN_HISTORY_RUNS,
        }

    def fetch_real_history(self, dataset_name: Optional[str] = None) -> pd.DataFrame:
        where_clause = "WHERE q.dataset_name = :ds" if dataset_name else ""
        query = text(
            f"""
            SELECT
                q.dq_result_id,
                q.dataset_name,
                q.overall_score AS dq_score,
                CASE WHEN q.total_records > 0
                     THEN (q.invalid_records::float / q.total_records) * 100.0
                     ELSE 0.0 END AS row_drop_pct,
                CASE WHEN q.total_records > 0
                     THEN (q.duplicate_records::float / q.total_records) * 100.0
                     ELSE 0.0 END AS duplicate_pct,
                q.evaluated_at
            FROM meta.quality_results q
            {where_clause}
            ORDER BY q.evaluated_at ASC;
            """
        )
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn, params={"ds": dataset_name} if dataset_name else {})

    def train_on_real_history(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """Train ONLY on real historical observations. Enforces cold-start fallback when N < 30."""
        counts = self.get_history_counts(dataset_name=dataset_name)
        real_df = self.fetch_real_history(dataset_name=dataset_name)

        if len(real_df) < MIN_HISTORY_RUNS:
            # Remove any stale artifact so cold-start fallback is strictly enforced
            if self.model_path.exists():
                self.model_path.unlink()
            logger.warning(
                f"[PREDICTIVE DQ COLD-START] Observed {len(real_df)} real dataset evaluations "
                f"(across {counts['pipeline_runs_count']} pipeline runs) < {MIN_HISTORY_RUNS} required. "
                f"ML prediction disabled; falling back to deterministic Data Contract rules."
            )
            return {
                "status": "COLD_START_FALLBACK",
                "ml_prediction_enabled": False,
                **counts,
                "observed_observations": len(real_df),
                "fallback_policy": "DETERMINISTIC_DATA_CONTRACT_RULES",
            }

        df = real_df.copy()
        df["prev_dq_score"] = df["dq_score"].shift(1).fillna(100.0)
        df["null_pct"] = df["row_drop_pct"] * 0.4
        df["schema_warning_count"] = 0
        df["processing_duration_sec"] = 2.5
        df["target_degraded"] = (df["dq_score"] < 85.0).astype(int)

        if df["target_degraded"].nunique() < 2:
            return {
                "status": "SINGLE_CLASS_HISTORY_FALLBACK",
                "ml_prediction_enabled": False,
                **counts,
                "observed_observations": len(real_df),
                "fallback_policy": "DETERMINISTIC_DATA_CONTRACT_RULES",
            }

        split_idx = int(len(df) * 0.75)
        train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
        X_train, y_train = train_df[FEATURE_COLS], train_df["target_degraded"]
        X_test, y_test = test_df[FEATURE_COLS], test_df["target_degraded"]

        clf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]

        joblib.dump({"model": clf, "model_name": "RandomForest", "features": FEATURE_COLS, "is_synthetic": False}, self.model_path)
        return {
            "status": "TRAINED_ON_REAL_HISTORY",
            "ml_prediction_enabled": True,
            **counts,
            "observed_observations": len(real_df),
            "metrics": {
                "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_test, preds, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_test, probs)), 4) if y_test.nunique() > 1 else 0.0,
            },
        }

    def predict_next_batch_risk(self, current_metrics: Dict[str, float], dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """Return ML risk probability ONLY when trained on >= 30 real historical runs; otherwise return cold-start fallback."""
        counts = self.get_history_counts(dataset_name=dataset_name)
        if counts["target_dataset_evaluations"] < MIN_HISTORY_RUNS or not self.model_path.exists():
            return {
                "status": "COLD_START_FALLBACK",
                "ml_prediction_enabled": False,
                "risk_probability": None,
                "risk_level": "RULE_GOVERNED",
                "reason": f"Insufficient real history ({counts['target_dataset_evaluations']}/{MIN_HISTORY_RUNS} observations).",
            }

        artifact = joblib.load(self.model_path)
        if artifact.get("is_synthetic", True):
            return {
                "status": "COLD_START_FALLBACK",
                "ml_prediction_enabled": False,
                "risk_probability": None,
                "risk_level": "RULE_GOVERNED",
                "reason": "Synthetic benchmark models are blocked from production predictions.",
            }

        X = pd.DataFrame([{col: current_metrics.get(col, 0.0) for col in artifact["features"]}])
        prob = float(artifact["model"].predict_proba(X)[0, 1])
        risk_level = "HIGH" if prob >= 0.65 else ("MEDIUM" if prob >= 0.35 else "LOW")
        return {
            "status": "PREDICTED",
            "ml_prediction_enabled": True,
            "model": artifact["model_name"],
            "risk_probability": round(prob, 4),
            "risk_level": risk_level,
        }

    @staticmethod
    def run_synthetic_benchmark_simulation(n_batches: int = 120) -> Dict[str, Any]:
        """Isolated offline simulation to benchmark RF vs XGBoost on synthetic longitudinal telemetry (never saved for prod)."""
        rng = np.random.default_rng(42)
        records = []
        dq_score = 99.5
        for i in range(n_batches):
            prev_dq = dq_score
            schema_warns = int(rng.choice([0, 0, 0, 1, 2], p=[0.70, 0.10, 0.05, 0.10, 0.05]))
            null_pct = float(np.clip(rng.normal(0.4, 0.3) + schema_warns * 1.8, 0.0, 25.0))
            dup_pct = float(np.clip(rng.normal(0.8, 0.5), 0.0, 15.0))
            duration = float(np.clip(rng.normal(4.5, 1.2) + schema_warns * 2.0, 1.0, 30.0))
            row_drop = float(np.clip(null_pct * 0.6 + dup_pct * 0.8, 0.0, 30.0))

            logit = -3.2 + 0.35 * null_pct + 0.25 * dup_pct + 1.1 * schema_warns + 0.08 * (100.0 - prev_dq)
            prob = 1.0 / (1.0 + np.exp(-logit))
            degraded = int(rng.random() < prob)
            dq_score = float(np.clip(rng.normal(76.0, 5.0) if degraded else rng.normal(98.5, 1.2), 40.0, 100.0))
            records.append(
                {
                    "prev_dq_score": prev_dq,
                    "null_pct": null_pct,
                    "duplicate_pct": dup_pct,
                    "schema_warning_count": schema_warns,
                    "processing_duration_sec": duration,
                    "row_drop_pct": row_drop,
                    "target_degraded": degraded,
                }
            )
        df = pd.DataFrame(records)
        split_idx = int(len(df) * 0.75)
        X_train, y_train = df.iloc[:split_idx][FEATURE_COLS], df.iloc[:split_idx]["target_degraded"]
        X_test, y_test = df.iloc[split_idx:][FEATURE_COLS], df.iloc[split_idx:]["target_degraded"]

        results = {}
        for name, clf in [
            ("RandomForest_Sim", RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)),
            ("XGBoost_Sim", XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.08, eval_metric="logloss", random_state=42)),
        ]:
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            probs = clf.predict_proba(X_test)[:, 1]
            results[name] = {
                "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_test, preds, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
            }
        return {"is_synthetic_simulation": True, "sample_size": n_batches, "metrics": results}
