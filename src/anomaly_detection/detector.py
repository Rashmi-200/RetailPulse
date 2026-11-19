from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.observability.tracker import ObservabilityTracker, utc_now
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridAnomalyDetector:
    """Detects statistical (Z-Score, NIST MAD, Tukey IQR) and multivariate ML (Isolation Forest, LOF) anomalies."""

    def __init__(self, engine: Engine, tracker: ObservabilityTracker):
        self.engine = engine
        self.tracker = tracker

    @staticmethod
    def detect_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Standard Three-Sigma Z-Score outlier detection (|z| > 3.0)."""
        mean = series.mean()
        std = series.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series(False, index=series.index)
        z_scores = (series - mean).abs() / std
        return z_scores > threshold

    @staticmethod
    def detect_modified_zscore_mad(series: pd.Series, threshold: float = 3.5) -> pd.Series:
        """NIST Standard Modified Z-Score using Median Absolute Deviation (|M_i| > 3.5)."""
        median = series.median()
        mad = (series - median).abs().median()
        if mad == 0 or np.isnan(mad):
            return pd.Series(False, index=series.index)
        modified_z = 0.6745 * (series - median).abs() / mad
        return modified_z > threshold

    @staticmethod
    def detect_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
        """Tukey's Fences Interquartile Range outlier detection (Q1 - 1.5*IQR, Q3 + 1.5*IQR)."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return pd.Series(False, index=series.index)
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        return (series < lower) | (series > upper)

    @staticmethod
    def detect_isolation_forest(features_df: pd.DataFrame, contamination: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_df.fillna(0))
        model = IsolationForest(n_estimators=150, contamination=contamination, random_state=42)
        preds = model.fit_predict(X_scaled)
        scores = -model.decision_function(X_scaled)
        return preds == -1, scores

    @staticmethod
    def detect_lof(features_df: pd.DataFrame, contamination: float = 0.03, n_neighbors: int = 20) -> tuple[np.ndarray, np.ndarray]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_df.fillna(0))
        k = min(n_neighbors, max(2, len(features_df) - 1))
        model = LocalOutlierFactor(n_neighbors=k, contamination=contamination)
        preds = model.fit_predict(X_scaled)
        scores = -model.negative_outlier_factor_
        return preds == -1, scores

    def scan_daily_order_metrics(
        self,
        orders_df: pd.DataFrame,
        items_df: pd.DataFrame,
        batch_id: str,
        run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        orders = orders_df.copy()
        orders["order_date"] = pd.to_datetime(orders["order_purchase_timestamp"]).dt.date

        order_rev = items_df.groupby("order_id", as_index=False)["total_item_value"].sum()
        merged = orders.merge(order_rev, on="order_id", how="left")
        merged["total_item_value"] = merged["total_item_value"].fillna(0.0)
        merged["is_canceled"] = (merged["order_status"] == "canceled").astype(int)

        daily = (
            merged.groupby("order_date", as_index=False)
            .agg(
                daily_orders=("order_id", "count"),
                daily_revenue=("total_item_value", "sum"),
                avg_order_value=("total_item_value", "mean"),
                cancel_rate=("is_canceled", "mean"),
            )
            .sort_values("order_date")
            .reset_index(drop=True)
        )

        daily = daily[daily["daily_orders"] >= 5].reset_index(drop=True)

        feature_cols = ["daily_orders", "daily_revenue", "avg_order_value", "cancel_rate"]
        z_mask = self.detect_zscore(daily["daily_orders"], threshold=3.0)
        mad_mask = self.detect_modified_zscore_mad(daily["daily_orders"], threshold=3.5)
        iqr_mask = self.detect_iqr(daily["daily_orders"], multiplier=1.5)
        iso_mask, iso_scores = self.detect_isolation_forest(daily[feature_cols], contamination=0.025)
        lof_mask, _ = self.detect_lof(daily[feature_cols], contamination=0.025)

        daily["is_zscore"] = z_mask
        daily["is_mad"] = mad_mask
        daily["is_iqr"] = iqr_mask
        daily["is_iso"] = iso_mask
        daily["iso_score"] = iso_scores
        daily["is_lof"] = lof_mask

        alerts: List[Dict[str, Any]] = []
        median_orders = float(daily["daily_orders"].median())
        q1_orders, q3_orders = float(daily["daily_orders"].quantile(0.25)), float(daily["daily_orders"].quantile(0.75))

        for _, row in daily[daily["is_iso"] | daily["is_zscore"] | daily["is_mad"]].iterrows():
            votes = int(row["is_zscore"]) + int(row["is_mad"]) + int(row["is_iqr"]) + int(row["is_iso"]) + int(row["is_lof"])
            severity = "HIGH" if votes >= 4 else ("MEDIUM" if votes >= 2 else "LOW")
            detector_used = "IsolationForest+Statistical" if (row["is_iso"] and (row["is_zscore"] or row["is_mad"] or row["is_iqr"])) else (
                "IsolationForest" if row["is_iso"] else "Statistical(Z/MAD)"
            )
            explanation = (
                f"Date {row['order_date']}: {int(row['daily_orders'])} orders "
                f"(median={median_orders:.0f}, revenue={row['daily_revenue']:.2f}, "
                f"cancel_rate={row['cancel_rate']:.2%}). Consensus votes={votes}/5."
            )
            alerts.append(
                {
                    "batch_id": batch_id,
                    "run_id": run_id,
                    "dataset_name": "orders",
                    "metric_name": "daily_order_volume_and_revenue",
                    "detector_type": detector_used,
                    "observed_value": float(row["daily_orders"]),
                    "expected_range": f"Tukey IQR [{q1_orders:.0f} - {q3_orders:.0f}]",
                    "anomaly_score": round(float(row["iso_score"]), 4),
                    "severity": severity,
                    "explanation": explanation,
                    "detected_at": utc_now(),
                }
            )

        if alerts:
            self._persist_alerts(alerts)
            high_alerts = [a for a in alerts if a["severity"] == "HIGH"]
            if high_alerts:
                self.tracker.create_incident(
                    incident_type="STATISTICAL_ML_ANOMALY",
                    severity="HIGH",
                    dataset_name="orders",
                    layer="silver",
                    batch_id=batch_id,
                    run_id=run_id,
                    evidence_refs={"high_severity_dates": [a["explanation"] for a in high_alerts[:5]]},
                    recommended_action="Verify whether order/revenue spike or drop reflects a genuine retail event (e.g., Black Friday) or upstream feed lag.",
                )

        logger.info(
            f"[ANOMALY SCAN] orders (daily): {len(alerts)} anomalies flagged "
            f"(High={sum(1 for a in alerts if a['severity']=='HIGH')}, "
            f"Medium={sum(1 for a in alerts if a['severity']=='MEDIUM')}, "
            f"Low={sum(1 for a in alerts if a['severity']=='LOW')})"
        )
        return alerts

    def _persist_alerts(self, alerts: List[Dict[str, Any]]) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO meta.anomaly_alerts (
                        batch_id, run_id, dataset_name, metric_name, detector_type,
                        observed_value, expected_range, anomaly_score, severity,
                        explanation, detected_at
                    ) VALUES (
                        :batch_id, :run_id, :dataset_name, :metric_name, :detector_type,
                        :observed_value, :expected_range, :anomaly_score, :severity,
                        :explanation, :detected_at
                    )
                    """
                ),
                alerts,
            )
