import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from src.anomaly_detection.detector import HybridAnomalyDetector


def evaluate_detectors_on_synthetic_scenarios(orders_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    """Inject labelled synthetic retail anomalies to benchmark Z-Score, NIST MAD, Tukey IQR, IsoForest, and LOF."""
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
    daily = daily[(daily["daily_orders"] >= 40) & (daily["daily_orders"] <= 350)].reset_index(drop=True)
    y_true = np.zeros(len(daily), dtype=int)

    rng = np.random.default_rng(42)
    anomaly_indices = rng.choice(len(daily), size=12, replace=False)

    for i, idx in enumerate(anomaly_indices):
        y_true[idx] = 1
        if i % 3 == 0:
            daily.loc[idx, "daily_orders"] = int(daily.loc[idx, "daily_orders"] * 0.2)
            daily.loc[idx, "daily_revenue"] = daily.loc[idx, "daily_revenue"] * 0.2
        elif i % 3 == 1:
            daily.loc[idx, "avg_order_value"] = daily.loc[idx, "avg_order_value"] * 4.5
            daily.loc[idx, "daily_revenue"] = daily.loc[idx, "daily_revenue"] * 4.5
        else:
            daily.loc[idx, "cancel_rate"] = 0.42
            daily.loc[idx, "daily_revenue"] = daily.loc[idx, "daily_revenue"] * 0.45

    feature_cols = ["daily_orders", "daily_revenue", "avg_order_value", "cancel_rate"]
    contamination = round(len(anomaly_indices) / len(daily), 4)

    z_pred = HybridAnomalyDetector.detect_zscore(daily["daily_orders"], threshold=3.0).astype(int)
    mad_pred = HybridAnomalyDetector.detect_modified_zscore_mad(daily["daily_orders"], threshold=3.5).astype(int)
    iqr_pred = HybridAnomalyDetector.detect_iqr(daily["daily_orders"], multiplier=1.5).astype(int)
    iso_pred, _ = HybridAnomalyDetector.detect_isolation_forest(daily[feature_cols], contamination=contamination)
    lof_pred, _ = HybridAnomalyDetector.detect_lof(daily[feature_cols], contamination=contamination)

    results = []
    for name, preds in [
        ("Standard Z-Score (|z| > 3.0)", z_pred),
        ("NIST Modified Z-Score (MAD > 3.5)", mad_pred),
        ("Tukey IQR Fences (1.5x IQR)", iqr_pred),
        ("Isolation Forest (Multivariate ML)", iso_pred.astype(int)),
        ("Local Outlier Factor (Multivariate ML)", lof_pred.astype(int)),
    ]:
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        fpr = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0
        results.append(
            {
                "detector": name,
                "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_true, preds, zero_division=0)), 4),
                "false_positive_rate": fpr,
            }
        )

    return pd.DataFrame(results)
