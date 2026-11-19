import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from src.anomaly_detection.detector import HybridAnomalyDetector


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = round(float(fp / (fp + tn)), 4) if (fp + tn) > 0 else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "false_positive_rate": fpr,
    }


def evaluate_detectors_on_synthetic_scenarios(orders_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fairly benchmark anomaly detectors across separated tracks:
      Track A (Univariate KPI): Sudden spikes, sudden drops, and step shifts on individual KPI series.
      Track B (Multivariate Coupling): Simultaneous shifts across orders, AOV, and cancel_rate.
      Track C (Clean Normal Baseline): Zero injected anomalies to measure true false-positive rate.
    """
    rng = np.random.default_rng(42)
    n_days = 200

    # --- Track C: Clean Normal Baseline (Stationary baseline with zero injected anomalies) ---
    clean_orders = pd.Series(rng.normal(loc=180.0, scale=15.0, size=n_days))
    clean_df = pd.DataFrame(
        {
            "daily_orders": clean_orders,
            "daily_revenue": clean_orders * rng.normal(140.0, 8.0, size=n_days),
            "avg_order_value": rng.normal(140.0, 8.0, size=n_days),
            "cancel_rate": np.clip(rng.normal(0.015, 0.004, size=n_days), 0.001, 0.05),
        }
    )

    # --- Track A: Univariate KPI Benchmark (6 sudden spikes, 4 sudden drops, 4 step shifts = 14 anomalies) ---
    uni_series = clean_orders.copy()
    y_uni = np.zeros(n_days, dtype=int)
    # 6 sudden spikes (+5.5 std)
    for idx in [15, 42, 78, 110, 145, 182]:
        uni_series.iloc[idx] = 180.0 + 5.5 * 15.0
        y_uni[idx] = 1
    # 4 sudden drops (-5.0 std)
    for idx in [28, 64, 125, 168]:
        uni_series.iloc[idx] = 180.0 - 5.0 * 15.0
        y_uni[idx] = 1
    # 4-day sustained step distribution shift (+4.2 std)
    for idx in [90, 91, 92, 93]:
        uni_series.iloc[idx] = 180.0 + 4.2 * 15.0
        y_uni[idx] = 1

    # --- Track B: Multivariate Coupling Benchmark (12 coupled feature anomalies) ---
    multi_df = clean_df.copy()
    y_multi = np.zeros(n_days, dtype=int)
    multi_indices = [12, 27, 45, 61, 79, 95, 112, 130, 148, 163, 177, 191]
    for i, idx in enumerate(multi_indices):
        y_multi[idx] = 1
        if i % 2 == 0:
            # Subtle order drop within normal univariate range + severe cancel_rate & AOV divergence
            multi_df.loc[idx, "daily_orders"] = 148.0
            multi_df.loc[idx, "avg_order_value"] = 245.0
            multi_df.loc[idx, "cancel_rate"] = 0.28
            multi_df.loc[idx, "daily_revenue"] = 148.0 * 245.0
        else:
            # High order volume paired with collapsed AOV and elevated cancellations
            multi_df.loc[idx, "daily_orders"] = 212.0
            multi_df.loc[idx, "avg_order_value"] = 48.0
            multi_df.loc[idx, "cancel_rate"] = 0.22
            multi_df.loc[idx, "daily_revenue"] = 212.0 * 48.0

    rows = []
    # Evaluate Univariate Statistical Detectors on Track A & Clean Baseline FPR
    for name, fn in [
        ("Standard Z-Score (|z| > 3.0)", lambda s: HybridAnomalyDetector.detect_zscore(s, threshold=3.0).astype(int)),
        ("NIST Modified Z-Score (MAD > 3.5)", lambda s: HybridAnomalyDetector.detect_modified_zscore_mad(s, threshold=3.5).astype(int)),
        ("Tukey IQR Fences (1.5x IQR)", lambda s: HybridAnomalyDetector.detect_iqr(s, multiplier=1.5).astype(int)),
    ]:
        preds_uni = fn(uni_series).values
        clean_preds = fn(clean_orders).values
        clean_fpr = round(float(clean_preds.sum() / len(clean_preds)), 4)
        m = _compute_metrics(y_uni, preds_uni)
        rows.append({"track": "Univariate_KPI", "detector": name, **m, "clean_baseline_fpr": clean_fpr})

    # Evaluate Multivariate ML Detectors on Track B & Clean Baseline FPR
    feature_cols = ["daily_orders", "daily_revenue", "avg_order_value", "cancel_rate"]
    contamination = round(len(multi_indices) / n_days, 4)
    for name, fn in [
        ("Isolation Forest (Multivariate)", lambda df: HybridAnomalyDetector.detect_isolation_forest(df[feature_cols], contamination=contamination)[0].astype(int)),
        ("Local Outlier Factor (Multivariate)", lambda df: HybridAnomalyDetector.detect_lof(df[feature_cols], contamination=contamination)[0].astype(int)),
    ]:
        preds_multi = fn(multi_df)
        clean_preds = fn(clean_df)
        clean_fpr = round(float(clean_preds.sum() / len(clean_preds)), 4)
        m = _compute_metrics(y_multi, preds_multi)
        rows.append({"track": "Multivariate_ML", "detector": name, **m, "clean_baseline_fpr": clean_fpr})

    return pd.DataFrame(rows)
