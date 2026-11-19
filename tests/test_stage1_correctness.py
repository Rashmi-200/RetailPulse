import numpy as np
import pandas as pd
from src.analytics.demand_forecasting import RetailDemandForecaster
from src.analytics.predictive_dq import MIN_HISTORY_RUNS, PredictiveDQEngine
from src.anomaly_detection.evaluator import evaluate_detectors_on_synthetic_scenarios
from src.database.db_connection import get_engine


def test_predictive_dq_enforces_cold_start_guardrail():
    engine = get_engine()
    pdq = PredictiveDQEngine(engine)
    counts = pdq.get_history_counts(dataset_name="order_reviews")
    assert counts["pipeline_runs_count"] >= 10
    assert counts["target_dataset_evaluations"] < MIN_HISTORY_RUNS

    train_out = pdq.train_on_real_history(dataset_name="order_reviews")
    assert train_out["status"] == "COLD_START_FALLBACK"
    assert train_out["ml_prediction_enabled"] is False

    pred_out = pdq.predict_next_batch_risk({"prev_dq_score": 99.73}, dataset_name="order_reviews")
    assert pred_out["status"] == "COLD_START_FALLBACK"
    assert pred_out["risk_probability"] is None


def test_anomaly_detectors_fair_separated_benchmark():
    bench_df = evaluate_detectors_on_synthetic_scenarios(pd.DataFrame(), pd.DataFrame())
    uni_rows = bench_df[bench_df["track"] == "Univariate_KPI"]
    multi_rows = bench_df[bench_df["track"] == "Multivariate_ML"]

    # Univariate statistical detectors must achieve F1 > 0.70 on univariate spikes/drops/shifts
    assert (uni_rows["f1_score"] > 0.70).all()
    # Clean baseline false positive rate must remain low (< 5%)
    assert (uni_rows["clean_baseline_fpr"] < 0.05).all()
    # Multivariate ML detectors must achieve F1 > 0.70 on coupled feature anomalies
    assert (multi_rows["f1_score"] > 0.70).all()


def test_forecasting_zero_target_leakage_and_distribution_bounds():
    engine = get_engine()
    forecaster = RetailDemandForecaster(engine)
    raw = forecaster.load_gold_daily_series()
    feat = forecaster.engineer_features(raw, target_col="daily_orders")

    # Verify lag_1 at row t equals daily_orders at row t-1 (no lookahead)
    assert np.allclose(feat["lag_1"].iloc[1:].values, feat["daily_orders"].iloc[:-1].values)
    # Verify rolling_mean_7 at row 7 equals exact mean of daily_orders[0..6] (excludes row 7 target)
    expected_rm7 = feat["daily_orders"].iloc[0:7].mean()
    assert np.isclose(feat["rolling_mean_7"].iloc[7], expected_rm7)

    res = forecaster.train_and_benchmark(target_col="daily_orders")
    assert res["distribution_validation"]["within_active_min_max"] is True
    assert res["distribution_validation"]["mean_within_p10_p90"] is True
