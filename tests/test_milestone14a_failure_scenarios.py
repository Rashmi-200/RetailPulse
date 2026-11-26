import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from deltalake import DeltaTable, write_deltalake
from sqlalchemy import text

from src.analytics.demand_forecasting import FEATURE_COLS, RetailDemandForecaster
from src.anomaly_detection.detector import HybridAnomalyDetector
from src.anomaly_detection.evaluator import evaluate_detectors_on_synthetic_scenarios
from src.contracts.contract_engine import DataContractEngine
from src.copilot.rag_copilot import HybridDataEngineeringCopilot
from src.database.db_connection import get_engine
from src.observability.tracker import ObservabilityTracker
from src.rca.rca_analyzer import RootCauseAnalyzer
from src.recovery.recovery_engine import SafeRecoveryEngine
from src.schema_evolution.registry import SchemaEvolutionRegistry
from src.utils.config_loader import load_yaml_config


def test_m14a_01_to_04_schema_and_ingestion_failures():
    engine = get_engine()
    tracker = ObservabilityTracker(engine)
    contracts_cfg = load_yaml_config("config/data_contracts.yaml")
    contract_engine = DataContractEngine(engine, contracts_cfg)
    schema_reg = SchemaEvolutionRegistry(engine, tracker)

    ds = "pytest_m14a_schema_orders"
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM meta.schema_versions WHERE dataset_name = :ds"), {"ds": ds})
        conn.execute(text("DELETE FROM meta.incidents WHERE dataset_name = :ds"), {"ds": ds})
        conn.execute(text("DELETE FROM meta.quality_results WHERE dataset_name = :ds"), {"ds": ds})
        conn.execute(text("DELETE FROM quarantine.quarantined_records WHERE dataset_name = :ds"), {"ds": ds})

    try:
        # Scenario 1: Bronze ingestion missing required column ('customer_id')
        bad_ingest_df = pd.DataFrame({
            "order_id": ["ord_1", "ord_2"],
            "order_status": ["delivered", "delivered"],
            "order_purchase_timestamp": pd.to_datetime(["2018-06-01", "2018-06-02"]),
        })
        with pytest.raises(ValueError, match="missing required columns"):
            contract_engine.validate_and_split(bad_ingest_df, "orders", "m14a_b1", layer="bronze", persist=False)

        # Baseline schema v1
        base_df = pd.DataFrame({
            "order_id": pd.Series(["ord_1"], dtype="object"),
            "customer_id": pd.Series(["cust_1"], dtype="object"),
            "freight_value": pd.Series([15.5], dtype="float64"),
        })
        req_cols = ["order_id", "customer_id"]
        v1 = schema_reg.check_and_register(base_df, ds, "bronze", required_columns=req_cols)
        assert v1.version_number == 1

        # Scenario 4: Column reordering + nullable column addition -> non-breaking, zero false-positive drift
        reordered_df = base_df[["freight_value", "order_id", "customer_id"]]
        v_reordered = schema_reg.check_and_register(reordered_df, ds, "bronze", required_columns=req_cols)
        assert v_reordered.version_number == 1 and v_reordered.is_breaking is False

        additive_df = base_df.copy()
        additive_df["promo_code"] = pd.Series([None], dtype="object")
        v2 = schema_reg.check_and_register(additive_df, ds, "bronze", required_columns=req_cols)
        assert v2.version_number == 2 and v2.is_breaking is False

        # Scenario 2: Breaking column removal -> blocked with ValueError + CRITICAL incident
        removed_df = base_df.drop(columns=["customer_id"])
        with pytest.raises(ValueError, match="Breaking schema drift"):
            schema_reg.check_and_register(removed_df, ds, "bronze", required_columns=req_cols)

        # Scenario 3: Breaking data type mutation (float64 -> object) -> blocked against latest accepted v2
        mutated_df = additive_df.copy()
        mutated_df["freight_value"] = mutated_df["freight_value"].astype("object")
        with pytest.raises(ValueError, match="Breaking schema drift"):
            schema_reg.check_and_register(mutated_df, ds, "bronze", required_columns=req_cols)
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM meta.schema_versions WHERE dataset_name = :ds"), {"ds": ds})
            conn.execute(text("DELETE FROM meta.incidents WHERE dataset_name = :ds"), {"ds": ds})


def test_m14a_05_and_06_contract_duplicates_nulls_and_dq_formula():
    engine = get_engine()
    tracker = ObservabilityTracker(engine)
    ds = "pytest_m14a_orders"
    contracts_cfg = load_yaml_config("config/data_contracts.yaml")
    contract_engine = DataContractEngine(engine, contracts_cfg)
    contract_engine.contracts[ds] = contract_engine.contracts["orders"]

    try:
        # Scenario 5: Duplicate primary keys injected (3 unique + 2 duplicates = 5 total)
        dup_df = pd.DataFrame({
            "order_id": ["o1", "o2", "o3", "o1", "o2"],
            "customer_id": ["c1", "c2", "c3", "c1", "c2"],
            "order_status": ["delivered"] * 5,
            "order_purchase_timestamp": pd.to_datetime(["2018-07-01"] * 5),
        })
        rep_dup = contract_engine.validate_and_split(dup_df, ds, "m14a_dup_batch", layer="silver", persist=True)
        assert rep_dup.valid_records == 3
        assert rep_dup.invalid_records == 2
        assert rep_dup.duplicate_records == 2

        # Scenario 6: Null PK + invalid domain status + verify 6-dimension unweighted mean formula
        bad_df = pd.DataFrame({
            "order_id": ["o10", None, "o12"],
            "customer_id": ["c10", "c11", "c12"],
            "order_status": ["delivered", "delivered", "INVALID_STATUS_XYZ"],
            "order_purchase_timestamp": pd.to_datetime(["2018-07-02"] * 3),
        })
        rep_bad = contract_engine.validate_and_split(bad_df, ds, "m14a_bad_batch", layer="silver", persist=True)
        assert rep_bad.valid_records == 1
        assert rep_bad.invalid_records == 2

        dims = rep_bad.dimension_scores
        expected_mean = round(sum(dims.values()) / 6.0, 2)
        assert round(rep_bad.overall_score, 2) == expected_mean
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM meta.quality_results WHERE dataset_name = :ds"), {"ds": ds})
            conn.execute(text("DELETE FROM meta.incidents WHERE dataset_name = :ds"), {"ds": ds})
            conn.execute(text("DELETE FROM quarantine.quarantined_records WHERE dataset_name = :ds"), {"ds": ds})


def test_m14a_07_and_08_anomaly_benchmarks_and_forecasting_leakage():
    engine = get_engine()

    # Scenario 7: Anomaly Detection Benchmark (Statistical + ML detectors across anomaly types)
    orders_silver = DeltaTable("data/silver/orders/_delta").to_pandas()
    items_silver = DeltaTable("data/silver/order_items/_delta").to_pandas()
    bench_df = evaluate_detectors_on_synthetic_scenarios(orders_silver, items_silver)
    assert len(bench_df) >= 3
    for col in ["precision", "recall", "f1_score", "false_positive_rate", "clean_baseline_fpr"]:
        assert col in bench_df.columns
    assert bench_df["f1_score"].max() >= 0.70

    # Also verify HybridAnomalyDetector statistical spike & drop detection
    s = pd.Series([100.0 + (i % 5) - 2.0 for i in range(30)] + [450.0, 5.0])
    mad_flags = HybridAnomalyDetector.detect_modified_zscore_mad(s)
    assert bool(mad_flags.iloc[-2]) is True and bool(mad_flags.iloc[-1]) is True

    # Scenario 8: Demand Forecasting Chronological Split, Feature Leakage Test & 7-Day Rolling-Origin Backtest
    forecaster = RetailDemandForecaster(engine)
    daily_df = forecaster.load_gold_daily_series()
    assert daily_df["order_date"].is_monotonic_increasing

    feat_df = RetailDemandForecaster.engineer_features(daily_df, target_col="daily_orders")
    for fcol in FEATURE_COLS:
        assert fcol in feat_df.columns

    # Strict causal leakage check: mutating future target values at index >= t must NOT change features at index < t
    mutated_daily = daily_df.copy()
    cutoff_idx = len(mutated_daily) - 10
    mutated_daily.loc[cutoff_idx:, "daily_orders"] = 999999.0
    feat_mutated = RetailDemandForecaster.engineer_features(mutated_daily, target_col="daily_orders")
    cutoff_date = mutated_daily.loc[cutoff_idx - 1, "order_date"]
    orig_past = feat_df[feat_df["order_date"] <= cutoff_date][FEATURE_COLS].reset_index(drop=True)
    mut_past = feat_mutated[feat_mutated["order_date"] <= cutoff_date][FEATURE_COLS].reset_index(drop=True)
    pd.testing.assert_frame_equal(orig_past, mut_past)

    # 7-day rolling-origin backtest
    model_results = forecaster.rolling_origin_backtest(feat_df, target_col="daily_orders", n_folds=3, horizon=7)
    assert len(model_results) == 4
    for mres in model_results:
        assert mres["folds"] == 3
        assert mres["MAE"] >= 0 and mres["RMSE"] >= 0


def test_m14a_09_to_11_recovery_atomicity_idempotency_and_audit():
    engine = get_engine()
    tracker = ObservabilityTracker(engine)
    rec_cfg = load_yaml_config("config/recovery_policies.yaml")
    contracts_cfg = load_yaml_config("config/data_contracts.yaml")

    probe_ds = "pytest_m14a_recovery"
    contracts_cfg["contracts"][probe_ds] = contracts_cfg["contracts"]["orders"]
    recovery = SafeRecoveryEngine(engine, tracker, rec_cfg, contracts_cfg)

    probe_dir = Path(f"data/silver/{probe_ds}")
    if probe_dir.exists():
        shutil.rmtree(probe_dir)
    delta_dir = probe_dir / "_delta"
    delta_dir.mkdir(parents=True, exist_ok=True)

    try:
        base_rows = pd.DataFrame({
            "order_id": [f"m14a_{i:03d}" for i in range(20)],
            "customer_id": [f"cust_{i:03d}" for i in range(20)],
            "order_status": ["delivered"] * 20,
            "order_purchase_timestamp": pd.to_datetime(["2018-08-01"] * 20),
        })
        write_deltalake(str(delta_dir), base_rows, mode="overwrite")

        replay_df = pd.DataFrame({
            "order_id": [f"m14a_{i:03d}" for i in range(40)],
            "customer_id": [f"cust_{i:03d}" for i in range(40)],
            "order_status": ["delivered"] * 40,
            "order_purchase_timestamp": pd.to_datetime(["2018-08-02"] * 40),
        })

        # Scenario 9: Unapproved recovery replay blocked
        with pytest.raises(PermissionError):
            recovery.replay_corrected_records(probe_ds, "b_unapproved", replay_df, approved_by="")
        assert len(DeltaTable(str(delta_dir)).to_pandas()) == 20

        # Scenario 10: Corrupted recovery batch atomically rejected before Silver write
        corrupt_df = replay_df.iloc[:5].copy()
        corrupt_df.loc[2, "order_id"] = None
        with pytest.raises(ValueError, match="Recovery aborted"):
            recovery.replay_corrected_records(probe_ds, "b_corrupt", corrupt_df, approved_by="sre@retailpulse.ai")
        assert len(DeltaTable(str(delta_dir)).to_pandas()) == 20

        # Scenario 11: Repeated + restarted idempotent replay converges to exact 40 rows with audit log
        recovery.replay_corrected_records(probe_ds, "b_valid", replay_df, approved_by="sre@retailpulse.ai")
        restarted = SafeRecoveryEngine(get_engine(), ObservabilityTracker(get_engine()), rec_cfg, contracts_cfg)
        restarted.replay_corrected_records(probe_ds, "b_valid", replay_df, approved_by="sre@retailpulse.ai")
        assert len(DeltaTable(str(delta_dir)).to_pandas()) == 40

        with engine.connect() as conn:
            audit_rows = conn.execute(
                text("SELECT status FROM meta.recovery_history WHERE dataset_name = :ds"),
                {"ds": probe_ds},
            ).scalars().all()
            assert audit_rows.count("SUCCESS") == 2
            assert audit_rows.count("FAILED_VALIDATION") == 1
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM meta.recovery_history WHERE dataset_name = :ds"), {"ds": probe_ds})
            conn.execute(text("DELETE FROM meta.quality_results WHERE dataset_name = :ds"), {"ds": probe_ds})
            conn.execute(text("DELETE FROM meta.incidents WHERE dataset_name = :ds"), {"ds": probe_ds})
            conn.execute(text("DELETE FROM quarantine.quarantined_records WHERE dataset_name = :ds"), {"ds": probe_ds})
        shutil.rmtree(probe_dir, ignore_errors=True)


def test_m14a_12_hardened_sql_copilot_and_rca_guardrail():
    engine = get_engine()
    copilot = HybridDataEngineeringCopilot(engine)

    # Scenario 12a: Hardened SQL Copilot blocks all write/DDL, stacked queries, comments, unsafe funcs & system schemas
    malicious_queries = [
        "DELETE FROM meta.incidents",
        "DROP TABLE meta.pipeline_runs",
        "INSERT INTO meta.incidents VALUES ('x')",
        "UPDATE meta.incidents SET status='RESOLVED'",
        "ALTER TABLE meta.incidents ADD COLUMN x INT",
        "TRUNCATE TABLE meta.incidents",
        "CREATE TABLE meta.evil (id INT)",
        "GRANT ALL ON meta.incidents TO PUBLIC",
        "SELECT * FROM meta.incidents; DROP TABLE meta.incidents",
        "SELECT * FROM meta.incidents -- comment injection",
        "SELECT pg_sleep(10)",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT * FROM pg_catalog.pg_tables",
        "SELECT * FROM information_schema.tables",
    ]
    for q in malicious_queries:
        with pytest.raises(PermissionError):
            copilot.validate_readonly_sql(q)

    # Valid SELECT executes cleanly inside READ ONLY transaction
    rows = copilot.execute_readonly_sql("SELECT COUNT(*) AS cnt FROM meta.pipeline_runs")
    assert len(rows) == 1 and int(rows[0]["cnt"]) >= 1

    # Database-level SELECT-only role verification on isolated probe table
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE IF NOT EXISTS meta.pytest_readonly_probe (id INT PRIMARY KEY, val TEXT)"))
        conn.execute(text("INSERT INTO meta.pytest_readonly_probe (id, val) VALUES (1, 'ok') ON CONFLICT DO NOTHING"))
        conn.execute(text("""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'retailpulse_readonly') THEN
                    CREATE ROLE retailpulse_readonly NOLOGIN;
                END IF;
            END $$;
        """))
        conn.execute(text("GRANT USAGE ON SCHEMA meta TO retailpulse_readonly"))
        conn.execute(text("GRANT SELECT ON meta.pytest_readonly_probe TO retailpulse_readonly"))

    try:
        with engine.connect() as conn:
            conn.execute(text("SET LOCAL ROLE retailpulse_readonly"))
            val = conn.execute(text("SELECT val FROM meta.pytest_readonly_probe WHERE id = 1")).scalar()
            assert val == "ok"
            with pytest.raises(Exception):
                conn.execute(text("INSERT INTO meta.pytest_readonly_probe VALUES (2, 'blocked')"))
            conn.rollback()
    finally:
        with engine.begin() as conn:
            conn.execute(text("REVOKE ALL ON meta.pytest_readonly_probe FROM retailpulse_readonly"))
            conn.execute(text("DROP TABLE IF EXISTS meta.pytest_readonly_probe"))

    # Scenario 12b: RCA insufficient-evidence guardrail on uncorroborated incident
    tracker = ObservabilityTracker(engine)
    uncorr_inc = tracker.create_incident(
        incident_type="UNCLASSIFIED_PROBE",
        severity="LOW",
        dataset_name="pytest_m14a_uncorroborated",
        layer="silver",
        run_id=None,
    )
    try:
        rca = RootCauseAnalyzer(engine)
        res = rca.analyze_incident(uncorr_inc)
        assert res["status"] == "INSUFFICIENT_EVIDENCE"
        assert res["evidence_completeness_score"] == 0.0
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM meta.incidents WHERE incident_id = :iid"), {"iid": uncorr_inc})
