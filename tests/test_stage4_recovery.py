import shutil
from pathlib import Path
import pandas as pd
import pytest
from deltalake import DeltaTable, write_deltalake
from sqlalchemy import text

from src.database.db_connection import get_engine
from src.observability.tracker import ObservabilityTracker
from src.rca.evidence_collector import RCAEvidenceCollector
from src.recovery.recovery_engine import SafeRecoveryEngine
from src.utils.config_loader import load_yaml_config


def test_execution_status_separated_from_dq_status():
    engine = get_engine()
    tracker = ObservabilityTracker(engine)
    with tracker.track_pipeline_run("pytest_status_separation_probe", trigger_type="pytest") as run_id:
        tracker.create_incident(
            incident_type="DATA_CONTRACT_VIOLATION",
            severity="HIGH",
            dataset_name="pytest_probe_ds",
            layer="silver",
            run_id=run_id,
        )
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT status, dq_status FROM meta.pipeline_runs WHERE run_id = :rid"),
            {"rid": run_id},
        ).mappings().fetchone()
        assert row["status"] == "SUCCESS"
        assert row["dq_status"] == "FAILED"
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM meta.incidents WHERE run_id = :rid"), {"rid": run_id})
        conn.execute(text("DELETE FROM meta.pipeline_runs WHERE run_id = :rid"), {"rid": run_id})


def test_rca_telemetry_source_count_consistency():
    engine = get_engine()
    collector = RCAEvidenceCollector(engine)
    with engine.connect() as conn:
        inc_id = conn.execute(text("SELECT incident_id FROM meta.incidents LIMIT 1")).scalar()
    bundle = collector.collect_for_incident(inc_id)
    assert bundle["telemetry_source_count"] == len(bundle["telemetry_sources"]) == 8


def test_extended_idempotent_recovery_seven_scenarios():
    engine = get_engine()
    tracker = ObservabilityTracker(engine)
    rec_cfg = load_yaml_config("config/recovery_policies.yaml")
    contracts_cfg = load_yaml_config("config/data_contracts.yaml")

    probe_ds = "pytest_recovery_orders"
    contracts_cfg["contracts"][probe_ds] = contracts_cfg["contracts"]["orders"]
    recovery = SafeRecoveryEngine(engine, tracker, rec_cfg, contracts_cfg)

    probe_dir = Path(f"data/silver/{probe_ds}")
    if probe_dir.exists():
        shutil.rmtree(probe_dir)
    delta_dir = probe_dir / "_delta"
    delta_dir.mkdir(parents=True, exist_ok=True)

    try:
        base_rows = pd.DataFrame({
            "order_id": [f"ord_{i:04d}" for i in range(50)],
            "customer_id": [f"cust_{i:04d}" for i in range(50)],
            "order_status": ["delivered"] * 50,
            "order_purchase_timestamp": pd.to_datetime(["2018-05-01 10:00:00"] * 50),
        })
        write_deltalake(str(delta_dir), base_rows, mode="overwrite")

        batch_100 = pd.DataFrame({
            "order_id": [f"ord_{i:04d}" for i in range(100)],
            "customer_id": [f"cust_{i:04d}" for i in range(100)],
            "order_status": ["delivered"] * 100,
            "order_purchase_timestamp": pd.to_datetime(["2018-05-02 12:00:00"] * 100),
        })

        # 1. Missing human approval blocked
        with pytest.raises(PermissionError):
            recovery.replay_corrected_records(probe_ds, "pytest_b_unapproved", batch_100, approved_by="")
        assert len(DeltaTable(str(delta_dir)).to_pandas()) == 50

        # 2. 100-record replay
        r1 = recovery.replay_corrected_records(probe_ds, "pytest_b_100", batch_100, approved_by="lead_de@retailpulse.ai")
        assert len(DeltaTable(str(delta_dir)).to_pandas()) == 100

        # 3. Repeated batch replay (idempotent)
        r2 = recovery.replay_corrected_records(probe_ds, "pytest_b_100", batch_100, approved_by="lead_de@retailpulse.ai")
        assert len(DeltaTable(str(delta_dir)).to_pandas()) == 100

        # 4. Duplicate primary keys inside input batch -> Atomically rejected before Silver write
        dup_batch = pd.concat([batch_100.iloc[:20], batch_100.iloc[:20]], ignore_index=True)
        with pytest.raises(ValueError, match="Recovery aborted"):
            recovery.replay_corrected_records(probe_ds, "pytest_b_dup", dup_batch, approved_by="lead_de@retailpulse.ai")
        assert len(DeltaTable(str(delta_dir)).to_pandas()) == 100

        # 5. Partial failure (1 invalid null PK mid-batch) -> Atomically rejected without corrupting target table
        bad_batch = batch_100.iloc[:10].copy()
        bad_batch.loc[5, "order_id"] = None
        with pytest.raises(ValueError, match="Recovery aborted"):
            recovery.replay_corrected_records(probe_ds, "pytest_b_partial", bad_batch, approved_by="lead_de@retailpulse.ai")
        assert len(DeltaTable(str(delta_dir)).to_pandas()) == 100

        # 6. Simulated application restart mid-recovery
        restarted = SafeRecoveryEngine(get_engine(), ObservabilityTracker(get_engine()), rec_cfg, contracts_cfg)
        r3 = restarted.replay_corrected_records(probe_ds, "pytest_b_restart", batch_100, approved_by="lead_de@retailpulse.ai")
        assert len(DeltaTable(str(delta_dir)).to_pandas()) == 100

        # 7. Audit history verification (3 successful + 2 FAILED_VALIDATION audit logs = 5 records)
        with engine.connect() as conn:
            cnt = conn.execute(text("SELECT COUNT(*) FROM meta.recovery_history WHERE dataset_name = :ds"), {"ds": probe_ds}).scalar()
            assert int(cnt) == 5
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM meta.recovery_history WHERE dataset_name = :ds"), {"ds": probe_ds})
            conn.execute(text("DELETE FROM meta.quality_results WHERE dataset_name = :ds"), {"ds": probe_ds})
            conn.execute(text("DELETE FROM meta.incidents WHERE dataset_name = :ds"), {"ds": probe_ds})
            conn.execute(text("DELETE FROM quarantine.quarantined_records WHERE dataset_name = :ds"), {"ds": probe_ds})
        shutil.rmtree(probe_dir, ignore_errors=True)
