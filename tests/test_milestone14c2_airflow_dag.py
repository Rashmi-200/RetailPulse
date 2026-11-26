import ast
from pathlib import Path
import pandas as pd
import pytest
from deltalake import write_deltalake
from sqlalchemy import text

from src.database.db_connection import get_engine
from src.orchestration.airflow_tasks import (
    NonRetriablePipelineError,
    RetailPulseOrchestrator,
    derive_batch_id,
    sanitize_run_id,
)
from src.utils.config_loader import load_yaml_config


def _cleanup_test_run(engine, dag_run_id: str, batch_id: str) -> None:
    rid = sanitize_run_id(dag_run_id)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM meta.anomaly_alerts WHERE batch_id = :bid"), {"bid": batch_id})
        conn.execute(text("DELETE FROM meta.incidents WHERE run_id = :rid OR batch_id = :bid"), {"rid": rid, "bid": batch_id})
        conn.execute(text("DELETE FROM meta.task_runs WHERE run_id = :rid"), {"rid": rid})
        conn.execute(text("DELETE FROM meta.batch_metadata WHERE batch_id = :bid"), {"bid": batch_id})
        conn.execute(text("DELETE FROM meta.pipeline_runs WHERE run_id = :rid"), {"rid": rid})
        conn.execute(text("DELETE FROM meta.schema_versions WHERE layer = 'silver' AND dataset_name IN ('orders', 'order_items')"))


def test_m14c2_01_and_02_dag_syntax_and_dependency_graph():
    """Verify DAG AST syntax, 8 required tasks, parallel branches after build_gold, and max_active_runs=1."""
    dag_path = Path("dags/retailpulse_medallion_dag.py")
    assert dag_path.exists(), "dags/retailpulse_medallion_dag.py missing"
    src = dag_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(dag_path))
    assert tree is not None

    expected_tasks = [
        "ingest_sources",
        "process_bronze",
        "transform_silver",
        "validate_data_quality",
        "build_gold",
        "detect_anomalies",
        "generate_forecasts",
        "finalize_pipeline",
    ]
    for tname in expected_tasks:
        assert f'task_id="{tname}"' in src, f"Missing task {tname}"

    assert "max_active_runs=1" in src, "Concurrent run protection (max_active_runs=1) must be configured"
    assert "ingest_sources >> process_bronze >> transform_silver >> validate_data_quality >> build_gold" in src
    assert "build_gold >> [detect_anomalies, generate_forecasts] >> finalize_pipeline" in src
    assert "TriggerRule.ALL_DONE" in src


def test_m14c2_03_stable_batch_id_across_retries():
    """Verify deterministic batch_id stays identical across retries of the same DAG run and differs across runs."""
    dag_id = "retailpulse_medallion_pipeline"
    run_a = "manual__2026-11-19T12:00:00+00:00"
    run_b = "manual__2026-11-19T13:00:00+00:00"

    b_try1 = derive_batch_id(dag_id, run_a)
    b_try2 = derive_batch_id(dag_id, run_a)
    b_next = derive_batch_id(dag_id, run_b)

    assert b_try1 == b_try2, "Batch ID must not change on retry of the same logical DAG run"
    assert b_try1 != b_next, "Distinct logical DAG runs must produce distinct batch IDs"


def test_m14c2_04_and_08_end_to_end_execution_and_idempotency(tmp_path):
    """Verify end-to-end 8-stage execution and idempotent repeated execution with zero duplicate alerts."""
    engine = get_engine()
    base_cfg = load_yaml_config("config/pipeline_config.yaml")
    cfg = {
        **base_cfg,
        "paths": {
            "raw_dir": str(tmp_path / "raw"),
            "bronze_dir": str(tmp_path / "bronze"),
            "silver_dir": str(tmp_path / "silver"),
            "gold_dir": str(tmp_path / "gold"),
            "quarantine_dir": str(tmp_path / "quarantine"),
        },
        "sources": {"csv_files": {}},
    }

    orders_delta = tmp_path / "silver" / "orders" / "_delta"
    items_delta = tmp_path / "silver" / "order_items" / "_delta"
    orders_delta.mkdir(parents=True, exist_ok=True)
    items_delta.mkdir(parents=True, exist_ok=True)
    ts = pd.date_range("2018-01-01", periods=60, freq="12h")
    write_deltalake(
        str(orders_delta),
        pd.DataFrame({
            "order_id": [f"tmp_ord_{i:04d}" for i in range(60)],
            "customer_id": [f"tmp_cust_{i:04d}" for i in range(60)],
            "order_status": ["delivered"] * 60,
            "order_purchase_timestamp": [f"2026-10-{(i % 6) + 1:02d} 10:00:00" for i in range(60)],
            "order_purchase_timestamp": ts,
        }),
        mode="overwrite",
    )
    write_deltalake(
        str(items_delta),
        pd.DataFrame({
            "order_id": [f"tmp_ord_{i:04d}" for i in range(60)],
            "order_item_id": [1] * 60,
            "seller_id": [f"seller_{i % 3}" for i in range(60)],
            "total_item_value": [125.0] * 60,
            "product_id": [f"tmp_prod_{i % 5}" for i in range(60)],
            "price": [110.0 + (i % 10) for i in range(60)],
            "freight_value": [15.0] * 60,
        }),
        mode="overwrite",
    )

    dag_id = "retailpulse_medallion_pipeline"
    dag_run_id = "pytest__m14c2_e2e_run_01"
    batch_id = derive_batch_id(dag_id, dag_run_id)
    rid = sanitize_run_id(dag_run_id)
    _cleanup_test_run(engine, dag_run_id, batch_id)

    try:
        orch = RetailPulseOrchestrator(engine=engine, pipeline_cfg=cfg)
        for _ in (1, 2):
            orch.task_ingest_sources(dag_id, dag_run_id, retry_count=0)
            orch.task_process_bronze(dag_id, dag_run_id, batch_id, retry_count=0)
            orch.task_transform_silver(dag_id, dag_run_id, batch_id, retry_count=0)
            orch.task_validate_data_quality(dag_id, dag_run_id, batch_id, retry_count=0)
            orch.task_build_gold(dag_id, dag_run_id, batch_id, retry_count=0)
            orch.task_detect_anomalies(dag_id, dag_run_id, batch_id, retry_count=0)
            orch.task_generate_forecasts(dag_id, dag_run_id, batch_id, retry_count=0)
            fin = orch.task_finalize_pipeline(dag_id, dag_run_id, batch_id, retry_count=0)
            assert fin["status"] == "SUCCESS" and fin["dq_status"] == "PASSED"

        with engine.connect() as conn:
            run_row = conn.execute(
                text("SELECT status, dq_status FROM meta.pipeline_runs WHERE run_id = :rid"),
                {"rid": rid},
            ).fetchone()
            assert run_row == ("SUCCESS", "PASSED")

            bm_count = conn.execute(
                text("SELECT COUNT(*) FROM meta.batch_metadata WHERE batch_id = :bid"),
                {"bid": batch_id},
            ).scalar()
            assert bm_count == 1, "Repeated DAG run must not duplicate batch_metadata"
    finally:
        _cleanup_test_run(engine, dag_run_id, batch_id)


def test_m14c2_05_06_07_and_10_retry_permanent_failure_and_dq_status_separation():
    """Verify transient retry -> success, permanent DQ/schema failure -> immediate fail, and separated status vs dq_status."""
    engine = get_engine()
    dag_id = "retailpulse_medallion_pipeline"

    run_transient = "pytest__m14c2_transient_01"
    batch_transient = derive_batch_id(dag_id, run_transient)
    rid_transient = sanitize_run_id(run_transient)
    _cleanup_test_run(engine, run_transient, batch_transient)

    run_dq_fail = "pytest__m14c2_dq_fail_01"
    batch_dq_fail = derive_batch_id(dag_id, run_dq_fail)
    rid_dq_fail = sanitize_run_id(run_dq_fail)
    _cleanup_test_run(engine, run_dq_fail, batch_dq_fail)

    try:
        def flaky_ingest(run_id, batch_id, retry_count):
            if retry_count == 0:
                raise ConnectionError("Simulated transient source timeout")
            return {"records_in": 10, "records_out": 10}

        orch_a = RetailPulseOrchestrator(engine=engine, stage_hooks={"ingest_sources": flaky_ingest})
        with pytest.raises(ConnectionError, match="transient source timeout"):
            orch_a.task_ingest_sources(dag_id, run_transient, retry_count=0)

        res_retry = orch_a.task_ingest_sources(dag_id, run_transient, retry_count=1)
        assert res_retry["batch_id"] == batch_transient

        with engine.connect() as conn:
            statuses = conn.execute(
                text("SELECT status, retry_count FROM meta.task_runs WHERE run_id = :rid AND task_name = 'ingest_sources' ORDER BY started_at"),
                {"rid": rid_transient},
            ).fetchall()
            assert statuses == [("FAILED", 0), ("SUCCESS", 1)]

        def fail_dq(run_id, batch_id, retry_count):
            return {"dq_status": "FAILED", "reason": "Breaking schema drift & duplicate PK rate 12.4%"}

        orch_b = RetailPulseOrchestrator(
            engine=engine,
            stage_hooks={
                "ingest_sources": lambda **kw: {"records_in": 5, "records_out": 5},
                "process_bronze": lambda **kw: {"records_in": 5, "records_out": 5},
                "transform_silver": lambda **kw: {"records_in": 5, "records_out": 5},
                "validate_data_quality": fail_dq,
            },
        )
        orch_b.task_ingest_sources(dag_id, run_dq_fail, retry_count=0)
        orch_b.task_process_bronze(dag_id, run_dq_fail, batch_dq_fail, retry_count=0)
        orch_b.task_transform_silver(dag_id, run_dq_fail, batch_dq_fail, retry_count=0)
        with pytest.raises(NonRetriablePipelineError, match="Breaking schema drift"):
            orch_b.task_validate_data_quality(dag_id, run_dq_fail, batch_dq_fail, retry_count=0)

        with pytest.raises(RuntimeError, match="status=FAILED"):
            orch_b.task_finalize_pipeline(
                dag_id,
                run_dq_fail,
                batch_dq_fail,
                upstream_states={"validate_data_quality": "failed", "build_gold": "upstream_failed"},
            )

        with engine.connect() as conn:
            final_row = conn.execute(
                text("SELECT status, dq_status FROM meta.pipeline_runs WHERE run_id = :rid"),
                {"rid": rid_dq_fail},
            ).fetchone()
            assert final_row == ("FAILED", "FAILED"), "DQ failure must never be marked SUCCESS"
    finally:
        _cleanup_test_run(engine, run_transient, batch_transient)
        _cleanup_test_run(engine, run_dq_fail, batch_dq_fail)


def _import_dag_module_with_stub():
    import importlib
    import sys
    import types

    if "airflow" not in sys.modules:
        af = types.ModuleType("airflow")
        af_exc = types.ModuleType("airflow.exceptions")
        af_op = types.ModuleType("airflow.operators")
        af_py = types.ModuleType("airflow.operators.python")
        af_tr = types.ModuleType("airflow.utils.trigger_rule")
        af_ut = types.ModuleType("airflow.utils")

        class AirflowFailException(Exception):
            pass

        class DAG:
            def __init__(self, *args, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False

        class PythonOperator:
            def __init__(self, *args, **kwargs):
                pass
            def __rshift__(self, other):
                return other
            def __rrshift__(self, other):
                return self

        class TriggerRule:
            ALL_DONE = "all_done"

        af.DAG = DAG
        af_exc.AirflowFailException = AirflowFailException
        af_py.PythonOperator = PythonOperator
        af_tr.TriggerRule = TriggerRule

        sys.modules["airflow"] = af
        sys.modules["airflow.exceptions"] = af_exc
        sys.modules["airflow.operators"] = af_op
        sys.modules["airflow.operators.python"] = af_py
        sys.modules["airflow.utils"] = af_ut
        sys.modules["airflow.utils.trigger_rule"] = af_tr

    return importlib.import_module("dags.retailpulse_medallion_dag")


def test_m14c2_11_all_done_finalize_failure_propagation_regression():
    """Task 2 regression: finalize_pipeline (TriggerRule.ALL_DONE) logs FAILED in DB and raises so DAG stays FAILED."""
    dag_mod = _import_dag_module_with_stub()
    run_finalize_pipeline = dag_mod.run_finalize_pipeline

    engine = get_engine()
    dag_id = "retailpulse_medallion_pipeline"
    dag_run_id = "pytest__m14c2_all_done_fail_prop_01"
    batch_id = derive_batch_id(dag_id, dag_run_id)
    rid = sanitize_run_id(dag_run_id)
    _cleanup_test_run(engine, dag_run_id, batch_id)

    try:
        orch = RetailPulseOrchestrator(
            engine=engine,
            stage_hooks={
                "ingest_sources": lambda **kw: {"records_in": 4, "records_out": 4},
                "generate_forecasts": lambda **kw: (_ for _ in ()).throw(RuntimeError("Simulated LightGBM failure")),
            },
        )
        orch.task_ingest_sources(dag_id, dag_run_id, retry_count=0)
        with pytest.raises(RuntimeError, match="Simulated LightGBM failure"):
            orch.task_generate_forecasts(dag_id, dag_run_id, batch_id, retry_count=0)

        class _FakeTI:
            def __init__(self, task_id, state):
                self.task_id = task_id
                self.state = state
                self.try_number = 1

        class _FakeDagRun:
            run_id = dag_run_id

            def get_task_instances(self):
                return [
                    _FakeTI("ingest_sources", "success"),
                    _FakeTI("generate_forecasts", "failed"),
                    _FakeTI("finalize_pipeline", "running"),
                ]

        with pytest.raises(RuntimeError, match="status=FAILED"):
            run_finalize_pipeline(dag_run=_FakeDagRun(), ti=_FakeTI("finalize_pipeline", "running"))

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT status, error_summary FROM meta.pipeline_runs WHERE run_id = :rid"),
                {"rid": rid},
            ).fetchone()
            assert row is not None and row[0] == "FAILED"
            assert "generate_forecasts" in (row[1] or "")
    finally:
        _cleanup_test_run(engine, dag_run_id, batch_id)


def test_m14c2_12_retry_exhaustion_and_non_retriable_fail_fast():
    """Task 3 regression: transient exhaustion records all attempts; NonRetriablePipelineError maps to AirflowFailException."""
    dag_mod = _import_dag_module_with_stub()
    from airflow.exceptions import AirflowFailException

    engine = get_engine()
    dag_id = "retailpulse_medallion_pipeline"
    dag_run_id = "pytest__m14c2_retry_exhaust_01"
    batch_id = derive_batch_id(dag_id, dag_run_id)
    rid = sanitize_run_id(dag_run_id)
    _cleanup_test_run(engine, dag_run_id, batch_id)

    try:
        def always_timeout(run_id, batch_id, retry_count):
            raise TimeoutError(f"DB timeout on attempt {retry_count}")

        orch = RetailPulseOrchestrator(engine=engine, stage_hooks={"ingest_sources": always_timeout})
        for attempt in (0, 1, 2):
            with pytest.raises(TimeoutError):
                orch.task_ingest_sources(dag_id, dag_run_id, retry_count=attempt)

        with engine.connect() as conn:
            attempts = conn.execute(
                text("SELECT retry_count, status FROM meta.task_runs WHERE run_id = :rid AND task_name = 'ingest_sources' ORDER BY retry_count"),
                {"rid": rid},
            ).fetchall()
            assert attempts == [(0, "FAILED"), (1, "FAILED"), (2, "FAILED")]

        def permanent_contract_err():
            raise NonRetriablePipelineError("Missing required column order_id")

        with pytest.raises(AirflowFailException, match="Permanent non-retriable validation failure"):
            dag_mod._wrap_call(permanent_contract_err)
    finally:
        _cleanup_test_run(engine, dag_run_id, batch_id)


def test_m14c2_13_unified_batch_id_across_ingestion_and_retries(tmp_path):
    """Issue 1 regression: one logical DAG run produces exactly one authoritative batch_id across ingestion, retries, and re-runs."""
    engine = get_engine()
    base_cfg = load_yaml_config("config/pipeline_config.yaml")

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = raw_dir / "sample_orders.csv"
    sample_csv.write_text("order_id,customer_id,order_status\no1,c1,delivered\no2,c2,delivered\n", encoding="utf-8")

    cfg = {
        **base_cfg,
        "paths": {
            "raw_dir": str(raw_dir),
            "bronze_dir": str(tmp_path / "bronze"),
            "silver_dir": str(tmp_path / "silver"),
            "gold_dir": str(tmp_path / "gold"),
            "quarantine_dir": str(tmp_path / "quarantine"),
        },
        "sources": {
            "csv_files": {
                "orders": {"file": "sample_orders.csv", "path": str(sample_csv)}
            }
        },
    }

    dag_id = "retailpulse_medallion_pipeline"
    run_1 = "pytest__m14c2_unified_batch_run_01"
    run_2 = "pytest__m14c2_unified_batch_run_02"
    batch_1 = derive_batch_id(dag_id, run_1)
    batch_2 = derive_batch_id(dag_id, run_2)
    rid_1 = sanitize_run_id(run_1)
    rid_2 = sanitize_run_id(run_2)
    _cleanup_test_run(engine, run_1, batch_1)
    _cleanup_test_run(engine, run_2, batch_2)

    try:
        orch = RetailPulseOrchestrator(engine=engine, pipeline_cfg=cfg)
        # Initial run + simulated retry (retry_count=1) + idempotent re-run
        res_try0 = orch.task_ingest_sources(dag_id, run_1, retry_count=0)
        res_try1 = orch.task_ingest_sources(dag_id, run_1, retry_count=1)
        res_rerun = orch.task_ingest_sources(dag_id, run_1, retry_count=0)

        assert res_try0["batch_id"] == batch_1
        assert res_try1["batch_id"] == batch_1
        assert res_rerun["batch_id"] == batch_1

        # Distinct logical run gets a distinct batch ID
        res_other = orch.task_ingest_sources(dag_id, run_2, retry_count=0)
        assert res_other["batch_id"] == batch_2
        assert batch_1 != batch_2

        with engine.connect() as conn:
            rows_run1 = conn.execute(
                text("SELECT batch_id FROM meta.batch_metadata WHERE run_id = :rid"),
                {"rid": rid_1},
            ).fetchall()
            assert rows_run1 == [(batch_1,)], f"Expected exactly 1 authoritative batch_id row, got {rows_run1}"

            rows_run2 = conn.execute(
                text("SELECT batch_id FROM meta.batch_metadata WHERE run_id = :rid"),
                {"rid": rid_2},
            ).fetchall()
            assert rows_run2 == [(batch_2,)]
    finally:
        _cleanup_test_run(engine, run_1, batch_1)
        _cleanup_test_run(engine, run_2, batch_2)
