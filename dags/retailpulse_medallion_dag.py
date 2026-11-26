from datetime import datetime, timedelta
from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

from src.orchestration.airflow_tasks import (
    NonRetriablePipelineError,
    RetailPulseOrchestrator,
    derive_batch_id,
)

DAG_ID = "retailpulse_medallion_pipeline"

default_args = {
    "owner": "retailpulse-de",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(seconds=15),
    "execution_timeout": timedelta(minutes=15),
}


def _ctx_ids(context):
    dag_run = context["dag_run"]
    ti = context["ti"]
    dag_run_id = dag_run.run_id
    batch_id = derive_batch_id(DAG_ID, dag_run_id)
    retry_count = max(int(ti.try_number) - 1, 0)
    return dag_run_id, batch_id, retry_count


def _wrap_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NonRetriablePipelineError as exc:
        raise AirflowFailException(f"Permanent non-retriable validation failure: {exc}") from exc


def run_ingest_sources(**context):
    dag_run_id, batch_id, retry_count = _ctx_ids(context)
    orch = RetailPulseOrchestrator()
    return _wrap_call(orch.task_ingest_sources, DAG_ID, dag_run_id, retry_count=retry_count)


def run_process_bronze(**context):
    dag_run_id, batch_id, retry_count = _ctx_ids(context)
    orch = RetailPulseOrchestrator()
    return _wrap_call(orch.task_process_bronze, DAG_ID, dag_run_id, batch_id, retry_count=retry_count)


def run_transform_silver(**context):
    dag_run_id, batch_id, retry_count = _ctx_ids(context)
    orch = RetailPulseOrchestrator()
    return _wrap_call(orch.task_transform_silver, DAG_ID, dag_run_id, batch_id, retry_count=retry_count)


def run_validate_data_quality(**context):
    dag_run_id, batch_id, retry_count = _ctx_ids(context)
    orch = RetailPulseOrchestrator()
    return _wrap_call(orch.task_validate_data_quality, DAG_ID, dag_run_id, batch_id, retry_count=retry_count)


def run_build_gold(**context):
    dag_run_id, batch_id, retry_count = _ctx_ids(context)
    orch = RetailPulseOrchestrator()
    return _wrap_call(orch.task_build_gold, DAG_ID, dag_run_id, batch_id, retry_count=retry_count)


def run_detect_anomalies(**context):
    dag_run_id, batch_id, retry_count = _ctx_ids(context)
    orch = RetailPulseOrchestrator()
    return _wrap_call(orch.task_detect_anomalies, DAG_ID, dag_run_id, batch_id, retry_count=retry_count)


def run_generate_forecasts(**context):
    dag_run_id, batch_id, retry_count = _ctx_ids(context)
    orch = RetailPulseOrchestrator()
    return _wrap_call(orch.task_generate_forecasts, DAG_ID, dag_run_id, batch_id, retry_count=retry_count)


def run_finalize_pipeline(**context):
    dag_run_id, batch_id, retry_count = _ctx_ids(context)
    dag_run = context["dag_run"]
    upstream_states = {
        ti.task_id: str(ti.state)
        for ti in dag_run.get_task_instances()
        if ti.task_id != "finalize_pipeline"
    }
    orch = RetailPulseOrchestrator()
    return orch.task_finalize_pipeline(
        DAG_ID, dag_run_id, batch_id, upstream_states=upstream_states, retry_count=retry_count
    )


with DAG(
    dag_id=DAG_ID,
    description="RetailPulse AI Medallion Lakehouse, Data Quality, Anomaly Detection & Demand Forecasting DAG",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["retailpulse", "medallion", "observability", "mlops"],
) as dag:
    ingest_sources = PythonOperator(task_id="ingest_sources", python_callable=run_ingest_sources)
    process_bronze = PythonOperator(task_id="process_bronze", python_callable=run_process_bronze)
    transform_silver = PythonOperator(task_id="transform_silver", python_callable=run_transform_silver)
    validate_data_quality = PythonOperator(
        task_id="validate_data_quality",
        python_callable=run_validate_data_quality,
        retries=0,
    )
    build_gold = PythonOperator(task_id="build_gold", python_callable=run_build_gold)
    detect_anomalies = PythonOperator(task_id="detect_anomalies", python_callable=run_detect_anomalies)
    generate_forecasts = PythonOperator(task_id="generate_forecasts", python_callable=run_generate_forecasts)
    finalize_pipeline = PythonOperator(
        task_id="finalize_pipeline",
        python_callable=run_finalize_pipeline,
        trigger_rule=TriggerRule.ALL_DONE,
        retries=0,
    )

    ingest_sources >> process_bronze >> transform_silver >> validate_data_quality >> build_gold
    build_gold >> [detect_anomalies, generate_forecasts] >> finalize_pipeline
