CREATE SCHEMA IF NOT EXISTS meta;
CREATE SCHEMA IF NOT EXISTS quarantine;

-- 1. Pipeline Runs
CREATE TABLE IF NOT EXISTS meta.pipeline_runs (
    run_id VARCHAR(64) PRIMARY KEY,
    pipeline_name VARCHAR(128) NOT NULL,
    trigger_type VARCHAR(32) DEFAULT 'manual',
    status VARCHAR(32) NOT NULL,
    started_at TIMESTAMP NOT NULL,
    ended_at TIMESTAMP,
    duration_seconds DOUBLE PRECISION,
    error_summary TEXT
);

-- 2. Task Runs
CREATE TABLE IF NOT EXISTS meta.task_runs (
    task_run_id VARCHAR(64) PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL REFERENCES meta.pipeline_runs(run_id) ON DELETE CASCADE,
    task_name VARCHAR(128) NOT NULL,
    layer VARCHAR(32),
    status VARCHAR(32) NOT NULL,
    retry_count INTEGER DEFAULT 0,
    records_in BIGINT DEFAULT 0,
    records_out BIGINT DEFAULT 0,
    throughput_rows_per_sec DOUBLE PRECISION,
    started_at TIMESTAMP NOT NULL,
    ended_at TIMESTAMP,
    duration_seconds DOUBLE PRECISION,
    exception_type VARCHAR(128),
    stack_trace TEXT
);

-- 3. Batch Metadata (Ingestion Watermarks & Reconciliation)
CREATE TABLE IF NOT EXISTS meta.batch_metadata (
    batch_id VARCHAR(64) PRIMARY KEY,
    run_id VARCHAR(64) REFERENCES meta.pipeline_runs(run_id),
    source_system VARCHAR(64) NOT NULL,
    dataset_name VARCHAR(128) NOT NULL,
    extraction_mode VARCHAR(32) NOT NULL,
    watermark_column VARCHAR(128),
    watermark_start VARCHAR(64),
    watermark_end VARCHAR(64),
    source_record_count BIGINT DEFAULT 0,
    target_record_count BIGINT DEFAULT 0,
    reconciliation_status VARCHAR(32) NOT NULL,
    bronze_path TEXT,
    ingested_at TIMESTAMP NOT NULL
);

-- 4. Dataset Metadata (Freshness & Inventory)
CREATE TABLE IF NOT EXISTS meta.dataset_metadata (
    dataset_key VARCHAR(128) PRIMARY KEY,
    layer VARCHAR(32) NOT NULL,
    dataset_name VARCHAR(128) NOT NULL,
    storage_format VARCHAR(32) NOT NULL,
    storage_path TEXT NOT NULL,
    latest_batch_id VARCHAR(64),
    row_count BIGINT DEFAULT 0,
    last_updated_at TIMESTAMP NOT NULL
);

-- 5. Schema Versions
CREATE TABLE IF NOT EXISTS meta.schema_versions (
    schema_version_id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(128) NOT NULL,
    layer VARCHAR(32) NOT NULL,
    version_number INTEGER NOT NULL,
    schema_hash VARCHAR(64) NOT NULL,
    schema_definition JSONB NOT NULL,
    change_type VARCHAR(32) DEFAULT 'initial',
    is_breaking BOOLEAN DEFAULT FALSE,
    diff_summary JSONB,
    registered_at TIMESTAMP NOT NULL,
    UNIQUE(dataset_name, layer, version_number)
);

-- 6. Data Quality Results
CREATE TABLE IF NOT EXISTS meta.quality_results (
    dq_result_id SERIAL PRIMARY KEY,
    batch_id VARCHAR(64) NOT NULL,
    run_id VARCHAR(64),
    dataset_name VARCHAR(128) NOT NULL,
    layer VARCHAR(32) NOT NULL,
    total_records BIGINT NOT NULL,
    valid_records BIGINT NOT NULL,
    invalid_records BIGINT NOT NULL,
    duplicate_records BIGINT DEFAULT 0,
    overall_score DOUBLE PRECISION NOT NULL,
    dimension_scores JSONB NOT NULL,
    rule_results JSONB NOT NULL,
    profile_summary JSONB,
    evaluated_at TIMESTAMP NOT NULL
);

-- 7. Anomaly Alerts
CREATE TABLE IF NOT EXISTS meta.anomaly_alerts (
    anomaly_id SERIAL PRIMARY KEY,
    batch_id VARCHAR(64) NOT NULL,
    run_id VARCHAR(64),
    dataset_name VARCHAR(128) NOT NULL,
    metric_name VARCHAR(128) NOT NULL,
    detector_type VARCHAR(64) NOT NULL,
    observed_value DOUBLE PRECISION,
    expected_range VARCHAR(128),
    anomaly_score DOUBLE PRECISION,
    severity VARCHAR(32) NOT NULL,
    explanation TEXT,
    detected_at TIMESTAMP NOT NULL
);

-- 8. Structured Incidents
CREATE TABLE IF NOT EXISTS meta.incidents (
    incident_id VARCHAR(64) PRIMARY KEY,
    batch_id VARCHAR(64),
    run_id VARCHAR(64),
    pipeline_name VARCHAR(128),
    task_name VARCHAR(128),
    dataset_name VARCHAR(128),
    layer VARCHAR(32),
    incident_type VARCHAR(64) NOT NULL,
    severity VARCHAR(32) NOT NULL,
    status VARCHAR(32) DEFAULT 'OPEN',
    rca_status VARCHAR(32) DEFAULT 'PENDING',
    evidence_refs JSONB,
    rca_report JSONB,
    recommended_action TEXT,
    detected_at TIMESTAMP NOT NULL,
    resolved_at TIMESTAMP
);

-- 9. Recovery History
CREATE TABLE IF NOT EXISTS meta.recovery_history (
    recovery_id VARCHAR(64) PRIMARY KEY,
    incident_id VARCHAR(64) REFERENCES meta.incidents(incident_id),
    batch_id VARCHAR(64) NOT NULL,
    dataset_name VARCHAR(128) NOT NULL,
    failure_category VARCHAR(64) NOT NULL,
    recovery_action VARCHAR(64) NOT NULL,
    approved_by VARCHAR(64),
    records_reprocessed BIGINT DEFAULT 0,
    status VARCHAR(32) NOT NULL,
    notes TEXT,
    executed_at TIMESTAMP NOT NULL
);

-- 10. OpenLineage-Compatible Events
CREATE TABLE IF NOT EXISTS meta.lineage_events (
    event_id VARCHAR(64) PRIMARY KEY,
    event_type VARCHAR(32) NOT NULL,
    event_time TIMESTAMP NOT NULL,
    run_id VARCHAR(64) NOT NULL,
    job_namespace VARCHAR(128) DEFAULT 'retailpulse.lakehouse',
    job_name VARCHAR(128) NOT NULL,
    batch_id VARCHAR(64),
    input_datasets JSONB NOT NULL,
    output_datasets JSONB NOT NULL,
    column_lineage JSONB
);
