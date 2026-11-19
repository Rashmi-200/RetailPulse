CREATE TABLE IF NOT EXISTS quarantine.quarantined_records (
    quarantine_id VARCHAR(64) PRIMARY KEY,
    batch_id VARCHAR(64) NOT NULL,
    run_id VARCHAR(64),
    dataset_name VARCHAR(128) NOT NULL,
    record_key VARCHAR(256),
    error_category VARCHAR(64) NOT NULL,
    rejection_reason TEXT NOT NULL,
    raw_payload JSONB NOT NULL,
    recovery_status VARCHAR(32) DEFAULT 'QUARANTINED',
    recommended_action VARCHAR(128),
    quarantined_at TIMESTAMP NOT NULL,
    resolved_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_quarantine_dataset_status
    ON quarantine.quarantined_records (dataset_name, recovery_status);
