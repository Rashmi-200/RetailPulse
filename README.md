# RetailPulse 2.0 – Intelligent Self-Monitoring Retail Data Lakehouse

An end-to-end, self-monitoring Data Engineering platform implementing the **Medallion Architecture (Bronze, Silver, Gold)** with automated data quality scoring, schema drift detection, statistical anomaly detection, smart quarantine/recovery, and Azure cloud integration.

> **Dataset Disclaimer:** This project uses the public [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) to simulate a multi-source retail environment. It does not contain proprietary data from any organization.

## Architecture Overview
- **Ingestion:** Multi-source batch & incremental extraction (CSV, PostgreSQL, Simulated REST API) with batch tracking.
- **Bronze Layer:** Immutable raw Parquet storage preserving source snapshots.
- **Silver Layer:** Validated, deduplicated, standardized records gated by automated Data Quality and Schema Drift checks.
- **Gold Layer:** Dimensional Star Schema (`FactOrders`, `DimCustomer` SCD Type 2, etc.) for BI and ML workloads.
- **Observability:** Pipeline telemetry, lineage tracking, and real-time monitoring via Streamlit.

## Continuous Integration (GitHub Actions CI)

The repository includes an automated CI workflow at `.github/workflows/ci.yml` triggered on pushes and pull requests to `main` and `dev-1`.

### Reproducing CI Locally (Isolated Database - Zero Impact on `retailpulse_db`)
1. Create an isolated PostgreSQL test database:
   - `psql -U postgres -c "DROP DATABASE IF EXISTS retailpulse_ci_test;"`
   - `psql -U postgres -c "CREATE DATABASE retailpulse_ci_test;"`
2. Run the bootstrap and full Pytest suite against the isolated database:
   - `$env:DB_NAME="retailpulse_ci_test"; python scripts/bootstrap_ci_env.py; pytest tests/ -v --tb=short -ra`
3. Clean up the temporary test database:
   - `Remove-Item Env:DB_NAME; psql -U postgres -c "DROP DATABASE IF EXISTS retailpulse_ci_test;"`

## Docker Compose & Apache Airflow Orchestration (Milestone 14C)

### 1. Start Services Safely
- Copy `.env.example` to `.env` and set strong local/Docker passwords (never commit `.env`).
- Start the isolated stack: `docker compose up -d --build`
- Verify container health: `docker compose ps`

### 2. Access Airflow & Trigger the Medallion DAG
- Airflow Webserver UI: `http://127.0.0.1:8080` (default local dev credentials: `admin` / `admin`).
- Unpause and trigger `retailpulse_medallion_pipeline` via CLI:
  - `docker exec airflow-scheduler airflow dags unpause retailpulse_medallion_pipeline`
  - `docker exec airflow-scheduler airflow dags trigger retailpulse_medallion_pipeline`

### 3. Inspect Logs & Interpret Failures
- Check scheduler/webserver logs: `docker compose logs --tail=50 airflow-scheduler`
- Query structured execution & DQ status in Docker PostgreSQL:
  - `docker exec retailpulse-postgres psql -U retailpulse_app -d retailpulse_docker_db -c "SELECT run_id, status, dq_status, error_summary FROM meta.pipeline_runs ORDER BY started_at DESC LIMIT 5;"`
- Permanent contract/schema failures raise `AirflowFailException` (`retries=0` on `validate_data_quality`) and record `dq_status = 'FAILED'` + an open incident in `meta.incidents`.

### 4. Stop Services & Manage Volumes
- Stop containers while preserving persistent PostgreSQL and Delta Lake volumes: `docker compose down`
- Stop containers and remove volumes (full clean reset): `docker compose down -v`

### 5. Run Full Automated Test Suite
- `pytest tests/ -v --tb=short -ra`
