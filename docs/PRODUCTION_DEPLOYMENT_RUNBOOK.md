# RetailPulse AI — Production Deployment & Operations Runbook

## 1. Architecture & Service Responsibilities
- **retailpulse-postgres (`postgres:16-alpine`, 512 MiB, 30s grace):** Application database (`gold`, `meta`, `quarantine`, `public`) backed by volume `retailpulse_pgdata`.
- **airflow-postgres (`postgres:16-alpine`, 512 MiB, 30s grace):** Airflow metadata database backed by volume `airflow_pgdata`.
- **retailpulse-api (`Dockerfile` / `python:3.11-slim`, 1024 MiB, 30s grace, `127.0.0.1:8000`):** FastAPI control plane (`src/api/control_plane.py`) enforcing `X-API-Key` authentication, per-minute rate limiting (`API_RATE_LIMIT_PER_MIN`), and sanitized error responses.
- **airflow-init (`Dockerfile.airflow` / `2.10.3`, 1024 MiB):** Validates Airflow secrets fail-closed, runs `airflow db migrate`, and provisions initial admin user (one-shot, exits `0`).
- **airflow-webserver (`Dockerfile.airflow` / `2.10.3`, 1024 MiB, `127.0.0.1:8080`):** Serves Airflow Web UI and REST API.
- **airflow-scheduler (`Dockerfile.airflow` / `2.10.3`, 1536 MiB, 30s grace):** Executes `retailpulse_medallion_pipeline` (`dags/retailpulse_medallion_dag.py`) via `LocalExecutor`.

## 2. Prerequisites & Secure Environment Setup
1. Copy `.env.example` to `.env` (`chmod 600 .env`; never commit `.env` to Git).
2. Generate API key: `python -c "import secrets; print('rp_live_' + secrets.token_urlsafe(32))"`
3. Generate Fernet key using `Fernet.generate_key()` (never use arbitrary strings): `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
4. Generate strong DB passwords (`DOCKER_DB_PASSWORD`, `AIRFLOW_DB_PASSWORD`) and set `APP_ENV=production` and `API_AUTH_REQUIRED=true`.

## 3. Service-Scoped Production Configuration Validation
- `docker-compose.yml` enforces `:?` required variables (`DOCKER_DB_PASSWORD`, `AIRFLOW_DB_PASSWORD`, `AIRFLOW_FERNET_KEY`, `RETAILPULSE_API_KEY`) so Compose fails closed when required secrets are missing.
- `validate_runtime_environment(service="api")` validates only API and application DB settings at FastAPI startup; `retailpulse-api` never receives `AIRFLOW_DB_*` or `AIRFLOW_FERNET_KEY`.
- `validate_runtime_environment(service="airflow")` validates Airflow secrets in `airflow-init` and rejects placeholders (`change_me*`, `replace_with*`, `your_*`, `admin`, `postgres`).
- `APP_ENV=development` preserves local CLI and `pytest` compatibility without requiring production secrets while validating `DB_PORT` (`1-65535`).

## 4. Docker Deployment, Startup, Shutdown & Health Checks
- **Start / Rebuild Stack:** `docker compose up -d --build`
- **Graceful Shutdown:** `docker compose stop` (`exec uvicorn` + `stop_grace_period: 30s` ensures `ExitCode=0`).
- **Critical Safety Rule:** Never run `docker compose down -v` (destroys `retailpulse_pgdata`, `airflow_pgdata`, `retailpulse_lakehouse_data`, and `retailpulse_logs`).
- **Health Checks:** `docker ps -a` (all 5 core services report `healthy`; `airflow-init` reports `Exited (0)`).

## 5. API Authentication Verification
- Public Health (`200 OK`): `curl -f http://127.0.0.1:8000/api/v1/health`
- Unauthenticated Protected Endpoint (`401 Unauthorized`): `curl -i http://127.0.0.1:8000/api/v1/runs`
- Authenticated Protected Endpoint (`200 OK`): `curl -f -H "X-API-Key: <RETAILPULSE_API_KEY>" http://127.0.0.1:8000/api/v1/runs`

## 6. Airflow DAG Operation & Eight-Task Verification
- **DAG File & ID:** `dags/retailpulse_medallion_dag.py` (`retailpulse_medallion_pipeline`).
- **Unpause & Trigger:** `docker exec airflow-scheduler airflow dags unpause retailpulse_medallion_pipeline` and `docker exec airflow-scheduler airflow dags trigger retailpulse_medallion_pipeline --run-id <run_id>`.
- **Verify All Eight Tasks:** `ingest_sources`, `process_bronze`, `transform_silver`, `validate_data_quality`, `build_gold`, `detect_anomalies`, `generate_forecasts`, and `finalize_pipeline` must reach `success`, and `meta.pipeline_runs` in `retailpulse-postgres` must record `SUCCESS`.

## 7. PostgreSQL Backup & Isolated Restore Procedures
- **Create Docker Backup (`scripts/backup_postgres.py`):**
  `python scripts/backup_postgres.py --docker-container retailpulse-postgres --backup-dir backups/postgres` (generates `.dump` + `.sha256` manifest).
- **Host-Target Isolated Restore Verification (`scripts/restore_postgres.py`):**
  `python scripts/restore_postgres.py --backup-file backups/postgres/<archive>.dump --target-db retailpulse_stage5_verify_tmp --create-isolated-db --drop-after-verify`
- **Docker-Container Isolated Restore Verification:**
  Copy `.dump` to `retailpulse-postgres:/tmp/`, create temporary database `retailpulse_stage5_verify_tmp`, execute `pg_restore -U <DOCKER_DB_USER> -d retailpulse_stage5_verify_tmp --no-owner --no-acl --exit-on-error /tmp/<archive>.dump`, verify row counts in `meta.pipeline_runs` / `gold.*`, and drop `retailpulse_stage5_verify_tmp` afterward. Never restore over live databases.

## 8. Troubleshooting, Database Recovery & Rollback
- **Exit Code 137 on Stop:** Inspect `docker inspect <container> --format "{{.State.ExitCode}} {{.State.OOMKilled}}"` to distinguish cgroup OOM (`OOMKilled=true`) from `SIGKILL` grace-period timeout (`OOMKilled=false`).
- **Pre-Existing Volume Password Rotation:** Rotate passwords inside PostgreSQL (`ALTER USER <role> WITH PASSWORD '<new>';`) before updating `.env` - never delete or recreate `pgdata` volumes.
- **Airflow `FutureWarning` (`core/sql_alchemy_conn`):** Informational warning emitted by Airflow 2.10 internals; `docker-compose.yml` already sets `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN`.
- **Rollback:** Stop writers (`docker compose stop airflow-scheduler retailpulse-api`), check out the last known good Git commit (`git checkout <good_commit>`), rebuild (`docker compose up -d --build`), or restore a `.sha256`-verified `.dump` into a clean target database.

## 9. Local Single-Host Deployment vs. Future Public Cloud Deployment
- **TLS & Network Exposure:** Current local deployment binds exclusively to `127.0.0.1` over HTTP; public cloud deployment must terminate **TLS 1.2+** at an external reverse proxy or Cloud ALB (Nginx, Caddy, AWS ALB, Azure Application Gateway) and never expose unencrypted HTTP ports publicly.
- **External Secrets Management:** Replace local `.env` files with **AWS Secrets Manager**, **Azure Key Vault**, or **HashiCorp Vault** with automated key rotation and IAM role binding.
- **Managed Database & Scaling:** Migrate containerized PostgreSQL to managed **RDS / Azure Database for PostgreSQL** with Multi-AZ failover, Point-in-Time Recovery (PITR), and encrypted cross-region backups, and scale Airflow via managed **MWAA / Astronomer** or Kubernetes executors.