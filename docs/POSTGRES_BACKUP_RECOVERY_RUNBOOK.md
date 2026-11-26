# RetailPulse AI — PostgreSQL Backup, Restore & Recovery Runbook

## 1. Scope & Database Isolation
- **Application Database:** `retailpulse_db` (local) / `retailpulse_docker_db` (Docker service `retailpulse-postgres`), containing `meta`, `quarantine`, `gold`, and `source_oltp` schemas.
- **Airflow Metadata Database:** `airflow_meta` (Docker service `airflow-postgres`). Strictly excluded by `scripts/backup_postgres.py` so Airflow scheduler state is never mixed with RetailPulse backups.
- **Credential Hygiene:** Passwords are passed strictly via `PGPASSWORD` in subprocess environments and redacted from any error logs.

## 2. Backup Procedures
- **Local PostgreSQL:** `python scripts/backup_postgres.py --backup-dir backups/postgres`
- **Docker PostgreSQL:** `python scripts/backup_postgres.py --backup-dir backups/postgres --docker-container retailpulse-postgres`
- Generates `<db>_YYYYMMDD_HHMMSS.dump`, `<db>_YYYYMMDD_HHMMSS.dump.sha256`, and validates archive readability via `pg_restore --list`.

## 3. Isolated Restore Verification Drill
`python scripts/restore_postgres.py --backup-file backups/postgres/<file>.dump --target-db retailpulse_restore_verify_tmp --create-isolated-db --drop-after-verify`

## 4. Disaster Recovery (Explicit Overwrite Confirmation)
Stop active writers (`docker compose stop retailpulse-api airflow-scheduler`), verify the backup in an isolated database first, then run:
`python scripts/restore_postgres.py --backup-file backups/postgres/<verified>.dump --target-db retailpulse_db --allow-overwrite-existing --confirm-target-db retailpulse_db`
