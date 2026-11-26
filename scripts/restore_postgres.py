from pathlib import Path
import sys
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import os
from pathlib import Path
import re
import subprocess
from typing import Dict, List, Optional
from dotenv import load_dotenv
from scripts.backup_postgres import compute_sha256, find_pg_binary, sanitize_output

load_dotenv()

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_PROTECTED_DB_NAMES = {"postgres", "template0", "template1", "retailpulse_db", "retailpulse_docker_db", "airflow_meta"}


class RestoreError(RuntimeError):
    """Raised when PostgreSQL restore fails validation or violates overwrite safeguards."""


def verify_backup_integrity(backup_file: Path) -> str:
    if not backup_file.exists() or not backup_file.is_file():
        raise RestoreError(f"Backup file does not exist: {backup_file}")
    if backup_file.stat().st_size == 0:
        raise RestoreError(f"Backup file is empty (0 bytes): {backup_file}")
    checksum_file = Path(f"{backup_file}.sha256")
    actual_sha = compute_sha256(backup_file)
    if checksum_file.exists():
        expected_sha = checksum_file.read_text(encoding="utf-8").strip().split()[0]
        if actual_sha.lower() != expected_sha.lower():
            raise RestoreError("Backup SHA-256 checksum mismatch; archive may be corrupted.")
    return actual_sha


def restore_backup(
    backup_file: str,
    target_db: str,
    *,
    host: Optional[str] = None,
    port: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    create_isolated_db: bool = False,
    drop_after_verify: bool = False,
    allow_overwrite_existing: bool = False,
    confirm_target_db: Optional[str] = None,
) -> Dict[str, object]:
    if not _IDENTIFIER_RE.fullmatch(target_db or ""):
        raise RestoreError(f"Invalid target database identifier: '{target_db}'")
    backup_path = Path(backup_file)
    sha256_hex = verify_backup_integrity(backup_path)

    protected = set(_PROTECTED_DB_NAMES)
    for k in ("DB_NAME", "DOCKER_DB_NAME", "AIRFLOW_DB_NAME"):
        if os.getenv(k, "").strip():
            protected.add(os.getenv(k, "").strip().lower())

    if target_db.lower() in protected and not (allow_overwrite_existing and confirm_target_db == target_db):
        raise RestoreError(f"Safeguard blocked restore into protected database '{target_db}'.")

    target_host = host or os.getenv("DB_HOST", "localhost")
    target_port = str(port or os.getenv("DB_PORT", "5432"))
    target_user = user or os.getenv("DB_USER", "postgres")
    target_pass = password if password is not None else os.getenv("DB_PASSWORD", "")
    secrets = (target_pass,)

    psql_bin = find_pg_binary("psql")
    pg_restore_bin = find_pg_binary("pg_restore")
    env = os.environ.copy()
    env["PGPASSWORD"] = target_pass

    def run_psql(db: str, sql: str) -> str:
        proc = subprocess.run(
            [psql_bin, "-h", target_host, "-p", target_port, "-U", target_user, "-d", db, "-v", "ON_ERROR_STOP=1", "-tA", "-c", sql],
            capture_output=True, text=True, env=env, timeout=60,
        )
        if proc.returncode != 0:
            raise RestoreError(f"psql failed on '{db}': {sanitize_output(proc.stderr, secrets).strip()}")
        return proc.stdout.strip()

    db_exists = run_psql("postgres", f"SELECT 1 FROM pg_database WHERE datname = '{target_db}';") == "1"
    if db_exists and not allow_overwrite_existing:
        raise RestoreError(f"Target database '{target_db}' already exists. Refusing to overwrite.")

    if create_isolated_db:
        if db_exists:
            run_psql("postgres", f"DROP DATABASE {target_db};")
        run_psql("postgres", f"CREATE DATABASE {target_db};")

    r_proc = subprocess.run(
        [pg_restore_bin, "-h", target_host, "-p", target_port, "-U", target_user, "-d", target_db, "--no-owner", "--no-acl", "--exit-on-error", str(backup_path)],
        capture_output=True, text=True, env=env, timeout=180,
    )
    if r_proc.returncode != 0:
        raise RestoreError(f"pg_restore failed (exit {r_proc.returncode}): {sanitize_output(r_proc.stderr, secrets).strip()}")

    schemas = [s for s in run_psql(target_db, "SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT LIKE 'pg_%' AND schema_name <> 'information_schema' ORDER BY 1;").splitlines() if s.strip()]
    tables = [t for t in run_psql(target_db, "SELECT table_schema || '.' || table_name FROM information_schema.tables WHERE table_schema IN ('meta', 'quarantine', 'gold', 'source_oltp') ORDER BY 1;").splitlines() if t.strip()]
    sample_counts = {t: int(run_psql(target_db, f"SELECT COUNT(*) FROM {t};") or 0) for t in tables[:10]}

    if drop_after_verify:
        run_psql("postgres", f"DROP DATABASE IF EXISTS {target_db};")

    return {"target_db": target_db, "sha256": sha256_hex, "restored_schemas": schemas, "restored_table_count": len(tables), "sample_row_counts": sample_counts, "dropped_after_verify": drop_after_verify}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backup-file", required=True)
    p.add_argument("--target-db", required=True)
    p.add_argument("--create-isolated-db", action="store_true")
    p.add_argument("--drop-after-verify", action="store_true")
    p.add_argument("--allow-overwrite-existing", action="store_true")
    p.add_argument("--confirm-target-db", default=None)
    a = p.parse_args()
    res = restore_backup(a.backup_file, a.target_db, create_isolated_db=a.create_isolated_db, drop_after_verify=a.drop_after_verify, allow_overwrite_existing=a.allow_overwrite_existing, confirm_target_db=a.confirm_target_db)
    print(f"[RESTORE VERIFIED] target_db={res['target_db']} schemas={res['restored_schemas']} tables={res['restored_table_count']} sample_counts={res['sample_row_counts']} dropped={res['dropped_after_verify']}")


if __name__ == "__main__":
    main()

