import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()


class BackupError(RuntimeError):
    """Raised when PostgreSQL backup creation or integrity validation fails."""


def sanitize_output(text: str, secrets: tuple[str, ...]) -> str:
    cleaned = text or ""
    for sec in secrets:
        if sec:
            cleaned = cleaned.replace(sec, "***REDACTED***")
    return cleaned


def find_pg_binary(binary_name: str) -> str:
    found = shutil.which(binary_name)
    if found:
        return found
    exe_name = f"{binary_name}.exe" if os.name == "nt" else binary_name
    pg_root = Path(r"C:\Program Files\PostgreSQL")
    if pg_root.exists():
        candidates = sorted(pg_root.glob(f"*/bin/{exe_name}"), reverse=True)
        if candidates:
            return str(candidates[0])
    raise BackupError(f"Required PostgreSQL client binary '{binary_name}' not found in PATH or standard install paths.")


def compute_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_backup(
    backup_dir: str = "backups/postgres",
    *,
    host: Optional[str] = None,
    port: Optional[str] = None,
    db_name: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    docker_container: Optional[str] = None,
) -> Dict[str, str]:
    target_host = host or os.getenv("DB_HOST", "localhost")
    target_port = str(port or os.getenv("DB_PORT", "5432"))
    target_db = db_name or (os.getenv("DOCKER_DB_NAME", "retailpulse_docker_db") if docker_container else os.getenv("DB_NAME", "retailpulse_db"))
    target_user = user or (os.getenv("DOCKER_DB_USER", "retailpulse_app") if docker_container else os.getenv("DB_USER", "postgres"))
    target_pass = password if password is not None else (
        os.getenv("DOCKER_DB_PASSWORD", "") if docker_container else os.getenv("DB_PASSWORD", "")
    )

    airflow_db = os.getenv("AIRFLOW_DB_NAME", "airflow_meta")
    if target_db.strip().lower() in {airflow_db.strip().lower(), "airflow_meta"}:
        raise BackupError(
            f"Refusing to back up Airflow metadata database ('{target_db}') using RetailPulse application backup script."
        )

    out_dir = Path(backup_dir)
    if out_dir.exists() and not out_dir.is_dir():
        raise BackupError(f"Backup path is not a directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dump_path = out_dir / f"{target_db}_{timestamp}.dump"
    checksum_path = out_dir / f"{target_db}_{timestamp}.dump.sha256"
    secrets = (target_pass,)

    if docker_container:
        cmd = [
            "docker", "exec",
            "-e", f"PGPASSWORD={target_pass}",
            docker_container,
            "pg_dump", "-U", target_user, "-d", target_db, "-Fc", "--no-owner", "--no-acl",
        ]
        with open(dump_path, "wb") as out_f:
            proc = subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, env=os.environ.copy(), timeout=120)
        if proc.returncode != 0:
            if dump_path.exists():
                dump_path.unlink()
            err = sanitize_output(proc.stderr.decode("utf-8", errors="ignore"), secrets)
            raise BackupError(f"Docker pg_dump failed (exit {proc.returncode}): {err.strip()}")
    else:
        pg_dump_bin = find_pg_binary("pg_dump")
        pg_restore_bin = find_pg_binary("pg_restore")
        env = os.environ.copy()
        env["PGPASSWORD"] = target_pass
        cmd = [
            pg_dump_bin, "-h", target_host, "-p", target_port, "-U", target_user,
            "-d", target_db, "-Fc", "--no-owner", "--no-acl", "-f", str(dump_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
        if proc.returncode != 0:
            if dump_path.exists():
                dump_path.unlink()
            err = sanitize_output(proc.stderr, secrets)
            raise BackupError(f"pg_dump failed (exit {proc.returncode}): {err.strip()}")

        list_proc = subprocess.run([pg_restore_bin, "--list", str(dump_path)], capture_output=True, text=True, timeout=60)
        if list_proc.returncode != 0 or not list_proc.stdout.strip():
            raise BackupError("Backup archive failed pg_restore --list structural validation.")

    if not dump_path.exists() or dump_path.stat().st_size == 0:
        raise BackupError("Backup file was not created or is empty (0 bytes).")

    sha256_hex = compute_sha256(dump_path)
    checksum_path.write_text(f"{sha256_hex}  {dump_path.name}\n", encoding="utf-8")

    return {
        "database": target_db,
        "backup_file": str(dump_path.resolve()),
        "checksum_file": str(checksum_path.resolve()),
        "sha256": sha256_hex,
        "size_bytes": str(dump_path.stat().st_size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and validate a timestamped PostgreSQL backup for RetailPulse.")
    parser.add_argument("--backup-dir", default=os.getenv("BACKUP_DIR", "backups/postgres"), help="Directory to store backups.")
    parser.add_argument("--db-name", default=None, help="Target RetailPulse database name.")
    parser.add_argument("--docker-container", default=None, help="Optional Docker PostgreSQL container name.")
    args = parser.parse_args()

    result = run_backup(backup_dir=args.backup_dir, db_name=args.db_name, docker_container=args.docker_container)
    print(f"[BACKUP VERIFIED] db={result['database']} size={result['size_bytes']}B sha256={result['sha256'][:16]}... file={result['backup_file']}")


if __name__ == "__main__":
    main()
