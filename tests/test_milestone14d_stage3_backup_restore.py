import os
from pathlib import Path
import tempfile
import pytest
from scripts.backup_postgres import BackupError, run_backup
from scripts.restore_postgres import RestoreError, restore_backup


def test_m14d_stage3_backup_isolated_restore_and_safeguards():
    with tempfile.TemporaryDirectory() as tmp_dir:
        backup_info = run_backup(backup_dir=tmp_dir)
        dump_path = Path(backup_info["backup_file"])
        assert dump_path.exists() and dump_path.stat().st_size > 0
        assert Path(backup_info["checksum_file"]).exists()
        assert len(backup_info["sha256"]) == 64

        with pytest.raises(RestoreError, match="Safeguard blocked restore"):
            restore_backup(str(dump_path), target_db="retailpulse_db")

        with pytest.raises(BackupError, match="Airflow metadata database"):
            run_backup(backup_dir=tmp_dir, db_name="airflow_meta")

        with pytest.raises(RestoreError, match="does not exist"):
            restore_backup(str(Path(tmp_dir) / "missing.dump"), target_db="retailpulse_restore_verify_tmp")

        bad_dir = Path(tmp_dir) / "not_a_dir.txt"
        bad_dir.write_text("x", encoding="utf-8")
        with pytest.raises(BackupError, match="not a directory"):
            run_backup(backup_dir=str(bad_dir))

        res = restore_backup(
            str(dump_path),
            target_db="retailpulse_restore_verify_tmp",
            create_isolated_db=True,
            drop_after_verify=True,
        )
        assert "meta" in res["restored_schemas"] and "quarantine" in res["restored_schemas"]
        assert res["restored_table_count"] >= 20
        assert res["dropped_after_verify"] is True

        pw = os.getenv("DB_PASSWORD", "")
        if pw:
            assert pw not in str(backup_info) and pw not in str(res)
