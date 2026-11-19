from typing import Optional
from sqlalchemy import text
from sqlalchemy.engine import Engine


class WatermarkManager:
    """Manages incremental extraction watermarks and batch idempotency checks."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def get_last_watermark(self, source_system: str, dataset_name: str) -> Optional[str]:
        query = text(
            """
            SELECT watermark_end
            FROM meta.batch_metadata
            WHERE source_system = :source_system
              AND dataset_name = :dataset_name
              AND reconciliation_status = 'MATCHED'
              AND watermark_end IS NOT NULL
            ORDER BY ingested_at DESC
            LIMIT 1;
            """
        )
        with self.engine.connect() as conn:
            result = conn.execute(
                query, {"source_system": source_system, "dataset_name": dataset_name}
            ).fetchone()
        return result[0] if result else None

    def is_batch_already_ingested(
        self, source_system: str, dataset_name: str, watermark_start: Optional[str], watermark_end: Optional[str]
    ) -> bool:
        """Prevent duplicate ingestion for the exact same source window."""
        if watermark_end is None:
            return False
        query = text(
            """
            SELECT 1
            FROM meta.batch_metadata
            WHERE source_system = :source_system
              AND dataset_name = :dataset_name
              AND COALESCE(watermark_start, '') = COALESCE(:watermark_start, '')
              AND watermark_end = :watermark_end
              AND reconciliation_status = 'MATCHED'
            LIMIT 1;
            """
        )
        with self.engine.connect() as conn:
            row = conn.execute(
                query,
                {
                    "source_system": source_system,
                    "dataset_name": dataset_name,
                    "watermark_start": watermark_start,
                    "watermark_end": watermark_end,
                },
            ).fetchone()
        return row is not None
