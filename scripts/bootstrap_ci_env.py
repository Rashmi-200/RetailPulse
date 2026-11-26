import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from deltalake import write_deltalake
from sqlalchemy import text
from src.database.db_connection import get_engine


def bootstrap_isolated_ci() -> None:
    """Initialize isolated schema & synthetic lakehouse fixtures ONLY when DB_NAME != 'retailpulse_db'."""
    db_name = os.getenv("DB_NAME", "retailpulse_db")
    if db_name == "retailpulse_db":
        raise RuntimeError("Refusing to run CI bootstrap against primary local development database 'retailpulse_db'.")

    engine = get_engine()
    with engine.begin() as conn:
        for ddl_file in ("sql/ddl/01_create_meta_tables.sql", "sql/ddl/02_create_quarantine_tables.sql"):
            p = Path(ddl_file)
            if p.exists():
                conn.execute(text(p.read_text(encoding="utf-8")))

        conn.execute(text("CREATE SCHEMA IF NOT EXISTS gold;"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS source_oltp;"))
        conn.execute(text("ALTER TABLE meta.pipeline_runs ADD COLUMN IF NOT EXISTS dq_status VARCHAR(20) DEFAULT 'PASSED';"))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS gold.dimcustomer (
                customer_sk SERIAL PRIMARY KEY,
                customer_id VARCHAR(64) NOT NULL,
                customer_unique_id VARCHAR(64) NOT NULL,
                customer_city VARCHAR(120),
                customer_state VARCHAR(10),
                effective_from TIMESTAMP NOT NULL,
                effective_to TIMESTAMP NULL,
                is_current BOOLEAN NOT NULL DEFAULT TRUE
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS gold.dimdate (
                date_sk INT PRIMARY KEY,
                full_date DATE NOT NULL UNIQUE,
                day_of_week INT NOT NULL,
                is_weekend BOOLEAN NOT NULL,
                month INT NOT NULL,
                year INT NOT NULL
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS gold.dimlocation (
                location_sk BIGINT,
                zip_code_prefix TEXT,
                city TEXT,
                state TEXT,
                latitude DOUBLE PRECISION,
                longitude DOUBLE PRECISION
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS gold.dimproduct (
                product_sk BIGINT,
                product_id TEXT,
                product_category_name TEXT,
                product_category_name_english TEXT,
                product_weight_g DOUBLE PRECISION,
                product_length_cm DOUBLE PRECISION,
                product_height_cm DOUBLE PRECISION,
                product_width_cm DOUBLE PRECISION
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS gold.dimseller (
                seller_sk BIGINT,
                seller_id TEXT,
                seller_zip_code_prefix TEXT,
                seller_city TEXT,
                seller_state TEXT
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS gold.factorders (
                order_id VARCHAR(64) PRIMARY KEY,
                customer_sk INT,
                order_date_sk INT REFERENCES gold.dimdate(date_sk),
                order_status VARCHAR(32),
                total_order_value NUMERIC(12, 2)
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS gold.factorderitems (
                order_id TEXT,
                order_item_id BIGINT,
                product_sk BIGINT,
                seller_sk BIGINT,
                customer_sk BIGINT,
                order_date_key BIGINT,
                price DOUBLE PRECISION,
                freight_value DOUBLE PRECISION,
                total_item_value DOUBLE PRECISION
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS gold.factpayments (
                order_id TEXT,
                payment_sequential BIGINT,
                customer_sk BIGINT,
                order_date_key BIGINT,
                payment_type TEXT,
                payment_installments BIGINT,
                payment_value DOUBLE PRECISION
            );
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS source_oltp.category_translation (
                product_category_name TEXT,
                product_category_name_english TEXT
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS source_oltp.customers (
                customer_id TEXT,
                customer_unique_id TEXT,
                customer_zip_code_prefix BIGINT,
                customer_city TEXT,
                customer_state TEXT
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS source_oltp.products (
                product_id TEXT,
                product_category_name TEXT,
                product_name_lenght DOUBLE PRECISION,
                product_description_lenght DOUBLE PRECISION,
                product_photos_qty DOUBLE PRECISION,
                product_weight_g DOUBLE PRECISION,
                product_length_cm DOUBLE PRECISION,
                product_height_cm DOUBLE PRECISION,
                product_width_cm DOUBLE PRECISION
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS source_oltp.sellers (
                seller_id TEXT,
                seller_zip_code_prefix BIGINT,
                seller_city TEXT,
                seller_state TEXT
            );
        """))

        for i in range(1, 13):
            conn.execute(
                text("""
                    INSERT INTO meta.pipeline_runs (run_id, pipeline_name, trigger_type, status, dq_status, started_at)
                    VALUES (:rid, 'ci_medallion_seed', 'ci', 'SUCCESS', 'PASSED', NOW() - (:offset || ' hours')::INTERVAL)
                    ON CONFLICT (run_id) DO NOTHING;
                """),
                {"rid": f"ci_seed_run_{i:03d}", "offset": str(13 - i)},
            )

        conn.execute(text("""
            INSERT INTO meta.incidents (incident_id, batch_id, run_id, incident_type, severity, dataset_name, layer, status, detected_at)
            VALUES ('ci_seed_inc_001', 'ci_batch_reviews_001', 'ci_seed_run_001', 'DATA_CONTRACT_VIOLATION', 'HIGH', 'order_reviews', 'silver', 'OPEN', NOW())
            ON CONFLICT (incident_id) DO NOTHING;
        """))

        dim_scores = json.dumps({
            "completeness": 99.80,
            "uniqueness": 99.66,
            "validity": 99.73,
            "consistency": 99.73,
            "timeliness": 99.73,
            "accuracy": 99.73,
        })
        rule_results = json.dumps([
            {"rule": "review_score_range", "passed": False, "failed_rows": 814},
        ])
        conn.execute(
            text("""
                INSERT INTO meta.quality_results (
                    batch_id, run_id, dataset_name, layer, total_records, valid_records,
                    invalid_records, duplicate_records, overall_score, dimension_scores, rule_results, evaluated_at
                )
                SELECT 'ci_batch_reviews_001', 'ci_seed_run_001', 'order_reviews', 'silver',
                       99224, 98410, 814, 120, 99.73, CAST(:dims AS JSONB), CAST(:rules AS JSONB), NOW()
                WHERE NOT EXISTS (
                    SELECT 1 FROM meta.quality_results WHERE batch_id = 'ci_batch_reviews_001'
                );
            """),
            {"dims": dim_scores, "rules": rule_results},
        )

        conn.execute(text("""
            INSERT INTO quarantine.quarantined_records (
                quarantine_id, batch_id, run_id, dataset_name, record_key,
                error_category, rejection_reason, raw_payload, recovery_status, quarantined_at
            )
            SELECT
                'ci_quar_rev_' || LPAD(gs::TEXT, 4, '0'),
                'ci_batch_reviews_001',
                'ci_seed_run_001',
                'order_reviews',
                'rev_bad_' || gs::TEXT,
                'CONTRACT_VIOLATION',
                'Invalid review_score or duplicate review_id',
                jsonb_build_object('review_id', 'rev_bad_' || gs::TEXT, 'review_score', 99),
                'QUARANTINED',
                NOW()
            FROM generate_series(1, 814) AS gs
            ON CONFLICT (quarantine_id) DO NOTHING;
        """))

        conn.execute(text("""
            INSERT INTO meta.anomaly_alerts (
                batch_id, run_id, dataset_name, metric_name, detector_type,
                observed_value, expected_range, anomaly_score, severity, explanation, detected_at
            )
            SELECT
                'ci_batch_orders_001', 'ci_seed_run_001', 'orders', 'daily_order_volume',
                'modified_z_score', 340.0, '[45.0, 85.0]', 4.85, 'HIGH',
                'KPI deviation detected in daily_order_volume on silver.orders', NOW()
            WHERE NOT EXISTS (
                SELECT 1 FROM meta.anomaly_alerts WHERE batch_id = 'ci_batch_orders_001'
            );
        """))

        dates = pd.date_range("2018-01-01", periods=120, freq="D")
        rng = np.random.default_rng(42)
        for idx, dt in enumerate(dates):
            d_sk = int(dt.strftime("%Y%m%d"))
            conn.execute(
                text("""
                    INSERT INTO gold.dimdate (date_sk, full_date, day_of_week, is_weekend, month, year)
                    VALUES (:sk, :fd, :dow, :wk, :mo, :yr)
                    ON CONFLICT (date_sk) DO NOTHING
                """),
                {"sk": d_sk, "fd": dt.date(), "dow": int(dt.dayofweek), "wk": bool(dt.dayofweek >= 5), "mo": int(dt.month), "yr": int(dt.year)},
            )
            n_orders = int(60 + 10 * np.sin(idx / 7.0) + rng.integers(-5, 6))
            for o_i in range(n_orders):
                oid = f"ci_ord_{d_sk}_{o_i:03d}"
                conn.execute(
                    text("""
                        INSERT INTO gold.factorders (order_id, order_date_sk, order_status, total_order_value)
                        VALUES (:oid, :sk, 'delivered', :val)
                        ON CONFLICT (order_id) DO NOTHING
                    """),
                    {"oid": oid, "sk": d_sk, "val": float(85.0 + (o_i % 25))},
                )

        conn.execute(text("""
            INSERT INTO gold.dimcustomer (customer_id, customer_unique_id, customer_city, customer_state, effective_from, effective_to, is_current)
            VALUES
              ('ci_c1', 'ci_u1', 'Sao Paulo', 'SP', '2018-01-01 00:00:00', '2018-03-01 00:00:00', FALSE),
              ('ci_c1', 'ci_u1', 'Campinas', 'SP', '2018-03-01 00:00:00', NULL, TRUE)
            ON CONFLICT DO NOTHING;
        """))

    orders_delta = Path("data/silver/orders/_delta")
    if not orders_delta.exists():
        orders_delta.mkdir(parents=True, exist_ok=True)
        ts = pd.date_range("2018-01-01", periods=200, freq="6h")
        write_deltalake(
            str(orders_delta),
            pd.DataFrame({
                "order_id": [f"ci_s_ord_{i:04d}" for i in range(200)],
                "customer_id": [f"ci_s_cust_{i:04d}" for i in range(200)],
                "order_status": ["delivered"] * 200,
                "order_purchase_timestamp": ts,
            }),
            mode="overwrite",
        )

    items_delta = Path("data/silver/order_items/_delta")
    if not items_delta.exists():
        items_delta.mkdir(parents=True, exist_ok=True)
        write_deltalake(
            str(items_delta),
            pd.DataFrame({
                "order_id": [f"ci_s_ord_{i:04d}" for i in range(200)],
                "order_item_id": [1] * 200,
                "product_id": [f"ci_prod_{i % 20:02d}" for i in range(200)],
                "price": [99.5 + (i % 15) for i in range(200)],
                "freight_value": [12.5] * 200,
            }),
            mode="overwrite",
        )

    reviews_delta = Path("data/silver/order_reviews/_delta")
    if not reviews_delta.exists():
        reviews_delta.mkdir(parents=True, exist_ok=True)
        write_deltalake(
            str(reviews_delta),
            pd.DataFrame({
                "review_id": [f"ci_rev_{i:05d}" for i in range(98410)],
                "order_id": [f"ci_s_ord_{i % 200:04d}" for i in range(98410)],
                "review_score": [(i % 5) + 1 for i in range(98410)],
            }),
            mode="overwrite",
        )

    Path("logs").mkdir(parents=True, exist_ok=True)
    log_file = Path("logs/retailpulse.log")
    if not log_file.exists():
        log_file.write_text(
            "2026-11-19 10:00:00 | WARNING | src.quarantine | Quarantined duplicate review_id in order_reviews batch ci_001\n",
            encoding="utf-8",
        )

    print(f"Isolated CI environment bootstrapped safely on DB_NAME={db_name}.")


if __name__ == "__main__":
    bootstrap_isolated_ci()