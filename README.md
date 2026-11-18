# RetailPulse 2.0 – Intelligent Self-Monitoring Retail Data Lakehouse

An end-to-end, self-monitoring Data Engineering platform implementing the **Medallion Architecture (Bronze, Silver, Gold)** with automated data quality scoring, schema drift detection, statistical anomaly detection, smart quarantine/recovery, and Azure cloud integration.

> **Dataset Disclaimer:** This project uses the public [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) to simulate a multi-source retail environment. It does not contain proprietary data from any organization.

## Architecture Overview
- **Ingestion:** Multi-source batch & incremental extraction (CSV, PostgreSQL, Simulated REST API) with batch tracking.
- **Bronze Layer:** Immutable raw Parquet storage preserving source snapshots.
- **Silver Layer:** Validated, deduplicated, standardized records gated by automated Data Quality and Schema Drift checks.
- **Gold Layer:** Dimensional Star Schema (`FactOrders`, `DimCustomer` SCD Type 2, etc.) for BI and ML workloads.
- **Observability:** Pipeline telemetry, lineage tracking, and real-time monitoring via Streamlit.
