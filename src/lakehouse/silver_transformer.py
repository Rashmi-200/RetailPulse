from pathlib import Path
from typing import Dict, Optional
import pandas as pd
from deltalake import DeltaTable, write_deltalake
from sqlalchemy.engine import Engine

from src.contracts.contract_engine import DataContractEngine
from src.observability.tracker import ObservabilityTracker, utc_now
from src.quarantine.quarantine_manager import QuarantineManager
from src.schema_evolution.registry import SchemaEvolutionRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SilverTransformer:
    """Transforms Bronze Delta tables into standardized, contract-validated Silver Delta tables."""

    def __init__(
        self,
        engine: Engine,
        config: dict,
        contracts_cfg: dict,
        tracker: ObservabilityTracker,
    ):
        self.engine = engine
        self.config = config
        self.contracts_cfg = contracts_cfg
        self.tracker = tracker
        self.bronze_dir = Path(config["paths"]["bronze_dir"])
        self.silver_dir = Path(config["paths"]["silver_dir"])
        self.schema_registry = SchemaEvolutionRegistry(engine, tracker)
        self.contract_engine = DataContractEngine(engine, contracts_cfg)
        self.quarantine_mgr = QuarantineManager(engine, tracker, config["paths"]["quarantine_dir"])

    def _standardize_dataset(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        clean = df.copy()
        clean["_silver_processed_at"] = utc_now().isoformat()

        if dataset_name == "orders":
            ts_cols = [
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ]
            for col in ts_cols:
                if col in clean.columns:
                    clean[col] = pd.to_datetime(clean[col], errors="coerce")

        elif dataset_name == "order_items":
            if "shipping_limit_date" in clean.columns:
                clean["shipping_limit_date"] = pd.to_datetime(clean["shipping_limit_date"], errors="coerce")
            clean["price"] = pd.to_numeric(clean["price"], errors="coerce")
            clean["freight_value"] = pd.to_numeric(clean["freight_value"], errors="coerce")
            clean["total_item_value"] = (clean["price"] + clean["freight_value"]).round(2)

        elif dataset_name == "payments":
            clean["payment_type"] = clean["payment_type"].astype(str).str.strip().str.lower()
            clean["payment_value"] = pd.to_numeric(clean["payment_value"], errors="coerce").round(2)
            clean["payment_installments"] = pd.to_numeric(clean["payment_installments"], errors="coerce").fillna(1).astype(int)

        elif dataset_name == "customers":
            clean["customer_city"] = clean["customer_city"].astype(str).str.strip().str.lower()
            clean["customer_state"] = clean["customer_state"].astype(str).str.strip().str.upper()
            clean["customer_zip_code_prefix"] = clean["customer_zip_code_prefix"].astype(str).str.zfill(5)

        elif dataset_name == "sellers":
            clean["seller_city"] = clean["seller_city"].astype(str).str.strip().str.lower()
            clean["seller_state"] = clean["seller_state"].astype(str).str.strip().str.upper()
            clean["seller_zip_code_prefix"] = clean["seller_zip_code_prefix"].astype(str).str.zfill(5)

        elif dataset_name == "products":
            clean["product_category_name"] = clean["product_category_name"].fillna("uncategorized").astype(str).str.strip().str.lower()
            num_cols = [
                "product_name_lenght",
                "product_description_lenght",
                "product_photos_qty",
                "product_weight_g",
                "product_length_cm",
                "product_height_cm",
                "product_width_cm",
            ]
            for col in num_cols:
                if col in clean.columns:
                    clean[col] = pd.to_numeric(clean[col], errors="coerce").fillna(0)

        elif dataset_name == "order_reviews":
            for col in ["review_creation_date", "review_answer_timestamp"]:
                if col in clean.columns:
                    clean[col] = pd.to_datetime(clean[col], errors="coerce")
            clean["review_score"] = pd.to_numeric(clean["review_score"], errors="coerce").astype(int)

        elif dataset_name == "geolocation":
            clean["geolocation_zip_code_prefix"] = clean["geolocation_zip_code_prefix"].astype(str).str.zfill(5)
            clean["geolocation_city"] = clean["geolocation_city"].astype(str).str.strip().str.lower()
            clean["geolocation_state"] = clean["geolocation_state"].astype(str).str.strip().str.upper()
            # Deduplicate zip prefix to median coordinates for clean dimensional lookup
            clean = (
                clean.groupby(["geolocation_zip_code_prefix", "geolocation_city", "geolocation_state"], as_index=False)
                .agg({"geolocation_lat": "median", "geolocation_lng": "median"})
            )
            clean["_silver_processed_at"] = utc_now().isoformat()

        elif dataset_name == "category_translation":
            clean["product_category_name"] = clean["product_category_name"].astype(str).str.strip().str.lower()
            clean["product_category_name_english"] = clean["product_category_name_english"].astype(str).str.strip().str.lower()

        return clean

    def process_dataset(self, run_id: str, dataset_name: str, valid_order_ids: Optional[set] = None) -> pd.DataFrame:
        contract = self.contracts_cfg["contracts"][dataset_name]
        bronze_delta = self.bronze_dir / dataset_name / "_delta"

        with self.tracker.track_task_run(run_id, f"silver_transform_{dataset_name}", layer="silver") as metrics:
            raw_df = DeltaTable(str(bronze_delta)).to_pandas()
            batch_id = str(raw_df["_batch_id"].iloc[-1]) if "_batch_id" in raw_df.columns else "unknown_batch"

            # 1. Schema Evolution Gate
            self.schema_registry.check_and_register(
                df=raw_df,
                dataset_name=dataset_name,
                layer="bronze",
                required_columns=contract.get("required_columns", []),
                batch_id=batch_id,
                run_id=run_id,
            )

            # 2. Data Contract Validation & Split
            report = self.contract_engine.validate_and_split(
                df=raw_df,
                dataset_name=dataset_name,
                batch_id=batch_id,
                run_id=run_id,
                layer="silver",
                persist=True,
            )

            valid_df = report.valid_df
            invalid_df = report.invalid_df

            # 3. Referential Integrity Check against Silver orders for child tables
            if valid_order_ids is not None and dataset_name in {"order_items", "payments", "order_reviews"}:
                orphan_mask = ~valid_df["order_id"].isin(valid_order_ids)
                if orphan_mask.any():
                    orphans = valid_df.loc[orphan_mask].copy()
                    orphans["_rejection_reason"] = "REFERENTIAL_INTEGRITY_MISSING_ORDER_ID"
                    invalid_df = pd.concat([invalid_df, orphans], ignore_index=True)
                    valid_df = valid_df.loc[~orphan_mask].copy()

            # 4. Route Invalid Records to Audited Quarantine
            if not invalid_df.empty:
                self.quarantine_mgr.quarantine_records(
                    invalid_df=invalid_df,
                    dataset_name=dataset_name,
                    batch_id=batch_id,
                    pk_cols=contract.get("primary_key", []),
                    run_id=run_id,
                )

            # 5. Standardize & Write Clean Silver Delta Table
            silver_df = self._standardize_dataset(valid_df, dataset_name)
            silver_delta = self.silver_dir / dataset_name / "_delta"
            silver_delta.mkdir(parents=True, exist_ok=True)
            write_deltalake(str(silver_delta), silver_df, mode="overwrite", schema_mode="overwrite")

            self.tracker.upsert_dataset_metadata(
                layer="silver",
                dataset_name=dataset_name,
                storage_format="delta",
                storage_path=str(silver_delta),
                latest_batch_id=batch_id,
                row_count=len(silver_df),
            )

            metrics["records_in"] = len(raw_df)
            metrics["records_out"] = len(silver_df)
            logger.info(
                f"[SILVER READY] {dataset_name}: {len(silver_df):,} clean rows written "
                f"(quarantined={len(invalid_df):,})"
            )
            return silver_df
