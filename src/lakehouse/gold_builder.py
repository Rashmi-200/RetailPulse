import hashlib
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from deltalake import DeltaTable, write_deltalake
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.lineage.lineage_tracker import OpenLineageTracker
from src.observability.tracker import ObservabilityTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GoldStarSchemaBuilder:
    """Builds Kimball Star Schema (SCD Type 2 DimCustomer, Dims, and Facts) in Gold Delta + PostgreSQL."""

    def __init__(self, engine: Engine, config: dict, tracker: ObservabilityTracker, lineage: OpenLineageTracker):
        self.engine = engine
        self.config = config
        self.tracker = tracker
        self.lineage = lineage
        self.silver_dir = Path(config["paths"]["silver_dir"])
        self.gold_dir = Path(config["paths"]["gold_dir"])

    def _read_silver(self, dataset_name: str) -> pd.DataFrame:
        return DeltaTable(str(self.silver_dir / dataset_name / "_delta")).to_pandas()

    def _write_gold_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        run_id: str,
        batch_id: str,
        upstream_silver: list[str],
        col_lineage: Dict[str, list[str]] | None = None,
    ) -> pd.DataFrame:
        gold_delta = self.gold_dir / table_name / "_delta"
        gold_delta.mkdir(parents=True, exist_ok=True)
        write_deltalake(str(gold_delta), df, mode="overwrite", schema_mode="overwrite")

        with self.engine.begin() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS gold;"))
        df.to_sql(
            name=table_name.lower(),
            con=self.engine,
            schema="gold",
            if_exists="replace",
            index=False,
            chunksize=10000,
        )

        self.tracker.upsert_dataset_metadata(
            layer="gold",
            dataset_name=table_name,
            storage_format="delta+postgres",
            storage_path=str(gold_delta),
            latest_batch_id=batch_id,
            row_count=len(df),
        )

        self.lineage.emit_event(
            event_type="COMPLETE",
            run_id=run_id,
            job_name=f"build_gold_{table_name}",
            inputs=[{"namespace": "retailpulse.lakehouse", "name": f"silver.{s}"} for s in upstream_silver],
            outputs=[{"namespace": "retailpulse.lakehouse", "name": f"gold.{table_name}"}],
            batch_id=batch_id,
            column_lineage=col_lineage or {},
        )
        logger.info(f"[GOLD READY] {table_name}: {len(df):,} rows written to Delta & PostgreSQL gold.{table_name.lower()}")
        return df

    def build_dim_customer_scd2(self, customers_df: pd.DataFrame, orders_df: pd.DataFrame) -> pd.DataFrame:
        """Build Slowly Changing Dimension Type 2 for DimCustomer tracking historical zip/city/state changes."""
        merged = customers_df.merge(
            orders_df[["customer_id", "order_purchase_timestamp"]],
            on="customer_id",
            how="left",
        )
        merged["effective_ts"] = pd.to_datetime(merged["order_purchase_timestamp"]).fillna(pd.Timestamp("2016-09-01"))
        merged = merged.sort_values(["customer_unique_id", "effective_ts", "customer_id"]).reset_index(drop=True)

        attr_str = (
            merged["customer_zip_code_prefix"].astype(str)
            + "|"
            + merged["customer_city"].astype(str)
            + "|"
            + merged["customer_state"].astype(str)
        )
        merged["record_hash"] = [hashlib.md5(s.encode("utf-8")).hexdigest() for s in attr_str]

        prev_uid = merged["customer_unique_id"].shift(1)
        prev_hash = merged["record_hash"].shift(1)
        is_new_version = (merged["customer_unique_id"] != prev_uid) | (merged["record_hash"] != prev_hash)
        scd_versions = merged.loc[is_new_version].copy().reset_index(drop=True)

        scd_versions["customer_sk"] = np.arange(1, len(scd_versions) + 1)
        scd_versions["effective_from"] = scd_versions["effective_ts"]
        next_effective = scd_versions.groupby("customer_unique_id")["effective_from"].shift(-1)
        scd_versions["effective_to"] = next_effective.fillna(pd.Timestamp("2099-12-31 23:59:59"))
        scd_versions["is_current"] = next_effective.isna()

        cols = [
            "customer_sk",
            "customer_unique_id",
            "customer_id",
            "customer_zip_code_prefix",
            "customer_city",
            "customer_state",
            "effective_from",
            "effective_to",
            "is_current",
            "record_hash",
        ]
        scd_df = scd_versions[cols]
        historical_versions = int((~scd_df["is_current"]).sum())
        logger.info(
            f"[SCD TYPE 2] DimCustomer built: {len(scd_df):,} total versions "
            f"({historical_versions:,} historical versions preserved)"
        )
        return scd_df

    def build_all(self, run_id: str) -> Dict[str, int]:
        with self.tracker.track_task_run(run_id, "build_gold_star_schema", layer="gold") as metrics:
            orders = self._read_silver("orders")
            items = self._read_silver("order_items")
            payments = self._read_silver("payments")
            reviews = self._read_silver("order_reviews")
            customers = self._read_silver("customers")
            products = self._read_silver("products")
            sellers = self._read_silver("sellers")
            geo = self._read_silver("geolocation")
            cat_trans = self._read_silver("category_translation")

            batch_id = str(orders["_batch_id"].iloc[-1])

            # 1. DimCustomer (SCD Type 2)
            dim_customer = self.build_dim_customer_scd2(customers, orders)
            self._write_gold_table(
                dim_customer,
                "DimCustomer",
                run_id,
                batch_id,
                ["customers", "orders"],
                {"gold.DimCustomer.is_current": ["silver.customers.customer_city", "silver.customers.customer_state"]},
            )

            cust_map = customers[["customer_id", "customer_unique_id"]].merge(
                dim_customer[dim_customer["is_current"]][["customer_unique_id", "customer_sk"]],
                on="customer_unique_id",
                how="left",
            )[["customer_id", "customer_sk"]]

            # 2. DimProduct
            dim_product = products.merge(cat_trans, on="product_category_name", how="left")
            dim_product["product_category_name_english"] = dim_product["product_category_name_english"].fillna(
                dim_product["product_category_name"]
            )
            dim_product = dim_product.reset_index(drop=True)
            dim_product["product_sk"] = np.arange(1, len(dim_product) + 1)
            dim_product = dim_product[
                [
                    "product_sk",
                    "product_id",
                    "product_category_name",
                    "product_category_name_english",
                    "product_weight_g",
                    "product_length_cm",
                    "product_height_cm",
                    "product_width_cm",
                ]
            ]
            self._write_gold_table(dim_product, "DimProduct", run_id, batch_id, ["products", "category_translation"])

            # 3. DimSeller
            dim_seller = sellers.copy().reset_index(drop=True)
            dim_seller["seller_sk"] = np.arange(1, len(dim_seller) + 1)
            dim_seller = dim_seller[
                ["seller_sk", "seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"]
            ]
            self._write_gold_table(dim_seller, "DimSeller", run_id, batch_id, ["sellers"])

            # 4. DimLocation
            dim_location = geo.copy().reset_index(drop=True)
            dim_location["location_sk"] = np.arange(1, len(dim_location) + 1)
            dim_location = dim_location.rename(
                columns={
                    "geolocation_zip_code_prefix": "zip_code_prefix",
                    "geolocation_city": "city",
                    "geolocation_state": "state",
                    "geolocation_lat": "latitude",
                    "geolocation_lng": "longitude",
                }
            )[["location_sk", "zip_code_prefix", "city", "state", "latitude", "longitude"]]
            self._write_gold_table(dim_location, "DimLocation", run_id, batch_id, ["geolocation"])

            # 5. DimDate
            min_date = pd.to_datetime(orders["order_purchase_timestamp"]).min().normalize()
            max_date = pd.to_datetime(orders["order_purchase_timestamp"]).max().normalize()
            date_range = pd.date_range(min_date, max_date, freq="D")
            dim_date = pd.DataFrame({"full_date": date_range})
            dim_date["date_key"] = dim_date["full_date"].dt.strftime("%Y%m%d").astype(int)
            dim_date["year"] = dim_date["full_date"].dt.year
            dim_date["quarter"] = dim_date["full_date"].dt.quarter
            dim_date["month"] = dim_date["full_date"].dt.month
            dim_date["month_name"] = dim_date["full_date"].dt.strftime("%B")
            dim_date["week_of_year"] = dim_date["full_date"].dt.isocalendar().week.astype(int)
            dim_date["day_of_week"] = dim_date["full_date"].dt.dayofweek
            dim_date["day_name"] = dim_date["full_date"].dt.strftime("%A")
            dim_date["is_weekend"] = dim_date["day_of_week"].isin([5, 6])
            dim_date = dim_date[
                ["date_key", "full_date", "year", "quarter", "month", "month_name", "week_of_year", "day_of_week", "day_name", "is_weekend"]
            ]
            self._write_gold_table(dim_date, "DimDate", run_id, batch_id, ["orders"])

            # 6. FactOrders
            items_agg = items.groupby("order_id", as_index=False).agg(
                total_items_count=("order_item_id", "count"),
                gross_order_value=("price", "sum"),
                total_freight_value=("freight_value", "sum"),
            )
            pay_agg = payments.groupby("order_id", as_index=False).agg(
                total_payment_value=("payment_value", "sum")
            )
            rev_agg = reviews.groupby("order_id", as_index=False).agg(
                avg_review_score=("review_score", "mean")
            )

            fact_orders = (
                orders.merge(cust_map, on="customer_id", how="left")
                .merge(items_agg, on="order_id", how="left")
                .merge(pay_agg, on="order_id", how="left")
                .merge(rev_agg, on="order_id", how="left")
            )
            purchase_ts = pd.to_datetime(fact_orders["order_purchase_timestamp"])
            delivered_ts = pd.to_datetime(fact_orders["order_delivered_customer_date"])
            estimated_ts = pd.to_datetime(fact_orders["order_estimated_delivery_date"])

            fact_orders["order_date_key"] = purchase_ts.dt.strftime("%Y%m%d").astype(int)
            fact_orders["delivery_days"] = (delivered_ts - purchase_ts).dt.total_seconds() / 86400.0
            fact_orders["is_late_delivery"] = (delivered_ts > estimated_ts).fillna(False)
            fact_orders["total_items_count"] = fact_orders["total_items_count"].fillna(0).astype(int)
            fact_orders["gross_order_value"] = fact_orders["gross_order_value"].fillna(0.0).round(2)
            fact_orders["total_freight_value"] = fact_orders["total_freight_value"].fillna(0.0).round(2)
            fact_orders["total_payment_value"] = fact_orders["total_payment_value"].fillna(0.0).round(2)
            fact_orders["avg_review_score"] = fact_orders["avg_review_score"].round(2)

            fact_orders = fact_orders[
                [
                    "order_id",
                    "customer_sk",
                    "order_date_key",
                    "order_status",
                    "order_purchase_timestamp",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "delivery_days",
                    "is_late_delivery",
                    "total_items_count",
                    "gross_order_value",
                    "total_freight_value",
                    "total_payment_value",
                    "avg_review_score",
                ]
            ]
            self._write_gold_table(
                fact_orders,
                "FactOrders",
                run_id,
                batch_id,
                ["orders", "order_items", "payments", "order_reviews", "customers"],
                {
                    "gold.FactOrders.gross_order_value": ["silver.order_items.price"],
                    "gold.FactOrders.total_payment_value": ["silver.payments.payment_value"],
                    "gold.FactOrders.is_late_delivery": [
                        "silver.orders.order_delivered_customer_date",
                        "silver.orders.order_estimated_delivery_date",
                    ],
                },
            )

            # 7. FactOrderItems
            fact_items = (
                items.merge(orders[["order_id", "customer_id", "order_purchase_timestamp"]], on="order_id", how="inner")
                .merge(cust_map, on="customer_id", how="left")
                .merge(dim_product[["product_id", "product_sk"]], on="product_id", how="left")
                .merge(dim_seller[["seller_id", "seller_sk"]], on="seller_id", how="left")
            )
            fact_items["order_date_key"] = pd.to_datetime(fact_items["order_purchase_timestamp"]).dt.strftime("%Y%m%d").astype(int)
            fact_items = fact_items[
                [
                    "order_id",
                    "order_item_id",
                    "product_sk",
                    "seller_sk",
                    "customer_sk",
                    "order_date_key",
                    "price",
                    "freight_value",
                    "total_item_value",
                ]
            ]
            self._write_gold_table(
                fact_items,
                "FactOrderItems",
                run_id,
                batch_id,
                ["order_items", "orders", "products", "sellers", "customers"],
            )

            # 8. FactPayments
            fact_payments = (
                payments.merge(orders[["order_id", "customer_id", "order_purchase_timestamp"]], on="order_id", how="inner")
                .merge(cust_map, on="customer_id", how="left")
            )
            fact_payments["order_date_key"] = pd.to_datetime(fact_payments["order_purchase_timestamp"]).dt.strftime("%Y%m%d").astype(int)
            fact_payments = fact_payments[
                [
                    "order_id",
                    "payment_sequential",
                    "customer_sk",
                    "order_date_key",
                    "payment_type",
                    "payment_installments",
                    "payment_value",
                ]
            ]
            self._write_gold_table(
                fact_payments,
                "FactPayments",
                run_id,
                batch_id,
                ["payments", "orders", "customers"],
            )

            total_gold_rows = (
                len(dim_customer)
                + len(dim_product)
                + len(dim_seller)
                + len(dim_location)
                + len(dim_date)
                + len(fact_orders)
                + len(fact_items)
                + len(fact_payments)
            )
            metrics["records_in"] = len(orders) + len(items) + len(payments)
            metrics["records_out"] = total_gold_rows

            return {
                "DimCustomer": len(dim_customer),
                "DimProduct": len(dim_product),
                "DimSeller": len(dim_seller),
                "DimLocation": len(dim_location),
                "DimDate": len(dim_date),
                "FactOrders": len(fact_orders),
                "FactOrderItems": len(fact_items),
                "FactPayments": len(fact_payments),
            }
