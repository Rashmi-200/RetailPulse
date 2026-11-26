import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


WORKLOAD_META = {
    "W1": "CSV Ingestion & Metadata Stamping",
    "W2": "Data Cleaning, Type Casting & Null Handling",
    "W3": "Data Quality Checks & Duplicate Detection",
    "W4": "Group-By Aggregations & Financial Rollups",
    "W5": "Silver-to-Gold Star Schema Joins",
    "W6": "SCD Type 2 Change Detection & Gold Analytical Summaries",
}


def load_raw_subset(raw_dir: Path, row_limit: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    files = {
        "orders": "olist_orders_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "products": "olist_products_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "geolocation": "olist_geolocation_dataset.csv",
        "category_translation": "product_category_name_translation.csv",
    }
    dfs: Dict[str, pd.DataFrame] = {}
    for key, fname in files.items():
        nrows = None if key == "category_translation" else row_limit
        dfs[key] = pd.read_csv(raw_dir / fname, nrows=nrows)
    return dfs


def run_w1_csv_ingestion(raw_dir: Path, tmp_out_dir: Path, row_limit: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "olist_orders_dataset.csv", nrows=row_limit)
    df["_batch_id"] = "bench_batch_001"
    df["_source_system"] = "olist_csv"
    out_path = tmp_out_dir / "pandas_w1_orders.parquet"
    df.to_parquet(out_path, index=False)
    return pd.read_parquet(out_path)


def run_w2_cleaning_and_casting(raw_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    orders = raw_dfs["orders"].copy()
    for col in [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")
    orders["order_status"] = orders["order_status"].astype(str).str.strip().str.lower()

    items = raw_dfs["order_items"].copy()
    items["price"] = pd.to_numeric(items["price"], errors="coerce").fillna(0.0).round(2)
    items["freight_value"] = pd.to_numeric(items["freight_value"], errors="coerce").fillna(0.0).round(2)
    items["total_item_value"] = (items["price"] + items["freight_value"]).round(2)

    customers = raw_dfs["customers"].copy()
    customers["customer_city"] = customers["customer_city"].fillna("").astype(str).str.strip().str.lower()
    customers["customer_state"] = customers["customer_state"].fillna("").astype(str).str.strip().str.upper()
    customers["customer_zip_code_prefix"] = (
        customers["customer_zip_code_prefix"].fillna(0).astype(int).astype(str).str.zfill(5)
    )

    products = raw_dfs["products"].copy()
    products["product_category_name"] = (
        products["product_category_name"].fillna("uncategorized").astype(str).str.strip().str.lower()
    )
    for col in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
        products[col] = pd.to_numeric(products[col], errors="coerce").fillna(0.0)

    payments = raw_dfs["payments"].copy()
    payments["payment_type"] = payments["payment_type"].fillna("unknown").astype(str).str.strip().str.lower()
    payments["payment_value"] = pd.to_numeric(payments["payment_value"], errors="coerce").fillna(0.0).round(2)
    payments["payment_installments"] = (
        pd.to_numeric(payments["payment_installments"], errors="coerce").fillna(1).astype(int)
    )

    reviews = raw_dfs["reviews"].copy()
    reviews["review_score"] = pd.to_numeric(reviews["review_score"], errors="coerce").fillna(3).astype(int)

    sellers = raw_dfs["sellers"].copy()
    sellers["seller_city"] = sellers["seller_city"].fillna("").astype(str).str.strip().str.lower()
    sellers["seller_state"] = sellers["seller_state"].fillna("").astype(str).str.strip().str.upper()

    geo = raw_dfs["geolocation"].copy()
    geo["geolocation_zip_code_prefix"] = (
        geo["geolocation_zip_code_prefix"].fillna(0).astype(int).astype(str).str.zfill(5)
    )
    geo["geolocation_city"] = geo["geolocation_city"].fillna("").astype(str).str.strip().str.lower()
    geo["geolocation_state"] = geo["geolocation_state"].fillna("").astype(str).str.strip().str.upper()

    cat_trans = raw_dfs["category_translation"].copy()
    cat_trans["product_category_name"] = cat_trans["product_category_name"].astype(str).str.strip().str.lower()
    cat_trans["product_category_name_english"] = (
        cat_trans["product_category_name_english"].astype(str).str.strip().str.lower()
    )

    return {
        "orders": orders,
        "order_items": items,
        "customers": customers,
        "products": products,
        "payments": payments,
        "reviews": reviews,
        "sellers": sellers,
        "geolocation": geo,
        "category_translation": cat_trans,
    }


def run_w3_dq_and_deduplication(clean_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    orders = clean_dfs["orders"]
    valid_orders = (
        orders[orders["order_id"].notna() & orders["customer_id"].notna()]
        .sort_values(["order_id", "order_purchase_timestamp"])
        .drop_duplicates(subset=["order_id"], keep="first")
        .copy()
    )
    valid_order_ids = set(valid_orders["order_id"])

    items = clean_dfs["order_items"]
    valid_items = (
        items[
            items["order_id"].isin(valid_order_ids)
            & items["order_item_id"].notna()
            & (items["price"] >= 0.0)
            & (items["freight_value"] >= 0.0)
        ]
        .sort_values(["order_id", "order_item_id"])
        .drop_duplicates(subset=["order_id", "order_item_id"], keep="first")
        .copy()
    )

    payments = clean_dfs["payments"]
    valid_payments = (
        payments[payments["order_id"].isin(valid_order_ids) & (payments["payment_value"] >= 0.0)]
        .sort_values(["order_id", "payment_sequential"])
        .drop_duplicates(subset=["order_id", "payment_sequential"], keep="first")
        .copy()
    )

    reviews = clean_dfs["reviews"]
    valid_reviews = (
        reviews[
            reviews["order_id"].isin(valid_order_ids)
            & reviews["review_score"].between(1, 5)
        ]
        .sort_values(["order_id", "review_id"])
        .drop_duplicates(subset=["order_id", "review_id"], keep="first")
        .copy()
    )

    return {
        "orders": valid_orders,
        "order_items": valid_items,
        "payments": valid_payments,
        "reviews": valid_reviews,
    }


def run_w4_groupby_aggregations(
    clean_dfs: Dict[str, pd.DataFrame], dq_dfs: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    geo = clean_dfs["geolocation"]
    geo_agg = (
        geo.groupby(["geolocation_zip_code_prefix", "geolocation_city", "geolocation_state"], as_index=False)
        .agg(
            point_count=("geolocation_lat", "count"),
            mean_lat=("geolocation_lat", "mean"),
            mean_lng=("geolocation_lng", "mean"),
            median_lat=("geolocation_lat", "median"),
            median_lng=("geolocation_lng", "median"),
        )
    )
    for col in ["mean_lat", "mean_lng", "median_lat", "median_lng"]:
        geo_agg[col] = geo_agg[col].round(6)

    items_agg = (
        dq_dfs["order_items"]
        .groupby("order_id", as_index=False)
        .agg(
            total_items_count=("order_item_id", "count"),
            gross_order_value=("price", "sum"),
            total_freight_value=("freight_value", "sum"),
        )
    )
    items_agg["gross_order_value"] = items_agg["gross_order_value"].round(2)
    items_agg["total_freight_value"] = items_agg["total_freight_value"].round(2)

    pay_agg = (
        dq_dfs["payments"]
        .groupby("order_id", as_index=False)
        .agg(total_payment_value=("payment_value", "sum"))
    )
    pay_agg["total_payment_value"] = pay_agg["total_payment_value"].round(2)

    rev_agg = (
        dq_dfs["reviews"]
        .groupby("order_id", as_index=False)
        .agg(avg_review_score=("review_score", "mean"))
    )
    rev_agg["avg_review_score"] = rev_agg["avg_review_score"].round(2)

    return {
        "geo_agg": geo_agg,
        "items_agg": items_agg,
        "pay_agg": pay_agg,
        "rev_agg": rev_agg,
    }


def run_w5_star_schema_joins(
    clean_dfs: Dict[str, pd.DataFrame],
    dq_dfs: Dict[str, pd.DataFrame],
    agg_dfs: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    orders = dq_dfs["orders"]
    customers = clean_dfs["customers"][["customer_id", "customer_unique_id", "customer_state"]].drop_duplicates(
        subset=["customer_id"]
    )
    products = clean_dfs["products"][["product_id", "product_category_name"]].drop_duplicates(subset=["product_id"])
    cat_trans = clean_dfs["category_translation"].drop_duplicates(subset=["product_category_name"])
    products_enriched = products.merge(cat_trans, on="product_category_name", how="left")
    products_enriched["product_category_name_english"] = products_enriched["product_category_name_english"].fillna(
        products_enriched["product_category_name"]
    )
    sellers = clean_dfs["sellers"][["seller_id", "seller_state"]].drop_duplicates(subset=["seller_id"])

    fact_orders = (
        orders.merge(customers, on="customer_id", how="left")
        .merge(agg_dfs["items_agg"], on="order_id", how="left")
        .merge(agg_dfs["pay_agg"], on="order_id", how="left")
        .merge(agg_dfs["rev_agg"], on="order_id", how="left")
    )
    fact_orders["delivery_days"] = (
        (fact_orders["order_delivered_customer_date"] - fact_orders["order_purchase_timestamp"]).dt.total_seconds()
        / 86400.0
    ).round(4)
    fact_orders["is_late_delivery"] = (
        fact_orders["order_delivered_customer_date"] > fact_orders["order_estimated_delivery_date"]
    ).fillna(False)
    fact_orders["total_items_count"] = fact_orders["total_items_count"].fillna(0).astype(int)
    fact_orders["gross_order_value"] = fact_orders["gross_order_value"].fillna(0.0).round(2)
    fact_orders["total_freight_value"] = fact_orders["total_freight_value"].fillna(0.0).round(2)
    fact_orders["total_payment_value"] = fact_orders["total_payment_value"].fillna(0.0).round(2)
    fact_orders["avg_review_score"] = fact_orders["avg_review_score"].round(2)

    fact_orders_out = fact_orders[
        [
            "order_id",
            "customer_unique_id",
            "customer_state",
            "order_status",
            "delivery_days",
            "is_late_delivery",
            "total_items_count",
            "gross_order_value",
            "total_freight_value",
            "total_payment_value",
            "avg_review_score",
        ]
    ].copy()

    fact_items = (
        dq_dfs["order_items"]
        .merge(orders[["order_id", "customer_id"]], on="order_id", how="inner")
        .merge(customers, on="customer_id", how="left")
        .merge(products_enriched[["product_id", "product_category_name_english"]], on="product_id", how="left")
        .merge(sellers, on="seller_id", how="left")
    )[
        [
            "order_id",
            "order_item_id",
            "product_id",
            "product_category_name_english",
            "seller_id",
            "seller_state",
            "customer_state",
            "price",
            "freight_value",
            "total_item_value",
        ]
    ].copy()

    return {"fact_orders": fact_orders_out, "fact_items": fact_items}


def run_w6_scd2_and_gold_analytics(
    clean_dfs: Dict[str, pd.DataFrame],
    dq_dfs: Dict[str, pd.DataFrame],
    star_dfs: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    customers = clean_dfs["customers"]
    orders = dq_dfs["orders"][["customer_id", "order_purchase_timestamp"]]

    merged = customers.merge(orders, on="customer_id", how="left")
    merged["effective_ts"] = merged["order_purchase_timestamp"].fillna(pd.Timestamp("2016-09-01"))
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

    scd_versions["effective_from"] = scd_versions["effective_ts"]
    next_effective = scd_versions.groupby("customer_unique_id")["effective_from"].shift(-1)
    scd_versions["effective_to"] = next_effective
    scd_versions["is_current"] = next_effective.isna()

    dim_customer_scd2 = scd_versions[
        [
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
    ].copy()

    fact_orders = star_dfs["fact_orders"]
    state_summary = (
        fact_orders.groupby(["customer_state", "order_status"], as_index=False, dropna=False)
        .agg(
            order_count=("order_id", "count"),
            total_gmv=("gross_order_value", "sum"),
            total_payments=("total_payment_value", "sum"),
            late_orders=("is_late_delivery", "sum"),
        )
    )
    state_summary["total_gmv"] = state_summary["total_gmv"].round(2)
    state_summary["total_payments"] = state_summary["total_payments"].round(2)
    state_summary["late_orders"] = state_summary["late_orders"].astype(int)
    state_summary["late_delivery_rate_pct"] = (
        (state_summary["late_orders"] / state_summary["order_count"].replace(0, np.nan)) * 100.0
    ).fillna(0.0).round(2)

    orders_dates = dq_dfs["orders"][["order_id", "order_purchase_timestamp"]].copy()
    orders_dates["order_date"] = orders_dates["order_purchase_timestamp"].dt.strftime("%Y-%m-%d")
    daily_joined = fact_orders.merge(orders_dates[["order_id", "order_date"]], on="order_id", how="inner")
    daily_summary = (
        daily_joined.groupby("order_date", as_index=False)
        .agg(
            order_count=("order_id", "count"),
            total_gmv=("gross_order_value", "sum"),
            total_payments=("total_payment_value", "sum"),
            late_orders=("is_late_delivery", "sum"),
        )
    )
    daily_summary["total_gmv"] = daily_summary["total_gmv"].round(2)
    daily_summary["total_payments"] = daily_summary["total_payments"].round(2)
    daily_summary["late_orders"] = daily_summary["late_orders"].astype(int)
    daily_summary["late_delivery_rate_pct"] = (
        (daily_summary["late_orders"] / daily_summary["order_count"].replace(0, np.nan)) * 100.0
    ).fillna(0.0).round(2)

    return {
        "dim_customer_scd2": dim_customer_scd2,
        "state_summary": state_summary,
        "daily_summary": daily_summary,
    }
