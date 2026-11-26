from pathlib import Path
from typing import Dict, Optional

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F


def load_raw_subset_spark(
    spark: SparkSession, raw_dir: Path, row_limit: Optional[int] = None
) -> Dict[str, DataFrame]:
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
    dfs: Dict[str, DataFrame] = {}
    for key, fname in files.items():
        use_multiline = (key == "reviews")
        df = spark.read.csv(str(raw_dir / fname), header=True, inferSchema=True, multiLine=use_multiline, escape='"')
        if row_limit is not None and key != "category_translation":
            df = df.limit(row_limit)
        dfs[key] = df
    return dfs


def run_w1_csv_ingestion_spark(
    spark: SparkSession, raw_dir: Path, tmp_out_dir: Path, row_limit: Optional[int] = None
) -> DataFrame:
    df = spark.read.csv(str(raw_dir / "olist_orders_dataset.csv"), header=True, inferSchema=True, multiLine=False, escape='"')
    if row_limit is not None:
        df = df.limit(row_limit)
    df = df.withColumn("_batch_id", F.lit("bench_batch_001")).withColumn("_source_system", F.lit("olist_csv"))
    out_path = str(tmp_out_dir / "pyspark_w1_orders.parquet")
    df.write.mode("overwrite").parquet(out_path)
    return spark.read.parquet(out_path)


def run_w2_cleaning_and_casting_spark(raw_dfs: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
    orders = raw_dfs["orders"]
    for col in [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]:
        orders = orders.withColumn(col, F.to_timestamp(F.col(col)))
    orders = orders.withColumn("order_status", F.lower(F.trim(F.col("order_status").cast("string"))))

    items = (
        raw_dfs["order_items"]
        .withColumn("price", F.round(F.coalesce(F.col("price").cast("double"), F.lit(0.0)), 2))
        .withColumn("freight_value", F.round(F.coalesce(F.col("freight_value").cast("double"), F.lit(0.0)), 2))
        .withColumn("total_item_value", F.round(F.col("price") + F.col("freight_value"), 2))
    )

    customers = (
        raw_dfs["customers"]
        .withColumn("customer_city", F.lower(F.trim(F.coalesce(F.col("customer_city").cast("string"), F.lit("")))))
        .withColumn("customer_state", F.upper(F.trim(F.coalesce(F.col("customer_state").cast("string"), F.lit("")))))
        .withColumn(
            "customer_zip_code_prefix",
            F.lpad(F.coalesce(F.col("customer_zip_code_prefix").cast("int"), F.lit(0)).cast("string"), 5, "0"),
        )
    )

    products = raw_dfs["products"].withColumn(
        "product_category_name",
        F.lower(F.trim(F.coalesce(F.col("product_category_name").cast("string"), F.lit("uncategorized")))),
    )
    for col in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
        products = products.withColumn(col, F.coalesce(F.col(col).cast("double"), F.lit(0.0)))

    payments = (
        raw_dfs["payments"]
        .withColumn("payment_type", F.lower(F.trim(F.coalesce(F.col("payment_type").cast("string"), F.lit("unknown")))))
        .withColumn("payment_value", F.round(F.coalesce(F.col("payment_value").cast("double"), F.lit(0.0)), 2))
        .withColumn("payment_installments", F.coalesce(F.col("payment_installments").cast("int"), F.lit(1)))
    )

    reviews = raw_dfs["reviews"].withColumn(
        "review_score", F.coalesce(F.col("review_score").cast("int"), F.lit(3))
    )

    sellers = (
        raw_dfs["sellers"]
        .withColumn("seller_city", F.lower(F.trim(F.coalesce(F.col("seller_city").cast("string"), F.lit("")))))
        .withColumn("seller_state", F.upper(F.trim(F.coalesce(F.col("seller_state").cast("string"), F.lit("")))))
    )

    geo = (
        raw_dfs["geolocation"]
        .withColumn(
            "geolocation_zip_code_prefix",
            F.lpad(F.coalesce(F.col("geolocation_zip_code_prefix").cast("int"), F.lit(0)).cast("string"), 5, "0"),
        )
        .withColumn("geolocation_city", F.lower(F.trim(F.coalesce(F.col("geolocation_city").cast("string"), F.lit("")))))
        .withColumn("geolocation_state", F.upper(F.trim(F.coalesce(F.col("geolocation_state").cast("string"), F.lit("")))))
    )

    cat_trans = (
        raw_dfs["category_translation"]
        .withColumn("product_category_name", F.lower(F.trim(F.col("product_category_name").cast("string"))))
        .withColumn(
            "product_category_name_english", F.lower(F.trim(F.col("product_category_name_english").cast("string")))
        )
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


def run_w3_dq_and_deduplication_spark(clean_dfs: Dict[str, DataFrame]) -> Dict[str, DataFrame]:
    w_ord = Window.partitionBy("order_id").orderBy(F.col("order_purchase_timestamp").asc_nulls_last())
    valid_orders = (
        clean_dfs["orders"]
        .filter(F.col("order_id").isNotNull() & F.col("customer_id").isNotNull())
        .withColumn("_rn", F.row_number().over(w_ord))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )
    valid_order_keys = valid_orders.select("order_id")

    w_item = Window.partitionBy("order_id", "order_item_id").orderBy(F.col("order_item_id").asc())
    valid_items = (
        clean_dfs["order_items"]
        .join(valid_order_keys, on="order_id", how="inner")
        .filter(F.col("order_item_id").isNotNull() & (F.col("price") >= 0.0) & (F.col("freight_value") >= 0.0))
        .withColumn("_rn", F.row_number().over(w_item))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

    w_pay = Window.partitionBy("order_id", "payment_sequential").orderBy(F.col("payment_sequential").asc())
    valid_payments = (
        clean_dfs["payments"]
        .join(valid_order_keys, on="order_id", how="inner")
        .filter(F.col("payment_value") >= 0.0)
        .withColumn("_rn", F.row_number().over(w_pay))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

    w_rev = Window.partitionBy("order_id", "review_id").orderBy(F.col("review_id").asc())
    valid_reviews = (
        clean_dfs["reviews"]
        .join(valid_order_keys, on="order_id", how="inner")
        .filter((F.col("review_score") >= 1) & (F.col("review_score") <= 5))
        .withColumn("_rn", F.row_number().over(w_rev))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

    return {
        "orders": valid_orders,
        "order_items": valid_items,
        "payments": valid_payments,
        "reviews": valid_reviews,
    }


def _exact_grouped_median(df: DataFrame, group_cols: list[str], val_col: str, out_col: str) -> DataFrame:
    w_ord = Window.partitionBy(*group_cols).orderBy(F.col(val_col).asc())
    w_all = Window.partitionBy(*group_cols)
    ranked = (
        df.filter(F.col(val_col).isNotNull())
        .withColumn("_rn", F.row_number().over(w_ord))
        .withColumn("_cnt", F.count(val_col).over(w_all))
        .filter(
            (F.col("_rn") == F.floor((F.col("_cnt") + 1) / 2))
            | (F.col("_rn") == F.ceil((F.col("_cnt") + 1) / 2))
        )
    )
    return ranked.groupBy(*group_cols).agg(F.round(F.avg(val_col), 6).alias(out_col))


def run_w4_groupby_aggregations_spark(
    clean_dfs: Dict[str, DataFrame], dq_dfs: Dict[str, DataFrame]
) -> Dict[str, DataFrame]:
    g_cols = ["geolocation_zip_code_prefix", "geolocation_city", "geolocation_state"]
    geo = clean_dfs["geolocation"]
    geo_base = geo.groupBy(*g_cols).agg(
        F.count("geolocation_lat").alias("point_count"),
        F.round(F.avg("geolocation_lat"), 6).alias("mean_lat"),
        F.round(F.avg("geolocation_lng"), 6).alias("mean_lng"),
    )
    med_lat = _exact_grouped_median(geo, g_cols, "geolocation_lat", "median_lat")
    med_lng = _exact_grouped_median(geo, g_cols, "geolocation_lng", "median_lng")
    geo_agg = geo_base.join(med_lat, on=g_cols, how="left").join(med_lng, on=g_cols, how="left")

    items_agg = dq_dfs["order_items"].groupBy("order_id").agg(
        F.count("order_item_id").alias("total_items_count"),
        F.round(F.sum("price"), 2).alias("gross_order_value"),
        F.round(F.sum("freight_value"), 2).alias("total_freight_value"),
    )

    pay_agg = dq_dfs["payments"].groupBy("order_id").agg(
        F.round(F.sum("payment_value"), 2).alias("total_payment_value")
    )

    rev_agg = dq_dfs["reviews"].groupBy("order_id").agg(
        F.round(F.avg("review_score"), 2).alias("avg_review_score")
    )

    return {
        "geo_agg": geo_agg,
        "items_agg": items_agg,
        "pay_agg": pay_agg,
        "rev_agg": rev_agg,
    }


def run_w5_star_schema_joins_spark(
    clean_dfs: Dict[str, DataFrame],
    dq_dfs: Dict[str, DataFrame],
    agg_dfs: Dict[str, DataFrame],
) -> Dict[str, DataFrame]:
    orders = dq_dfs["orders"]
    customers = clean_dfs["customers"].select("customer_id", "customer_unique_id", "customer_state").dropDuplicates(
        ["customer_id"]
    )
    products = clean_dfs["products"].select("product_id", "product_category_name").dropDuplicates(["product_id"])
    cat_trans = clean_dfs["category_translation"].dropDuplicates(["product_category_name"])
    products_enriched = products.join(cat_trans, on="product_category_name", how="left").withColumn(
        "product_category_name_english",
        F.coalesce(F.col("product_category_name_english"), F.col("product_category_name")),
    )
    sellers = clean_dfs["sellers"].select("seller_id", "seller_state").dropDuplicates(["seller_id"])

    fact_orders = (
        orders.join(customers, on="customer_id", how="left")
        .join(agg_dfs["items_agg"], on="order_id", how="left")
        .join(agg_dfs["pay_agg"], on="order_id", how="left")
        .join(agg_dfs["rev_agg"], on="order_id", how="left")
        .withColumn(
            "delivery_days",
            F.round(
                (
                    F.unix_timestamp("order_delivered_customer_date")
                    - F.unix_timestamp("order_purchase_timestamp")
                )
                / 86400.0,
                4,
            ),
        )
        .withColumn(
            "is_late_delivery",
            F.coalesce(F.col("order_delivered_customer_date") > F.col("order_estimated_delivery_date"), F.lit(False)),
        )
        .withColumn("total_items_count", F.coalesce(F.col("total_items_count"), F.lit(0)).cast("int"))
        .withColumn("gross_order_value", F.round(F.coalesce(F.col("gross_order_value"), F.lit(0.0)), 2))
        .withColumn("total_freight_value", F.round(F.coalesce(F.col("total_freight_value"), F.lit(0.0)), 2))
        .withColumn("total_payment_value", F.round(F.coalesce(F.col("total_payment_value"), F.lit(0.0)), 2))
        .withColumn("avg_review_score", F.round(F.col("avg_review_score"), 2))
        .select(
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
        )
    )

    fact_items = (
        dq_dfs["order_items"]
        .join(orders.select("order_id", "customer_id"), on="order_id", how="inner")
        .join(customers, on="customer_id", how="left")
        .join(products_enriched.select("product_id", "product_category_name_english"), on="product_id", how="left")
        .join(sellers, on="seller_id", how="left")
        .select(
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
        )
    )

    return {"fact_orders": fact_orders, "fact_items": fact_items}


def run_w6_scd2_and_gold_analytics_spark(
    clean_dfs: Dict[str, DataFrame],
    dq_dfs: Dict[str, DataFrame],
    star_dfs: Dict[str, DataFrame],
) -> Dict[str, DataFrame]:
    customers = clean_dfs["customers"]
    orders = dq_dfs["orders"].select("customer_id", "order_purchase_timestamp")

    merged = (
        customers.join(orders, on="customer_id", how="left")
        .withColumn(
            "effective_ts",
            F.coalesce(F.col("order_purchase_timestamp"), F.to_timestamp(F.lit("2016-09-01 00:00:00"))),
        )
        .withColumn(
            "record_hash",
            F.md5(
                F.concat_ws(
                    "|",
                    F.col("customer_zip_code_prefix").cast("string"),
                    F.col("customer_city").cast("string"),
                    F.col("customer_state").cast("string"),
                )
            ),
        )
    )

    w_uid = Window.partitionBy("customer_unique_id").orderBy(
        F.col("effective_ts").asc(), F.col("customer_id").asc()
    )
    with_prev = merged.withColumn("_prev_hash", F.lag("record_hash", 1).over(w_uid))
    scd_versions = with_prev.filter(
        F.col("_prev_hash").isNull() | (F.col("record_hash") != F.col("_prev_hash"))
    ).drop("_prev_hash")

    w_ver = Window.partitionBy("customer_unique_id").orderBy(
        F.col("effective_ts").asc(), F.col("customer_id").asc()
    )
    dim_customer_scd2 = (
        scd_versions.withColumn("effective_from", F.col("effective_ts"))
        .withColumn("effective_to", F.lead("effective_from", 1).over(w_ver))
        .withColumn("is_current", F.col("effective_to").isNull())
        .select(
            "customer_unique_id",
            "customer_id",
            "customer_zip_code_prefix",
            "customer_city",
            "customer_state",
            "effective_from",
            "effective_to",
            "is_current",
            "record_hash",
        )
    )

    fact_orders = star_dfs["fact_orders"]
    state_summary = (
        fact_orders.groupBy("customer_state", "order_status")
        .agg(
            F.count("order_id").alias("order_count"),
            F.round(F.sum("gross_order_value"), 2).alias("total_gmv"),
            F.round(F.sum("total_payment_value"), 2).alias("total_payments"),
            F.sum(F.col("is_late_delivery").cast("int")).alias("late_orders"),
        )
        .withColumn(
            "late_delivery_rate_pct",
            F.bround(F.coalesce((F.col("late_orders") / F.col("order_count")) * 100.0, F.lit(0.0)), 2),
        )
    )

    orders_dates = dq_dfs["orders"].select(
        "order_id", F.date_format("order_purchase_timestamp", "yyyy-MM-dd").alias("order_date")
    )
    daily_summary = (
        fact_orders.join(orders_dates, on="order_id", how="inner")
        .groupBy("order_date")
        .agg(
            F.count("order_id").alias("order_count"),
            F.round(F.sum("gross_order_value"), 2).alias("total_gmv"),
            F.round(F.sum("total_payment_value"), 2).alias("total_payments"),
            F.sum(F.col("is_late_delivery").cast("int")).alias("late_orders"),
        )
        .withColumn(
            "late_delivery_rate_pct",
            F.bround(F.coalesce((F.col("late_orders") / F.col("order_count")) * 100.0, F.lit(0.0)), 2),
        )
    )

    return {
        "dim_customer_scd2": dim_customer_scd2,
        "state_summary": state_summary,
        "daily_summary": daily_summary,
    }
