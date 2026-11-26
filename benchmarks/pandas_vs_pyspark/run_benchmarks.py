import argparse
import gc
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import psutil
import yaml
from pyspark.sql import SparkSession

from correctness_validation import reports_to_dict, validate_workload_parity
from metrics_collector import MetricsCollector
from pandas_benchmark import (
    WORKLOAD_META,
    load_raw_subset,
    run_w1_csv_ingestion,
    run_w2_cleaning_and_casting,
    run_w3_dq_and_deduplication,
    run_w4_groupby_aggregations,
    run_w5_star_schema_joins,
    run_w6_scd2_and_gold_analytics,
)
from pyspark_benchmark import (
    load_raw_subset_spark,
    run_w1_csv_ingestion_spark,
    run_w2_cleaning_and_casting_spark,
    run_w3_dq_and_deduplication_spark,
    run_w4_groupby_aggregations_spark,
    run_w5_star_schema_joins_spark,
    run_w6_scd2_and_gold_analytics_spark,
)


def create_spark_session(cfg: dict) -> tuple[SparkSession, float]:
    t0 = time.perf_counter()
    builder = SparkSession.builder.appName(cfg["spark"]["app_name"]).master(cfg["spark"]["master"])
    for k, v in cfg["spark"]["configs"].items():
        builder = builder.config(k, v)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    init_sec = time.perf_counter() - t0
    return spark, init_sec


def get_free_ram_mb() -> float:
    return round(psutil.virtual_memory().available / (1024**2), 2)


def run_stage3_smoke_verification(cfg: dict) -> Dict[str, Any]:
    raw_dir = Path(cfg["benchmark"]["raw_data_dir"])
    results_dir = Path(cfg["benchmark"]["results_dir"])
    collector = MetricsCollector(str(results_dir))

    row_limit = 5000
    orders_bytes = (raw_dir / "olist_orders_dataset.csv").stat().st_size
    total_raw_bytes = sum(p.stat().st_size for p in raw_dir.glob("*.csv"))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        raw_pd = load_raw_subset(raw_dir, row_limit=row_limit)
        total_in_rows = sum(len(df) for df in raw_pd.values())

        with collector.measure(
            engine="pandas",
            scale="stage3_smoke_5k",
            run_type="warm",
            run_index=1,
            workload_id="W1",
            workload_name=WORKLOAD_META["W1"],
            input_rows=len(raw_pd["orders"]),
            input_bytes=orders_bytes,
        ) as m:
            pd_w1 = run_w1_csv_ingestion(raw_dir, tmp_path, row_limit=row_limit)
            m["output_rows"] = len(pd_w1)

        with collector.measure(
            engine="pandas",
            scale="stage3_smoke_5k",
            run_type="warm",
            run_index=1,
            workload_id="W2",
            workload_name=WORKLOAD_META["W2"],
            input_rows=total_in_rows,
            input_bytes=total_raw_bytes,
        ) as m:
            pd_w2 = run_w2_cleaning_and_casting(raw_pd)
            m["output_rows"] = sum(len(df) for df in pd_w2.values())

        with collector.measure(
            engine="pandas",
            scale="stage3_smoke_5k",
            run_type="warm",
            run_index=1,
            workload_id="W3",
            workload_name=WORKLOAD_META["W3"],
            input_rows=sum(len(pd_w2[k]) for k in ["orders", "order_items", "payments", "reviews"]),
            input_bytes=total_raw_bytes,
        ) as m:
            pd_w3 = run_w3_dq_and_deduplication(pd_w2)
            m["output_rows"] = sum(len(df) for df in pd_w3.values())

        with collector.measure(
            engine="pandas",
            scale="stage3_smoke_5k",
            run_type="warm",
            run_index=1,
            workload_id="W4",
            workload_name=WORKLOAD_META["W4"],
            input_rows=len(pd_w2["geolocation"]) + len(pd_w3["order_items"]) + len(pd_w3["payments"]) + len(pd_w3["reviews"]),
            input_bytes=total_raw_bytes,
        ) as m:
            pd_w4 = run_w4_groupby_aggregations(pd_w2, pd_w3)
            m["output_rows"] = sum(len(df) for df in pd_w4.values())

        with collector.measure(
            engine="pandas",
            scale="stage3_smoke_5k",
            run_type="warm",
            run_index=1,
            workload_id="W5",
            workload_name=WORKLOAD_META["W5"],
            input_rows=len(pd_w3["orders"]) + len(pd_w3["order_items"]),
            input_bytes=total_raw_bytes,
        ) as m:
            pd_w5 = run_w5_star_schema_joins(pd_w2, pd_w3, pd_w4)
            m["output_rows"] = len(pd_w5["fact_orders"]) + len(pd_w5["fact_items"])

        with collector.measure(
            engine="pandas",
            scale="stage3_smoke_5k",
            run_type="warm",
            run_index=1,
            workload_id="W6",
            workload_name=WORKLOAD_META["W6"],
            input_rows=len(pd_w2["customers"]) + len(pd_w5["fact_orders"]),
            input_bytes=total_raw_bytes,
        ) as m:
            pd_w6 = run_w6_scd2_and_gold_analytics(pd_w2, pd_w3, pd_w5)
            m["output_rows"] = len(pd_w6["dim_customer_scd2"]) + len(pd_w6["state_summary"]) + len(pd_w6["daily_summary"])

        gc.collect()

        spark, spark_init_sec = create_spark_session(cfg)
        try:
            raw_sp = load_raw_subset_spark(spark, raw_dir, row_limit=row_limit)

            with collector.measure(
                engine="pyspark",
                scale="stage3_smoke_5k",
                run_type="warm",
                run_index=1,
                workload_id="W1",
                workload_name=WORKLOAD_META["W1"],
                input_rows=len(raw_pd["orders"]),
                input_bytes=orders_bytes,
                spark_init_sec=spark_init_sec,
            ) as m:
                sp_w1_df = run_w1_csv_ingestion_spark(spark, raw_dir, tmp_path, row_limit=row_limit)
                sp_w1 = sp_w1_df.toPandas()
                m["output_rows"] = len(sp_w1)

            with collector.measure(
                engine="pyspark",
                scale="stage3_smoke_5k",
                run_type="warm",
                run_index=1,
                workload_id="W2",
                workload_name=WORKLOAD_META["W2"],
                input_rows=total_in_rows,
                input_bytes=total_raw_bytes,
            ) as m:
                sp_w2_dfs = run_w2_cleaning_and_casting_spark(raw_sp)
                sp_w2_items = sp_w2_dfs["order_items"].toPandas()
                m["output_rows"] = sum(df.count() for df in sp_w2_dfs.values())

            with collector.measure(
                engine="pyspark",
                scale="stage3_smoke_5k",
                run_type="warm",
                run_index=1,
                workload_id="W3",
                workload_name=WORKLOAD_META["W3"],
                input_rows=sum(len(pd_w2[k]) for k in ["orders", "order_items", "payments", "reviews"]),
                input_bytes=total_raw_bytes,
            ) as m:
                sp_w3_dfs = run_w3_dq_and_deduplication_spark(sp_w2_dfs)
                sp_w3_items = sp_w3_dfs["order_items"].toPandas()
                m["output_rows"] = sum(df.count() for df in sp_w3_dfs.values())

            with collector.measure(
                engine="pyspark",
                scale="stage3_smoke_5k",
                run_type="warm",
                run_index=1,
                workload_id="W4",
                workload_name=WORKLOAD_META["W4"],
                input_rows=len(pd_w2["geolocation"]) + len(pd_w3["order_items"]) + len(pd_w3["payments"]) + len(pd_w3["reviews"]),
                input_bytes=total_raw_bytes,
            ) as m:
                sp_w4_dfs = run_w4_groupby_aggregations_spark(sp_w2_dfs, sp_w3_dfs)
                sp_w4_geo = sp_w4_dfs["geo_agg"].toPandas()
                m["output_rows"] = sum(df.count() for df in sp_w4_dfs.values())

            with collector.measure(
                engine="pyspark",
                scale="stage3_smoke_5k",
                run_type="warm",
                run_index=1,
                workload_id="W5",
                workload_name=WORKLOAD_META["W5"],
                input_rows=len(pd_w3["orders"]) + len(pd_w3["order_items"]),
                input_bytes=total_raw_bytes,
            ) as m:
                sp_w5_dfs = run_w5_star_schema_joins_spark(sp_w2_dfs, sp_w3_dfs, sp_w4_dfs)
                sp_w5_fact_orders = sp_w5_dfs["fact_orders"].toPandas()
                sp_w5_fact_items = sp_w5_dfs["fact_items"].toPandas()
                m["output_rows"] = len(sp_w5_fact_orders) + len(sp_w5_fact_items)

            with collector.measure(
                engine="pyspark",
                scale="stage3_smoke_5k",
                run_type="warm",
                run_index=1,
                workload_id="W6",
                workload_name=WORKLOAD_META["W6"],
                input_rows=len(pd_w2["customers"]) + len(pd_w5["fact_orders"]),
                input_bytes=total_raw_bytes,
            ) as m:
                sp_w6_dfs = run_w6_scd2_and_gold_analytics_spark(sp_w2_dfs, sp_w3_dfs, sp_w5_dfs)
                sp_w6_scd2 = sp_w6_dfs["dim_customer_scd2"].toPandas()
                sp_w6_state = sp_w6_dfs["state_summary"].toPandas()
                sp_w6_daily = sp_w6_dfs["daily_summary"].toPandas()
                m["output_rows"] = len(sp_w6_scd2) + len(sp_w6_state) + len(sp_w6_daily)
        finally:
            spark.stop()

    parity_reports = [
        validate_workload_parity("W1", WORKLOAD_META["W1"], pd_w1, sp_w1, ["order_id"]),
        validate_workload_parity("W2", WORKLOAD_META["W2"] + " (order_items)", pd_w2["order_items"], sp_w2_items, ["order_id", "order_item_id"]),
        validate_workload_parity("W3", WORKLOAD_META["W3"] + " (deduped order_items)", pd_w3["order_items"], sp_w3_items, ["order_id", "order_item_id"]),
        validate_workload_parity("W4", WORKLOAD_META["W4"] + " (geo_agg exact median & mean)", pd_w4["geo_agg"], sp_w4_geo, ["geolocation_zip_code_prefix", "geolocation_city", "geolocation_state"]),
        validate_workload_parity("W5", WORKLOAD_META["W5"] + " (FactOrders)", pd_w5["fact_orders"], sp_w5_fact_orders, ["order_id"]),
        validate_workload_parity("W6", WORKLOAD_META["W6"] + " (DimCustomer SCD2)", pd_w6["dim_customer_scd2"], sp_w6_scd2, ["customer_unique_id", "customer_id"]),
    ]

    additional_parity_reports = [
        validate_workload_parity("W5_FactOrderItems", WORKLOAD_META["W5"] + " (FactOrderItems)", pd_w5["fact_items"], sp_w5_fact_items, ["order_id", "order_item_id"]),
        validate_workload_parity("W6_DailyRevenueAndLateSummary", WORKLOAD_META["W6"] + " (Daily Revenue & Late-Delivery Summary)", pd_w6["daily_summary"], sp_w6_daily, ["order_date"]),
        validate_workload_parity("W6_StateRevenueAndLateSummary", WORKLOAD_META["W6"] + " (State Revenue & Late-Delivery Summary)", pd_w6["state_summary"], sp_w6_state, ["customer_state", "order_status"]),
    ]
    additional_parity_summary = reports_to_dict(additional_parity_reports)
    (results_dir / "stage3_additional_parity_validation.json").write_text(json.dumps(additional_parity_summary, indent=2), encoding="utf-8")

    parity_summary = reports_to_dict(parity_reports)
    parity_path = results_dir / "stage3_parity_validation.json"
    parity_path.write_text(json.dumps(parity_summary, indent=2), encoding="utf-8")
    saved_paths = collector.save(prefix="stage3_smoke")

    assert parity_summary["all_passed"], f"Parity validation failed: {json.dumps(parity_summary, indent=2)}"
    assert additional_parity_summary["all_passed"], f"Additional parity validation failed: {json.dumps(additional_parity_summary, indent=2)}"

    output = {
        "status": "STAGE_3_VERIFIED",
        "spark_init_sec": round(spark_init_sec, 4),
        "all_workloads_parity_passed": True,
        "saved_artifacts": saved_paths,
    }
    print(json.dumps(output, indent=2))
    return output


def run_full_scaled_benchmarks(cfg: dict, target_scales: List[str] | None = None) -> Dict[str, Any]:
    scaled_root = Path(cfg["benchmark"]["scaled_data_dir"])
    results_dir = Path(cfg["benchmark"]["results_dir"])
    collector = MetricsCollector(str(results_dir))

    cold_runs = int(cfg["benchmark"]["repetitions"]["cold_runs"])
    warm_runs = int(cfg["benchmark"]["repetitions"]["warm_runs"])
    run_schedule = [("cold", i + 1) for i in range(cold_runs)] + [("warm", i + 1) for i in range(warm_runs)]

    # Pre-flight minimum free RAM guards for the 3.83 GB Docker Desktop VM
    scale_ram_guards = {
        "1x": 1500.0,
        "5x": 1700.0,
        "10x": 1900.0,
        "20x": 2300.0,
    }

    execution_log: List[Dict[str, Any]] = []
    all_scale_parity: Dict[str, Any] = {}

    for sc in cfg["benchmark"]["scales"]:
        scale_name = sc["name"]
        if target_scales and scale_name not in target_scales:
            continue

        scale_dir = scaled_root / scale_name
        if not scale_dir.exists():
            execution_log.append({"scale": scale_name, "status": "SKIPPED_DATASET_MISSING"})
            continue

        gc.collect()
        free_ram_before = get_free_ram_mb()
        req_ram = scale_ram_guards.get(scale_name, 1800.0)
        if free_ram_before < req_ram:
            entry = {
                "scale": scale_name,
                "status": "GUARDED_SKIP_INSUFFICIENT_RAM",
                "free_ram_mb": free_ram_before,
                "required_min_free_ram_mb": req_ram,
            }
            execution_log.append(entry)
            print(json.dumps(entry, indent=2), flush=True)
            continue

        orders_bytes = (scale_dir / "olist_orders_dataset.csv").stat().st_size
        total_raw_bytes = sum(p.stat().st_size for p in scale_dir.glob("*.csv"))

        scale_status: Dict[str, Any] = {
            "scale": scale_name,
            "free_ram_before_mb": free_ram_before,
            "pandas_status": "PENDING",
            "pyspark_status": "PENDING",
            "parity_status": "PENDING",
        }

        pd_last_outputs: Dict[str, Any] = {}
        sp_last_outputs: Dict[str, Any] = {}

        # --- 1. PANDAS RUNS (1 cold + 3 warm) ---
        try:
            for run_type, run_idx in run_schedule:
                gc.collect()
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir)
                    raw_pd = load_raw_subset(scale_dir, row_limit=None)
                    total_in_rows = sum(len(df) for df in raw_pd.values())

                    with collector.measure(
                        engine="pandas",
                        scale=scale_name,
                        run_type=run_type,
                        run_index=run_idx,
                        workload_id="W1",
                        workload_name=WORKLOAD_META["W1"],
                        input_rows=len(raw_pd["orders"]),
                        input_bytes=orders_bytes,
                    ) as m:
                        pd_w1 = run_w1_csv_ingestion(scale_dir, tmp_path, row_limit=None)
                        m["output_rows"] = len(pd_w1)

                    with collector.measure(
                        engine="pandas",
                        scale=scale_name,
                        run_type=run_type,
                        run_index=run_idx,
                        workload_id="W2",
                        workload_name=WORKLOAD_META["W2"],
                        input_rows=total_in_rows,
                        input_bytes=total_raw_bytes,
                    ) as m:
                        pd_w2 = run_w2_cleaning_and_casting(raw_pd)
                        m["output_rows"] = sum(len(df) for df in pd_w2.values())

                    del raw_pd
                    gc.collect()

                    with collector.measure(
                        engine="pandas",
                        scale=scale_name,
                        run_type=run_type,
                        run_index=run_idx,
                        workload_id="W3",
                        workload_name=WORKLOAD_META["W3"],
                        input_rows=sum(len(pd_w2[k]) for k in ["orders", "order_items", "payments", "reviews"]),
                        input_bytes=total_raw_bytes,
                    ) as m:
                        pd_w3 = run_w3_dq_and_deduplication(pd_w2)
                        m["output_rows"] = sum(len(df) for df in pd_w3.values())

                    with collector.measure(
                        engine="pandas",
                        scale=scale_name,
                        run_type=run_type,
                        run_index=run_idx,
                        workload_id="W4",
                        workload_name=WORKLOAD_META["W4"],
                        input_rows=len(pd_w2["geolocation"]) + len(pd_w3["order_items"]) + len(pd_w3["payments"]) + len(pd_w3["reviews"]),
                        input_bytes=total_raw_bytes,
                    ) as m:
                        pd_w4 = run_w4_groupby_aggregations(pd_w2, pd_w3)
                        m["output_rows"] = sum(len(df) for df in pd_w4.values())

                    with collector.measure(
                        engine="pandas",
                        scale=scale_name,
                        run_type=run_type,
                        run_index=run_idx,
                        workload_id="W5",
                        workload_name=WORKLOAD_META["W5"],
                        input_rows=len(pd_w3["orders"]) + len(pd_w3["order_items"]),
                        input_bytes=total_raw_bytes,
                    ) as m:
                        pd_w5 = run_w5_star_schema_joins(pd_w2, pd_w3, pd_w4)
                        m["output_rows"] = len(pd_w5["fact_orders"]) + len(pd_w5["fact_items"])

                    with collector.measure(
                        engine="pandas",
                        scale=scale_name,
                        run_type=run_type,
                        run_index=run_idx,
                        workload_id="W6",
                        workload_name=WORKLOAD_META["W6"],
                        input_rows=len(pd_w2["customers"]) + len(pd_w5["fact_orders"]),
                        input_bytes=total_raw_bytes,
                    ) as m:
                        pd_w6 = run_w6_scd2_and_gold_analytics(pd_w2, pd_w3, pd_w5)
                        m["output_rows"] = len(pd_w6["dim_customer_scd2"]) + len(pd_w6["state_summary"]) + len(pd_w6["daily_summary"])

                    if run_type == "warm" and run_idx == warm_runs:
                        pd_last_outputs = {
                            "W1": pd_w1,
                            "W2_order_items": pd_w2["order_items"],
                            "W3_order_items": pd_w3["order_items"],
                            "W4_geo_agg": pd_w4["geo_agg"],
                            "W5_fact_orders": pd_w5["fact_orders"],
                            "W5_fact_items": pd_w5["fact_items"],
                            "W6_scd2": pd_w6["dim_customer_scd2"],
                            "W6_daily": pd_w6["daily_summary"],
                            "W6_state": pd_w6["state_summary"],
                        }
                    else:
                        del pd_w1, pd_w2, pd_w3, pd_w4, pd_w5, pd_w6
                        gc.collect()

            scale_status["pandas_status"] = "COMPLETED"
        except (MemoryError, Exception) as exc:
            scale_status["pandas_status"] = f"FAILED_OR_OOM: {type(exc).__name__}: {exc}"
            pd_last_outputs = {}
            gc.collect()

        # --- 2. PYSPARK RUNS (1 cold + 3 warm, separated Spark init) ---
        gc.collect()
        spark, spark_init_sec = create_spark_session(cfg)
        scale_status["spark_init_sec"] = round(spark_init_sec, 4)
        try:
            for run_type, run_idx in run_schedule:
                spark.catalog.clearCache()
                gc.collect()
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir)
                    gen_meta = json.loads((results_dir / "stage4_dataset_generation.json").read_text(encoding="utf-8"))
                    rc = next(x["row_counts"] for x in gen_meta["scales"] if x["scale"] == scale_name)
                    orders_in_rows = rc["orders"]
                    total_in_rows = sum(rc.values())

                    raw_sp = load_raw_subset_spark(spark, scale_dir, row_limit=None)

                    with collector.measure(
                        engine="pyspark",
                        scale=scale_name,
                        run_type=run_type,
                        run_index=run_idx,
                        workload_id="W1",
                        workload_name=WORKLOAD_META["W1"],
                        input_rows=orders_in_rows,
                        input_bytes=orders_bytes,
                        spark_init_sec=spark_init_sec if (run_type == "cold" and run_idx == 1) else 0.0,
                    ) as m:
                        sp_w1_df = run_w1_csv_ingestion_spark(spark, scale_dir, tmp_path, row_limit=None)
                        m["output_rows"] = sp_w1_df.count()

                    with collector.measure(
                        engine="pyspark",
                        scale=scale_name,
                        run_type=run_type,
                        run_index=run_idx,
                        workload_id="W2",
                        workload_name=WORKLOAD_META["W2"],
                        input_rows=total_in_rows,
                        input_bytes=total_raw_bytes,
                    ) as m:
                        sp_w2_dfs = run_w2_cleaning_and_casting_spark(raw_sp)
                        m["output_rows"] = sp_w2_dfs["order_items"].count()

                    w3_in_rows = sum(rc[k] for k in ["orders", "order_items", "payments", "reviews"])
                    with collector.measure(
                        engine="pyspark",
                        scale=scale_name,
                        run_type=run_type,
                        run_index=run_idx,
                        workload_id="W3",
                        workload_name=WORKLOAD_META["W3"],
                        input_rows=w3_in_rows,
                        input_bytes=total_raw_bytes,
                    ) as m:
                        sp_w3_dfs = run_w3_dq_and_deduplication_spark(sp_w2_dfs)
                        m["output_rows"] = sp_w3_dfs["order_items"].count()

                    w4_in_rows = rc["geolocation"] + rc["order_items"] + rc["payments"] + rc["reviews"]
                    with collector.measure(
                        engine="pyspark",
                        scale=scale_name,
                        run_type=run_type,
                        run_index=run_idx,
                        workload_id="W4",
                        workload_name=WORKLOAD_META["W4"],
                        input_rows=w4_in_rows,
                        input_bytes=total_raw_bytes,
                    ) as m:
                        sp_w4_dfs = run_w4_groupby_aggregations_spark(sp_w2_dfs, sp_w3_dfs)
                        m["output_rows"] = sp_w4_dfs["geo_agg"].count()

                    w5_in_rows = rc["orders"] + rc["order_items"]
                    with collector.measure(
                        engine="pyspark",
                        scale=scale_name,
                        run_type=run_type,
                        run_index=run_idx,
                        workload_id="W5",
                        workload_name=WORKLOAD_META["W5"],
                        input_rows=w5_in_rows,
                        input_bytes=total_raw_bytes,
                    ) as m:
                        sp_w5_dfs = run_w5_star_schema_joins_spark(sp_w2_dfs, sp_w3_dfs, sp_w4_dfs)
                        m["output_rows"] = sp_w5_dfs["fact_orders"].count()

                    w6_in_rows = rc["customers"] + rc["orders"]
                    with collector.measure(
                        engine="pyspark",
                        scale=scale_name,
                        run_type=run_type,
                        run_index=run_idx,
                        workload_id="W6",
                        workload_name=WORKLOAD_META["W6"],
                        input_rows=w6_in_rows,
                        input_bytes=total_raw_bytes,
                    ) as m:
                        sp_w6_dfs = run_w6_scd2_and_gold_analytics_spark(sp_w2_dfs, sp_w3_dfs, sp_w5_dfs)
                        m["output_rows"] = sp_w6_dfs["dim_customer_scd2"].count()

                    # Materialize Spark outputs to Pandas ONLY on final run when Pandas succeeded (for parity validation)
                    if run_type == "warm" and run_idx == warm_runs and pd_last_outputs:
                        sp_last_outputs = {
                            "W1": sp_w1_df.toPandas(),
                            "W2_order_items": sp_w2_dfs["order_items"].toPandas(),
                            "W3_order_items": sp_w3_dfs["order_items"].toPandas(),
                            "W4_geo_agg": sp_w4_dfs["geo_agg"].toPandas(),
                            "W5_fact_orders": sp_w5_dfs["fact_orders"].toPandas(),
                            "W5_fact_items": sp_w5_dfs["fact_items"].toPandas(),
                            "W6_scd2": sp_w6_dfs["dim_customer_scd2"].toPandas(),
                            "W6_daily": sp_w6_dfs["daily_summary"].toPandas(),
                            "W6_state": sp_w6_dfs["state_summary"].toPandas(),
                        }

            scale_status["pyspark_status"] = "COMPLETED"
        except Exception as exc:
            scale_status["pyspark_status"] = f"FAILED_OR_OOM: {type(exc).__name__}: {exc}"
        finally:
            spark.stop()
            gc.collect()

        # --- 3. PARITY VALIDATION AT THIS SCALE ---
        if pd_last_outputs and sp_last_outputs:
            scale_reports = [
                validate_workload_parity(workload_id=f"{scale_name}_W1", workload_name=WORKLOAD_META["W1"], pandas_df=pd_last_outputs["W1"], pyspark_df=sp_last_outputs["W1"], sort_keys=["order_id"]),
                validate_workload_parity(workload_id=f"{scale_name}_W2_order_items", workload_name=WORKLOAD_META["W2"], pandas_df=pd_last_outputs["W2_order_items"], pyspark_df=sp_last_outputs["W2_order_items"], sort_keys=["order_id", "order_item_id"]),
                validate_workload_parity(workload_id=f"{scale_name}_W3_order_items", workload_name=WORKLOAD_META["W3"], pandas_df=pd_last_outputs["W3_order_items"], pyspark_df=sp_last_outputs["W3_order_items"], sort_keys=["order_id", "order_item_id"]),
                validate_workload_parity(workload_id=f"{scale_name}_W4_geo_agg", workload_name=WORKLOAD_META["W4"], pandas_df=pd_last_outputs["W4_geo_agg"], pyspark_df=sp_last_outputs["W4_geo_agg"], sort_keys=["geolocation_zip_code_prefix", "geolocation_city", "geolocation_state"]),
                validate_workload_parity(workload_id=f"{scale_name}_W5_fact_orders", workload_name=WORKLOAD_META["W5"] + " (FactOrders)", pandas_df=pd_last_outputs["W5_fact_orders"], pyspark_df=sp_last_outputs["W5_fact_orders"], sort_keys=["order_id"]),
                validate_workload_parity(workload_id=f"{scale_name}_W5_fact_items", workload_name=WORKLOAD_META["W5"] + " (FactOrderItems)", pandas_df=pd_last_outputs["W5_fact_items"], pyspark_df=sp_last_outputs["W5_fact_items"], sort_keys=["order_id", "order_item_id"]),
                validate_workload_parity(workload_id=f"{scale_name}_W6_scd2", workload_name=WORKLOAD_META["W6"] + " (DimCustomer SCD2)", pandas_df=pd_last_outputs["W6_scd2"], pyspark_df=sp_last_outputs["W6_scd2"], sort_keys=["customer_unique_id", "customer_id"]),
                validate_workload_parity(workload_id=f"{scale_name}_W6_daily", workload_name=WORKLOAD_META["W6"] + " (Daily Summary)", pandas_df=pd_last_outputs["W6_daily"], pyspark_df=sp_last_outputs["W6_daily"], sort_keys=["order_date"]),
                validate_workload_parity(workload_id=f"{scale_name}_W6_state", workload_name=WORKLOAD_META["W6"] + " (State Summary)", pandas_df=pd_last_outputs["W6_state"], pyspark_df=sp_last_outputs["W6_state"], sort_keys=["customer_state", "order_status"]),
            ]
            p_dict = reports_to_dict(scale_reports)
            all_scale_parity[scale_name] = p_dict
            scale_status["parity_status"] = "ALL_PASSED" if p_dict["all_passed"] else "PARITY_MISMATCH"
        else:
            scale_status["parity_status"] = "SKIPPED_SINGLE_ENGINE_BOUNDARY"

        del pd_last_outputs, sp_last_outputs
        gc.collect()

        execution_log.append(scale_status)
        print(json.dumps(scale_status, indent=2), flush=True)

        # Save per-scale checkpoint
        collector.save(prefix=f"stage4_{scale_name}")
        (results_dir / f"stage4_{scale_name}_parity.json").write_text(json.dumps(all_scale_parity.get(scale_name, {}), indent=2), encoding="utf-8")
        (results_dir / f"stage4_{scale_name}_status.json").write_text(json.dumps(scale_status, indent=2), encoding="utf-8")

        # Merge all available per-scale checkpoints into unified stage4_full_* files
        merged_raw = []
        merged_summary = []
        merged_parity = {}
        merged_exec = []
        for s_item in cfg["benchmark"]["scales"]:
            s_name = s_item["name"]
            r_file = results_dir / f"stage4_{s_name}_raw_metrics.json"
            sum_file = results_dir / f"stage4_{s_name}_summary_metrics.json"
            p_file = results_dir / f"stage4_{s_name}_parity.json"
            st_file = results_dir / f"stage4_{s_name}_status.json"
            if r_file.exists():
                merged_raw.extend(json.loads(r_file.read_text(encoding="utf-8")))
            if sum_file.exists():
                merged_summary.extend(json.loads(sum_file.read_text(encoding="utf-8")))
            if p_file.exists():
                p_obj = json.loads(p_file.read_text(encoding="utf-8"))
                if p_obj:
                    merged_parity[s_name] = p_obj
            if st_file.exists():
                merged_exec.append(json.loads(st_file.read_text(encoding="utf-8")))

        import pandas as pd
        (results_dir / "stage4_full_raw_metrics.json").write_text(json.dumps(merged_raw, indent=2), encoding="utf-8")
        if merged_raw:
            pd.DataFrame(merged_raw).to_csv(results_dir / "stage4_full_raw_metrics.csv", index=False)
        (results_dir / "stage4_full_summary_metrics.json").write_text(json.dumps(merged_summary, indent=2), encoding="utf-8")
        if merged_summary:
            pd.DataFrame(merged_summary).to_csv(results_dir / "stage4_full_summary_metrics.csv", index=False)
        (results_dir / "stage4_parity_validation.json").write_text(json.dumps(merged_parity, indent=2), encoding="utf-8")
        (results_dir / "stage4_execution_summary.json").write_text(json.dumps({"scales": merged_exec}, indent=2), encoding="utf-8")

    return {"scales": execution_log}


def main() -> None:
    parser = argparse.ArgumentParser(description="RetailPulse Pandas vs PySpark Benchmark Runner")
    parser.add_argument("--mode", choices=["stage3-smoke", "full"], default="stage3-smoke")
    parser.add_argument("--scales", nargs="*", default=None, help="Optional subset of scales e.g. 1x 5x 10x 20x")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    if args.mode == "stage3-smoke":
        run_stage3_smoke_verification(cfg)
    elif args.mode == "full":
        run_full_scaled_benchmarks(cfg, target_scales=args.scales)


if __name__ == "__main__":
    main()
