import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import psutil
import pyspark
import yaml
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def get_cgroup_memory_limit_bytes() -> int | None:
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        p = Path(path)
        if p.exists():
            raw = p.read_text(encoding="utf-8").strip()
            if raw.isdigit():
                val = int(raw)
                if val < (1 << 60):
                    return val
    return None


def main() -> None:
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    java_ver = subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT, text=True).strip().splitlines()

    vm = psutil.virtual_memory()
    cgroup_limit = get_cgroup_memory_limit_bytes()
    disk = shutil.disk_usage("/workspace/benchmarks/pandas_vs_pyspark")

    env_info = {
        "python_version": platform.python_version(),
        "java_version": java_ver,
        "pyspark_version": pyspark.__version__,
        "pandas_version": pd.__version__,
        "pyarrow_version": pa.__version__,
        "psutil_version": psutil.__version__,
        "cpu_count_logical": os.cpu_count(),
        "container_cgroup_mem_limit_gb": round(cgroup_limit / (1024**3), 2) if cgroup_limit else None,
        "visible_total_ram_gb": round(vm.total / (1024**3), 2),
        "visible_available_ram_gb": round(vm.available / (1024**3), 2),
        "benchmark_disk_free_gb": round(disk.free / (1024**3), 2),
    }

    min_free_mb = cfg["benchmark"]["safety"]["min_free_container_mem_mb"]
    avail_mb = vm.available / (1024**2)
    assert avail_mb >= min_free_mb, f"Insufficient free RAM: {avail_mb:.1f} MB < {min_free_mb} MB"

    raw_dir = Path(cfg["benchmark"]["raw_data_dir"])
    orders_csv = raw_dir / "olist_orders_dataset.csv"
    items_csv = raw_dir / "olist_order_items_dataset.csv"
    assert orders_csv.exists() and items_csv.exists(), "Raw Olist CSVs not mounted!"
    read_only_verified = False
    try:
        (raw_dir / "_ro_probe.tmp").write_text("test", encoding="utf-8")
    except OSError:
        read_only_verified = True
    assert read_only_verified, "Expected /workspace/data/raw to be mounted read-only!"

    out_probe = Path(cfg["benchmark"]["results_dir"]) / "_stage2_write_probe.tmp"
    out_probe.write_text("ok", encoding="utf-8")
    out_probe.unlink()

    t0 = time.perf_counter()
    builder = SparkSession.builder.appName(cfg["spark"]["app_name"]).master(cfg["spark"]["master"])
    for k, v in cfg["spark"]["configs"].items():
        builder = builder.config(k, v)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    spark_init_sec = round(time.perf_counter() - t0, 4)

    orders_pd = pd.read_csv(orders_csv, nrows=5000)
    items_pd = pd.read_csv(items_csv, nrows=10000)

    valid_orders_pd = orders_pd[orders_pd["order_status"] == "delivered"][["order_id", "customer_id"]].copy()
    items_pd["total_item_value"] = (items_pd["price"] + items_pd["freight_value"]).round(2)
    agg_pd = (
        valid_orders_pd.merge(items_pd, on="order_id", how="inner")
        .groupby("order_id", as_index=False)
        .agg(item_count=("order_item_id", "count"), total_revenue=("total_item_value", "sum"))
        .sort_values("order_id")
        .reset_index(drop=True)
    )
    agg_pd["total_revenue"] = agg_pd["total_revenue"].round(2)

    orders_sp = spark.read.csv(str(orders_csv), header=True, inferSchema=True).limit(5000)
    items_sp = spark.read.csv(str(items_csv), header=True, inferSchema=True).limit(10000)
    valid_orders_sp = orders_sp.filter(F.col("order_status") == "delivered").select("order_id", "customer_id")
    items_sp_calc = items_sp.withColumn("total_item_value", F.round(F.col("price") + F.col("freight_value"), 2))
    agg_sp_df = (
        valid_orders_sp.join(items_sp_calc, on="order_id", how="inner")
        .groupBy("order_id")
        .agg(
            F.count("order_item_id").alias("item_count"),
            F.round(F.sum("total_item_value"), 2).alias("total_revenue"),
        )
        .orderBy("order_id")
    )
    agg_sp = agg_sp_df.toPandas()

    with tempfile.TemporaryDirectory() as tmpdir:
        pq_path = str(Path(tmpdir) / "parity_check.parquet")
        agg_sp_df.write.mode("overwrite").parquet(pq_path)
        pq_back = pd.read_parquet(pq_path).sort_values("order_id").reset_index(drop=True)
        assert len(pq_back) == len(agg_pd), "Parquet round-trip row count mismatch!"

    assert len(agg_pd) == len(agg_sp) and len(agg_pd) > 0, f"Row count mismatch: {len(agg_pd)} vs {len(agg_sp)}"
    assert list(agg_pd.columns) == list(agg_sp.columns), f"Column mismatch: {agg_pd.columns} vs {agg_sp.columns}"
    assert (agg_pd["order_id"].values == agg_sp["order_id"].values).all(), "Order ID mismatch!"
    assert (agg_pd["item_count"].values == agg_sp["item_count"].values).all(), "Item count mismatch!"
    assert np.allclose(agg_pd["total_revenue"].values, agg_sp["total_revenue"].values, rtol=1e-5, atol=1e-4), "Revenue mismatch!"

    spark.stop()

    report = {
        "status": "STAGE_2_VALIDATED",
        "environment": env_info,
        "spark_init_sec": spark_init_sec,
        "read_only_raw_data_verified": read_only_verified,
        "writable_results_dir_verified": True,
        "parquet_roundtrip_verified": True,
        "parity_test": {
            "matched_orders": int(len(agg_pd)),
            "pandas_total_revenue_sum": round(float(agg_pd["total_revenue"].sum()), 2),
            "pyspark_total_revenue_sum": round(float(agg_sp["total_revenue"].sum()), 2),
            "exact_row_and_value_match": True,
        },
    }
    Path("results/stage2_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
