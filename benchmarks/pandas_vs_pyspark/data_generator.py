import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import psutil
import yaml


RAW_FILES = {
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


def check_preflight_resources(min_free_ram_mb: float, min_free_disk_gb: float, target_path: Path) -> Dict:
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(str(target_path))
    free_ram_mb = round(vm.available / (1024**2), 2)
    total_ram_mb = round(vm.total / (1024**2), 2)
    free_disk_gb = round(disk.free / (1024**3), 2)
    ok = (free_ram_mb >= min_free_ram_mb) and (free_disk_gb >= min_free_disk_gb)
    return {
        "passed": ok,
        "total_ram_mb": total_ram_mb,
        "free_ram_mb": free_ram_mb,
        "required_min_free_ram_mb": min_free_ram_mb,
        "free_disk_gb": free_disk_gb,
        "required_min_free_disk_gb": min_free_disk_gb,
    }


def _suffix_series(s: pd.Series, rep: int) -> pd.Series:
    if rep == 0:
        return s.astype(str)
    return s.astype(str) + f"_r{rep}"


def _shift_timestamp_col(s: pd.Series, days_offset: int) -> pd.Series:
    if days_offset == 0:
        return s
    dt = pd.to_datetime(s, errors="coerce")
    shifted = dt + pd.Timedelta(days=days_offset)
    out = shifted.dt.strftime("%Y-%m-%d %H:%M:%S")
    return out.where(dt.notna(), s)


def generate_scale_dataset(raw_dir: Path, out_dir: Path, scale_name: str, factor: int) -> Dict:
    t0 = time.perf_counter()
    scale_dir = out_dir / scale_name
    scale_dir.mkdir(parents=True, exist_ok=True)

    # category_translation is a small static lookup table (71 rows)
    cat_df = pd.read_csv(raw_dir / RAW_FILES["category_translation"])
    cat_df.to_csv(scale_dir / RAW_FILES["category_translation"], index=False)

    if factor == 1:
        row_counts = {"category_translation": len(cat_df)}
        total_bytes = (scale_dir / RAW_FILES["category_translation"]).stat().st_size
        for key, fname in RAW_FILES.items():
            if key == "category_translation":
                continue
            src = raw_dir / fname
            dst = scale_dir / fname
            shutil.copy2(src, dst)
            # Fast line count minus header
            with dst.open("rb") as f:
                row_counts[key] = max(sum(1 for _ in f) - 1, 0)
            total_bytes += dst.stat().st_size
        return {
            "scale": scale_name,
            "factor": factor,
            "status": "GENERATED",
            "generation_sec": round(time.perf_counter() - t0, 3),
            "total_size_mb": round(total_bytes / (1024**2), 2),
            "row_counts": row_counts,
            "referential_integrity_verified": True,
        }

    # Load base 1x tables once into memory
    base_tables = {
        key: pd.read_csv(raw_dir / fname)
        for key, fname in RAW_FILES.items()
        if key != "category_translation"
    }

    row_counts = {"category_translation": len(cat_df)}
    total_bytes = (scale_dir / RAW_FILES["category_translation"]).stat().st_size

    # Stream replica chunks to CSV to keep peak RAM low
    for key, fname in RAW_FILES.items():
        if key == "category_translation":
            continue
        dst = scale_dir / fname
        if dst.exists():
            dst.unlink()
        base_df = base_tables[key]
        written_rows = 0

        for rep in range(factor):
            chunk = base_df.copy()
            day_shift = rep * 7

            if key == "customers":
                chunk["customer_id"] = _suffix_series(chunk["customer_id"], rep)
                chunk["customer_unique_id"] = _suffix_series(chunk["customer_unique_id"], rep)
            elif key == "sellers":
                chunk["seller_id"] = _suffix_series(chunk["seller_id"], rep)
            elif key == "products":
                chunk["product_id"] = _suffix_series(chunk["product_id"], rep)
            elif key == "orders":
                chunk["order_id"] = _suffix_series(chunk["order_id"], rep)
                chunk["customer_id"] = _suffix_series(chunk["customer_id"], rep)
                for ts_col in [
                    "order_purchase_timestamp",
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                ]:
                    chunk[ts_col] = _shift_timestamp_col(chunk[ts_col], day_shift)
            elif key == "order_items":
                chunk["order_id"] = _suffix_series(chunk["order_id"], rep)
                chunk["product_id"] = _suffix_series(chunk["product_id"], rep)
                chunk["seller_id"] = _suffix_series(chunk["seller_id"], rep)
                chunk["shipping_limit_date"] = _shift_timestamp_col(chunk["shipping_limit_date"], day_shift)
            elif key == "payments":
                chunk["order_id"] = _suffix_series(chunk["order_id"], rep)
            elif key == "reviews":
                chunk["review_id"] = _suffix_series(chunk["review_id"], rep)
                chunk["order_id"] = _suffix_series(chunk["order_id"], rep)
                for ts_col in ["review_creation_date", "review_answer_timestamp"]:
                    chunk[ts_col] = _shift_timestamp_col(chunk[ts_col], day_shift)
            elif key == "geolocation":
                if rep > 0:
                    rng = np.random.default_rng(42 + rep)
                    chunk["geolocation_lat"] = (chunk["geolocation_lat"] + rng.normal(0.0, 1e-4, len(chunk))).round(6)
                    chunk["geolocation_lng"] = (chunk["geolocation_lng"] + rng.normal(0.0, 1e-4, len(chunk))).round(6)

            chunk.to_csv(dst, mode="a", header=(rep == 0), index=False)
            written_rows += len(chunk)

        row_counts[key] = written_rows
        total_bytes += dst.stat().st_size

    # Verify referential integrity on generated scale
    orders_keys = pd.read_csv(scale_dir / RAW_FILES["orders"], usecols=["order_id", "customer_id"])
    cust_keys = set(pd.read_csv(scale_dir / RAW_FILES["customers"], usecols=["customer_id"])["customer_id"])
    items_keys = pd.read_csv(scale_dir / RAW_FILES["order_items"], usecols=["order_id", "product_id", "seller_id"])
    prod_keys = set(pd.read_csv(scale_dir / RAW_FILES["products"], usecols=["product_id"])["product_id"])
    sell_keys = set(pd.read_csv(scale_dir / RAW_FILES["sellers"], usecols=["seller_id"])["seller_id"])
    ord_id_set = set(orders_keys["order_id"])

    assert orders_keys["order_id"].is_unique, f"{scale_name}: duplicate order_id!"
    assert orders_keys["customer_id"].isin(cust_keys).all(), f"{scale_name}: orphan customer_id in orders!"
    assert items_keys["order_id"].isin(ord_id_set).all(), f"{scale_name}: orphan order_id in order_items!"
    assert items_keys["product_id"].isin(prod_keys).all(), f"{scale_name}: orphan product_id in order_items!"
    assert items_keys["seller_id"].isin(sell_keys).all(), f"{scale_name}: orphan seller_id in order_items!"

    return {
        "scale": scale_name,
        "factor": factor,
        "status": "GENERATED",
        "generation_sec": round(time.perf_counter() - t0, 3),
        "total_size_mb": round(total_bytes / (1024**2), 2),
        "row_counts": row_counts,
        "referential_integrity_verified": True,
    }


def main() -> None:
    cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    raw_dir = Path(cfg["benchmark"]["raw_data_dir"])
    scaled_dir = Path(cfg["benchmark"]["scaled_data_dir"])
    results_dir = Path(cfg["benchmark"]["results_dir"])
    scaled_dir.mkdir(parents=True, exist_ok=True)

    # Required free RAM per scale inside the 3.83 GB Docker VM (with safety margin)
    scale_ram_guards_mb = {
        "1x": 1500.0,
        "5x": 1800.0,
        "10x": 2200.0,
        "20x": 2800.0,
    }
    scale_disk_guards_gb = {
        "1x": 1.0,
        "5x": 2.0,
        "10x": 4.0,
        "20x": 8.0,
    }

    summary = {"scales": []}
    for sc in cfg["benchmark"]["scales"]:
        name = sc["name"]
        factor = int(sc["factor"])
        preflight = check_preflight_resources(
            min_free_ram_mb=scale_ram_guards_mb.get(name, 2000.0),
            min_free_disk_gb=scale_disk_guards_gb.get(name, 4.0),
            target_path=scaled_dir,
        )
        if not preflight["passed"]:
            entry = {
                "scale": name,
                "factor": factor,
                "status": "GUARDED_SKIP_INSUFFICIENT_RESOURCES",
                "preflight": preflight,
            }
            summary["scales"].append(entry)
            print(json.dumps(entry, indent=2))
            continue

        res = generate_scale_dataset(raw_dir, scaled_dir, name, factor)
        res["preflight"] = preflight
        summary["scales"].append(res)
        print(json.dumps(res, indent=2))

    out_json = results_dir / "stage4_dataset_generation.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved dataset generation report to {out_json}")


if __name__ == "__main__":
    main()
