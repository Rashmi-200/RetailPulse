import csv
import gc
import json
import os
import resource
import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import psutil


@dataclass
class WorkloadRunMetric:
    engine: str
    scale: str
    run_type: str  # "cold" or "warm"
    run_index: int
    workload_id: str
    workload_name: str
    input_rows: int
    output_rows: int
    input_bytes: int
    spark_init_sec: float
    execution_sec: float
    total_sec: float
    throughput_rows_per_sec: float
    rss_before_mb: float
    rss_after_mb: float
    rss_delta_mb: float
    peak_process_rss_mb: float
    cpu_time_sec: float
    cpu_utilization_pct: float
    status: str = "SUCCESS"
    notes: str = ""


class MetricsCollector:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.records: List[WorkloadRunMetric] = []
        self._proc = psutil.Process(os.getpid())

    def _current_rss_mb(self) -> float:
        return self._proc.memory_info().rss / (1024 * 1024)

    def _peak_rss_mb(self) -> float:
        # Linux ru_maxrss is in KiB
        ru_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return max(self._current_rss_mb(), ru_kb / 1024.0)

    @contextmanager
    def measure(
        self,
        *,
        engine: str,
        scale: str,
        run_type: str,
        run_index: int,
        workload_id: str,
        workload_name: str,
        input_rows: int,
        input_bytes: int,
        spark_init_sec: float = 0.0,
        notes: str = "",
    ):
        gc.collect()
        rss_before = self._current_rss_mb()
        cpu_before = self._proc.cpu_times()
        t0 = time.perf_counter()
        ctx: Dict[str, Any] = {"output_rows": 0, "status": "SUCCESS", "notes": notes}
        try:
            yield ctx
        finally:
            exec_sec = max(time.perf_counter() - t0, 1e-6)
            cpu_after = self._proc.cpu_times()
            rss_after = self._current_rss_mb()
            peak_rss = self._peak_rss_mb()
            cpu_time = max(
                (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system),
                0.0,
            )
            num_cpus = os.cpu_count() or 1
            cpu_pct = min((cpu_time / (exec_sec * num_cpus)) * 100.0, 100.0)
            out_rows = int(ctx.get("output_rows", 0))
            throughput = round(input_rows / exec_sec, 2) if input_rows > 0 else 0.0
            total_sec = exec_sec + spark_init_sec

            self.records.append(
                WorkloadRunMetric(
                    engine=engine,
                    scale=scale,
                    run_type=run_type,
                    run_index=run_index,
                    workload_id=workload_id,
                    workload_name=workload_name,
                    input_rows=int(input_rows),
                    output_rows=out_rows,
                    input_bytes=int(input_bytes),
                    spark_init_sec=round(spark_init_sec, 4),
                    execution_sec=round(exec_sec, 4),
                    total_sec=round(total_sec, 4),
                    throughput_rows_per_sec=throughput,
                    rss_before_mb=round(rss_before, 2),
                    rss_after_mb=round(rss_after, 2),
                    rss_delta_mb=round(rss_after - rss_before, 2),
                    peak_process_rss_mb=round(peak_rss, 2),
                    cpu_time_sec=round(cpu_time, 4),
                    cpu_utilization_pct=round(cpu_pct, 2),
                    status=str(ctx.get("status", "SUCCESS")),
                    notes=str(ctx.get("notes", "")),
                )
            )

    def summarize(self) -> List[Dict[str, Any]]:
        grouped: Dict[tuple, List[WorkloadRunMetric]] = {}
        for r in self.records:
            grouped.setdefault((r.engine, r.scale, r.workload_id, r.workload_name, r.run_type), []).append(r)

        summary_rows: List[Dict[str, Any]] = []
        for (engine, scale, wid, wname, rtype), items in grouped.items():
            times = [x.execution_sec for x in items]
            tputs = [x.throughput_rows_per_sec for x in items]
            mems = [x.peak_process_rss_mb for x in items]
            cpus = [x.cpu_utilization_pct for x in items]
            summary_rows.append(
                {
                    "engine": engine,
                    "scale": scale,
                    "run_type": rtype,
                    "workload_id": wid,
                    "workload_name": wname,
                    "repetitions": len(items),
                    "input_rows": items[0].input_rows,
                    "output_rows": items[-1].output_rows,
                    "spark_init_sec": items[0].spark_init_sec,
                    "exec_mean_sec": round(statistics.mean(times), 4),
                    "exec_median_sec": round(statistics.median(times), 4),
                    "exec_std_sec": round(statistics.stdev(times), 4) if len(times) > 1 else 0.0,
                    "throughput_mean_rows_per_sec": round(statistics.mean(tputs), 2),
                    "peak_rss_max_mb": round(max(mems), 2),
                    "cpu_util_mean_pct": round(statistics.mean(cpus), 2),
                }
            )
        return summary_rows

    def save(self, prefix: str = "stage3_smoke") -> Dict[str, str]:
        raw_dicts = [asdict(r) for r in self.records]
        summary_dicts = self.summarize()

        raw_json = self.results_dir / f"{prefix}_raw_metrics.json"
        raw_csv = self.results_dir / f"{prefix}_raw_metrics.csv"
        sum_json = self.results_dir / f"{prefix}_summary_metrics.json"
        sum_csv = self.results_dir / f"{prefix}_summary_metrics.csv"

        raw_json.write_text(json.dumps(raw_dicts, indent=2), encoding="utf-8")
        sum_json.write_text(json.dumps(summary_dicts, indent=2), encoding="utf-8")

        if raw_dicts:
            with raw_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(raw_dicts[0].keys()))
                writer.writeheader()
                writer.writerows(raw_dicts)

        if summary_dicts:
            with sum_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_dicts[0].keys()))
                writer.writeheader()
                writer.writerows(summary_dicts)

        return {
            "raw_json": str(raw_json),
            "raw_csv": str(raw_csv),
            "summary_json": str(sum_json),
            "summary_csv": str(sum_csv),
        }
