"""Times dataset generation, model fitting, prediction and interval construction on this machine.

Uses fold 1 of the family-holdout protocol from simulation_study.py. Run it while no other
heavy job is running, otherwise the timings include contention.
"""
from __future__ import annotations

import ctypes
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

import simulation_study as sim

SEED = 20261028
MODELS = ("Ridge", "GradientBoosting")
FEATURES = ("M1_size_memory", "M3_full_descriptors")
SINGLE_QUERIES = 200


def machine_info() -> dict:
    info = {"os": platform.platform(), "python": platform.python_version(),
            "logical_cpus": os.cpu_count(), "cpu": platform.processor()}
    if os.name == "nt":
        import winreg
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            info["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        info["ram_gb"] = round(status.ullTotalPhys / 2 ** 30, 1)
    return info


def main() -> None:
    t0 = time.perf_counter()
    data = sim.make_dataset(SEED, 72, 3)
    generation_s = time.perf_counter() - t0
    train, cal, test, test_family, cal_family = sim._family_holdout_split(data, 0, SEED, 24)

    rows = []
    for feat in FEATURES:
        x = sim._design(data, sim._feature_columns(sim.FEATURE_CONFIGS[feat]))
        mean, scale = x[train].mean(0), x[train].std(0)
        scale[scale < 1e-10] = 1.0
        x_tr, x_ca, x_te = [(x[i] - mean) / scale for i in (train, cal, test)]
        for model in MODELS:
            for target in sim.TARGETS:
                y = data[target].to_numpy(dtype=float)
                t = time.perf_counter()
                predict = sim.MODEL_BUILDERS[model](x_tr, y[train])
                fit_s = time.perf_counter() - t

                t = time.perf_counter()
                p_ca = predict(x_ca)
                p_te = predict(x_te)
                batch_s = (time.perf_counter() - t) / 2

                t = time.perf_counter()
                radius = sim._conformal_radius(np.abs(y[cal] - p_ca), .90)
                lower, upper = p_te - radius, p_te + radius
                interval_s = time.perf_counter() - t

                single = []
                for i in range(SINGLE_QUERIES):
                    t = time.perf_counter()
                    predict(x_te[i:i + 1])
                    single.append(time.perf_counter() - t)
                single_ms = np.array(single) * 1e3
                rows.append({
                    "model": model, "feature_config": feat, "target": target,
                    "train_rows": int(len(train)), "fit_s": fit_s,
                    "batch_rows": int(len(test)), "batch_predict_s": batch_s,
                    "batch_us_per_row": batch_s / len(test) * 1e6,
                    "single_query_median_ms": float(np.median(single_ms)),
                    "single_query_p95_ms": float(np.quantile(single_ms, .95)),
                    "interval_construction_s": interval_s,
                })
                print(f"{model:16s} {feat:22s} {target:12s} fit {fit_s:7.2f}s  "
                      f"single {np.median(single_ms):7.3f} ms", flush=True)

    out = {"machine": machine_info(), "seed": SEED, "fold": 1, "test_family": test_family,
           "calibration_family": cal_family, "dataset_rows": int(len(data)),
           "dataset_generation_s": generation_s, "single_queries_per_fit": SINGLE_QUERIES,
           "timings": rows}
    Path("results").mkdir(exist_ok=True)
    Path("results/latency_benchmark.json").write_text(json.dumps(out, indent=2))
    print("Saved results/latency_benchmark.json")


if __name__ == "__main__":
    main()
