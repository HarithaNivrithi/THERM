"""
Simulation study: static workload descriptors for predicting inference duration,
energy and temperature change under workload-family shift.

Every workload, device profile and outcome is produced by the first-order RC
simulator in this file. Nothing here is a physical-device measurement, and the
descriptive thermal analyses are properties of the simulator, not of hardware.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. SYNTHETIC WORKLOAD FAMILIES  (20 operator-mix archetypes)
# ─────────────────────────────────────────────────────────────────────────────
# Operator mix columns: [conv, matmul, attention, activation, norm]
FAMILIES = {
    "vision_cnn":            np.array([.58, .04, .01, .24, .13]),
    "object_detector":       np.array([.47, .14, .02, .23, .14]),
    "audio_cnn":             np.array([.34, .16, .02, .28, .20]),
    "mlp":                   np.array([.02, .71, .01, .12, .14]),
    "recurrent":             np.array([.03, .42, .24, .17, .14]),
    "transformer":           np.array([.01, .46, .34, .09, .10]),
    "vision_transformer":    np.array([.22, .31, .28, .13, .06]),
    "diffusion":             np.array([.29, .24, .25, .14, .08]),
    "sparse_cnn":            np.array([.52, .06, .01, .30, .11]),
    "quantized_mlp":         np.array([.01, .78, .01, .11, .09]),
    "semantic_segmentation": np.array([.52, .12, .01, .25, .10]),
    "super_resolution":      np.array([.64, .08, .01, .19, .08]),
    "pose_estimation":       np.array([.43, .20, .02, .25, .10]),
    "video_action":          np.array([.28, .30, .12, .20, .10]),
    "speech_transformer":    np.array([.01, .30, .46, .12, .11]),
    "keyword_spotter":       np.array([.54, .10, .01, .27, .08]),
    "graph_neural_network":  np.array([.03, .39, .04, .42, .12]),
    "graph_transformer":     np.array([.02, .29, .42, .15, .12]),
    "recommendation_mlp":    np.array([.01, .69, .01, .17, .12]),
    "multimodal_fusion":     np.array([.15, .37, .26, .13, .09]),
}
OP_NAMES = ("conv", "matmul", "attention", "activation", "norm")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DEVICE PROFILES  (5 virtual devices)
# ─────────────────────────────────────────────────────────────────────────────
DEVICE_PLANT = {
    "cpu":        {"throughput": 78.,  "bandwidth": 1500., "idle_w": 1.0, "compute_w": 5.4,
                   "memory_w": 1.0, "r": 2.6, "c": 24., "mix": [1.0, 1.0, 1.2, .95, 1.0]},
    "gpu":        {"throughput": 235., "bandwidth": 3900., "idle_w": 1.8, "compute_w": 9.4,
                   "memory_w": 1.7, "r": 2.1, "c": 31., "mix": [.72, .80, .90, .95, 1.18]},
    "npu":        {"throughput": 205., "bandwidth": 2300., "idle_w": .8,  "compute_w": 4.0,
                   "memory_w": 1.2, "r": 3.0, "c": 19., "mix": [.62, .92, 1.65, .75, 1.05]},
    "mobile_cpu": {"throughput": 32.,  "bandwidth": 800.,  "idle_w": 0.4, "compute_w": 2.8,
                   "memory_w": 0.6, "r": 3.8, "c": 12., "mix": [1.0, 1.0, 1.1, .95, 1.0]},
    "dsp":        {"throughput": 160., "bandwidth": 2000., "idle_w": 0.6, "compute_w": 3.2,
                   "memory_w": 0.9, "r": 2.8, "c": 16., "mix": [.55, .88, 1.40, .80, 1.10]},
}
TAU = {d: p["r"] * p["c"] for d, p in DEVICE_PLANT.items()}
CONTEXTS = [(20., 27.), (30., 48.), (40., 64.), (25., 38.), (35., 55.)]  # (ambient, start)
SENSOR_NOISE_C = .22
REGIMES = ("transient", "transition", "saturation")
TARGETS = ("duration_s", "energy_j", "temp_rise_c")
POSITIVE_TARGETS = ("duration_s", "energy_j")


def _regime_from_ratio(ratio):
    ratio = np.asarray(ratio, dtype=float)
    return np.where(ratio < .3, "transient", np.where(ratio > 2.0, "saturation", "transition"))


# ─────────────────────────────────────────────────────────────────────────────
# 3. WORKLOAD GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def _make_workloads(seed: int, per_family: int = 72) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for family, center in FAMILIES.items():
        for index in range(per_family):
            mix = rng.dirichlet(np.maximum(center * 65, .35))
            ops = float(np.exp(rng.uniform(np.log(180), np.log(9000))))
            params = float(ops * np.exp(rng.normal(-1.35, .45)))
            bytes_per_op = float(np.exp(rng.normal(-4.9 + .7 * mix[2] + .35 * mix[1], .48)))
            memory = float(max(8, ops * bytes_per_op))
            depth = float(max(3, rng.lognormal(np.log(15 + 70 * mix[2] + 25 * mix[0]), .25)))
            # Higher values mean a greater zero fraction. Only the sparse family
            # draws from a high-sparsity distribution; quantization is not modelled.
            sparsity = float(np.clip(rng.beta(8, 2) if "sparse" in family else rng.beta(2, 8), 0.02, 0.98))
            rows.append({
                "workload_id": f"{family}-{index:03d}", "family": family,
                "ops_m": ops, "parameters_m": params, "memory_mb": memory,
                "graph_depth": depth, "sparsity": sparsity,
                **{f"mix_{name}": float(v) for name, v in zip(OP_NAMES, mix)}
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 4. THERMAL PLANT
# ─────────────────────────────────────────────────────────────────────────────
def _plant_observation(workload: dict, device_kind: str, ambient: float,
                       start_temp: float, rng: np.random.Generator) -> dict:
    p = DEVICE_PLANT[device_kind]
    mix = np.array([workload[f"mix_{name}"] for name in OP_NAMES])
    weighted_rate = float(np.dot(mix, p["mix"]))

    compute_s = workload["ops_m"] / p["throughput"] * weighted_rate
    compute_s *= (1.0 - 0.45 * workload["sparsity"])
    mem_s = workload["memory_mb"] / p["bandwidth"] * (1 + .35 * mix[2] + .12 * mix[4])
    duration = (compute_s + mem_s) * float(rng.lognormal(0, .035))

    compute_load = min(1.0, max(.08, .35 + .50 * weighted_rate))
    memory_load = min(1.0, workload["memory_mb"] / max(1, duration * p["bandwidth"]))
    power = p["idle_w"] + p["compute_w"] * compute_load + p["memory_w"] * memory_load
    power *= float(rng.normal(1, .025))

    tau = p["r"] * p["c"]
    decay = math.exp(-duration / tau)
    ideal_final = ambient + (start_temp - ambient) * decay + power * p["r"] * (1 - decay)
    final_temp = ideal_final + float(rng.normal(0, SENSOR_NOISE_C))
    temp_rise = final_temp - start_temp

    ratio = duration / tau
    regime = "transient" if ratio < 0.3 else ("saturation" if ratio > 2.0 else "transition")

    # Net rise the package would reach at steady state from this initial condition.
    # Negative when the start temperature is above the new steady state.
    ss_delta = power * p["r"] - (start_temp - ambient)
    saturation_fraction = temp_rise / ss_delta if abs(ss_delta) > 1e-9 else float("nan")

    # A first-order RC response under constant power is monotone in time, so the
    # maximum temperature during the run is either the start or the end value.
    return {
        "device": device_kind, "ambient_c": ambient, "start_temp_c": start_temp,
        "duration_s": duration, "power_w": power, "energy_j": duration * power,
        "final_temp_c": final_temp, "peak_temp_c": max(start_temp, final_temp),
        "temp_rise_c": temp_rise,
        "ideal_final_temp_c": ideal_final, "ideal_peak_temp_c": max(start_temp, ideal_final),
        "ideal_temp_rise_c": ideal_final - start_temp,
        "steady_state_rise_c": power * p["r"],
        "tau_s": tau, "duration_over_tau": ratio,
        "thermal_regime": regime, "saturation_fraction": saturation_fraction,
        "ss_temp_delta": ss_delta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def make_dataset(seed: int = 42, per_family: int = 72, repeats: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 101)
    rows = []
    for w in _make_workloads(seed, per_family):
        for device in DEVICE_PLANT:
            for ambient, start in CONTEXTS:
                for repeat in range(repeats):
                    obs = _plant_observation(w, device, ambient, start, rng)
                    rows.append({**w, **obs, "repeat": repeat + 1})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 6. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def _feature_columns(kind: str) -> list[str]:
    devices = [f"device_{d}" for d in DEVICE_PLANT]
    context = ["ambient_c", "start_temp_c"] + devices
    if kind == "size_only":
        return ["log_ops", *context]
    if kind == "size_memory":
        return ["log_ops", "log_parameters", "log_memory", *context]
    if kind == "size_memory_no_thermal_state":
        return ["log_ops", "log_parameters", "log_memory", *devices]
    if kind == "size_memory_sparsity":
        return ["log_ops", "log_parameters", "log_memory", "sparsity", *context]
    mixes = [f"mix_{op}" for op in OP_NAMES]
    interactions = [f"ops_x_{op}" for op in OP_NAMES]
    interactions += [f"memory_x_{op}" for op in OP_NAMES]
    interactions += [f"device_{d}_x_{op}" for d in DEVICE_PLANT for op in OP_NAMES]
    return ["log_ops", "log_parameters", "log_memory", "log_depth", "sparsity",
            *mixes, *context, *interactions]


def _design(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    out = frame.copy()
    out["log_ops"] = np.log1p(out.ops_m)
    out["log_parameters"] = np.log1p(out.parameters_m)
    out["log_memory"] = np.log1p(out.memory_mb)
    out["log_depth"] = np.log1p(out.graph_depth)
    for device in DEVICE_PLANT:
        out[f"device_{device}"] = (out.device == device).astype(float)
    for name in OP_NAMES:
        out[f"ops_x_{name}"] = out.log_ops * out[f"mix_{name}"]
        out[f"memory_x_{name}"] = out.log_memory * out[f"mix_{name}"]
        for device in DEVICE_PLANT:
            out[f"device_{device}_x_{name}"] = out[f"device_{device}"] * out[f"mix_{name}"]
    return out[columns].to_numpy(dtype=float)


FEATURE_CONFIGS = {
    "M0_size_only":           "size_only",
    "M1_size_memory":         "size_memory",
    "M1n_no_thermal_state":   "size_memory_no_thermal_state",
    "M2_size_memory_sparse":  "size_memory_sparsity",
    "M3_full_descriptors":    "full",
}


# ─────────────────────────────────────────────────────────────────────────────
# 7. MODELS
# ─────────────────────────────────────────────────────────────────────────────
def _ridge(x_tr, y_tr, alpha=1.0, **_):
    z = np.column_stack([np.ones(len(x_tr)), x_tr])
    pen = np.eye(z.shape[1]) * alpha
    pen[0, 0] = 0
    coef = np.linalg.solve(z.T @ z + pen, z.T @ y_tr)
    return lambda x: np.column_stack([np.ones(len(x)), x]) @ coef


def _lasso(x_tr, y_tr, alpha=0.01, **_):
    from sklearn.linear_model import Lasso
    m = Lasso(alpha=alpha, max_iter=5000, fit_intercept=True)
    m.fit(x_tr, y_tr)
    return m.predict


def _rf(x_tr, y_tr, **_):
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(n_estimators=120, max_features="sqrt",
                              min_samples_leaf=5, random_state=42, n_jobs=1)
    m.fit(x_tr, y_tr)
    return m.predict


def _gbm(x_tr, y_tr, **_):
    from sklearn.ensemble import GradientBoostingRegressor
    m = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, random_state=42)
    m.fit(x_tr, y_tr)
    return m.predict


def _gp(x_tr, y_tr, gp_max_rows=1000, **_):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    # Exact GP is O(n^3); fit on a fixed random subset of training rows.
    rng = np.random.default_rng(0)
    idx = rng.choice(len(x_tr), min(gp_max_rows, len(x_tr)), replace=False)
    m = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=0)
    m.fit(x_tr[idx], y_tr[idx])
    return m.predict


# RidgeLog is Ridge fitted on log(target); only defined for strictly positive targets.
MODEL_BUILDERS = {
    "Ridge":            _ridge,
    "RidgeLog":         _ridge,
    "Lasso":            _lasso,
    "RandomForest":     _rf,
    "GradientBoosting": _gbm,
    "GaussianProcess":  _gp,
}
LOG_TARGET_MODELS = {"RidgeLog"}


# ─────────────────────────────────────────────────────────────────────────────
# 8. CONFORMAL INTERVALS
# ─────────────────────────────────────────────────────────────────────────────
def _conformal_radius(scores: np.ndarray, coverage: float = .90) -> float:
    n = len(scores)
    rank = int(math.ceil((n + 1) * coverage))
    if rank > n:
        return float("inf")
    return float(np.partition(scores, rank - 1)[rank - 1])


def _group_radii(scores_cal, groups_cal, groups_test, global_radius, coverage, min_group):
    """Mondrian split conformal; groups with too few calibration rows use the global radius."""
    radii = np.full(len(groups_test), global_radius, dtype=float)
    fallback = np.zeros(len(groups_test), dtype=bool)
    for g in np.unique(groups_test):
        mask_c = groups_cal == g
        mask_t = groups_test == g
        if mask_c.sum() >= min_group:
            radii[mask_t] = _conformal_radius(scores_cal[mask_c], coverage)
        else:
            fallback[mask_t] = True
    return radii, fallback


# ─────────────────────────────────────────────────────────────────────────────
# 9. DESCRIPTIVE SIMULATOR ANALYSES
# ─────────────────────────────────────────────────────────────────────────────
def analyse_regime_device_counts(data: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(data.device, data.thermal_regime).reindex(columns=list(REGIMES), fill_value=0)
    counts = counts.reindex(list(DEVICE_PLANT))
    counts["total"] = counts.sum(axis=1)
    counts["tau_s"] = [TAU[d] for d in counts.index]
    stats = data.groupby("device").agg(max_duration_s=("duration_s", "max"),
                                       max_duration_over_tau=("duration_over_tau", "max"))
    return counts.join(stats).reset_index()


def analyse_energy_thermal_alignment(data: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Compare the energy-minimising device with several thermal criteria per task instance.

    A task instance is one workload in one context and repeat, run on all devices.
    """
    keys = ["workload_id", "ambient_c", "start_temp_c", "repeat"]
    devices = list(DEVICE_PLANT)
    wide = data.pivot_table(index=keys, columns="device",
                            values=["energy_j", "final_temp_c", "ideal_peak_temp_c",
                                    "steady_state_rise_c"]).reset_index()

    def block(col):
        return wide[col][devices].to_numpy()

    energy = block("energy_j")
    final = block("final_temp_c")
    peak = block("ideal_peak_temp_c")
    ss = block("steady_state_rise_c")
    n = len(energy)
    rows = np.arange(n)
    e_opt = energy.argmin(1)

    final_min, final_max = final.argmin(1), final.argmax(1)
    ss_min, ss_max = ss.argmin(1), ss.argmax(1)
    tol = 1e-9
    peak_min_val = peak.min(1, keepdims=True)
    peak_max_val = peak.max(1, keepdims=True)
    in_coolest_peak = peak[rows, e_opt] <= peak_min_val[:, 0] + tol
    n_at_max = (peak >= peak_max_val - tol).sum(1)
    unique_hottest_peak = (peak[rows, e_opt] >= peak_max_val[:, 0] - tol) & (n_at_max == 1)
    all_peaks_tied = (peak.max(1) - peak.min(1)) <= tol

    summary = {
        "task_instances": int(n),
        "final_temp_measured": {
            "energy_opt_is_coolest_share": float(np.mean(e_opt == final_min)),
            "energy_opt_not_coolest_share": float(np.mean(e_opt != final_min)),
            "energy_opt_is_hottest_share": float(np.mean(e_opt == final_max)),
            "energy_opt_is_hottest_count": int(np.sum(e_opt == final_max)),
        },
        "peak_temp_ideal": {
            "energy_opt_in_coolest_set_share": float(np.mean(in_coolest_peak)),
            "energy_opt_unique_hottest_share": float(np.mean(unique_hottest_peak)),
            "all_devices_tied_at_start_temp_share": float(np.mean(all_peaks_tied)),
        },
        "steady_state_rise": {
            "energy_opt_is_lowest_share": float(np.mean(e_opt == ss_min)),
            "energy_opt_is_highest_share": float(np.mean(e_opt == ss_max)),
        },
    }

    def freq(idx):
        return np.bincount(idx, minlength=len(devices)) / n

    choice = pd.DataFrame({
        "device": devices,
        "lowest_energy_share": freq(e_opt),
        "lowest_final_temp_share": freq(final_min),
        "highest_final_temp_share": freq(final_max),
        "lowest_steady_state_rise_share": freq(ss_min),
        "highest_steady_state_rise_share": freq(ss_max),
    })

    ctx = wide[["ambient_c", "start_temp_c"]].copy()
    ctx.columns = ["ambient_c", "start_temp_c"]
    ctx["aligned_final"] = e_opt == final_min
    ctx["energy_opt_in_coolest_peak"] = in_coolest_peak
    ctx["all_peaks_tied"] = all_peaks_tied
    by_context = ctx.groupby(["ambient_c", "start_temp_c"]).mean().reset_index()
    by_context.columns = ["ambient_c", "start_temp_c", "energy_opt_is_lowest_final_share",
                          "energy_opt_in_coolest_peak_share", "all_peaks_tied_share"]
    return summary, choice, by_context


def analyse_cooling(data: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cooling = data.temp_rise_c < 0
    cool_rows = data[cooling]
    worst = data.loc[data.temp_rise_c.idxmin()]
    summary = {
        "rows": int(len(data)),
        "net_cooling_share_measured": float(cooling.mean()),
        "net_cooling_share_noise_free": float((data.ideal_temp_rise_c < 0).mean()),
        "mean_temp_change_cooling_rows_c": float(data.temp_rise_c[cooling].mean()),
        "mean_temp_change_heating_rows_c": float(data.temp_rise_c[~cooling].mean()),
        "min_temp_change_c": float(worst.temp_rise_c),
        "min_temp_change_device": str(worst.device),
        "max_temp_change_c": float(data.temp_rise_c.max()),
    }
    by_context = data.assign(cooling=cooling).groupby(["ambient_c", "start_temp_c"]).agg(
        cooling_share=("cooling", "mean"),
        mean_temp_change_c=("temp_rise_c", "mean"),
        mean_steady_state_rise_c=("steady_state_rise_c", "mean"),
    ).reset_index()
    by_context["start_elevation_c"] = by_context.start_temp_c - by_context.ambient_c
    rate = -cool_rows.temp_rise_c / cool_rows.duration_s
    by_device = data.assign(cooling=cooling).groupby("device").agg(
        cooling_share=("cooling", "mean"),
        mean_temp_change_c=("temp_rise_c", "mean"),
        mean_steady_state_rise_c=("steady_state_rise_c", "mean"),
        mean_duration_s=("duration_s", "mean"),
        mean_power_w=("power_w", "mean"),
        mean_energy_j=("energy_j", "mean"),
    )
    by_device["mean_cooling_rate_c_per_s"] = rate.groupby(cool_rows.device).mean()
    by_device["median_cooling_rate_c_per_s"] = rate.groupby(cool_rows.device).median()
    return summary, by_context, by_device.reindex(list(DEVICE_PLANT)).reset_index()


def analyse_sign_rule(data: pd.DataFrame) -> dict:
    """Check the RC sign rule: net cooling iff P*R < (T0 - Ta)."""
    predicted = (data.ss_temp_delta < 0).to_numpy()
    measured = (data.temp_rise_c < 0).to_numpy()
    ideal = (data.ideal_temp_rise_c < 0).to_numpy()
    mismatch = predicted != measured
    near = np.abs(data.ideal_temp_rise_c.to_numpy()) < 3 * SENSOR_NOISE_C
    return {
        "agreement_with_measured_sign": float(np.mean(~mismatch)),
        "agreement_with_noise_free_sign": float(np.mean(predicted == ideal)),
        "mismatch_rows": int(mismatch.sum()),
        "mismatch_share_with_abs_true_change_below_3_sigma": float(near[mismatch].mean()) if mismatch.any() else float("nan"),
        "sensor_noise_sd_c": SENSOR_NOISE_C,
    }


def analyse_device_thermal_by_regime(data: pd.DataFrame) -> pd.DataFrame:
    """Mean temperature change per family x regime x device, with row counts."""
    g = data.groupby(["family", "thermal_regime", "device"]).agg(
        rows=("temp_rise_c", "size"), mean_temp_change_c=("temp_rise_c", "mean")).reset_index()
    return g


def analyse_regime_rank_support(data: pd.DataFrame) -> pd.DataFrame:
    """Check whether each regime contains every device before ranking devices."""
    rows = []
    for family, family_data in data.groupby("family"):
        stats = {}
        for regime in ("transient", "saturation"):
            group = family_data[family_data.thermal_regime == regime]
            stats[regime] = (group.groupby("device").size().to_dict(),
                             group.groupby("device").temp_rise_c.mean())
        t_counts, t_rises = stats["transient"]
        s_counts, s_rises = stats["saturation"]
        t_ok = all(t_counts.get(d, 0) > 0 for d in DEVICE_PLANT)
        s_ok = all(s_counts.get(d, 0) > 0 for d in DEVICE_PLANT)
        rows.append({
            "family": family,
            "transient_devices_present": len(t_counts),
            "saturation_devices_present": len(s_counts),
            "saturation_devices": ";".join(sorted(s_counts)),
            "transient_rows": int(sum(t_counts.values())),
            "saturation_rows": int(sum(s_counts.values())),
            "ranking_comparable": bool(t_ok and s_ok),
            "hottest_transient_among_present": str(t_rises.idxmax()) if len(t_rises) else None,
            "hottest_saturation_among_present": str(s_rises.idxmax()) if len(s_rises) else None,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 10. SPLITS
# ─────────────────────────────────────────────────────────────────────────────
def _family_holdout_split(data, fold_no, seed, train_per_family):
    names = tuple(FAMILIES)
    test_family = names[fold_no]
    cal_family = names[(fold_no + 1) % len(names)]
    train_families = [f for f in names if f not in (test_family, cal_family)]
    rng = np.random.default_rng(seed + (fold_no + 1) * 100003)
    ids = []
    for family in train_families:
        available = data.loc[data.family == family, "workload_id"].unique()
        ids.extend(rng.choice(available, size=min(train_per_family, len(available)), replace=False))
    train_idx = np.flatnonzero(data.workload_id.isin(ids).to_numpy())
    cal_idx = np.flatnonzero((data.family == cal_family).to_numpy())
    test_idx = np.flatnonzero((data.family == test_family).to_numpy())
    return train_idx, cal_idx, test_idx, test_family, cal_family


def _random_workload_split(data, fold_no, seed, n_train, n_cal, n_test):
    """Same split sizes as family holdout, but workloads drawn from all families."""
    rng = np.random.default_rng(seed + (fold_no + 1) * 100003 + 7)
    perm = rng.permutation(data.workload_id.unique())
    test_ids, cal_ids = perm[:n_test], perm[n_test:n_test + n_cal]
    train_ids = perm[n_test + n_cal:n_test + n_cal + n_train]
    wid = data.workload_id
    return (np.flatnonzero(wid.isin(train_ids).to_numpy()),
            np.flatnonzero(wid.isin(cal_ids).to_numpy()),
            np.flatnonzero(wid.isin(test_ids).to_numpy()),
            "random", "random")


# ─────────────────────────────────────────────────────────────────────────────
# 11. ONE FOLD
# ─────────────────────────────────────────────────────────────────────────────
def _run_fold(protocol, fold_no, split, arrays, x_all, model_names, feature_names,
              coverage, min_group, gp_max_rows):
    train_idx, cal_idx, test_idx, test_label, cal_label = split
    device = arrays["device"]
    true_regime = arrays["thermal_regime"]
    tau = arrays["tau_s"]
    workload = arrays["workload_id"]
    family = arrays["family"]
    detail, workload_rows, skipped, convergence = [], [], [], []
    t0 = time.time()

    for feat in feature_names:
        x = x_all[feat]
        mean = x[train_idx].mean(axis=0)
        scale = x[train_idx].std(axis=0)
        scale[scale < 1e-10] = 1.0
        x_tr, x_ca, x_te = [(x[i] - mean) / scale for i in (train_idx, cal_idx, test_idx)]

        for model in model_names:
            log_target = model in LOG_TARGET_MODELS
            dur_pred = None
            for target in TARGETS:
                if log_target and target not in POSITIVE_TARGETS:
                    continue
                y = arrays[target]
                y_tr, y_ca, y_te = y[train_idx], y[cal_idx], y[test_idx]
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        predict = MODEL_BUILDERS[model](x_tr, np.log(y_tr) if log_target else y_tr,
                                                        gp_max_rows=gp_max_rows)
                        p_ca, p_te = predict(x_ca), predict(x_te)
                except Exception as exc:  # recorded in metadata, never silently dropped
                    skipped.append({"protocol": protocol, "fold": fold_no + 1, "model": model,
                                    "feature_config": feat, "target": target, "error": repr(exc)})
                    continue
                n_conv = sum(1 for w in caught if "Convergence" in type(w.message).__name__)
                if n_conv:
                    convergence.append({"protocol": protocol, "fold": fold_no + 1, "model": model,
                                        "feature_config": feat, "target": target, "warnings": n_conv})
                if log_target:
                    p_ca, p_te = np.exp(p_ca), np.exp(p_te)
                if target == "duration_s":
                    dur_pred = (p_ca, p_te)

                res_ca = np.abs(y_ca - p_ca)
                res_te = np.abs(y_te - p_te)
                g_radius = _conformal_radius(res_ca, coverage)
                half = {"global": (np.full(len(test_idx), g_radius), np.zeros(len(test_idx), bool))}
                half["device_mondrian"] = _group_radii(res_ca, device[cal_idx], device[test_idx],
                                                       g_radius, coverage, min_group)
                if dur_pred is not None:
                    reg_ca = _regime_from_ratio(dur_pred[0] / tau[cal_idx])
                    reg_te = _regime_from_ratio(dur_pred[1] / tau[test_idx])
                    half["predicted_regime_mondrian"] = _group_radii(res_ca, reg_ca, reg_te,
                                                                     g_radius, coverage, min_group)
                if target in POSITIVE_TARGETS:
                    floor = float(np.quantile(y_tr, .05))
                    s_ca, s_te = np.maximum(p_ca, floor), np.maximum(p_te, floor)
                    q = _conformal_radius(res_ca / s_ca, coverage)
                    half["normalized"] = (q * s_te, np.zeros(len(test_idx), bool))

                strata = [("all", "all", np.ones(len(test_idx), bool))]
                strata += [("true_regime", r, true_regime[test_idx] == r) for r in REGIMES]
                strata += [("device", d, device[test_idx] == d) for d in DEVICE_PLANT]
                for method, (hw, fb) in half.items():
                    covered = res_te <= hw
                    for s_type, s_name, mask in strata:
                        n = int(mask.sum())
                        if n == 0:
                            continue
                        detail.append({
                            "protocol": protocol, "fold": fold_no + 1, "test_group": test_label,
                            "calibration_group": cal_label, "target": target, "model": model,
                            "feature_config": feat, "method": method,
                            "stratum_type": s_type, "stratum": s_name, "n": n,
                            "covered_sum": int(covered[mask].sum()),
                            "width_sum": float((2 * hw[mask]).sum()),
                            "fallback_rows": int(fb[mask].sum()),
                            "abs_err_sum": float(res_te[mask].sum()),
                            "sq_err_sum": float((res_te[mask] ** 2).sum()),
                        })

                per_wl = pd.DataFrame({"workload_id": workload[test_idx], "family": family[test_idx],
                                       "abs_err": res_te}).groupby(["workload_id", "family"]).abs_err.mean()
                for (wl, fam), err in per_wl.items():
                    workload_rows.append({"protocol": protocol, "fold": fold_no + 1, "target": target,
                                          "model": model, "feature_config": feat,
                                          "workload_id": wl, "family": fam, "mae": float(err)})
    print(f"  done {protocol} fold {fold_no + 1} ({test_label}) in {time.time() - t0:.0f}s", flush=True)
    return detail, workload_rows, skipped, convergence


# ─────────────────────────────────────────────────────────────────────────────
# 12. AGGREGATION AND BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────
def _aggregate(detail: pd.DataFrame) -> pd.DataFrame:
    keys = ["protocol", "target", "model", "feature_config", "method", "stratum_type", "stratum"]
    s = detail.groupby(keys)[["n", "covered_sum", "width_sum", "fallback_rows",
                              "abs_err_sum", "sq_err_sum"]].sum().reset_index()
    s["coverage"] = s.covered_sum / s.n
    s["mean_width"] = s.width_sum / s.n
    s["mae"] = s.abs_err_sum / s.n
    s["rmse"] = np.sqrt(s.sq_err_sum / s.n)
    s["fallback_share"] = s.fallback_rows / s.n
    fold = detail[detail.stratum_type == "all"].assign(
        fold_cov=lambda d: d.covered_sum / d.n, fold_mae=lambda d: d.abs_err_sum / d.n)
    spread = fold.groupby(keys).agg(fold_coverage_min=("fold_cov", "min"),
                                    fold_coverage_max=("fold_cov", "max"),
                                    fold_mae_sd=("fold_mae", "std")).reset_index()
    return s.merge(spread, on=keys, how="left").drop(columns=["covered_sum", "width_sum",
                                                               "abs_err_sum", "sq_err_sum"])


def paired_bootstrap(werr, target, base, comp, protocol="family_holdout", samples=2000, seed=0):
    """Workload-level paired bootstrap of MAE(base) - MAE(comp); positive favours comp."""
    sub = werr[(werr.protocol == protocol) & (werr.target == target)]
    a = sub[(sub.model == base[0]) & (sub.feature_config == base[1])].groupby("workload_id").mae.mean()
    b = sub[(sub.model == comp[0]) & (sub.feature_config == comp[1])].groupby("workload_id").mae.mean()
    gain = (a - b).dropna().to_numpy()
    if len(gain) == 0:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.choice(gain, (samples, len(gain)), replace=True).mean(1)
    return {"target": target, "protocol": protocol,
            "baseline": f"{base[0]}/{base[1]}", "comparison": f"{comp[0]}/{comp[1]}",
            "baseline_mae": float(a.mean()), "comparison_mae": float(b.mean()),
            "mean_gain": float(gain.mean()), "ci_low": float(np.quantile(draws, .025)),
            "ci_high": float(np.quantile(draws, .975)), "workloads": int(len(gain))}


def protocol_bootstrap(werr, target, model, feat, samples=2000, seed=0):
    """Unpaired workload bootstrap of MAE(random split) - MAE(family holdout)."""
    sub = werr[(werr.target == target) & (werr.model == model) & (werr.feature_config == feat)]
    fam = sub[sub.protocol == "family_holdout"].mae.to_numpy()
    rnd = sub[sub.protocol == "random_workload"].mae.to_numpy()
    if len(fam) == 0 or len(rnd) == 0:
        return None
    rng = np.random.default_rng(seed)
    diff = (rng.choice(rnd, (samples, len(rnd))).mean(1) - rng.choice(fam, (samples, len(fam))).mean(1))
    return {"target": target, "model": model, "feature_config": feat,
            "family_holdout_mae": float(fam.mean()), "random_split_mae": float(rnd.mean()),
            "random_minus_family": float(rnd.mean() - fam.mean()),
            "ci_low": float(np.quantile(diff, .025)), "ci_high": float(np.quantile(diff, .975)),
            "family_test_units": int(len(fam)), "random_test_units": int(len(rnd))}


# ─────────────────────────────────────────────────────────────────────────────
# 13. MAIN STUDY RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float))


def run_study(seed, per_family, repeats, out, train_per_family, models, features,
              random_models, random_features, coverage, min_group, gp_max_rows, jobs):
    for name in (*models, *random_models):
        if name not in MODEL_BUILDERS:
            raise ValueError(f"Unknown model {name}")
    for name in (*features, *random_features):
        if name not in FEATURE_CONFIGS:
            raise ValueError(f"Unknown feature config {name}")
    started = time.time()
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Generating dataset …", flush=True)
    data = make_dataset(seed, per_family, repeats)
    n_workloads = data.workload_id.nunique()
    print(f"  {n_workloads} workloads × {len(DEVICE_PLANT)} devices × {len(CONTEXTS)} contexts × "
          f"{repeats} repeats = {len(data)} rows", flush=True)

    print("Descriptive simulator analyses …", flush=True)
    analyse_regime_device_counts(data).to_csv(out_path / "regime_device_counts.csv", index=False)
    align, choice, align_ctx = analyse_energy_thermal_alignment(data)
    _write_json(out_path / "energy_thermal_alignment.json", align)
    choice.to_csv(out_path / "device_choice_frequencies.csv", index=False)
    align_ctx.to_csv(out_path / "energy_thermal_alignment_by_context.csv", index=False)
    cool, cool_ctx, cool_dev = analyse_cooling(data)
    _write_json(out_path / "cooling_summary.json", cool)
    cool_ctx.to_csv(out_path / "cooling_by_context.csv", index=False)
    cool_dev.to_csv(out_path / "device_thermal_summary.csv", index=False)
    _write_json(out_path / "sign_rule_check.json", analyse_sign_rule(data))
    analyse_regime_rank_support(data).to_csv(out_path / "regime_rank_support.csv", index=False)
    analyse_device_thermal_by_regime(data).to_csv(out_path / "device_thermal_by_regime.csv", index=False)

    print("Building design matrices …", flush=True)
    all_features = list(dict.fromkeys([*features, *random_features]))
    x_all = {f: _design(data, _feature_columns(FEATURE_CONFIGS[f])) for f in all_features}
    arrays = {c: data[c].to_numpy() for c in ("device", "thermal_regime", "workload_id", "family")}
    arrays.update({c: data[c].to_numpy(dtype=float) for c in ("tau_s", *TARGETS)})

    n_families = len(FAMILIES)
    n_train = train_per_family * (n_families - 2)
    tasks = []
    for fold_no in range(n_families):
        split = _family_holdout_split(data, fold_no, seed, train_per_family)
        tasks.append(("family_holdout", fold_no, split, models, features))
    if random_models and random_features:
        for fold_no in range(n_families):
            split = _random_workload_split(data, fold_no, seed, n_train, per_family, per_family)
            tasks.append(("random_workload", fold_no, split, random_models, random_features))
    split_sizes = sorted({(p, len(s[0]), len(s[1]), len(s[2])) for p, _, s, _, _ in tasks})
    print(f"Running {len(tasks)} folds on {jobs} worker(s); split sizes (rows) {split_sizes}", flush=True)

    results = Parallel(n_jobs=jobs, verbose=0)(
        delayed(_run_fold)(protocol, fold_no, split, arrays, x_all, list(m), list(f),
                           coverage, min_group, gp_max_rows)
        for protocol, fold_no, split, m, f in tasks)

    detail = pd.DataFrame([r for d, _, _, _ in results for r in d])
    werr = pd.DataFrame([r for _, w, _, _ in results for r in w])
    skipped = [r for _, _, s, _ in results for r in s]
    convergence = pd.DataFrame([r for _, _, _, c in results for r in c],
                               columns=["protocol", "fold", "model", "feature_config", "target", "warnings"])
    convergence.to_csv(out_path / "convergence_warnings.csv", index=False)
    fits_with_warnings = convergence.groupby("model").size().to_dict()
    detail.to_csv(out_path / "fold_strata_detail.csv", index=False)
    werr.to_csv(out_path / "workload_errors.csv", index=False)
    summary = _aggregate(detail)
    summary.to_csv(out_path / "prediction_summary.csv", index=False)
    fam = detail[(detail.protocol == "family_holdout") & (detail.stratum_type == "all")
                 & (detail.method == "global")].copy()
    fam["mae"] = fam.abs_err_sum / fam.n
    fam["coverage"] = fam.covered_sum / fam.n
    fam[["target", "model", "feature_config", "fold", "test_group", "calibration_group",
         "n", "mae", "coverage"]].to_csv(out_path / "family_fold_metrics.csv", index=False)

    comparisons = []
    for target in TARGETS:
        pairs = [
            (("GradientBoosting", "M1_size_memory"), ("GradientBoosting", "M2_size_memory_sparse")),
            (("GradientBoosting", "M2_size_memory_sparse"), ("GradientBoosting", "M3_full_descriptors")),
            (("Ridge", "M3_full_descriptors"), ("GradientBoosting", "M3_full_descriptors")),
            (("Ridge", "M1_size_memory"), ("RidgeLog", "M1_size_memory")),
            (("RidgeLog", "M3_full_descriptors"), ("GradientBoosting", "M3_full_descriptors")),
            (("GradientBoosting", "M1n_no_thermal_state"), ("GradientBoosting", "M1_size_memory")),
            (("Ridge", "M1n_no_thermal_state"), ("Ridge", "M1_size_memory")),
        ]
        for base, comp in pairs:
            r = paired_bootstrap(werr, target, base, comp, seed=seed)
            if r:
                comparisons.append(r)
    pd.DataFrame(comparisons).to_csv(out_path / "bootstrap_comparisons.csv", index=False)
    proto = [protocol_bootstrap(werr, t, m, f, seed=seed)
             for t in TARGETS for m in random_models for f in random_features]
    pd.DataFrame([p for p in proto if p]).to_csv(out_path / "protocol_comparison.csv", index=False)

    meta = {
        "status": "SYNTHETIC SIMULATION ONLY - not physical-device measurements",
        "seed": seed, "per_family": per_family, "repeats": repeats,
        "families": list(FAMILIES), "devices": list(DEVICE_PLANT), "contexts": CONTEXTS,
        "tau_s": TAU, "sensor_noise_sd_c": SENSOR_NOISE_C,
        "total_workloads": int(n_workloads), "total_rows": int(len(data)),
        "models": list(models), "feature_configs": {f: len(_feature_columns(FEATURE_CONFIGS[f]))
                                                     for f in all_features},
        "random_protocol_models": list(random_models), "random_protocol_features": list(random_features),
        "training_workloads_per_training_family": train_per_family,
        "training_workloads_per_fold": n_train,
        "split_sizes_rows": [list(s) for s in split_sizes],
        "coverage_target": coverage, "mondrian_min_calibration_rows": min_group,
        "gp_max_training_rows": gp_max_rows,
        "skipped_fits": skipped,
        "fits_with_convergence_warnings_by_model": fits_with_warnings,
        "runtime_s": round(time.time() - started, 1),
        "calibration_protocol": "family holdout: the next family in list order is the calibration family; "
                                "random_workload: 72 random workloads; same train/cal/test sizes",
    }
    _write_json(out_path / "metadata.json", meta)
    print(f"\nSaved outputs to {out_path} in {meta['runtime_s']} s; skipped fits: {len(skipped)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--per-family", type=int, default=72)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--train-workloads-per-family", type=int, default=24)
    parser.add_argument("--models", nargs="+", default=list(MODEL_BUILDERS))
    parser.add_argument("--features", nargs="+", default=list(FEATURE_CONFIGS))
    parser.add_argument("--random-models", nargs="*", default=["Ridge", "RidgeLog", "GradientBoosting"])
    parser.add_argument("--random-features", nargs="*", default=["M1_size_memory", "M3_full_descriptors"])
    parser.add_argument("--coverage", type=float, default=.90)
    parser.add_argument("--mondrian-min-rows", type=int, default=30)
    parser.add_argument("--gp-max-rows", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    run_study(args.seed, args.per_family, args.repeats, args.out, args.train_workloads_per_family,
              tuple(args.models), tuple(args.features), tuple(args.random_models),
              tuple(args.random_features), args.coverage, args.mondrian_min_rows,
              args.gp_max_rows, args.jobs)
