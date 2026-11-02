"""Builds Course_Based_Project_Report.docx from the files in results/.

Every reported number is read from results/ or from the configuration in simulation_study.py.
Sections that depend on outputs that do not exist yet are marked PENDING.
"""
from __future__ import annotations

import importlib.metadata as md
import inspect
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

import simulation_study as sim
from latency_benchmark import machine_info

R = Path("results")
FIG = Path("report_figures")
FIG.mkdir(exist_ok=True)
OUT = Path("Course_Based_Project_Report.docx")

TITLE = ("Reliable Pre-Deployment Prediction of Edge-AI Inference Time, Energy and Temperature "
         "Using Family-Holdout Evaluation and Regime-Aware Uncertainty Intervals")
PER_FAMILY, REPEATS, TRAIN_PER_FAMILY, SEED, COVERAGE, MIN_GROUP = 72, 3, 24, 20261028, .90, 30

# ───────────────────────────── data ─────────────────────────────
reg = pd.read_csv(R / "regime_device_counts.csv")
align = json.loads((R / "energy_thermal_alignment.json").read_text())
choice = pd.read_csv(R / "device_choice_frequencies.csv")
align_ctx = pd.read_csv(R / "energy_thermal_alignment_by_context.csv")
cool = json.loads((R / "cooling_summary.json").read_text())
cool_ctx = pd.read_csv(R / "cooling_by_context.csv")
dev = pd.read_csv(R / "device_thermal_summary.csv")
sign = json.loads((R / "sign_rule_check.json").read_text())
rank = pd.read_csv(R / "regime_rank_support.csv")
dtr = pd.read_csv(R / "device_thermal_by_regime.csv")

DEVICES = list(sim.DEVICE_PLANT)
FAMILIES = list(sim.FAMILIES)
N_ROWS = cool["rows"]
N_WORKLOADS = len(FAMILIES) * PER_FAMILY
N_TASKS = align["task_instances"]
assert N_ROWS == N_WORKLOADS * len(DEVICES) * len(sim.CONTEXTS) * REPEATS
N_TRAIN_ROWS = (len(FAMILIES) - 2) * TRAIN_PER_FAMILY * len(DEVICES) * len(sim.CONTEXTS) * REPEATS
N_FAMILY_ROWS = PER_FAMILY * len(DEVICES) * len(sim.CONTEXTS) * REPEATS

PRED = (R / "prediction_summary.csv").exists() and (R / "metadata.json").exists()
LAT = (R / "latency_benchmark.json").exists()
if PRED:
    ps = pd.read_csv(R / "prediction_summary.csv")
    proto = pd.read_csv(R / "protocol_comparison.csv")
    boot = pd.read_csv(R / "bootstrap_comparisons.csv")
    ffm = pd.read_csv(R / "family_fold_metrics.csv")
    conv = pd.read_csv(R / "convergence_warnings.csv")
    meta = json.loads((R / "metadata.json").read_text())
    detail_all = pd.read_csv(R / "fold_strata_detail.csv",
                             usecols=["protocol", "fold", "target", "model", "feature_config",
                                      "method", "stratum_type"])
    assert [N_TRAIN_ROWS, N_FAMILY_ROWS, N_FAMILY_ROWS] == meta["split_sizes_rows"][0][1:]
if LAT:
    lat = json.loads((R / "latency_benchmark.json").read_text())

DEV_LABEL = {"cpu": "cpu (edge CPU)", "gpu": "gpu (mobile GPU)", "npu": "npu (edge NPU)",
             "mobile_cpu": "mobile_cpu (mobile CPU)", "dsp": "dsp (embedded DSP)"}
FEAT_ORDER = list(sim.FEATURE_CONFIGS)
FEAT_SHORT = {"M0_size_only": "M0", "M1n_no_thermal_state": "M1n", "M1_size_memory": "M1",
              "M2_size_memory_sparse": "M2", "M3_full_descriptors": "M3"}
FEAT_DIM = {f: len(sim._feature_columns(k)) for f, k in sim.FEATURE_CONFIGS.items()}
MODEL_ORDER = list(sim.MODEL_BUILDERS)
MODEL_SHORT = {"Ridge": "Ridge", "RidgeLog": "RidgeLog", "Lasso": "Lasso", "RandomForest": "RF",
               "GradientBoosting": "GBM", "GaussianProcess": "GP"}
TARGETS = list(sim.TARGETS)
TARGET_LABEL = {"duration_s": "Inference duration", "energy_j": "Inference energy",
                "temp_rise_c": "Temperature change"}
UNIT = {"duration_s": "s", "energy_j": "J", "temp_rise_c": "°C"}
METHOD_LABEL = {"global": "Global split conformal", "device_mondrian": "Device-Mondrian",
                "predicted_regime_mondrian": "Predicted-regime Mondrian",
                "normalized": "Normalized (scaled residual)"}
REF = ("GradientBoosting", "M3_full_descriptors")


def pct(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def num(x, d=2):
    return f"{x:,.{d}f}"


def cfg(model, feat):
    return f"{MODEL_SHORT[model]}/{FEAT_SHORT[feat]}"


def pick(target, model, feat, method="global", stype="all", stratum="all", protocol="family_holdout"):
    if not PRED:
        return None
    m = ps[(ps.protocol == protocol) & (ps.target == target) & (ps.model == model)
           & (ps.feature_config == feat) & (ps.method == method)
           & (ps.stratum_type == stype) & (ps.stratum == stratum)]
    return None if m.empty else m.iloc[0]


def bootrow(target, base, comp):
    if not PRED:
        return None
    m = boot[(boot.target == target) & (boot.baseline == f"{base[0]}/{base[1]}")
             & (boot.comparison == f"{comp[0]}/{comp[1]}")]
    return None if m.empty else m.iloc[0]


# ───────────────────────────── derived descriptive values ─────────────────────────────
reg_tot = reg[["transient", "transition", "saturation"]].sum()
sat_devices = reg.loc[reg.saturation > 0, "device"].tolist()
n_comparable = int(rank.ranking_comparable.sum())
hot_transient = rank.hottest_transient_among_present.value_counts()
fm = align["final_temp_measured"]
pk = align["peak_temp_ideal"]
ssr = align["steady_state_rise"]
dsp_energy = float(choice.set_index("device").loc["dsp", "lowest_energy_share"])
npu_energy = float(choice.set_index("device").loc["npu", "lowest_energy_share"])
mean_ss = float(cool_ctx.mean_steady_state_rise_c.mean())
regime_dev = (dtr.assign(w=dtr.rows * dtr.mean_temp_change_c)
              .groupby(["thermal_regime", "device"])[["rows", "w"]].sum())
regime_dev["mean_temp_change_c"] = regime_dev.w / regime_dev.rows

# ───────────────────────────── derived prediction values ─────────────────────────────
best = {}
if PRED:
    glob = ps[(ps.protocol == "family_holdout") & (ps.stratum_type == "all") & (ps.method == "global")]
    for t in TARGETS:
        best[t] = glob[glob.target == t].sort_values("mae").iloc[0]
    fits_done = detail_all[(detail_all.method == "global") & (detail_all.stratum_type == "all")]
    n_fits_done = len(fits_done.drop_duplicates())
    n_skipped = len(meta["skipped_fits"])
    conv_by_model = conv.groupby("model").agg(fits=("warnings", "size"), warnings=("warnings", "sum"))


# ───────────────────────────── figures ─────────────────────────────
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
COL = {"train": "#4C72B0", "cal": "#DD8452", "test": "#C44E52"}


def _box(ax, x, y, w, h, text, fc, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec="#333333", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#333333"))


def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    for x, label in ((0.15, "INPUT LAYER"), (2.75, "SIMULATION LAYER"), (5.35, "LEARNING LAYER"),
                     (7.95, "EVALUATION & OUTPUT")):
        ax.text(x + 1.1, 6.2, label, ha="center", fontsize=9, fontweight="bold", color="#444444")
    _box(ax, .15, 4.5, 2.2, 1.3, f"Workload generator\n{len(FAMILIES)} families × {PER_FAMILY}\n"
                                 "Dirichlet operator mix,\nlog-uniform op count", "#E8F0FB")
    _box(ax, .15, 2.7, 2.2, 1.3, f"Device plant\n{len(DEVICES)} virtual devices\nthroughput, power,\n"
                                 "thermal R and C", "#E8F0FB")
    _box(ax, .15, 0.9, 2.2, 1.3, f"Operating contexts\n{len(sim.CONTEXTS)} (ambient, start) pairs\n"
                                 f"× {REPEATS} repeats", "#E8F0FB")
    _box(ax, 2.75, 3.3, 2.2, 1.9, "First-order RC\nthermal simulator\nT(t)=Ta+(T0−Ta)e^(−t/τ)\n"
                                  f"+PR(1−e^(−t/τ))\nnoise σ={sim.SENSOR_NOISE_C} °C", "#FDF2E3")
    _box(ax, 2.75, 1.1, 2.2, 1.4, f"Simulated dataset\n{N_ROWS:,} rows\nduration, power, energy,\n"
                                  "temperatures, regime", "#FDF2E3")
    _box(ax, 5.35, 4.8, 2.2, 1.1, "Descriptive audit\nregimes, cooling, sign rule,\n"
                                  "energy vs thermal, ranks", "#EAF5EA")
    _box(ax, 5.35, 3.4, 2.2, 1.0, "Feature engineering\nM0, M1n, M1, M2, M3", "#EAF5EA")
    _box(ax, 5.35, 2.0, 2.2, 1.0, f"Model training\n{len(MODEL_ORDER)} regressors × 3 targets", "#EAF5EA")
    _box(ax, 5.35, 0.5, 2.2, 1.1, f"Conformal calibration\n{int(COVERAGE * 100)}% intervals,\n4 methods",
         "#EAF5EA")
    _box(ax, 7.95, 3.0, 2.3, 2.0, f"Evaluation\nfamily holdout ({len(FAMILIES)} folds)\n"
                                  f"random control ({len(FAMILIES)} folds)\nMAE, RMSE, coverage\n"
                                  "paired bootstrap CIs", "#F3E8F7")
    _box(ax, 7.95, 0.6, 2.3, 1.7, "Outputs\nCSV / JSON results,\ntables, figures,\nthis report", "#F3E8F7")
    for y in (5.15, 3.35, 1.55):
        _arrow(ax, 2.35, y, 2.75, 4.25 if y > 2 else 3.6)
    _arrow(ax, 3.85, 3.3, 3.85, 2.5)
    _arrow(ax, 4.95, 1.8, 5.35, 5.3)
    _arrow(ax, 4.95, 1.8, 5.35, 3.9)
    _arrow(ax, 6.45, 3.4, 6.45, 3.0)
    _arrow(ax, 6.45, 2.0, 6.45, 1.6)
    _arrow(ax, 7.55, 1.05, 7.95, 3.4)
    _arrow(ax, 7.55, 5.35, 7.95, 4.6)
    _arrow(ax, 9.1, 3.0, 9.1, 2.3)
    path = FIG / "fig_architecture.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_methodology():
    steps = [
        "1. Define 20 workload families, 5 virtual devices, 5 thermal contexts",
        f"2. Generate {N_WORKLOADS:,} workloads ({PER_FAMILY} per family)",
        f"3. Simulate every workload × device × context × repeat  →  {N_ROWS:,} rows",
        "4. Audit the simulator: regimes, cooling, sign rule, energy vs thermal, rank support",
        "5. Build feature sets M0, M1n, M1, M2, M3 from static descriptors",
        f"6. Split: family holdout ({len(FAMILIES)} folds) and size-matched random control",
        f"7. Train {len(MODEL_ORDER)} regressors for duration, energy and temperature change",
        f"8. Calibrate {int(COVERAGE * 100)}% conformal intervals (global, device, predicted regime, normalized)",
        "9. Evaluate: MAE, RMSE, coverage by regime / device, paired bootstrap CIs",
        "10. Generate tables, figures and report directly from result files",
    ]
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) * 1.0 + .2)
    ax.axis("off")
    colors = ["#E8F0FB"] * 3 + ["#FDF2E3"] + ["#EAF5EA"] * 4 + ["#F3E8F7"] * 2
    for i, (s, c) in enumerate(zip(steps, colors)):
        y = len(steps) - 1 - i + .25
        _box(ax, .4, y, 9.2, .62, s, c, fs=9)
        if i < len(steps) - 1:
            _arrow(ax, 5, y, 5, y - .36)
    path = FIG / "fig_methodology.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_split():
    folds = [0, 1, len(FAMILIES) - 1]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for row, k in enumerate(folds):
        y = len(folds) - row
        cal = (k + 1) % len(FAMILIES)
        for j in range(len(FAMILIES)):
            role = "test" if j == k else ("cal" if j == cal else "train")
            ax.add_patch(plt.Rectangle((j, y), .92, .8, fc=COL[role], ec="white",
                                       hatch="///" if role == "train" else None, alpha=.9))
        ax.text(-.3, y + .4, f"Fold {k + 1}", ha="right", va="center", fontsize=9)
    for j, f in enumerate(FAMILIES):
        ax.text(j + .46, .9, f, rotation=70, ha="right", va="top", fontsize=6.5)
    ax.text(10, 4.25, "…folds 3–19 follow the same pattern…", ha="center", fontsize=8, style="italic")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COL["train"], hatch="///"),
               plt.Rectangle((0, 0), 1, 1, fc=COL["cal"]), plt.Rectangle((0, 0), 1, 1, fc=COL["test"])]
    ax.legend(handles, [f"Training: {TRAIN_PER_FAMILY} of {PER_FAMILY} workloads per family",
                        f"Calibration: next family, all {PER_FAMILY} workloads",
                        f"Test: held-out family, all {PER_FAMILY} workloads"],
              loc="upper center", bbox_to_anchor=(.5, 1.28), ncol=3, fontsize=8, frameon=False)
    ax.set_xlim(-2.2, len(FAMILIES))
    ax.set_ylim(-2.6, 4.5)
    ax.axis("off")
    path = FIG / "fig_split.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_rc_response():
    p = sim.DEVICE_PLANT["mobile_cpu"]
    power = float(dev.set_index("device").loc["mobile_cpu", "mean_power_w"])
    tau = p["r"] * p["c"]
    t = np.linspace(0, 5 * tau, 400)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for (ta, t0), c in zip([sim.CONTEXTS[0], sim.CONTEXTS[2]], ["#C44E52", "#4C72B0"]):
        temp = ta + (t0 - ta) * np.exp(-t / tau) + power * p["r"] * (1 - np.exp(-t / tau))
        ss = ta + power * p["r"]
        kind = "heats" if ss > t0 else "cools"
        ax.plot(t, temp, color=c, lw=2, label=f"Ta={ta:.0f} °C, T0={t0:.0f} °C: device {kind} toward {ss:.1f} °C")
        ax.axhline(ss, color=c, ls=":", lw=1)
    for m, lab in ((.3, "t/τ = 0.3"), (2.0, "t/τ = 2.0")):
        ax.axvline(m * tau, color="#777777", ls="--", lw=1)
        ax.text(m * tau, ax.get_ylim()[1] if False else 66, lab, fontsize=8, ha="left", color="#555555")
    ax.set_xlabel("Time since start of inference (s)")
    ax.set_ylabel("Package temperature (°C)")
    ax.set_title(f"mobile_cpu: R={p['r']} K/W, C={p['c']} J/K, τ={tau:.1f} s, P={power:.2f} W", fontsize=9)
    ax.legend(fontsize=8, loc="center right")
    path = FIG / "fig_rc_response.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_regimes():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    bottom = np.zeros(len(reg))
    for regime, c in zip(("transient", "transition", "saturation"), ("#8DB6E2", "#F2B26B", "#C44E52")):
        axes[0].bar(reg.device, reg[regime], bottom=bottom, color=c, label=regime)
        bottom += reg[regime].to_numpy()
    axes[0].set_ylabel("Simulated rows")
    axes[0].set_title("(a) Thermal regime of each run, by device")
    axes[0].legend(fontsize=8)
    axes[1].bar(reg.device, reg.max_duration_over_tau, color=["#C44E52" if v > 2 else "#8DB6E2"
                                                               for v in reg.max_duration_over_tau])
    axes[1].axhline(2.0, color="k", ls="--", lw=1)
    axes[1].text(-.4, 2.1, "saturation threshold t/τ = 2.0", fontsize=8)
    for i, v in enumerate(reg.max_duration_over_tau):
        axes[1].text(i, v + .08, f"{v:.2f}", ha="center", fontsize=8)
    axes[1].set_ylabel("Maximum duration / τ")
    axes[1].set_title("(b) Longest run relative to thermal time constant")
    for a in axes:
        a.tick_params(axis="x", rotation=20)
    path = FIG / "fig_regimes.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_cooling():
    labels = [f"{a:.0f}/{s:.0f}" for a, s in zip(cool_ctx.ambient_c, cool_ctx.start_temp_c)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar(labels, cool_ctx.cooling_share * 100, color="#4C72B0")
    for i, v in enumerate(cool_ctx.cooling_share * 100):
        axes[0].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8)
    axes[0].set_ylabel("Rows with net cooling (%)")
    axes[0].set_xlabel("Context: ambient / start temperature (°C)")
    axes[0].set_title("(a) Share of runs that end cooler than they started")
    x = np.arange(len(labels))
    axes[1].bar(x - .2, cool_ctx.start_elevation_c, .4, label="Start elevation T0 − Ta", color="#DD8452")
    axes[1].bar(x + .2, cool_ctx.mean_steady_state_rise_c, .4, label="Mean steady-state rise P·R",
                color="#55A868")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("°C")
    axes[1].set_xlabel("Context: ambient / start temperature (°C)")
    axes[1].set_title("(b) Why: start elevation versus mean P·R")
    axes[1].legend(fontsize=8)
    path = FIG / "fig_cooling.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_devices():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(dev))
    axes[0].bar(x - .2, dev.mean_steady_state_rise_c, .4, label="Mean P·R (steady-state rise)", color="#55A868")
    axes[0].bar(x + .2, dev.mean_temp_change_c, .4, label="Mean observed ΔT", color="#C44E52")
    axes[0].axhline(0, color="k", lw=.8)
    axes[0].set_xticks(x, dev.device, rotation=20)
    axes[0].set_ylabel("°C")
    axes[0].set_title("(a) Steady-state rise vs observed change")
    axes[0].legend(fontsize=8)
    axes[1].bar(dev.device, dev.mean_energy_j, color="#8172B2")
    for i, v in enumerate(dev.mean_energy_j):
        axes[1].text(i, v + 3, f"{v:.1f}", ha="center", fontsize=8)
    axes[1].set_ylabel("Mean simulated energy per inference (J)")
    axes[1].set_title("(b) Simulated energy per inference")
    axes[1].tick_params(axis="x", rotation=20)
    path = FIG / "fig_devices.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_energy_thermal():
    items = [("Energy-optimal = coolest\n(end temperature)", fm["energy_opt_is_coolest_share"]),
             ("Energy-optimal = hottest\n(end temperature)", fm["energy_opt_is_hottest_share"]),
             ("Energy-optimal in coolest set\n(true peak)", pk["energy_opt_in_coolest_set_share"]),
             ("Energy-optimal = unique hottest\n(true peak)", pk["energy_opt_unique_hottest_share"]),
             ("Energy-optimal = lowest P·R", ssr["energy_opt_is_lowest_share"]),
             ("All devices tied at\nstart temperature (peak)", pk["all_devices_tied_at_start_temp_share"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.35, 1]})
    vals = [v * 100 for _, v in items]
    colors = ["#8DB6E2", "#C44E52", "#55A868", "#C44E52", "#55A868", "#BBBBBB"]
    axes[0].barh([k for k, _ in items][::-1], vals[::-1], color=colors[::-1])
    for i, v in enumerate(vals[::-1]):
        axes[0].text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)
    axes[0].set_xlim(0, 110)
    axes[0].set_xlabel(f"Share of {N_TASKS:,} task instances (%)")
    axes[0].set_title("(a) The answer depends on the temperature definition")
    labels = [f"{a:.0f}/{s:.0f}" for a, s in zip(align_ctx.ambient_c, align_ctx.start_temp_c)]
    axes[1].plot(labels, align_ctx.energy_opt_is_lowest_final_share * 100, "o-", label="= coolest (end temp)")
    axes[1].plot(labels, align_ctx.energy_opt_in_coolest_peak_share * 100, "s-", label="in coolest set (peak)")
    axes[1].plot(labels, align_ctx.all_peaks_tied_share * 100, "^--", label="all peaks tied")
    axes[1].set_ylabel("%")
    axes[1].set_xlabel("Context: ambient / start (°C)")
    axes[1].set_title("(b) By thermal context")
    axes[1].legend(fontsize=8)
    path = FIG / "fig_energy_thermal.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_mae():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    fam = ps[(ps.protocol == "family_holdout") & (ps.stratum_type == "all") & (ps.method == "global")]
    for ax, t in zip(axes, TARGETS):
        for m in MODEL_ORDER:
            s = fam[(fam.target == t) & (fam.model == m)].set_index("feature_config").reindex(FEAT_ORDER)
            if s.mae.notna().any():
                ax.plot([FEAT_SHORT[f] for f in FEAT_ORDER], s.mae, "o-", label=MODEL_SHORT[m])
        if t != "temp_rise_c":
            ax.set_yscale("log")
        ax.set_title(f"{TARGET_LABEL[t]} MAE ({UNIT[t]})")
        ax.set_xlabel("Feature set")
    axes[0].legend(fontsize=7)
    path = FIG / "fig_mae.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_coverage_regime():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    regimes = list(sim.REGIMES)
    for ax, t in zip(axes, TARGETS):
        methods = [m for m in METHOD_LABEL if pick(t, *REF, method=m) is not None]
        w = .8 / len(methods)
        for i, m in enumerate(methods):
            vals = [getattr(pick(t, *REF, method=m, stype="true_regime", stratum=r), "coverage", np.nan)
                    for r in regimes]
            ax.bar(np.arange(3) + (i - (len(methods) - 1) / 2) * w, np.array(vals) * 100, w,
                   label=METHOD_LABEL[m])
        ax.axhline(COVERAGE * 100, color="k", ls="--", lw=1)
        ax.set_xticks(np.arange(3), regimes)
        ax.set_title(f"{TARGET_LABEL[t]} ({cfg(*REF)})")
    axes[0].set_ylabel("Empirical coverage (%)")
    axes[0].legend(fontsize=7, loc="lower left")
    path = FIG / "fig_coverage_regime.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_protocol():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, t in zip(axes, TARGETS):
        s = proto[proto.target == t]
        labels = [cfg(m, f) for m, f in zip(s.model, s.feature_config)]
        x = np.arange(len(s))
        ax.bar(x - .2, s.family_holdout_mae, .4, label="Family holdout", color=COL["test"])
        ax.bar(x + .2, s.random_split_mae, .4, label="Random split", color=COL["train"])
        ax.set_xticks(x, labels, rotation=30, fontsize=7)
        ax.set_title(f"{TARGET_LABEL[t]} MAE ({UNIT[t]})")
    axes[0].legend(fontsize=8)
    path = FIG / "fig_protocol.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_families():
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    for ax, t in zip(axes, TARGETS):
        s = ffm[(ffm.target == t) & (ffm.model == REF[0]) & (ffm.feature_config == REF[1])]
        s = s.set_index("test_group").reindex(FAMILIES)
        ax.barh(FAMILIES, s.mae, color="#4C72B0")
        ax.set_title(f"{TARGET_LABEL[t]} MAE ({UNIT[t]})")
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", labelsize=7)
    path = FIG / "fig_families.png"
    fig.savefig(path)
    plt.close(fig)
    return path


figs = {"architecture": fig_architecture(), "methodology": fig_methodology(), "split": fig_split(),
        "rc": fig_rc_response(), "regimes": fig_regimes(), "cooling": fig_cooling(),
        "devices": fig_devices(), "energy": fig_energy_thermal()}
if PRED:
    figs.update(mae=fig_mae(), coverage=fig_coverage_regime(), protocol=fig_protocol(),
                families=fig_families())

# ───────────────────────────── docx helpers ─────────────────────────────
doc = Document()
FONT = "Times New Roman"
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.54)
    s.top_margin = s.bottom_margin = Cm(2.3)


def _set_font(style, size, bold=None, italic=None):
    style.font.name = FONT
    style.font.size = Pt(size)
    style.font.color.rgb = RGBColor(0, 0, 0)
    if bold is not None:
        style.font.bold = bold
    if italic is not None:
        style.font.italic = italic
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        rfonts.attrib.pop(qn(attr), None)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), FONT)


normal = doc.styles["Normal"]
_set_font(normal, 12)
normal.paragraph_format.line_spacing = 1.5
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
for lvl, size in ((1, 16), (2, 13.5), (3, 12)):
    st = doc.styles[f"Heading {lvl}"]
    _set_font(st, size, bold=True, italic=(lvl == 3))
    st.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.keep_with_next = True
for name in ("List Bullet", "Caption"):
    _set_font(doc.styles[name], 12 if name == "List Bullet" else 10.5)

counter = {"fig": 0, "tab": 0, "eq": 0}


def _shade(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


def _runs(p, text, size=None, italic=False):
    parts = text.split("**")
    for i, part in enumerate(parts):
        if part:
            r = p.add_run(part)
            r.bold = i % 2 == 1
            r.italic = italic
            if size:
                r.font.size = Pt(size)
    return p


def para(text, italic=False, align=None, size=None):
    p = _runs(doc.add_paragraph(), text, size=size, italic=italic)
    if align:
        p.alignment = align
    return p


def bullets(items):
    for it in items:
        p = _runs(doc.add_paragraph(style="List Bullet"), it)
        p.paragraph_format.space_after = Pt(3)


def numbered(items):
    for i, it in enumerate(items, 1):
        p = _runs(doc.add_paragraph(), f"{i}. " + it)
        p.paragraph_format.left_indent = Cm(.8)
        p.paragraph_format.first_line_indent = Cm(-.6)
        p.paragraph_format.space_after = Pt(3)


def h(text, level=1):
    doc.add_heading(text, level)


def caption(kind, text):
    counter[kind] += 1
    label = "Figure" if kind == "fig" else "Table"
    p = doc.add_paragraph(style="Caption")
    r = p.add_run(f"{label} {counter[kind]}: ")
    r.bold = True
    p.add_run(text)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = kind == "tab"
    return counter[kind]


def figure(path, text, width=16):
    doc.add_picture(str(path), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.keep_with_next = True
    return caption("fig", text)


def table(header, rows, text, widths=None, size=9.5, bold_rows=()):
    n = caption("tab", text)
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, hd in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        _runs(c.paragraphs[0], f"**{hd}**", size=size)
        _shade(c, "D9E2F3")
    trpr = t.rows[0]._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader")
    th.set(qn("w:val"), "true")
    trpr.append(th)
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            txt = str(v)
            _runs(cells[i].paragraphs[0], f"**{txt}**" if ri in bold_rows and txt else txt, size=size)
    for row in t.rows:
        for i, c in enumerate(row.cells):
            for p in c.paragraphs:
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.space_after = Pt(1)
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            if widths:
                c.width = Cm(widths[i])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return n


def note(text, fill="FFF2CC"):
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    c = t.rows[0].cells[0]
    c.text = ""
    _runs(c.paragraphs[0], text, size=10.5)
    c.paragraphs[0].paragraph_format.line_spacing = 1.15
    _shade(c, fill)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def pending(what):
    note(f"**PENDING:** {what} The prediction benchmark had not finished writing its output files "
         "when this document was generated. Re-run `python make_report.py` after it finishes; "
         "this box will be replaced by the measured values.", fill="FBE4E4")


def equation(text):
    counter["eq"] += 1
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = True
    p.add_run(f"\t\t({counter['eq']})")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return counter["eq"]


def code(src, size=8.5):
    for line in src.rstrip().splitlines():
        p = doc.add_paragraph()
        r = p.add_run(line if line else " ")
        r.font.name = "Consolas"
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
        r.font.size = Pt(size)
        pf = p.paragraph_format
        pf.line_spacing = 1.0
        pf.space_after = Pt(0)
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        ppr = p._p.get_or_add_pPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), "F2F2F2")
        ppr.append(shd)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def field(paragraph, instr, placeholder=""):
    run = paragraph.add_run()
    for tag, val in (("begin", None), ("instr", instr), ("separate", None), ("text", placeholder),
                     ("end", None)):
        if tag == "instr":
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = val
        elif tag == "text":
            el = OxmlElement("w:t")
            el.text = val
        else:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), tag)
        run._r.append(el)


def src_lines(path):
    return sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())


# ───────────────────────────── title page ─────────────────────────────
for _ in range(2):
    doc.add_paragraph()
p = para("COURSE-BASED PROJECT DOCUMENTATION WITH INNOVATION", align=WD_ALIGN_PARAGRAPH.CENTER, size=13)
p.runs[0].bold = True
doc.add_paragraph()
p = para(TITLE, align=WD_ALIGN_PARAGRAPH.CENTER, size=18)
p.runs[0].bold = True
doc.add_paragraph()
para("A simulation-based study with a full audit of an earlier edge-AI thermal benchmark",
     italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
for _ in range(2):
    doc.add_paragraph()
for line in ("Submitted by: [Student Name]", "Register Number: [Register Number]",
             "Course: [Course Name and Code]", "Faculty Guide: [Guide Name, Designation]",
             "Department of [Department Name]", "[Institution Name]", "October 2026"):
    para(line, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
note("**Data provenance statement.** All devices, workloads and measurements in this report are produced "
     "by a software simulator (first-order RC thermal model) written for this project. No physical edge "
     "hardware was used. Every number in the report is read automatically from the result files "
     "generated by the project code (folder results/) at the time the document was built.",
     fill="E2EFDA")
doc.add_page_break()

h("Table of Contents", 1)
field(doc.add_paragraph(), 'TOC \\o "1-3" \\h \\z \\u',
      "Right-click here and choose Update Field to build the table of contents.")
doc.add_page_break()

h("List of Abbreviations", 1)
table(["Abbreviation", "Meaning"], [
    ("CI", "Confidence interval (95%, from bootstrap)"),
    ("CPU / GPU / NPU / DSP", "Central / graphics / neural / digital-signal processing unit"),
    ("GBM", "Gradient boosting machine (gradient boosted decision trees)"),
    ("GP", "Gaussian process regression"),
    ("MAE / RMSE", "Mean absolute error / root mean squared error"),
    ("Mondrian conformal", "Conformal prediction calibrated separately inside each group"),
    ("P·R", "Power × thermal resistance = steady-state temperature rise above ambient"),
    ("RC model", "Resistor–capacitor (lumped) thermal model"),
    ("RF", "Random forest"),
    ("Ta, T0", "Ambient temperature, temperature at the start of the run"),
    ("τ (tau)", "Thermal time constant τ = R·C"),
    ("ΔT", "Temperature change during one inference run (end − start)"),
], "Abbreviations and symbols used in this report", widths=[4.5, 11.5])
doc.add_page_break()

# ───────────────────────────── 1. Title ─────────────────────────────
h("1. Title of the Project", 1)
para(f"**{TITLE}**")
para("**Project type:** Simulation study and machine-learning benchmark with a reproducibility audit.")
para("**Domain:** Edge artificial intelligence, embedded systems, thermal-aware computing, machine learning "
     "with uncertainty quantification.")
para("**Keywords:** edge AI, inference cost prediction, RC thermal model, workload-family shift, "
     "leave-one-family-out evaluation, split conformal prediction, Mondrian conformal prediction, "
     "bootstrap confidence intervals, reproducibility.")
para("The title states the three things the project does. It predicts the cost of running a neural network "
     "on an edge device before the network is deployed (time, energy and temperature). It evaluates the "
     "predictors on workload families they have never seen (family holdout). And it attaches uncertainty "
     "intervals whose reliability is checked separately in each thermal regime.")

# ───────────────────────────── 2. Abstract ─────────────────────────────
h("2. Abstract", 1)
para("Edge schedulers must decide which processor should run a neural network before it runs, so they rely on "
     "predicted inference duration, energy and temperature change. This project began as a review of an "
     "existing draft study that claimed five \"discoveries\" about edge accelerators. The draft reported a "
     "\"Green-Thermal Paradox\", universal net cooling, a device-ranking inversion and a collapse of conformal "
     "coverage. All of these came from a synthetic simulator, but the draft presented them as empirical "
     "findings, and its text disagreed with its code in several places.")
para(f"We rebuilt the simulator so that it matches its documentation, and ran it at full scale. The dataset "
     f"has {len(FAMILIES)} workload families, {N_WORKLOADS:,} workloads, {len(DEVICES)} virtual devices, "
     f"{len(sim.CONTEXTS)} thermal contexts and {REPEATS} repeats, giving {N_ROWS:,} rows. We then audited "
     f"every claim against the model equations:")
bullets([
    f"**Device-ranking inversion is not supported.** Only {', '.join(sat_devices)} ever reaches the saturation "
    f"regime, so device rankings in that regime can be compared in {n_comparable} of {len(rank)} families.",
    f"**The \"absolute paradox\" is a definition artifact.** The energy-optimal device is the hottest by "
    f"end-of-run temperature in {pct(fm['energy_opt_is_hottest_share'])} of task instances. With the "
    f"physically correct peak temperature it is the unique hottest device in "
    f"{pct(pk['energy_opt_unique_hottest_share'])} of them, and it lies in the coolest set in "
    f"{pct(pk['energy_opt_in_coolest_set_share'])}.",
    f"**Net cooling is a consequence of the chosen start temperatures.** It occurs in "
    f"{pct(cool['net_cooling_share_measured'])} of runs. The cooling sign rule holds in "
    f"{pct(sign['agreement_with_noise_free_sign'])} of noise-free runs because it is an algebraic identity "
    f"of the model.",
])
if PRED:
    b = {t: best[t] for t in TARGETS}
    para("We then built a prediction benchmark: six regressors, five static feature sets, family-holdout "
         "cross-validation, a size-matched random-split control and four split-conformal interval methods. "
         "Under family holdout, the best mean absolute errors were:")
    bullets([f"{TARGET_LABEL[t]}: {num(b[t].mae, 3)} {UNIT[t]} ({cfg(b[t].model, b[t].feature_config)}), "
             f"with 90% global-conformal coverage of {pct(b[t].coverage)}." for t in TARGETS])
    para("The abstract does not summarise per-regime coverage, random-split optimism or the thermal-state "
         "ablation. Those results are reported with bootstrap confidence intervals in Section 12.")
else:
    para("A prediction benchmark is also included: six regressors, five static feature sets, "
         "family-holdout cross-validation, a size-matched random-split control and four split-conformal "
         "interval methods. Its numerical results are marked PENDING in this version of the document.")
para("Our contribution is methodological. We provide a corrected, documented simulator and an audit procedure "
     "that separates model-implied identities from findings. We also provide an evaluation protocol whose "
     "every number can be traced to a generated file. None of the results are hardware measurements.")

# ───────────────────────────── 3. Problem statement ─────────────────────────────
h("3. Problem Statement", 1)
h("3.1 Background", 2)
para("Modern edge systems-on-chip combine several kinds of processor in one package: general-purpose CPU "
     "cores, a mobile GPU, a neural processing unit (NPU) and a digital signal processor (DSP). A scheduler "
     "that receives a neural-network inference request has to choose one of them. The choice affects three "
     "quantities a user cares about:")
bullets([
    "**Latency:** how long the inference takes.",
    "**Energy:** how much of the battery it uses.",
    "**Temperature:** how hot the package becomes, and therefore whether thermal throttling follows.",
])
para("Profiling every network on every processor is too slow. Schedulers therefore use static descriptors of "
     "the network instead, such as operation count, parameter count, memory footprint, operator mix and "
     "sparsity, to predict these quantities in advance.")
h("3.2 The problem", 2)
para("Static predictors are usually evaluated in ways that overstate how well they will work on new models. "
     "Three weaknesses are common:")
numbered([
    "**Random splits leak information.** Train and test sets are drawn from the same architecture families. "
    "Real deployments meet new families, so reported accuracy can be optimistic.",
    "**One error number hides the hard cases.** Uncertainty intervals are calibrated once over all runs. An "
    "interval that is valid on average can still be badly invalid for long runs, where temperature "
    "approaches its steady state.",
    "**Simulator properties get reported as discoveries.** When results come from a simulator, a quantity "
    "built into the simulator's equations can be presented as an empirical finding.",
])
para("The draft study that this project started from showed all three weaknesses, and its claims did not "
     "match its code.")
para("**Problem statement:** Design and implement a reproducible pipeline that does four things. It simulates "
     "edge-AI inference cost and thermal behaviour, and checks each claimed thermal phenomenon against the "
     "simulator equations. It predicts inference duration, energy and temperature change from static "
     "workload descriptors for unseen workload families. And it quantifies how reliable 90% uncertainty "
     "intervals are in each thermal regime and on each device.")
h("3.3 Why the problem matters", 2)
bullets([
    "A scheduler that trusts an over-confident temperature prediction can push a device into throttling. "
    "Throttling degrades latency for every later request.",
    "A paper or report that presents model-implied identities as findings misleads readers and does not "
    "survive peer review.",
    "Students and researchers without hardware still need a principled way to study these trade-offs. A "
    "simulator that is honest about its assumptions provides that.",
])
h("3.4 Scope and constraints", 2)
bullets([
    "No physical edge hardware was available. All devices are virtual devices defined by a small set of "
    "parameters (Table 6), and all outcomes are simulated.",
    "Thermal behaviour is modelled with a single-node RC network. Multi-core coupling, dynamic voltage and "
    "frequency scaling (DVFS) and throttling controllers are outside the scope.",
    "Workloads are synthetic operator-mix profiles of 20 architecture families, not compiled real networks.",
])

# ───────────────────────────── 4. Existing system ─────────────────────────────
h("4. Existing System", 1)
h("4.1 Approaches in the literature", 2)
para("Several lines of prior work bear on this problem:")
bullets([
    "**Analytical performance models.** The Roofline model [1] bounds attainable performance by "
    "computational intensity and memory bandwidth. Our simulator's duration term, compute time plus "
    "memory-transfer time, follows the same reasoning.",
    "**Learned latency predictors.** nn-Meter [2] predicts edge inference latency by detecting kernels and "
    "using kernel-level predictors. Li, Paolieri and Golubchik [3] study inference latency prediction at "
    "the edge. PerfSAGE [4] uses graph neural networks to predict performance for arbitrary models on edge "
    "devices. These works focus mainly on latency, and they usually evaluate on architecture variants "
    "close to the training data.",
    "**Compact thermal models.** HotSpot [5] models chip temperature with RC networks. Our single-node model "
    "is the simplest member of that family. Recent work such as ThermE [6] manages shared thermal headroom "
    "for sustained inference on edge SoCs.",
    "**Uncertainty quantification.** Split conformal prediction [7, 8, 9] produces intervals with a "
    "guaranteed average coverage when calibration and test data are exchangeable. Mondrian conformal "
    "prediction [7] calibrates separately within groups, which aims at group-wise validity.",
])
h("4.2 The baseline system examined in this project", 2)
para("The direct starting point is an existing draft paper together with its simulator script. The draft "
     "describes 10 workload families, 54,000 rows, five virtual devices and four feature tiers. It also "
     "describes five models evaluated with leave-one-family-out cross-validation and global split-conformal "
     "intervals. It reports five \"discoveries\":")
bullets([
    "an 86.38% Pareto misalignment and a 3.50% \"absolute paradox\" between energy-optimal and coolest "
    "devices;",
    "net cooling in 66.16% of runs, with an 89.49% accurate sign rule;",
    "conformal coverage collapsing to 2.8%–34.8% in the saturation regime;",
    "a device-ranking inversion in 100% of families;",
    "an 88.8% error reduction for gradient boosting over linear models, with sparsity cutting error by "
    "about 50%.",
])
h("4.3 Limitations of the existing system", 2)
para("Reading the draft paper alongside its code revealed the following limitations:")
table(["#", "Limitation", "Evidence in the original code / paper"], [
    ("1", "Paper and code disagree on scale",
     f"The code defines {len(FAMILIES)} families, which gives {N_ROWS:,} rows; the paper reports 10 families and 54,000 rows."),
    ("2", "Sparsity distribution misreported",
     "The paper states Beta(2,5); the code uses Beta(8,2) for the sparse family and Beta(2,8) for the others."),
    ("3", "\"Peak\" temperature is the end-of-run value",
     "The code stores the end temperature as peak_temp_c. For a cooling run, the true maximum is the start temperature."),
    ("4", "Rankings compared across absent devices",
     "The inversion claim compares devices in the saturation regime without checking that each device appears there."),
    ("5", "Identities reported as discoveries",
     "The cooling sign rule follows algebraically from the RC equation. The paper reports it as an 89.49% empirical result."),
    ("6", "No random-split control",
     "The paper motivates family holdout by random-split optimism but never measures that optimism."),
    ("7", "No statistical testing",
     "Model and feature comparisons are reported as point MAEs without confidence intervals."),
    ("8", "Warnings silenced",
     "warnings.filterwarnings(\"ignore\") hides solver convergence problems."),
    ("9", "Only global intervals",
     "Coverage is calibrated once for all rows. No regime-aware or device-aware method is tried, even though the paper diagnoses regime failure."),
], "Limitations of the existing system identified during the audit", widths=[.8, 4.4, 10.8], size=9)

# ───────────────────────────── 5. Proposed system ─────────────────────────────
h("5. Proposed System", 1)
para("The proposed system is a reproducible Python pipeline. It corrects and extends the baseline simulator, "
     "audits the simulator's claimed phenomena, and benchmarks static-descriptor predictors under "
     "workload-family shift with calibrated uncertainty. It has four main parts:")
numbered([
    "**Corrected simulator.** It matches its documentation: 20 families, the true sparsity distributions, a "
    "correct peak temperature max(T0, T_end), and noise-free reference values stored next to each noisy "
    "measurement.",
    "**Claim-audit module.** It tests each claimed phenomenon directly against the simulator equations: "
    "regime coverage per device, net cooling by context, the sign rule with and without noise, energy versus "
    "thermal optimality under three temperature definitions, and rank support before any device ranking.",
    "**Prediction benchmark.** It trains six regressors on five feature sets for three targets, with "
    "leave-one-family-out evaluation and a size-matched random-workload control. Four conformal interval "
    "methods are compared, stratified by true regime and by device.",
    "**Reporting and verification.** Tables, figures and this report are generated directly from the result "
    "files. A verification script checks that every number in the paper text appears in a generated table.",
])
table(["Aspect", "Existing system", "Proposed system"], [
    ("Simulator vs documentation", "Inconsistent", "Consistent; configuration exported to metadata.json"),
    ("Peak temperature", "End-of-run value", "max(start, end), correct for a monotone RC response"),
    ("Claims", "Reported as discoveries", "Each tested against the model equations"),
    ("Evaluation", "Family holdout only", "Family holdout plus a size-matched random-split control"),
    ("Uncertainty", "Global split conformal", "Global, device-Mondrian, predicted-regime Mondrian, normalized"),
    ("Statistics", "Point estimates", "Workload-level paired bootstrap 95% CIs"),
    ("Diagnostics", "Warnings ignored", "Convergence warnings and skipped fits recorded"),
], "Summary of the existing system and the proposed system", widths=[4, 5, 7])

# ───────────────────────────── 6. Innovation ─────────────────────────────
h("6. Innovation", 1)
para("Conformal prediction, Mondrian calibration, grouped cross-validation and bootstrap testing are all "
     "established techniques [7–9, 15]. The innovation in this project lies in how they are combined and "
     "applied to audit and improve an edge-AI thermal benchmark.")
innovations = [
    ("I1. Claim audit against model equations",
     "Every claimed phenomenon is re-derived from the simulator's own equations. It is classed as a model "
     "identity, a definition artifact, a property of the chosen parameters, or an empirical result. The same "
     "procedure works for any simulation-based study."),
    ("I2. Physically correct peak-temperature definition",
     "For a first-order RC system under constant power, temperature moves monotonically from T0 toward the "
     "steady state. The true peak during a run is therefore max(T0, T_end). Using it exposes the "
     "\"absolute paradox\" as an artifact of the definition."),
    ("I3. Rank-support check before ranking devices",
     "Device rankings within a regime are computed only when every device actually appears in that regime. "
     "This prevents comparisons between devices that are present and devices that are absent."),
    ("I4. Size-matched random-split control",
     "Random-workload splits use exactly the same numbers of training, calibration and test rows as the family "
     "folds. The optimism of random splits is then measured, not assumed."),
    ("I5. Deployable regime-aware conformal intervals",
     "The true thermal regime is unknown before a run, because it depends on the run's duration. The intervals "
     "are therefore grouped by the regime implied by the predicted duration, with explicit fallback when a "
     f"group has fewer than {MIN_GROUP} calibration rows. The share of rows that fall back is reported."),
    ("I6. Thermal-state ablation",
     "Feature set M1n removes the ambient and start temperatures from M1. This measures how much of the "
     "temperature-prediction skill comes from knowing the thermal state rather than the workload."),
    ("I7. End-to-end number traceability",
     "Tables, figures and this report are generated from result files. A verifier rejects any number in the "
     "paper text that does not appear in a generated table."),
]
table(["Innovation", "Description"], innovations, "Innovations introduced in this project",
      widths=[4.5, 11.5], size=9.5)

# ───────────────────────────── 7. Objectives ─────────────────────────────
h("7. Objectives", 1)
h("7.1 Primary objectives", 2)
numbered([
    "Reproduce the baseline simulator and make it consistent with its documentation.",
    "Test each claimed thermal phenomenon against the RC equations, and classify it as a finding, a "
    "model identity or an artifact.",
    "Predict inference duration, energy and temperature change from static workload descriptors for "
    "unseen workload families.",
    f"Measure whether {int(COVERAGE * 100)}% conformal intervals keep their coverage in each thermal regime "
    "and on each device, and test whether regime-aware and device-aware calibration helps.",
])
h("7.2 Secondary objectives", 2)
numbered([
    "Quantify the optimism of random splits relative to family holdout at equal sample sizes.",
    "Quantify the contribution of sparsity, full descriptors and thermal-state features, with 95% "
    "bootstrap confidence intervals.",
    "Record solver convergence problems and failed fits instead of hiding them.",
    "Produce a report in which every number can be traced to a generated result file.",
])
h("7.3 Success criteria", 2)
table(["Objective", "Measurable criterion"], [
    ("Simulator consistency", "Configuration in metadata.json matches the text; noise-free sign rule = 100%"),
    ("Claim audit", "Each claim has a measured value and a verdict (Section 13)"),
    ("Prediction accuracy", "MAE per target under family holdout, with fold spread"),
    ("Interval reliability", f"Coverage per regime and device compared against the {int(COVERAGE * 100)}% target"),
    ("Statistical validity", "95% bootstrap CI for every model or feature comparison claimed"),
    ("Traceability", "Report generated from results/; no hand-typed result numbers"),
], "Objectives and how their success is measured", widths=[5, 11])

# ───────────────────────────── 8. Methodology ─────────────────────────────
h("8. Methodology", 1)
h("8.1 Overall workflow", 2)
n_meth = figure(figs["methodology"], "End-to-end methodology of the project", width=13.5)
para(f"Figure {n_meth} shows the ten stages. Stages 1–3 generate data. Stage 4 audits the simulator. "
     "Stages 5–9 build and evaluate predictors. Stage 10 produces the documentation. Each stage writes its "
     "results to disk, so later stages and this report read the same files.")

h("8.2 Thermal model", 2)
para("Each device package is modelled as a single thermal resistance R (K/W) in parallel with a thermal "
     "capacitance C (J/K). During an inference the device dissipates constant power P. The governing "
     "equation is:")
equation("C · dT/dt = P − (T − Ta) / R")
para("Starting from temperature T0 at time 0, the closed-form solution is:")
equation("T(t) = Ta + (T0 − Ta) · e^(−t/τ) + P·R · (1 − e^(−t/τ)),   τ = R·C")
para("The temperature change after a run of duration t is therefore:")
equation("ΔT = T(t) − T0 = (P·R − (T0 − Ta)) · (1 − e^(−t/τ))")
para("Two consequences follow directly from Equation 3, and both matter for the audit:")
bullets([
    "Because 1 − e^(−t/τ) > 0, the sign of ΔT is the sign of P·R − (T0 − Ta). A run cools the device "
    "exactly when the starting elevation above ambient exceeds the steady-state rise. This is an identity "
    "of the model, not an empirical regularity.",
    "T(t) is monotone in t. The highest temperature during the run is therefore max(T0, T(t)), not T(t).",
])
para(f"A measured end temperature adds Gaussian sensor noise with standard deviation "
     f"{sim.SENSOR_NOISE_C} °C. Each run is assigned a thermal regime from t/τ: transient below 0.3, "
     "saturation above 2.0, and transition between them.")
n_rc = figure(figs["rc"], "Thermal response of the simulated mobile_cpu computed from Equation 2, using its R, "
                          "C and mean simulated power, for the coolest and hottest contexts")
para(f"Figure {n_rc} illustrates both behaviours on one device. Starting at 27 °C in a 20 °C room, the device "
     "heats toward its steady state. Starting at 64 °C in a 40 °C room, the same workload cools the device, "
     "because the steady state is lower than the start temperature.")

h("8.3 Workload generation", 2)
para(f"Each of the {len(FAMILIES)} families has a characteristic operator mix over five operator classes: "
     "convolution, matrix multiplication, attention, activation and normalization. For each family, "
     f"{PER_FAMILY} workloads are drawn. The steps are:")
bullets([
    "**Operator mix:** drawn from a Dirichlet distribution centred on the family mix, with concentration "
    "65 × centre and a floor of 0.35.",
    "**Operation count:** log-uniform between 180 and 9,000 million.",
    "**Parameter count:** the operation count times a log-normal factor.",
    "**Memory footprint:** the operation count times a bytes-per-op factor that rises with the attention "
    "and matmul shares.",
    "**Graph depth:** log-normal, increasing with the attention and convolution shares.",
    "**Sparsity:** Beta(8,2) for the sparse family and Beta(2,8) for all other families.",
])
fam_rows = [(i + 1, f, *[f"{v:.2f}" for v in mix]) for i, (f, mix) in enumerate(sim.FAMILIES.items())]
table(["#", "Family", "conv", "matmul", "attention", "activation", "norm"], fam_rows,
      "The 20 synthetic workload families and their centre operator mixes", size=9,
      widths=[.8, 4.6, 1.9, 1.9, 1.9, 1.9, 1.9])

h("8.4 Virtual device plant", 2)
para("Each device has a throughput, a memory bandwidth, three power coefficients and thermal R and C, "
     "plus a per-operator efficiency vector. Duration is compute time plus memory-transfer time, with 3.5% "
     "log-normal jitter. Power is idle power plus load-weighted compute and memory power, with 2.5% "
     "Gaussian jitter. Energy is power × duration.")
dev_rows = []
for d, p in sim.DEVICE_PLANT.items():
    dev_rows.append((DEV_LABEL[d], f"{p['throughput']:.0f}", f"{p['bandwidth']:.0f}", f"{p['idle_w']:.1f}",
                     f"{p['compute_w']:.1f}", f"{p['memory_w']:.1f}", f"{p['r']:.1f}", f"{p['c']:.0f}",
                     f"{sim.TAU[d]:.1f}"))
table(["Device", "Throughput (Mops/s)", "Bandwidth (MB/s)", "Idle W", "Compute W", "Memory W",
       "R (K/W)", "C (J/K)", "τ (s)"], dev_rows, "Virtual device parameters used by the simulator", size=8.5)
ctx_rows = [(i + 1, f"{a:.0f}", f"{s:.0f}", f"{s - a:.0f}") for i, (a, s) in enumerate(sim.CONTEXTS)]
table(["Context", "Ambient Ta (°C)", "Start T0 (°C)", "Start elevation T0 − Ta (°C)"], ctx_rows,
      "Thermal operating contexts", widths=[2.5, 4, 4, 5.5])
para(f"Every workload is run on every device, in every context, {REPEATS} times. This gives "
     f"{N_WORKLOADS:,} × {len(DEVICES)} × {len(sim.CONTEXTS)} × {REPEATS} = {N_ROWS:,} rows. Grouping "
     f"the rows by workload, context and repeat gives {N_TASKS:,} task instances; each task instance has "
     "one outcome per device.")

h("8.5 Descriptive audit analyses", 2)
bullets([
    "**Regime counts per device:** rows in each regime, and the maximum t/τ each device reaches.",
    "**Net cooling:** the share of rows with ΔT < 0, overall and by context.",
    "**Sign rule:** agreement between sign(P·R − (T0 − Ta)) and the sign of the measured and of the "
    "noise-free ΔT. Mismatches are tested against a 3σ noise band.",
    "**Energy versus thermal optimality:** for each task instance, whether the energy-minimising device is "
    "the coolest or the hottest. Three temperature definitions are used: end temperature, true peak "
    "temperature and steady-state rise P·R.",
    "**Rank support:** for each family, which devices appear in the transient and saturation regimes. A "
    "ranking is called comparable only if all five devices appear in both.",
])

h("8.6 Feature sets", 2)
table(["Set", "Dims", "Features"], [
    ("M0", FEAT_DIM["M0_size_only"], "log(1+ops), Ta, T0, 5 device indicators"),
    ("M1n", FEAT_DIM["M1n_no_thermal_state"], "log ops, log parameters, log memory, 5 device indicators (no Ta, T0)"),
    ("M1", FEAT_DIM["M1_size_memory"], "M1n + Ta, T0"),
    ("M2", FEAT_DIM["M2_size_memory_sparse"], "M1 + sparsity"),
    ("M3", FEAT_DIM["M3_full_descriptors"], "M2 + log depth, 5 operator-mix shares, ops×mix, memory×mix, device×mix interactions"),
], "Static feature sets (dimension counts are computed from the code)", widths=[1.5, 1.5, 13])
para("All features are standardised using the training rows of each fold only, so no test information "
     "leaks into the scaling.")

h("8.7 Regression models", 2)
table(["Model", "Configuration"], [
    ("Ridge", "Closed-form least squares with L2 penalty α = 1.0 (intercept unpenalised)"),
    ("RidgeLog", "Ridge fitted to log(target) and exponentiated; duration and energy only (they are positive)"),
    ("Lasso", "scikit-learn Lasso, α = 0.01, max_iter = 5000"),
    ("Random forest (RF)", "120 trees, max_features = sqrt, min_samples_leaf = 5"),
    ("Gradient boosting (GBM)", "200 trees, depth 4, learning rate 0.05, subsample 0.8"),
    ("Gaussian process (GP)", "Constant × RBF + White kernel, normalize_y; fitted on a fixed random subset of "
                              f"{meta['gp_max_training_rows'] if PRED else 2000} training rows"),
], "Regression models and hyperparameters", widths=[4.5, 11.5])

h("8.8 Evaluation protocols", 2)
n_split = figure(figs["split"], "Leave-one-family-out split: training, calibration and test families in each fold")
para(f"**Family holdout (primary).** In fold k, family k is the test set ({N_FAMILY_ROWS:,} rows) and "
     f"family k+1 is the calibration set ({N_FAMILY_ROWS:,} rows). From each of the remaining "
     f"{len(FAMILIES) - 2} families, {TRAIN_PER_FAMILY} workloads are sampled for training "
     f"({N_TRAIN_ROWS:,} rows). Figure {n_split} shows the arrangement. The test family is never seen in "
     "training or calibration, which mimics deployment to a new architecture class.")
para("**Random-workload control.** In each of 20 folds, workloads are drawn at random from all families, "
     "with exactly the same training, calibration and test row counts as the family folds. The gap between "
     "the two protocols measures how optimistic a random split is.")

h("8.9 Uncertainty intervals", 2)
para("Split conformal prediction uses the absolute residuals sᵢ = |yᵢ − ŷᵢ| of the n calibration rows. "
     "The interval half-width q is the ⌈(n+1)(1−α)⌉-th smallest residual, with α = 0.10:")
equation("Ĉ(x) = [ ŷ(x) − q ,  ŷ(x) + q ]")
table(["Method", "How q is chosen"], [
    ("Global", "One q from all calibration rows"),
    ("Device-Mondrian", f"Separate q per device (global q if fewer than {MIN_GROUP} calibration rows)"),
    ("Predicted-regime Mondrian", "Separate q per regime, where the regime comes from the predicted duration / τ "
                                  f"(global q if fewer than {MIN_GROUP} rows)"),
    ("Normalized", "q applied to residual / max(ŷ, 5th percentile of training y); duration and energy only"),
], "Conformal interval methods compared", widths=[5, 11])
para("Coverage is the share of test rows whose true value lies inside the interval. It is reported overall, "
     "per true regime and per device, together with the mean interval width and the share of rows that used "
     "the fallback radius.")
h("8.10 Statistical comparison", 2)
para("Each workload's MAE is averaged over its test rows. Model or feature comparisons use a paired "
     "bootstrap over workloads (2,000 resamples) of MAE(baseline) − MAE(comparison); a positive value "
     "favours the comparison. The random-versus-family protocol gap uses an unpaired workload bootstrap. A "
     "difference is called significant only if its 95% CI excludes zero.")
h("8.11 Reproducibility", 2)
para(f"All randomness flows from seed {SEED}. Running `python simulation_study.py` with its defaults "
     "regenerates every result file. The 40 folds run in parallel with joblib, and their results are "
     "deterministic for a given seed.")

# ───────────────────────────── 9. Requirements ─────────────────────────────
h("9. Hardware and Software Requirements", 1)
para("Only the technologies actually used in this project are listed. They were read from the machine "
     "and the Python environment when this report was generated.")
mi = machine_info()
table(["Item", "Specification used"], [
    ("Computer", "Personal laptop / desktop (existing machine, no purchase)"),
    ("Processor", mi.get("cpu", "unknown")),
    ("Logical CPUs", mi.get("logical_cpus")),
    ("RAM", f"{mi.get('ram_gb', 'unknown')} GB"),
    ("Operating system", mi.get("os")),
    ("Edge hardware", "None. All five devices are simulated"),
], "Hardware requirements (as used)", widths=[4.5, 11.5])
sw = [("Python", mi["python"], "Programming language")]
for pkg, purpose in (("numpy", "Numerical arrays, random number generation"),
                     ("pandas", "Tabular data, grouping, CSV output"),
                     ("scikit-learn", "Lasso, random forest, gradient boosting, Gaussian process"),
                     ("scipy", "Numerical routines used by scikit-learn"),
                     ("joblib", "Parallel execution of the 40 folds"),
                     ("matplotlib", "Figures in this report"),
                     ("python-docx", "Generation of this report")):
    sw.append((pkg, md.version(pkg), purpose))
table(["Software", "Version", "Purpose"], sw, "Software requirements (versions read from the environment)",
      widths=[3.5, 2.5, 10])
para("All software is free and open source. No commercial licence, cloud service or GPU was used.")

# ───────────────────────────── 10. Architecture ─────────────────────────────
h("10. System Architecture", 1)
n_arch = figure(figs["architecture"], "System architecture of the proposed pipeline")
para(f"Figure {n_arch} shows the four layers:")
bullets([
    "**Input layer.** Holds the configuration: workload families, the device plant and the thermal "
    "contexts. All of it is defined in one file, so the paper text and the code cannot drift apart.",
    "**Simulation layer.** Turns every workload–device–context–repeat combination into one observation, "
    "using the closed-form RC solution plus sensor noise.",
    "**Learning layer.** Has two branches. The audit branch reads the simulated data directly. The "
    "prediction branch builds feature matrices, trains regressors per fold and calibrates the four "
    "interval methods.",
    "**Evaluation and output layer.** Aggregates per-fold results into coverage, width, MAE and RMSE, "
    "computes bootstrap CIs, and writes CSV/JSON files. Tables, figures and this report are produced "
    "from those files.",
])
table(["Component", "Input", "Output file(s)"], [
    ("Workload generator", "Family mixes, seed", "In-memory workload table"),
    ("RC simulator", "Workloads, device plant, contexts", f"Dataset ({N_ROWS:,} rows)"),
    ("Audit module", "Dataset", "regime_device_counts.csv, cooling_*.csv/json, sign_rule_check.json, "
                                "energy_thermal_alignment*.{json,csv}, regime_rank_support.csv"),
    ("Feature builder", "Dataset", "Design matrices M0–M3"),
    ("Fold runner", "Matrices, split", "fold_strata_detail.csv, workload_errors.csv, convergence_warnings.csv"),
    ("Aggregator / bootstrap", "Fold outputs", "prediction_summary.csv, bootstrap_comparisons.csv, "
                                              "protocol_comparison.csv, family_fold_metrics.csv"),
    ("Reporting", "results/", "paper_tables.md, report_figures/*.png, this .docx"),
], "Data flow between components", widths=[3.8, 4.2, 8], size=9)

# ───────────────────────────── 11. Implementation ─────────────────────────────
h("11. Implementation", 1)
para("The project is implemented in Python as the following scripts (non-blank line counts measured from "
     "the files):")
table(["File", "Lines", "Role"], [
    ("simulation_study.py", src_lines("simulation_study.py"), "Simulator, audit analyses, prediction benchmark"),
    ("build_tables.py", src_lines("build_tables.py"), "Markdown tables for the paper from results/"),
    ("verify_paper.py", src_lines("verify_paper.py"), "Checks that every number in the paper is traceable"),
    ("latency_benchmark.py", src_lines("latency_benchmark.py"), "Fit / prediction timing on this machine"),
    ("make_report.py", src_lines("make_report.py"), "Figures and this Word report"),
], "Source files of the implementation", widths=[4.5, 2, 9.5])


def module(no, name, purpose, inputs, outputs, logic, src=None):
    h(f"11.{no} Module {no}: {name}", 2)
    para(f"**Purpose.** {purpose}")
    para(f"**Inputs.** {inputs}")
    para(f"**Outputs.** {outputs}")
    para("**Key logic.**")
    bullets(logic)
    if src:
        para("**Code excerpt (from simulation_study.py):**")
        code(src)


plant_src = inspect.getsource(sim._plant_observation)
start = plant_src.index("    tau = p")
end = plant_src.index("    ratio = duration / tau")
module(1, "Workload Generator",
       "Create a realistic spread of synthetic neural-network workloads for each architecture family.",
       f"The family operator-mix centres, the number of workloads per family ({PER_FAMILY}) and the seed.",
       f"{N_WORKLOADS:,} workload records with ops, parameters, memory, depth, sparsity and five mix shares.",
       ["Dirichlet sampling around each family centre keeps families distinct but overlapping.",
        "Log-uniform operation counts span a 50× size range.",
        "The memory footprint depends on the operator mix, so memory-bound families (attention-heavy) "
        "differ from compute-bound ones."])
module(2, "RC Thermal Plant Simulator",
       "Produce duration, power, energy and temperature trajectories for each workload on each device.",
       "One workload record, a device name, the ambient and start temperatures, and a random generator.",
       "Duration, power, energy, end temperature, peak temperature, noise-free versions, P·R, τ, t/τ and regime.",
       ["Duration is a roofline-style compute term plus a memory term, with multiplicative jitter.",
        "Power depends on compute and memory load.",
        "Temperature uses the closed-form RC solution; the peak is max(T0, T_end)."],
       plant_src[start:end])
module(3, "Claim-Audit (Descriptive Analysis)",
       "Test each phenomenon claimed by the baseline study directly against the simulated data and equations.",
       "The full simulated dataset.",
       "Ten CSV/JSON files summarising regimes, cooling, the sign rule, energy–thermal alignment and rank support.",
       ["Uses noise-free columns to separate model identities from noise effects.",
        "Pivots task instances across devices to compare energy-optimal and thermally optimal choices.",
        "Checks that every device appears in a regime before ranking devices in it."],
       inspect.getsource(sim._regime_from_ratio))
module(4, "Feature Engineering and Model Training",
       "Turn workload descriptors into feature matrices and fit six regressors per fold and target.",
       "The dataset and the feature-set definitions M0–M3.",
       "Prediction functions for the calibration and test rows of every fold.",
       ["Log transforms for size features; one-hot device indicators; operator-mix interactions in M3.",
        "Standardisation uses training rows only.",
        "Each fit runs inside warnings.catch_warnings(record=True). Convergence warnings are counted, and "
        "failed fits are recorded rather than dropped silently."])
module(5, "Evaluation and Uncertainty Quantification",
       "Create the splits, calibrate conformal intervals and compute stratified metrics.",
       "Predictions, calibration residuals, true and predicted regimes, and devices.",
       "Per-fold, per-stratum coverage, width, errors and fallback counts; aggregated summaries; bootstrap CIs.",
       ["Split conformal radius from the ⌈(n+1)(1−α)⌉-th calibration residual.",
        f"Mondrian groups with fewer than {MIN_GROUP} calibration rows fall back to the global radius.",
        "40 folds run in parallel through joblib."],
       inspect.getsource(sim._conformal_radius) + "\n\n" + inspect.getsource(sim._group_radii))
module(6, "Reporting and Verification",
       "Produce documentation whose numbers are guaranteed to come from the result files.",
       "The results/ folder.",
       "paper_tables.md, report figures, this Word report and a verification verdict for the paper text.",
       ["Tables and figures are generated from CSV/JSON, never typed by hand.",
        "verify_paper.py extracts every number from the paper text and fails if one is not in a generated table.",
        "Sections whose inputs are missing are marked PENDING instead of being estimated."])

# ───────────────────────────── 12. Results ─────────────────────────────
h("12. Results and Performance", 1)
note("All results below are produced by the simulator described in Section 8. They describe the behaviour of "
     "the model, not of physical devices.", fill="E2EFDA")
h("12.1 Dataset summary", 2)
table(["Quantity", "Value"], [
    ("Workload families", len(FAMILIES)), ("Workloads", f"{N_WORKLOADS:,}"), ("Devices", len(DEVICES)),
    ("Thermal contexts", len(sim.CONTEXTS)), ("Repeats", REPEATS), ("Rows", f"{N_ROWS:,}"),
    ("Task instances (workload × context × repeat)", f"{N_TASKS:,}"),
    ("Rows per fold: train / calibration / test", f"{N_TRAIN_ROWS:,} / {N_FAMILY_ROWS:,} / {N_FAMILY_ROWS:,}"),
], "Size of the simulated dataset and of each fold", widths=[9, 7])

h("12.2 Thermal regimes", 2)
n_reg = figure(figs["regimes"], "Thermal regime distribution by device and maximum t/τ reached")
table(["Device", "Transient", "Transition", "Saturation", "τ (s)", "Max duration (s)", "Max t/τ"],
      [(r.device, f"{r.transient:,}", f"{r.transition:,}", f"{r.saturation:,}", f"{r.tau_s:.1f}",
        f"{r.max_duration_s:.1f}", f"{r.max_duration_over_tau:.2f}") for r in reg.itertuples()]
      + [("Total", f"{reg_tot.transient:,}", f"{reg_tot.transition:,}", f"{reg_tot.saturation:,}", "", "", "")],
      "Rows per thermal regime and device", bold_rows=(len(reg),))
para(f"Across all rows, {pct(reg_tot.transient / N_ROWS)} are transient, "
     f"{pct(reg_tot.transition / N_ROWS)} are transition and {pct(reg_tot.saturation / N_ROWS)} are "
     f"saturation. Figure {n_reg}(b) shows the key fact for the audit: only "
     f"{', '.join(sat_devices)} ever exceeds t/τ = 2.0. Every other device stays below the threshold even "
     f"on its longest run; the highest of them is "
     f"{reg[reg.device != sat_devices[0]].max_duration_over_tau.max():.2f}. The saturation regime is "
     "therefore a single-device regime in this simulator.")

h("12.3 Device ranking support", 2)
table(["Family", "Devices in transient", "Devices in saturation", "Saturation rows", "Comparable",
       "Hottest in transient"],
      [(r.family, r.transient_devices_present, r.saturation_devices_present, r.saturation_rows,
        "Yes" if r.ranking_comparable else "No", r.hottest_transient_among_present) for r in rank.itertuples()],
      "Rank support per family: can devices be ranked in both regimes?", size=8.5)
table(["Regime", *DEVICES],
      [(rg, *[f"{regime_dev.loc[(rg, d), 'mean_temp_change_c']:+.2f} ({int(regime_dev.loc[(rg, d), 'rows']):,})"
              if (rg, d) in regime_dev.index else "absent" for d in DEVICES]) for rg in sim.REGIMES],
      "Mean temperature change in °C (rows) by regime and device, pooled over families", size=8.5)
para(f"In all {len(rank)} families every device appears in the transient regime, and the hottest transient "
     f"device is {hot_transient.index[0]} in {hot_transient.iloc[0]} of them. But in every family the "
     f"saturation regime contains only {sat_devices[0]}. So the claimed \"universal device-ranking "
     f"inversion\" compares one present device against four absent ones; comparable rankings exist in "
     f"{n_comparable} of {len(rank)} families. The claim is **not supported** by the simulator.")

h("12.4 Net cooling during inference", 2)
n_cool = figure(figs["cooling"], "Net cooling by thermal context and the reason behind it")
table(["Ambient (°C)", "Start (°C)", "Start elevation (°C)", "Mean P·R (°C)", "Cooling share", "Mean ΔT (°C)"],
      [(f"{r.ambient_c:.0f}", f"{r.start_temp_c:.0f}", f"{r.start_elevation_c:.0f}",
        f"{r.mean_steady_state_rise_c:.2f}", pct(r.cooling_share), f"{r.mean_temp_change_c:+.2f}")
       for r in cool_ctx.itertuples()], "Net cooling by thermal context")
para(f"{pct(cool['net_cooling_share_measured'])} of all rows end cooler than they started; the noise-free "
     f"share is {pct(cool['net_cooling_share_noise_free'])}. The mean change is "
     f"{cool['mean_temp_change_cooling_rows_c']:+.2f} °C in cooling rows and "
     f"{cool['mean_temp_change_heating_rows_c']:+.2f} °C in heating rows. The extremes are "
     f"{cool['min_temp_change_c']:+.2f} °C (on {cool['min_temp_change_device']}) and "
     f"{cool['max_temp_change_c']:+.2f} °C.")
para(f"Figure {n_cool}(b) explains the result. The mean steady-state rise P·R is about {mean_ss:.1f} °C in "
     f"every context, while the chosen start elevations range from "
     f"{cool_ctx.start_elevation_c.min():.0f} to {cool_ctx.start_elevation_c.max():.0f} °C. Whenever the "
     "start elevation is larger than P·R, the device must cool. The cooling share is therefore set by the "
     f"context design: it rises from {pct(cool_ctx.cooling_share.iloc[0])} to "
     f"{pct(cool_ctx.cooling_share.iloc[-1])} as the start elevation grows. It is a property of the "
     "chosen parameters, not a discovery.")
table(["Check", "Value"], [
    ("Agreement of sign rule with noise-free ΔT", pct(sign["agreement_with_noise_free_sign"], 2)),
    ("Agreement of sign rule with measured (noisy) ΔT", pct(sign["agreement_with_measured_sign"], 2)),
    ("Mismatching rows", f"{sign['mismatch_rows']:,}"),
    (f"Mismatches whose true |ΔT| < 3σ = {3 * sign['sensor_noise_sd_c']:.2f} °C",
     pct(sign["mismatch_share_with_abs_true_change_below_3_sigma"], 2)),
], "Verification of the cooling sign rule ΔT < 0 ⇔ P·R < T0 − Ta", widths=[11, 5])
para("The sign rule agrees with the noise-free outcome in every row, because it is Equation 3 rearranged. "
     "Almost all mismatches with the noisy measurement are runs whose true change is within three sensor "
     "standard deviations of zero. The \"89.49% accuracy\" reported in the baseline therefore measures sensor "
     "noise, not the validity of a theorem.")

h("12.5 Energy-optimal versus thermally optimal device", 2)
n_en = figure(figs["energy"], "Is the energy-optimal device also the coolest? The answer depends on the temperature definition")
table(["Criterion", "Share of task instances"], [
    ("Energy-optimal device has the lowest end temperature", pct(fm["energy_opt_is_coolest_share"])),
    ("Energy-optimal device does not have the lowest end temperature", pct(fm["energy_opt_not_coolest_share"])),
    ("Energy-optimal device has the highest end temperature", f"{pct(fm['energy_opt_is_hottest_share'])} "
                                                             f"({fm['energy_opt_is_hottest_count']:,})"),
    ("Energy-optimal device is in the coolest set by true peak", pct(pk["energy_opt_in_coolest_set_share"])),
    ("Energy-optimal device is the unique hottest by true peak", pct(pk["energy_opt_unique_hottest_share"])),
    ("All devices tied at the start temperature (true peak)", pct(pk["all_devices_tied_at_start_temp_share"])),
    ("Energy-optimal device has the lowest P·R", pct(ssr["energy_opt_is_lowest_share"])),
    ("Energy-optimal device has the highest P·R", pct(ssr["energy_opt_is_highest_share"])),
], f"Energy versus thermal optimality over {N_TASKS:,} task instances", widths=[11, 5])
table(["Device", "Lowest energy", "Lowest end temp", "Highest end temp", "Lowest P·R", "Highest P·R"],
      [(r.device, pct(r.lowest_energy_share), pct(r.lowest_final_temp_share), pct(r.highest_final_temp_share),
        pct(r.lowest_steady_state_rise_share), pct(r.highest_steady_state_rise_share))
       for r in choice.itertuples()], "How often each device is selected under each criterion", size=9)
para(f"The energy-optimal device is always dsp ({pct(dsp_energy)}) or npu ({pct(npu_energy)}). By end "
     f"temperature it is the coolest device in only {pct(fm['energy_opt_is_coolest_share'])} of task "
     "instances. That number is what the baseline reports as a paradox. But end temperature is the wrong "
     "quantity: in a cooling run the device's hottest moment is the start, which is the same for all "
     f"devices. With the true peak, the energy-optimal device is in the coolest set in "
     f"{pct(pk['energy_opt_in_coolest_set_share'])} of instances and is never the unique hottest "
     f"({pct(pk['energy_opt_unique_hottest_share'])}). Figure {n_en}(b) shows that peaks tie in most of the "
     "hot contexts, because every device cools from the same start temperature.")
para("By steady-state rise P·R, the quantity that governs sustained thermal stress, the energy-optimal device "
     f"is the lowest in {pct(ssr['energy_opt_is_lowest_share'])} of instances and the highest in "
     f"{pct(ssr['energy_opt_is_highest_share'])}. In this simulator, minimising energy is therefore largely "
     "compatible with minimising thermal stress.")

h("12.6 Device-level thermal and energy behaviour", 2)
figure(figs["devices"], "Mean steady-state rise, observed temperature change and simulated energy per device")
table(["Device", "Cooling share", "Mean ΔT (°C)", "Mean P·R (°C)", "Mean duration (s)", "Mean power (W)",
       "Mean energy (J)", "Mean cooling rate (°C/s)"],
      [(r.device, pct(r.cooling_share), f"{r.mean_temp_change_c:+.2f}", f"{r.mean_steady_state_rise_c:.2f}",
        f"{r.mean_duration_s:.2f}", f"{r.mean_power_w:.2f}", f"{r.mean_energy_j:.1f}",
        f"{r.mean_cooling_rate_c_per_s:.3f}") for r in dev.itertuples()],
      "Per-device simulated behaviour averaged over all rows", size=8.5)

h("12.7 Prediction accuracy under family holdout", 2)
if PRED:
    n_mae = figure(figs["mae"], "Family-holdout MAE by model and feature set (log scale for duration and energy)")
    fam_glob = ps[(ps.protocol == "family_holdout") & (ps.stratum_type == "all") & (ps.method == "global")]
    rows_mae, bold_idx = [], []
    for t in TARGETS:
        for m in MODEL_ORDER:
            s = fam_glob[(fam_glob.target == t) & (fam_glob.model == m)].set_index("feature_config")
            if s.empty:
                continue
            if m == best[t].model:
                bold_idx.append(len(rows_mae))
            rows_mae.append((TARGET_LABEL[t] + f" ({UNIT[t]})", MODEL_SHORT[m],
                             *[num(s.loc[f, "mae"], 3) if f in s.index else "–" for f in FEAT_ORDER]))
    table(["Target", "Model", *[FEAT_SHORT[f] for f in FEAT_ORDER]], rows_mae,
          "Family-holdout MAE (20 folds pooled); the row of the best model for each target is bold",
          size=8.5, bold_rows=tuple(bold_idx))
    para(f"Figure {n_mae} and the table above give the pooled MAE over all 20 held-out families. The best "
         "configuration for each target is:")
    bullets([f"**{TARGET_LABEL[t]}:** {cfg(best[t].model, best[t].feature_config)}, MAE "
             f"{num(best[t].mae, 3)} {UNIT[t]}, RMSE {num(best[t].rmse, 3)} {UNIT[t]}, fold-to-fold MAE "
             f"standard deviation {num(best[t].fold_mae_sd, 3)}." for t in TARGETS])
    r_d, g_d = pick("duration_s", "Ridge", "M3_full_descriptors"), pick("duration_s", *REF)
    para(f"For duration with full descriptors, Ridge reaches {num(r_d.mae, 3)} s and GBM {num(g_d.mae, 3)} s, "
         f"a {pct(1 - g_d.mae / r_d.mae)} reduction. The baseline paper reported 88.8% for the same "
         "comparison on its 10-family setting.")
    n_fam = figure(figs["families"], f"Per-family MAE of {cfg(*REF)} when each family is held out")
    para(f"Figure {n_fam} shows that difficulty varies strongly between held-out families. This is why "
         "results pooled over random splits can be misleading.")
else:
    pending("Family-holdout MAE and RMSE for every model, feature set and target.")

h("12.8 Reliability of the 90% intervals", 2)
if PRED:
    rows_cov = []
    for t in TARGETS:
        for m in MODEL_ORDER:
            s = fam_glob[(fam_glob.target == t) & (fam_glob.model == m)].set_index("feature_config")
            if not s.empty:
                rows_cov.append((TARGET_LABEL[t], MODEL_SHORT[m],
                                 *[pct(s.loc[f, "coverage"]) if f in s.index else "–" for f in FEAT_ORDER]))
    table(["Target", "Model", *[FEAT_SHORT[f] for f in FEAT_ORDER]], rows_cov,
          "Global split-conformal coverage under family holdout (target 90%)", size=8.5)
    n_cov = figure(figs["coverage"], f"Coverage by true thermal regime for {cfg(*REF)} and each interval method")
    rows_reg = []
    for t in TARGETS:
        for m in METHOD_LABEL:
            cells = []
            for rg in sim.REGIMES:
                r = pick(t, *REF, method=m, stype="true_regime", stratum=rg)
                cells.append("–" if r is None else f"{pct(r.coverage)} / {num(r.mean_width, 2)}")
            if any(c != "–" for c in cells):
                fb = pick(t, *REF, method=m, stype="true_regime", stratum="saturation")
                rows_reg.append((TARGET_LABEL[t], METHOD_LABEL[m], *cells,
                                 pct(fb.fallback_share) if fb is not None else "–"))
    table(["Target", "Method", "Transient", "Transition", "Saturation", "Fallback (sat.)"], rows_reg,
          f"Coverage / mean interval width by true regime for {cfg(*REF)}", size=8)
    sat_n = pick("duration_s", *REF, stype="true_regime", stratum="saturation")
    g_sat = sat_n
    pr_sat = pick("duration_s", *REF, method="predicted_regime_mondrian", stype="true_regime", stratum="saturation")
    para(f"The saturation stratum has {int(sat_n.n):,} test rows over all folds, all of them on "
         f"{sat_devices[0]}. For duration, global intervals of {cfg(*REF)} cover {pct(g_sat.coverage)} of "
         f"saturation rows; predicted-regime Mondrian intervals cover {pct(pr_sat.coverage)}. Figure {n_cov} "
         "compares all methods and targets. The baseline reported 16.3%–34.8% for nonlinear models in its own "
         "configuration.")
    rows_dev = []
    for t in TARGETS:
        for m in ("global", "device_mondrian"):
            rows_dev.append((TARGET_LABEL[t], METHOD_LABEL[m],
                             *[pct(pick(t, *REF, method=m, stype="device", stratum=d).coverage) for d in DEVICES]))
    table(["Target", "Method", *DEVICES], rows_dev, f"Coverage by device for {cfg(*REF)}", size=8)
else:
    pending("Coverage, interval width and fallback share by method, true regime and device.")

h("12.9 Random split versus family holdout", 2)
if PRED:
    n_pr = figure(figs["protocol"], "MAE under family holdout versus a size-matched random split")
    table(["Target", "Configuration", "Family MAE", "Random MAE", "Random − family", "95% CI"],
          [(TARGET_LABEL[r.target], cfg(r.model, r.feature_config), num(r.family_holdout_mae, 3),
            num(r.random_split_mae, 3), f"{r.random_minus_family:+.3f}", f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]")
           for r in proto.itertuples()], "Optimism of random splits (negative = random split looks better)", size=8.5)
    n_opt = int(((proto.ci_high < 0)).sum())
    para(f"In {n_opt} of {len(proto)} configurations the random split gives a significantly lower MAE than "
         f"family holdout (95% CI entirely below zero). Figure {n_pr} shows the size of the gap. This is the "
         "optimism the baseline assumed but never measured.")
else:
    pending("Random-split minus family-holdout MAE with 95% CIs.")

h("12.10 Feature and model comparisons", 2)
if PRED:
    table(["Target", "Baseline", "Comparison", "Baseline MAE", "Comparison MAE", "Gain", "95% CI"],
          [(TARGET_LABEL[r.target], r.baseline.replace("GradientBoosting", "GBM"),
            r.comparison.replace("GradientBoosting", "GBM"), num(r.baseline_mae, 3), num(r.comparison_mae, 3),
            f"{r.mean_gain:+.3f}", f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]") for r in boot.itertuples()],
          "Paired workload bootstrap: gain = MAE(baseline) − MAE(comparison); positive favours comparison", size=7.5)
    sp = bootrow("duration_s", ("GradientBoosting", "M1_size_memory"), ("GradientBoosting", "M2_size_memory_sparse"))
    th = bootrow("temp_rise_c", ("GradientBoosting", "M1n_no_thermal_state"), ("GradientBoosting", "M1_size_memory"))
    para(f"Adding sparsity (GBM M1 → M2) changes duration MAE by {sp.mean_gain:+.3f} s "
         f"({pct(sp.mean_gain / sp.baseline_mae)} of the baseline; 95% CI [{sp.ci_low:+.3f}, {sp.ci_high:+.3f}]). "
         f"The baseline claimed about 50%. The thermal-state ablation (GBM M1n → M1) changes "
         f"temperature-change MAE by {th.mean_gain:+.3f} °C (95% CI [{th.ci_low:+.3f}, {th.ci_high:+.3f}]), "
         f"from {num(th.baseline_mae, 3)} to {num(th.comparison_mae, 3)} °C. Most of the temperature skill "
         "therefore comes from knowing the ambient and start temperatures, as Equation 3 predicts.")
else:
    pending("Paired bootstrap comparisons, including the M1n vs M1 thermal-state ablation.")

h("12.11 Performance metrics required by the project template", 2)
perf = []
if PRED:
    perf.append(("Accuracy", "; ".join(f"{TARGET_LABEL[t]} MAE {num(best[t].mae, 3)} {UNIT[t]} "
                                        f"(coverage {pct(best[t].coverage)})" for t in TARGETS),
                 "prediction_summary.csv"))
else:
    perf.append(("Accuracy", "PENDING", "prediction_summary.csv"))
perf.append(("Accuracy of thermal sign rule", f"{pct(sign['agreement_with_noise_free_sign'], 2)} noise-free; "
                                             f"{pct(sign['agreement_with_measured_sign'], 2)} with sensor noise",
             "sign_rule_check.json"))
if LAT:
    tl = pd.DataFrame(lat["timings"])
    g = tl[(tl.model == REF[0]) & (tl.feature_config == REF[1])]
    perf.append(("Response time (one prediction)",
                 f"{cfg(*REF)}: median {g.single_query_median_ms.median():.2f} ms per single-workload query "
                 f"(p95 up to {g.single_query_p95_ms.max():.2f} ms)", "latency_benchmark.json"))
    perf.append(("Processing time (training)",
                 f"{cfg(*REF)} fit on {int(g.train_rows.iloc[0]):,} rows: {g.fit_s.min():.1f}–{g.fit_s.max():.1f} s "
                 f"per target; batch prediction {g.batch_us_per_row.median():.2f} µs/row",
                 "latency_benchmark.json"))
    perf.append(("Processing time (data)", f"Dataset generation of {lat['dataset_rows']:,} rows: "
                                           f"{lat['dataset_generation_s']:.1f} s", "latency_benchmark.json"))
else:
    perf.append(("Response time / training time", "PENDING (latency benchmark not yet run)", "latency_benchmark.json"))
if PRED:
    perf.append(("Processing time (full study)", f"{meta['runtime_s'] / 60:.1f} min for all 40 folds on "
                                                 f"{mi.get('logical_cpus', '?') - 1} worker processes", "metadata.json"))
perf.append(("Energy consumption (simulated device)",
             f"Mean per inference: {dev.mean_energy_j.min():.1f} J ({dev.loc[dev.mean_energy_j.idxmin(), 'device']}) "
             f"to {dev.mean_energy_j.max():.1f} J ({dev.loc[dev.mean_energy_j.idxmax(), 'device']})",
             "device_thermal_summary.csv"))
perf.append(("Energy consumption (host computer)", "Not measured; no power meter was used", "–"))
perf.append(("Cost (₹)", "₹0: open-source software only, existing computer, no hardware purchased", "Section 9"))
if PRED:
    ffm_best = {t: ffm[(ffm.target == t) & (ffm.model == best[t].model) & (ffm.feature_config == best[t].feature_config)]
                for t in TARGETS}
    perf.append(("Success rate (fits)", f"{n_fits_done:,} of {n_fits_done + n_skipped:,} model fits completed "
                                        f"({pct(n_fits_done / (n_fits_done + n_skipped), 2)})",
                 "fold_strata_detail.csv, metadata.json"))
    perf.append(("Success rate (interval validity)",
                 "; ".join(f"{TARGET_LABEL[t]}: {int((ffm_best[t].coverage >= COVERAGE).sum())} of "
                           f"{len(ffm_best[t])} held-out families reach ≥ 90% coverage" for t in TARGETS),
                 "family_fold_metrics.csv"))
else:
    perf.append(("Success rate", "PENDING", "metadata.json"))
table(["Metric", "Measured value", "Source file"], perf,
      "Performance metrics measured in this project", widths=[4, 8.5, 3.5], size=9)

h("12.12 Solver diagnostics", 2)
if PRED:
    if conv_by_model.empty:
        para("No convergence warnings were raised by any fit.")
    else:
        table(["Model", "Fits with warnings", "Total warnings"],
              [(m, int(r.fits), int(r.warnings)) for m, r in conv_by_model.iterrows()],
              "Convergence warnings recorded during training", widths=[6, 5, 5])
    para(f"Skipped (failed) fits: {n_skipped}. Total study runtime: {meta['runtime_s'] / 60:.1f} minutes.")
else:
    pending("Convergence-warning and skipped-fit counts.")

# ───────────────────────────── 13. Comparison ─────────────────────────────
h("13. Comparison with Existing System", 1)
para("The baseline's numbers were produced from a 10-family configuration that does not match its own code. "
     f"This project re-measures each claim on the consistent {len(FAMILIES)}-family, {N_ROWS:,}-row "
     "configuration. The values are therefore re-measurements, not exact replications. The verdict column "
     "states what the re-measurement shows.")
cmp_rows = [
    ("Energy-optimal ≠ coolest device", "86.38%",
     f"{pct(fm['energy_opt_not_coolest_share'])} (end temp); {pct(1 - pk['energy_opt_in_coolest_set_share'])} (true peak)",
     "Depends on the temperature definition; mostly aligned with the true peak"),
    ("\"Absolute paradox\" (energy-optimal is hottest)", "3.50%",
     f"{pct(fm['energy_opt_is_hottest_share'])} (end temp); {pct(pk['energy_opt_unique_hottest_share'])} (true peak)",
     "Artifact of calling the end temperature the peak"),
    ("Net cooling during inference", "66.16%", pct(cool["net_cooling_share_measured"]),
     "Reproduced; set by the chosen start temperatures"),
    ("Sign-rule accuracy", "89.49%",
     f"{pct(sign['agreement_with_noise_free_sign'], 2)} noise-free; {pct(sign['agreement_with_measured_sign'], 2)} noisy",
     "Model identity; mismatches are sensor noise"),
    ("Device-ranking inversion", "100% of families", f"Comparable in {n_comparable}/{len(rank)} families",
     "Not supported; only one device saturates"),
]
if PRED:
    cmp_rows += [
        ("Saturation coverage collapse (duration)", "2.8%–34.8%",
         f"{cfg(*REF)}: {pct(g_sat.coverage)} global, {pct(pr_sat.coverage)} predicted-regime",
         "Measured on a single-device stratum"),
        ("GBM vs Ridge duration MAE (M3)", "−88.8%", f"{pct(1 - g_d.mae / r_d.mae)} reduction",
         "Re-measured under family holdout"),
        ("Sparsity gain (GBM M1→M2, duration)", "≈ 50%", f"{pct(sp.mean_gain / sp.baseline_mae)}, CI "
                                                        f"[{sp.ci_low:+.3f}, {sp.ci_high:+.3f}] s",
         "Re-measured with a bootstrap CI"),
        ("Random-split optimism", "Assumed, not measured",
         f"Significant in {n_opt}/{len(proto)} configurations", "Now measured"),
    ]
else:
    cmp_rows += [("Coverage, model gains, random-split optimism", "Various", "PENDING", "PENDING")]
table(["Claim", "Existing system", "This project", "Verdict"], cmp_rows,
      "Claim-by-claim comparison with the existing system", widths=[4.2, 2.8, 5, 4], size=8.5)
table(["Capability", "Existing", "Proposed"], [
    ("Configuration consistency", "No", "Yes"), ("Correct peak temperature", "No", "Yes"),
    ("Regime support check", "No", "Yes"), ("Random-split control", "No", "Yes (size-matched)"),
    ("Interval methods", "1", "4"), ("Stratified coverage (regime, device)", "Regime only", "Regime and device"),
    ("Bootstrap confidence intervals", "No", "Yes"), ("Thermal-state ablation", "No", "Yes (M1n)"),
    ("Recorded convergence problems", "No (suppressed)", "Yes"), ("Generated, verifiable numbers", "No", "Yes"),
], "Capability comparison", widths=[8, 4, 4])

# ───────────────────────────── 14. Innovation validation ─────────────────────────────
h("14. Innovation Validation", 1)
para("Each innovation from Section 6 was tested. The status is based only on the measured results above.")
val = [
    ("I1 Claim audit", "Classify each baseline claim", "5 descriptive claims classified (Section 13)", "Validated"),
    ("I2 Correct peak", "Compare the hottest-device share under end temp and true peak",
     f"{pct(fm['energy_opt_is_hottest_share'])} vs {pct(pk['energy_opt_unique_hottest_share'])}", "Validated"),
    ("I3 Rank support", "Count families where all devices appear in both regimes",
     f"{n_comparable}/{len(rank)}", "Validated (claim refuted)"),
]
if PRED:
    val += [
        ("I4 Random control", "CI of random − family MAE excludes 0", f"{n_opt}/{len(proto)} configurations",
         "Validated" if n_opt else "Not validated"),
        ("I5 Regime-aware intervals", "Saturation coverage: predicted-regime vs global (duration)",
         f"{pct(pr_sat.coverage)} vs {pct(g_sat.coverage)}",
         "Improves" if pr_sat.coverage > g_sat.coverage else "Does not improve"),
        ("I6 Thermal-state ablation", "CI of the M1n → M1 gain for ΔT excludes 0",
         f"{th.mean_gain:+.3f} °C [{th.ci_low:+.3f}, {th.ci_high:+.3f}]",
         "Validated" if th.ci_low > 0 or th.ci_high < 0 else "Not significant"),
    ]
else:
    val += [("I4–I6", "Prediction-based tests", "PENDING", "PENDING")]
val.append(("I7 Traceability", "Report built only from results/", "All result numbers read from files by "
                                                                  "make_report.py", "Validated"))
table(["Innovation", "Test", "Result", "Status"], val, "Validation of each innovation",
      widths=[3.5, 5.5, 4.5, 2.5], size=8.5)
para("A negative or neutral result is reported as such. For example, if regime-aware calibration does not "
     "raise saturation coverage, that is itself a finding: with so few saturation rows, all on one device, "
     "calibration cannot fix a group the calibration family barely contains.")

# ───────────────────────────── 15. Future ─────────────────────────────
h("15. Future Enhancement", 1)
numbered([
    "**Hardware validation.** Run real networks (exported to ONNX) on physical boards, for example an NVIDIA "
    "Jetson Orin Nano and a Raspberry Pi 5. Log on-die temperature sensors at 10 Hz or faster and power with "
    "an external meter, and record the true pre-run temperature for every run. Fit R and C per device from "
    "step responses before comparing with the simulator.",
    "**Richer thermal model.** Use multi-node RC networks with core-to-core coupling, DVFS, and throttling "
    "controllers that change power during a run.",
    "**Real workload descriptors.** Extract operator counts, memory traffic and sparsity from compiled graphs "
    "instead of synthetic profiles.",
    "**Better conditional coverage.** Try conformalized quantile regression, locally weighted conformal "
    "methods, and calibration families chosen to include long runs.",
    "**Scheduler integration.** Use the intervals inside a device-selection policy and measure deadline misses "
    "and throttling events in closed-loop simulation.",
    "**Wider saturation coverage.** Redesign device parameters or workload sizes so that every device reaches "
    "saturation. Regime-wise device rankings could then actually be tested.",
])

# ───────────────────────────── 16. Conclusion ─────────────────────────────
h("16. Conclusion", 1)
para("This project began from a draft study whose headline discoveries did not survive inspection. Once the "
     "simulator was made consistent and run at full scale, the descriptive claims fell into three groups:")
bullets([
    f"**Unsupported.** The device-ranking inversion is comparable in {n_comparable} of {len(rank)} families.",
    f"**Definition artifact.** The absolute paradox falls from {pct(fm['energy_opt_is_hottest_share'])} to "
    f"{pct(pk['energy_opt_unique_hottest_share'])} with the correct peak temperature.",
    "**Consequences of the model and its parameters.** Net cooling follows from the chosen start "
    "temperatures, and the sign rule is an identity of the model.",
])
if PRED:
    para(f"The prediction benchmark shows how well static descriptors predict cost for unseen families. The best "
         f"family-holdout MAEs are {num(best['duration_s'].mae, 3)} s for duration, "
         f"{num(best['energy_j'].mae, 3)} J for energy and {num(best['temp_rise_c'].mae, 3)} °C for "
         "temperature change. It also measures how much random splits overstate that skill, and where 90% "
         "intervals fail.")
else:
    para("The prediction benchmark measures how well static descriptors predict cost for unseen families, "
         "and where 90% intervals fail; its numbers are pending in this version.")
para("The lasting contribution is methodological. It is a reproducible pipeline that separates what a "
     "simulator assumes from what it reveals, evaluates predictors under realistic family shift, and "
     "produces documentation whose numbers can be traced to files. The results describe a simulator, not "
     "physical devices. Hardware validation (Section 15) is the necessary next step before any claim about "
     "real edge accelerators.")

# ───────────────────────────── references ─────────────────────────────
h("References", 1)
refs = [
    "S. Williams, A. Waterman, and D. Patterson, \"Roofline: An insightful visual performance model for multicore architectures,\" Communications of the ACM, vol. 52, no. 4, pp. 65–76, 2009.",
    "L. L. Zhang et al., \"nn-Meter: Towards accurate latency prediction of deep-learning model inference on diverse edge devices,\" in Proc. ACM MobiSys, 2021, pp. 81–93.",
    "Z. Li, M. Paolieri, and L. Golubchik, \"Inference latency prediction at the edge,\" arXiv:2210.02620, 2022.",
    "Y. Chai et al., \"PerfSAGE: Generalized inference performance predictor for arbitrary deep learning models on edge devices,\" arXiv:2301.10999, 2023.",
    "W. Huang et al., \"HotSpot: A compact thermal modeling methodology for early-stage VLSI design,\" IEEE Trans. VLSI Systems, vol. 14, no. 5, pp. 501–513, 2006.",
    "R. Lu, Y. Wang, and B. Liu, \"ThermE: Predictive management of shared thermal headroom for sustained LLM inference on thermally constrained edge SoCs,\" arXiv:2610.00267, 2026.",
    "V. Vovk, A. Gammerman, and G. Shafer, Algorithmic Learning in a Random World. Springer, 2005.",
    "J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani, and L. Wasserman, \"Distribution-free predictive inference for regression,\" J. American Statistical Association, vol. 113, no. 523, pp. 1094–1111, 2018.",
    "A. N. Angelopoulos and S. Bates, \"A gentle introduction to conformal prediction and distribution-free uncertainty quantification,\" Foundations and Trends in Machine Learning, vol. 16, no. 4, pp. 494–591, 2023.",
    "J. H. Friedman, \"Greedy function approximation: A gradient boosting machine,\" Annals of Statistics, vol. 29, no. 5, pp. 1189–1232, 2001.",
    "L. Breiman, \"Random forests,\" Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.",
    "C. E. Rasmussen and C. K. I. Williams, Gaussian Processes for Machine Learning. MIT Press, 2006.",
    "R. Tibshirani, \"Regression shrinkage and selection via the lasso,\" J. Royal Statistical Society B, vol. 58, no. 1, pp. 267–288, 1996.",
    "A. E. Hoerl and R. W. Kennard, \"Ridge regression: Biased estimation for nonorthogonal problems,\" Technometrics, vol. 12, no. 1, pp. 55–67, 1970.",
    "B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. Chapman & Hall, 1993.",
    "F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" J. Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
]
for i, r in enumerate(refs, 1):
    p = para(f"[{i}] {r}")
    p.paragraph_format.left_indent = Cm(.9)
    p.paragraph_format.first_line_indent = Cm(-.9)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT

# ───────────────────────────── appendices ─────────────────────────────
h("Appendix A: How to Reproduce", 1)
code("python simulation_study.py            # simulate, audit, train, evaluate -> results/\n"
     "python latency_benchmark.py           # timing on this machine -> results/latency_benchmark.json\n"
     "python build_tables.py                # results/paper_tables.md\n"
     "python make_report.py                 # report_figures/*.png and this .docx", size=9)
h("Appendix B: Result File Inventory", 1)
inv = []
for f in sorted(R.iterdir()):
    if f.suffix == ".csv":
        info = f"{sum(1 for _ in f.open(encoding='utf-8')) - 1:,} rows"
    else:
        info = "JSON / text"
    inv.append((f.name, info, f"{f.stat().st_size / 1024:,.1f} KB"))
table(["File", "Contents", "Size"], inv, "Files in results/ when this report was generated", widths=[8, 4, 4], size=9)

# footer page numbers and auto-update of the TOC on open
for s in doc.sections:
    fp = s.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field(fp, "PAGE", "1")
upd = OxmlElement("w:updateFields")
upd.set(qn("w:val"), "true")
doc.settings.element.append(upd)

doc.save(OUT)
print(f"Saved {OUT} (prediction results {'included' if PRED else 'PENDING'}, "
      f"latency {'included' if LAT else 'PENDING'}); {counter['fig']} figures, {counter['tab']} tables")
