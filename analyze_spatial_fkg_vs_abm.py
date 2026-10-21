#!/usr/bin/env python3
"""
Analyze fixed-terminal-time Spatial-FKG and agent-based benchmark results.

Benchmark design
----------------
- Spatial-FKG: 3 independent mean estimates per terminal time.
- Agent-based method: 100 stochastic realizations per terminal time.
- Terminal times: T = 20, 25, 30.

The analysis compares intracellular-state distributions, immune-label
distributions, tumor-relative immune spatial organization, CPU time,
and ABM ensemble convergence.

Pass the benchmark result directory with ``--results``. Figures are
displayed but not written to disk.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEFAULT_RESULTS = Path("benchmark_results")

TERMINAL_TIMES = (20, 25, 30)

R_NEIGH = 2
R_EDGE = 1
THETA_EDGE = 1.0

# Signed-distance bins shown/computed for the morphology profile.
DISTANCE_MIN = -20
DISTANCE_MAX = 20

# Scalar peripheral organization summary.
BAND_WIDTH = 3

CONVERGENCE_NS = (10, 50, 100)
N_SUBSETS = 100
RNG_SEED = 20260830
FKG_REFERENCE_REP = 1

# Expected directory layout under --results:
#   benchmark_results/
#     fkg/T20, T25, T30
#     edge_abm/T20, T25, T30
MODEL_DIRS = {
    "FKG": "fkg",
    "ABM-edge": "edge_abm",
}
EXPECTED_RUNS = {
    "FKG": 3,
    "ABM-edge": 100,
}

DISPLAY_NAMES = {
    "FKG": "Spatial-FKG",
    "ABM-edge": "Agent-based method",
}

REP_RE = re.compile(r"_rep(\d+)")


# -----------------------------------------------------------------------------
# File discovery / loading
# -----------------------------------------------------------------------------
def replicate_id(path: Path) -> int:
    m = REP_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse replicate number from: {path.name}")
    return int(m.group(1))


def discover_bundles(results_root: Path, model_label: str, T: int) -> List[Path]:
    d = results_root / MODEL_DIRS[model_label] / f"T{T}"
    if not d.is_dir():
        raise FileNotFoundError(f"Missing result directory: {d}")

    files = sorted(d.glob("*_benchmark_bundle.npz"), key=replicate_id)
    expected = EXPECTED_RUNS[model_label]

    if len(files) != expected:
        raise RuntimeError(
            f"{model_label}, T={T}: found {len(files)} bundles; "
            f"expected {expected}. Directory: {d}"
        )

    reps = [replicate_id(p) for p in files]
    if reps != list(range(1, expected + 1)):
        raise RuntimeError(
            f"{model_label}, T={T}: replicate IDs are not exactly "
            f"1..{expected}. Found: {reps}"
        )

    return files


def fixed_fkg_reference_path(results_root: Path, T: int) -> Path:
    """Return the pre-specified FKG scientific reference replicate."""
    paths = discover_bundles(results_root, "FKG", T)
    matches = [p for p in paths if replicate_id(p) == FKG_REFERENCE_REP]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one FKG reference rep {FKG_REFERENCE_REP:03d} "
            f"for T={T}; found {len(matches)}."
        )
    return matches[0]


def load_bundle(path: Path) -> Dict[str, np.ndarray]:
    required = {
        "marginal_x1",
        "marginal_x2",
        "marginal_x3",
        "tumor_grid",
        "immune_grid",
        "immune_label_mass",
    }

    with np.load(path, allow_pickle=False) as z:
        missing = required.difference(z.files)
        if missing:
            raise KeyError(f"{path} is missing keys: {sorted(missing)}")

        return {
            k: np.asarray(z[k], dtype=float)
            for k in required
        }


def read_runtime_summary_cpu_seconds(bundle_path: Path) -> float:
    p = Path(
        str(bundle_path).replace(
            "_benchmark_bundle.npz",
            "_runtime_summary.csv",
        )
    )

    if not p.exists():
        return float("nan")

    with p.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 1 or "cpu_seconds" not in rows[0]:
        return float("nan")

    try:
        return float(rows[0]["cpu_seconds"])
    except (TypeError, ValueError):
        return float("nan")


def extract_run_id(bundle_path: Path) -> str | None:
    stem = bundle_path.name.replace("_benchmark_bundle.npz", "")

    for token in ("edge_abm_T", "fkg_T"):
        idx = stem.find(token)
        if idx >= 0:
            return stem[idx:]

    return None


def read_true_cpu_seconds(bundle_path: Path) -> Tuple[float, str]:
    run_id = extract_run_id(bundle_path)

    if run_id is not None:
        time_file = bundle_path.parent / f"{run_id}_slurm_time.txt"

        if time_file.exists():
            user_t = None
            sys_t = None

            for line in time_file.read_text(errors="replace").splitlines():
                s = line.strip()

                if s.startswith("User time (seconds):"):
                    try:
                        user_t = float(s.split(":", 1)[1].strip())
                    except ValueError:
                        pass

                elif s.startswith("System time (seconds):"):
                    try:
                        sys_t = float(s.split(":", 1)[1].strip())
                    except ValueError:
                        pass

            if user_t is not None and sys_t is not None:
                return user_t + sys_t, "GNU time: user+system"

    return (
        read_runtime_summary_cpu_seconds(bundle_path),
        "runtime_summary fallback",
    )


# -----------------------------------------------------------------------------
# Generic vector helpers
# -----------------------------------------------------------------------------
def pad_vectors(vectors: List[np.ndarray]) -> np.ndarray:
    nmax = max(len(v) for v in vectors)
    out = np.zeros((len(vectors), nmax), dtype=float)

    for i, v in enumerate(vectors):
        out[i, :len(v)] = v

    return out


def pad_and_mean(vectors: List[np.ndarray]) -> np.ndarray:
    return pad_vectors(vectors).mean(axis=0)


def normalize_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    s = float(v.sum())

    if s <= 0:
        return np.zeros_like(v, dtype=float)

    return v / s


# -----------------------------------------------------------------------------
# Exact Spatial-FKG-style edge operator
# -----------------------------------------------------------------------------
def make_disk_kernel(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return ((x * x + y * y) <= radius * radius).astype(float)


def conv2_same_edgepad(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Same padding convention as the FKG script: np.pad(..., mode="edge").
    Vectorized over the small kernel support.
    """
    h, w = a.shape
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2

    ap = np.pad(a, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(a, dtype=float)

    for ki in range(kh):
        for kj in range(kw):
            kval = float(k[ki, kj])
            if kval == 0:
                continue

            out += kval * ap[ki:ki + h, kj:kj + w]

    return out


def tumor_region_and_edge(tumor_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    kernel_neigh = make_disk_kernel(R_NEIGH)
    N_t = (
        conv2_same_edgepad(
            tumor_grid.astype(float),
            kernel_neigh,
        )
        / float(kernel_neigh.sum())
    )

    core = N_t >= THETA_EDGE

    kernel_edge = make_disk_kernel(R_EDGE).astype(bool)
    h, w = N_t.shape
    kh, kw = kernel_edge.shape
    ph, pw = kh // 2, kw // 2

    padded = np.pad(
        N_t,
        ((ph, ph), (pw, pw)),
        mode="edge",
    )

    local_min = np.full_like(N_t, np.inf, dtype=float)

    for ki in range(kh):
        for kj in range(kw):
            if not kernel_edge[ki, kj]:
                continue

            local_min = np.minimum(
                local_min,
                padded[ki:ki + h, kj:kj + w],
            )

    edge = core & (local_min < THETA_EDGE)

    return core, edge


# -----------------------------------------------------------------------------
# Tumor-relative immune organization
# -----------------------------------------------------------------------------
DISTANCE_BINS = np.arange(DISTANCE_MIN, DISTANCE_MAX + 1, dtype=int)


def signed_distance_map(
    tumor_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Signed Euclidean distance relative to the run-specific Spatial-FKG-style edge.

    Convention:
      edge sites          -> 0
      inside tumor core   -> negative
      outside tumor core  -> positive

    Distance is measured to the detected edge.
    """
    core, edge = tumor_region_and_edge(tumor_grid)

    if not np.any(edge):
        raise RuntimeError("No tumor edge detected for a run.")

    # distance_transform_edt is zero where its input is False and positive
    # inside True regions. Using ~edge gives distance to the edge.
    dist_to_edge = distance_transform_edt(~edge)

    signed = dist_to_edge.astype(float)
    signed[core] *= -1.0
    signed[edge] = 0.0

    return signed, core, edge


def immune_signed_distance_profile(
    tumor_grid: np.ndarray,
    immune_grid: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Return:
      profile[k] = fraction of total immune mass whose signed distance rounds
                   to DISTANCE_BINS[k]
      band_fraction = fraction of immune mass with |distance| <= BAND_WIDTH

    Immune mass beyond [DISTANCE_MIN, DISTANCE_MAX] is omitted from the plotted
    profile but remains in the denominator, so the profile retains its physical
    interpretation as fraction of TOTAL immune mass.
    """
    immune_total = float(immune_grid.sum())

    if immune_total <= 0:
        return np.full(len(DISTANCE_BINS), np.nan), float("nan")

    signed, _, _ = signed_distance_map(tumor_grid)

    # Assign each lattice site to the nearest integer-distance bin.
    d_int = np.rint(signed).astype(int)

    profile = np.zeros(len(DISTANCE_BINS), dtype=float)

    for k, d in enumerate(DISTANCE_BINS):
        profile[k] = float(immune_grid[d_int == d].sum()) / immune_total

    band_fraction = float(
        immune_grid[np.abs(signed) <= BAND_WIDTH].sum()
        / immune_total
    )

    return profile, band_fraction


# -----------------------------------------------------------------------------
# Per-run precomputation
# -----------------------------------------------------------------------------
def preprocess_bundle(path: Path) -> Dict[str, np.ndarray | float]:
    b = load_bundle(path)

    profile, band_fraction = immune_signed_distance_profile(
        b["tumor_grid"],
        b["immune_grid"],
    )

    m3 = np.asarray(b["marginal_x3"], dtype=float)
    x3_total = float(m3.sum())
    x3_fraction = (
        float(m3[1] / x3_total)
        if x3_total > 0 and len(m3) > 1
        else float("nan")
    )

    immune_label_raw = np.asarray(
        b["immune_label_mass"],
        dtype=float,
    )
    immune_label_composition = normalize_vector(immune_label_raw)

    return {
        "marginal_x1": np.asarray(b["marginal_x1"], dtype=float),
        "marginal_x2": np.asarray(b["marginal_x2"], dtype=float),
        "marginal_x3": m3,
        "x3_fraction": x3_fraction,
        "tumor_grid": np.asarray(b["tumor_grid"], dtype=float),
        "immune_grid": np.asarray(b["immune_grid"], dtype=float),
        "immune_profile": profile,
        "band_fraction": band_fraction,
        "immune_label_raw": immune_label_raw,
        "immune_label_composition": immune_label_composition,
    }


def average_preprocessed(
    runs: List[Dict[str, np.ndarray | float]],
) -> Dict[str, np.ndarray | float]:
    m1 = pad_and_mean(
        [np.asarray(r["marginal_x1"], float) for r in runs]
    )
    m2 = pad_and_mean(
        [np.asarray(r["marginal_x2"], float) for r in runs]
    )
    m3 = pad_and_mean(
        [np.asarray(r["marginal_x3"], float) for r in runs]
    )

    x3_total = float(m3.sum())
    x3_fraction = (
        float(m3[1] / x3_total)
        if x3_total > 0 and len(m3) > 1
        else float("nan")
    )

    tumor_grid = np.mean(
        np.stack(
            [np.asarray(r["tumor_grid"], float) for r in runs],
            axis=0,
        ),
        axis=0,
    )

    immune_grid = np.mean(
        np.stack(
            [np.asarray(r["immune_grid"], float) for r in runs],
            axis=0,
        ),
        axis=0,
    )

    immune_profile = np.mean(
        np.stack(
            [np.asarray(r["immune_profile"], float) for r in runs],
            axis=0,
        ),
        axis=0,
    )

    band_fraction = float(
        np.mean([float(r["band_fraction"]) for r in runs])
    )

    immune_label_raw = pad_and_mean(
        [np.asarray(r["immune_label_raw"], float) for r in runs]
    )

    # Average run-level normalized compositions, so total immune mass does not
    # dominate the composition comparison.
    label_comp_matrix = pad_vectors(
        [
            np.asarray(r["immune_label_composition"], float)
            for r in runs
        ]
    )
    immune_label_composition = label_comp_matrix.mean(axis=0)

    return {
        "marginal_x1": m1,
        "marginal_x2": m2,
        "marginal_x3": m3,
        "x3_fraction": x3_fraction,
        "tumor_grid": tumor_grid,
        "immune_grid": immune_grid,
        "immune_profile": immune_profile,
        "band_fraction": band_fraction,
        "immune_label_raw": immune_label_raw,
        "immune_label_composition": immune_label_composition,
        "tumor_grid_mass": float(tumor_grid.sum()),
        "immune_grid_mass": float(immune_grid.sum()),
    }


def ensemble_average_with_timing(
    paths: List[Path],
) -> Tuple[Dict[str, np.ndarray | float | str], List[Dict[str, np.ndarray | float]]]:
    runs = [preprocess_bundle(p) for p in paths]
    avg = average_preprocessed(runs)

    cpu_values = []
    cpu_sources = []

    for p in paths:
        value, source = read_true_cpu_seconds(p)
        cpu_values.append(value)
        cpu_sources.append(source)

    cpu_values = np.asarray(cpu_values, dtype=float)

    if np.any(~np.isfinite(cpu_values)):
        bad = [
            str(paths[i])
            for i in np.where(~np.isfinite(cpu_values))[0]
        ]
        raise RuntimeError(
            "Could not read CPU timing for:\n  " + "\n  ".join(bad)
        )

    avg["cpu_mean"] = float(cpu_values.mean())
    avg["cpu_total"] = float(cpu_values.sum())
    avg["cpu_std"] = (
        float(cpu_values.std(ddof=1))
        if len(cpu_values) > 1
        else 0.0
    )
    avg["cpu_source"] = "; ".join(sorted(set(cpu_sources)))

    return avg, runs


# -----------------------------------------------------------------------------
# Discrepancies
# -----------------------------------------------------------------------------
def normalize_probability(v: np.ndarray) -> np.ndarray:
    """
    Normalize a nonnegative vector to a probability distribution.
    """
    v = np.asarray(v, dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.clip(v, 0.0, None)

    s = float(v.sum())
    if s <= 0:
        return np.zeros_like(v, dtype=float)

    return v / s


def align_two(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = max(len(a), len(b))

    aa = np.zeros(n, dtype=float)
    bb = np.zeros(n, dtype=float)

    aa[:len(a)] = a
    bb[:len(b)] = b

    return aa, bb


def total_variation_distance(
    value: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Total-variation distance between two discrete probability distributions:

        TV(P,Q) = 0.5 * sum_i |P_i - Q_i|

    TV lies in [0,1].  100*TV can be read as the percentage of probability
    mass that would need to be redistributed across categories/bins to make
    the two distributions identical.
    """
    value, reference = align_two(
        np.asarray(value, dtype=float),
        np.asarray(reference, dtype=float),
    )

    p = normalize_probability(value)
    q = normalize_probability(reference)

    if p.sum() <= 0 and q.sum() <= 0:
        return 0.0
    if p.sum() <= 0 or q.sum() <= 0:
        return 1.0

    return 0.5 * float(np.abs(p - q).sum())


def profile_probability_with_outside(profile: np.ndarray) -> np.ndarray:
    """
    Convert the tumor-relative immune profile into a complete probability
    vector without silently renormalizing away immune mass outside the plotted
    signed-distance window.

    The stored profile already uses TOTAL immune mass in its denominator, so

        sum(profile) <= 1.

    We append one final category containing all immune mass outside
    [DISTANCE_MIN, DISTANCE_MAX].
    """
    profile = np.asarray(profile, dtype=float)
    profile = np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)
    profile = np.clip(profile, 0.0, None)

    inside_mass = float(profile.sum())
    outside_mass = max(0.0, 1.0 - inside_mass)

    complete = np.concatenate(
        [profile, np.array([outside_mass], dtype=float)]
    )

    # Numerical safeguard only.
    return normalize_probability(complete)


def metric_bundle(avg):
    return {
        "x1": normalize_probability(avg["marginal_x1"]),
        "x2": normalize_probability(avg["marginal_x2"]),
        "x3_fraction": float(avg["x3_fraction"]),
        "profile": profile_probability_with_outside(
            avg["immune_profile"]
        ),
        "immune_labels": normalize_probability(
            avg["immune_label_composition"]
        ),
    }


def compute_distances(summary):
    """
    Two headline benchmark discrepancies.

    Scalar discrepancy:
        equal-weight mean of

          1) x1 total-variation distance,
          2) x2 total-variation distance,
          3) absolute x3=1 fraction difference,
          4) immune-label total-variation distance.

        The x3 term is an absolute fraction difference, avoiding the unstable
        division by the small FKG x3=1 fraction used in relative error.

    Cancer-immune organization discrepancy:
        total-variation distance between the tumor-relative immune profiles,
        with all immune mass outside the plotted distance window retained as
        one additional "outside" category.

    All components lie in [0,1], so multiplying by 100 gives a percentage
    discrepancy on a common scale.
    """
    out = {T: {} for T in TERMINAL_TIMES}

    for T in TERMINAL_TIMES:
        current = metric_bundle(summary["ABM-edge"][T])
        reference = metric_bundle(summary["FKG"][T])

        out[T]["x1"] = total_variation_distance(
            current["x1"], reference["x1"]
        )
        out[T]["x2"] = total_variation_distance(
            current["x2"], reference["x2"]
        )
        out[T]["x3"] = abs(
            current["x3_fraction"] - reference["x3_fraction"]
        )
        out[T]["immune_labels"] = total_variation_distance(
            current["immune_labels"],
            reference["immune_labels"],
        )
        out[T]["profile"] = total_variation_distance(
            current["profile"],
            reference["profile"],
        )

        out[T]["general"] = float(np.mean([
            out[T]["x1"],
            out[T]["x2"],
            out[T]["x3"],
            out[T]["immune_labels"],
        ]))
        out[T]["spatial"] = out[T]["profile"]

    return out


# -----------------------------------------------------------------------------
# ABM convergence to FKG
# -----------------------------------------------------------------------------
def convergence_errors(
    preloaded: Dict[str, Dict[int, List[Dict[str, np.ndarray | float]]]]
):
    """
    Stability of the agent-based ensemble relative to the 3-run mean Spatial-FKG.

      - scalar discrepancy:
          mean of TV(x1), TV(x2), absolute x3 fraction difference,
          and TV(immune-label composition)

      - cancer-immune organization discrepancy:
          TV distance of the tumor-relative immune profile

    Shading in the plot is the 10--90% interval across random subsets.
    """
    rng = np.random.default_rng(RNG_SEED)
    out = {T: {"general": {}, "spatial": {}} for T in TERMINAL_TIMES}

    for T in TERMINAL_TIMES:
        fkg_ref = metric_bundle(
            average_preprocessed(preloaded["FKG"][T])
        )
        abm_runs = preloaded["ABM-edge"][T]
        n_total = len(abm_runs)

        for n in CONVERGENCE_NS:
            general_values = []
            spatial_values = []

            if n == n_total:
                subsets = [np.arange(n_total)]
            else:
                subsets = [
                    np.sort(rng.choice(n_total, size=n, replace=False))
                    for _ in range(N_SUBSETS)
                ]

            for idx in subsets:
                avg = average_preprocessed([abm_runs[i] for i in idx])
                cur = metric_bundle(avg)

                d_x1 = total_variation_distance(
                    cur["x1"], fkg_ref["x1"]
                )
                d_x2 = total_variation_distance(
                    cur["x2"], fkg_ref["x2"]
                )
                d_x3 = abs(
                    cur["x3_fraction"] - fkg_ref["x3_fraction"]
                )
                d_label = total_variation_distance(
                    cur["immune_labels"],
                    fkg_ref["immune_labels"],
                )
                d_profile = total_variation_distance(
                    cur["profile"],
                    fkg_ref["profile"],
                )

                general_values.append(float(np.mean([
                    d_x1, d_x2, d_x3, d_label
                ])))
                spatial_values.append(d_profile)

            out[T]["general"][n] = np.asarray(
                general_values, dtype=float
            )
            out[T]["spatial"][n] = np.asarray(
                spatial_values, dtype=float
            )

    return out


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_cpu(summary, distances):
    """Plot CPU comparisons as two separate compact figures.

    Keeping the per-run and aggregate comparisons in separate figures makes
    each panel easier to resize and position independently in Illustrator.
    """
    Tvals = np.arange(len(TERMINAL_TIMES), dtype=float)
    width = 0.32

    fkg_means = np.array([summary["FKG"][T]["cpu_mean"] for T in TERMINAL_TIMES])
    fkg_stds = np.array([summary["FKG"][T]["cpu_std"] for T in TERMINAL_TIMES])
    abm_means = np.array([summary["ABM-edge"][T]["cpu_mean"] for T in TERMINAL_TIMES])
    abm_stds = np.array([summary["ABM-edge"][T]["cpu_std"] for T in TERMINAL_TIMES])
    abm_totals = np.array([summary["ABM-edge"][T]["cpu_total"] for T in TERMINAL_TIMES])

    colors = {"FKG": "steelblue", "ABM-edge": "0.72"}

    fig, ax = plt.subplots(figsize=(6.0, 4.6))

    ax.bar(
        Tvals - width / 2, fkg_means, width, yerr=fkg_stds, capsize=3,
        color=colors["FKG"], label="Spatial-FKG"
    )
    ax.bar(
        Tvals + width / 2, abm_means, width, yerr=abm_stds, capsize=3,
        color=colors["ABM-edge"], label="Agent-based method"
    )

    ax.set_yscale("log")
    ax.set_xticks(Tvals, [f"T={T}" for T in TERMINAL_TIMES])
    ax.set_ylabel("CPU time per run (seconds)")
    ax.set_title("Average CPU time")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.set_axisbelow(True)
    ax.legend()

    for i, T in enumerate(TERMINAL_TIMES):
        y_text = max(fkg_means[i], abm_means[i]) * 0.45
        ax.text(
            Tvals[i] - 0.34, y_text,
            f"Cell-state: {100.0 * distances[T]['general']:.1f}%\n"
            f"T-I organization: {100.0 * distances[T]['spatial']:.1f}%",
            fontsize=8, va="top"
        )

    fig.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(6.0, 4.6))
    ax2.bar(
        Tvals - width / 2, fkg_means, width, yerr=fkg_stds, capsize=3,
        color=colors["FKG"], label="Spatial-FKG"
    )
    ax2.bar(
        Tvals + width / 2, abm_totals, width,
        color=colors["ABM-edge"], label="Agent-based method (100 runs)"
    )

    ax2.set_yscale("log")
    ax2.set_xticks(Tvals, [f"T={T}" for T in TERMINAL_TIMES])
    ax2.set_ylabel("CPU time for reported mean (seconds)")
    ax2.set_title("Aggregate CPU time")
    ax2.grid(axis="y", alpha=0.25, which="both")
    ax2.set_axisbelow(True)
    ax2.legend()

    for i, T in enumerate(TERMINAL_TIMES):
        y_text = max(fkg_means[i], abm_totals[i]) * 0.45
        ax2.text(
            Tvals[i] - 0.34, y_text,
            f"Cell-state: {100.0 * distances[T]['general']:.1f}%\n"
            f"T-I organization: {100.0 * distances[T]['spatial']:.1f}%",
            fontsize=8, va="top"
        )

    fig2.tight_layout()

def plot_marginals(summary, key: str, symbol: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    for ax, T in zip(axes, TERMINAL_TIMES):
        xmax = 0

        for model in ("FKG", "ABM-edge"):
            v = np.asarray(summary[model][T][key], float)
            pos = np.where(v > 0)[0]

            if pos.size:
                xmax = max(xmax, int(pos[-1]))

        for model in ("FKG", "ABM-edge"):
            v = np.asarray(summary[model][T][key], float)

            ax.plot(
                np.arange(min(len(v), xmax + 1)),
                v[:xmax + 1],
                label=DISPLAY_NAMES[model],
            )

        ax.set_title(f"T={T}")
        ax.set_xlabel(symbol)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Ensemble-averaged marginal mass")
    axes[-1].legend()

    fig.suptitle(f"Averaged {symbol} marginal")
    fig.tight_layout()


def plot_tumor_relative_immune_profiles(summary):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), sharey=True)

    for ax, T in zip(axes, TERMINAL_TIMES):
        for model in ("FKG", "ABM-edge"):
            profile = np.asarray(
                summary[model][T]["immune_profile"],
                float,
            )

            ax.plot(
                DISTANCE_BINS,
                profile,
                marker="o",
                markersize=3,
                label=DISPLAY_NAMES[model],
            )

        ax.axvline(0, linestyle="--", linewidth=1)
        ax.axvspan(
            -BAND_WIDTH,
            BAND_WIDTH,
            alpha=0.08,
        )
        ax.set_title(f"T={T}")
        ax.set_xlabel(
            "Signed distance from tumor edge\n"
            "(negative = tumor interior)"
        )
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Fraction of total immune mass")
    axes[-1].legend()

    fig.suptitle(
        "Tumor-relative immune spatial profile\n"
        "run-level profiles averaged within each model"
    )
    fig.tight_layout()


def plot_band_fraction(summary):
    x = np.arange(len(TERMINAL_TIMES), dtype=float)
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.5, 5))

    for j, model in enumerate(("FKG", "ABM-edge")):
        vals = [
            summary[model][T]["band_fraction"]
            for T in TERMINAL_TIMES
        ]

        ax.bar(
            x + (j - 0.5) * width,
            vals,
            width,
            label=DISPLAY_NAMES[model],
        )

    ax.set_xticks(x, [f"T={T}" for T in TERMINAL_TIMES])
    ax.set_ylabel(
        rf"Immune fraction within $|d|\leq {BAND_WIDTH}$"
    )
    ax.set_title("Peripheral immune-band fraction")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()


def plot_immune_label_distributions(summary):
    """
    Figure 1: raw mean immune-label mass.
    Figure 2: mean run-normalized immune-label composition.
    """
    # Raw mass
    fig_raw, axes_raw = plt.subplots(
        1, 3, figsize=(15, 4.5)
    )

    for ax, T in zip(axes_raw, TERMINAL_TIMES):
        xmax = 0

        for model in ("FKG", "ABM-edge"):
            v = np.asarray(
                summary[model][T]["immune_label_raw"],
                float,
            )
            pos = np.where(v > 0)[0]
            if pos.size:
                xmax = max(xmax, int(pos[-1]))

        for model in ("FKG", "ABM-edge"):
            v = np.asarray(
                summary[model][T]["immune_label_raw"],
                float,
            )
            ax.plot(
                np.arange(min(len(v), xmax + 1)),
                v[:xmax + 1],
                marker="o",
                markersize=3,
                label=DISPLAY_NAMES[model],
            )

        ax.set_title(f"T={T}")
        ax.set_xlabel(r"Immune label $z=x_2$")
        ax.grid(alpha=0.25)

    axes_raw[0].set_ylabel("Mean immune-label mass")
    axes_raw[-1].legend()
    fig_raw.suptitle("Averaged immune-cell label distribution: raw mass")
    fig_raw.tight_layout()

    # Normalized composition
    fig_norm, axes_norm = plt.subplots(
        1, 3, figsize=(15, 4.5)
    )

    for ax, T in zip(axes_norm, TERMINAL_TIMES):
        xmax = 0

        for model in ("FKG", "ABM-edge"):
            v = np.asarray(
                summary[model][T]["immune_label_composition"],
                float,
            )
            pos = np.where(v > 0)[0]
            if pos.size:
                xmax = max(xmax, int(pos[-1]))

        for model in ("FKG", "ABM-edge"):
            v = np.asarray(
                summary[model][T]["immune_label_composition"],
                float,
            )
            ax.plot(
                np.arange(min(len(v), xmax + 1)),
                v[:xmax + 1],
                marker="o",
                markersize=3,
                label=DISPLAY_NAMES[model],
            )

        ax.set_title(f"T={T}")
        ax.set_xlabel(r"Immune label $z=x_2$")
        ax.grid(alpha=0.25)

    axes_norm[0].set_ylabel("Mean normalized immune-label fraction")
    axes_norm[-1].legend()
    fig_norm.suptitle(
        "Averaged immune-cell label distribution: normalized composition"
    )
    fig_norm.tight_layout()


def plot_scalar_quantities(summary):
    x = np.arange(len(TERMINAL_TIMES), dtype=float)
    width = 0.34

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for j, model in enumerate(("FKG", "ABM-edge")):
        x3 = [
            summary[model][T]["x3_fraction"]
            for T in TERMINAL_TIMES
        ]
        imass = [
            summary[model][T]["immune_grid_mass"]
            for T in TERMINAL_TIMES
        ]
        tmass = [
            summary[model][T]["tumor_grid_mass"]
            for T in TERMINAL_TIMES
        ]

        axes[0].bar(
            x + (j - 0.5) * width,
            x3,
            width,
            label=DISPLAY_NAMES[model],
        )
        axes[1].bar(
            x + (j - 0.5) * width,
            tmass,
            width,
            label=DISPLAY_NAMES[model],
        )
        axes[2].bar(
            x + (j - 0.5) * width,
            imass,
            width,
            label=DISPLAY_NAMES[model],
        )

    for ax in axes:
        ax.set_xticks(x, [f"T={T}" for T in TERMINAL_TIMES])
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_title(r"Immune-escaped tumor fraction $x_3=1$")
    axes[0].set_ylabel("Fraction")

    axes[1].set_title("Final tumor-grid mass")
    axes[1].set_ylabel("Tumor mass")

    axes[2].set_title("Final immune-grid mass")
    axes[2].set_ylabel("Immune mass")
    axes[2].legend()

    fig.tight_layout()


def plot_distances(distances):
    x = np.arange(len(TERMINAL_TIMES), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    scalar_vals = [
        100.0 * distances[T]["general"]
        for T in TERMINAL_TIMES
    ]
    organization_vals = [
        100.0 * distances[T]["spatial"]
        for T in TERMINAL_TIMES
    ]

    axes[0].bar(x, scalar_vals, width=0.55)
    axes[0].set_xticks(x, [f"T={T}" for T in TERMINAL_TIMES])
    axes[0].set_ylabel("Scalar discrepancy from Spatial-FKG (%)")
    axes[0].set_title(
        "Scalar discrepancy from Spatial-FKG\n"
        "mean of x1, x2, x3 and immune-label discrepancies"
    )
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0, 50)

    axes[1].bar(x, organization_vals, width=0.55)
    axes[1].set_xticks(x, [f"T={T}" for T in TERMINAL_TIMES])
    axes[1].set_ylabel("Cancer-immune organization discrepancy (%)")
    axes[1].set_title(
        "Cancer-immune organization discrepancy\n"
        "tumor-relative immune profile"
    )
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_ylim(0, 50)

    fig.suptitle("Agent-based method comparison relative to 3-run mean Spatial-FKG")
    fig.tight_layout()

def plot_convergence(conv, summary):
    metrics = ("general", "spatial")
    ns = np.asarray(CONVERGENCE_NS, dtype=float)

    fig, axes = plt.subplots(
        2, len(TERMINAL_TIMES),
        figsize=(14, 8),
        sharex=True,
    )

    for row, metric in enumerate(metrics):
        for col, T in enumerate(TERMINAL_TIMES):
            ax = axes[row, col]
            med, lo, hi = [], [], []

            for n in CONVERGENCE_NS:
                vals = 100.0 * np.asarray(conv[T][metric][n], float)
                med.append(float(np.median(vals)))
                lo.append(float(np.quantile(vals, 0.10)))
                hi.append(float(np.quantile(vals, 0.90)))

            med = np.asarray(med)
            lo = np.asarray(lo)
            hi = np.asarray(hi)

            ax.plot(ns, med, marker="o")
            ax.fill_between(ns, lo, hi, alpha=0.2)

            ax.set_xscale("log")
            ax.set_xticks(
                CONVERGENCE_NS,
                [str(n) for n in CONVERGENCE_NS],
            )
            ax.set_ylim(0, 50)
            ax.grid(alpha=0.25, which="both")

            if row == 0:
                mean_final_abm_tumor = summary["ABM-edge"][T]["tumor_grid_mass"]
                mean_final_fkg_tumor_mass = summary["FKG"][T]["tumor_grid_mass"]
                ax.set_title(
                    f"T={T}\n"
                    f"Agent-based method mean tumor cells = {mean_final_abm_tumor:,.0f}\n"
                    f"Spatial-FKG mean tumor mass (3 runs) = {mean_final_fkg_tumor_mass:,.0f}"
                )

            if col == 0:
                if metric == "general":
                    ax.set_ylabel("Scalar discrepancy from Spatial-FKG (%)")
                else:
                    ax.set_ylabel("Cancer-immune organization discrepancy (%)")

            if row == 1:
                ax.set_xlabel("Number of agent-based-method realizations")

    fig.suptitle(
        "Agent-based method ensemble convergence to 3-run mean Spatial-FKG\n"
        "Shading = 10--90% interval across random subsets (n=10,50); "
        "n=100 is the full ensemble"
    )
    fig.tight_layout()

# -----------------------------------------------------------------------------
# Console reporting
# -----------------------------------------------------------------------------
def print_summary(summary):
    print("\n" + "=" * 115)
    print("ENSEMBLE-AVERAGED RESULTS")
    print("=" * 115)

    print(
        f"{'Model':<12} {'T':>5} {'runs':>6} "
        f"{'CPU mean':>12} {'x3=1':>11} "
        f"{'band frac':>12} {'tumor mass':>14} {'immune mass':>14}"
    )
    print("-" * 115)

    for model in ("FKG", "ABM-edge"):
        for T in TERMINAL_TIMES:
            s = summary[model][T]

            print(
                f"{model:<12} {T:>5} {EXPECTED_RUNS[model]:>6d} "
                f"{s['cpu_mean']:>12.6g} "
                f"{s['x3_fraction']:>11.6g} "
                f"{s['band_fraction']:>12.6g} "
                f"{s['tumor_grid_mass']:>14.6g} "
                f"{s['immune_grid_mass']:>14.6g}"
            )

    print("\nCPU COST OF MEAN-POPULATION ESTIMATES (not equal-accuracy claims)")
    print("One Spatial-FKG run versus the SUM of 100 ABM realization CPU times:")
    for T in TERMINAL_TIMES:
        fkg_one = summary["FKG"][T]["cpu_mean"]
        abm_one = summary["ABM-edge"][T]["cpu_mean"]
        abm_all = summary["ABM-edge"][T]["cpu_total"]
        print(f"  T={T}: FKG one={fkg_one:.3f} s; "
              f"ABM one (mean)={abm_one:.3f} s; "
              f"ABM 100 (sum)={abm_all:.3f} CPU s; "
              f"aggregate/one-FKG={abm_all/fkg_one:.2f}x")


def print_distances(distances):
    print("\\n" + "=" * 90)
    print("HEADLINE DISCREPANCIES RELATIVE TO SPATIAL FKG")
    print("=" * 90)
    print(
        f"{'T':>5} {'scalar discrepancy':>20} "
        f"{'organization disc.':>22}"
    )
    print("-" * 90)

    for T in TERMINAL_TIMES:
        print(
            f"{T:>5} "
            f"{100*distances[T]['general']:>19.3f}% "
            f"{100*distances[T]['spatial']:>21.3f}%"
        )


def print_convergence(conv):
    print("\\n" + "=" * 105)
    print("AGENT-BASED-METHOD CONVERGENCE TO SPATIAL FKG")
    print(
        f"n=10,50: {N_SUBSETS} random subsets; "
        "n=100: full ABM ensemble."
    )
    print("Intervals are 10th--90th percentiles across random subsets.")
    print("=" * 105)

    for T in TERMINAL_TIMES:
        print(f"\\nT={T}")

        for metric in ("general", "spatial"):
            pieces = []

            for n in CONVERGENCE_NS:
                vals = 100.0 * np.asarray(conv[T][metric][n], float)

                med = np.median(vals)
                lo = np.quantile(vals, 0.10)
                hi = np.quantile(vals, 0.90)

                pieces.append(
                    f"n={n}: {med:.3f}% "
                    f"[{lo:.3f}, {hi:.3f}]"
                )

            print(f"  {metric:<12}: " + ", ".join(pieces))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"Results root (default: {DEFAULT_RESULTS})",
    )

    args = parser.parse_args()
    results_root = args.results.expanduser().resolve()

    print(f"Reading benchmark results from:\n  {results_root}")
    print("Precomputing run-level tumor-relative immune profiles...\n")

    summary = {
        model: {}
        for model in MODEL_DIRS
    }

    preloaded = {
        "FKG": {},
        "ABM-edge": {},
    }

    for T in TERMINAL_TIMES:
        # FKG scientific quantities, including final tumor mass:
        # average over all 3 FKG realizations.
        # FKG CPU mean/SD: also computed from the same 3 runs.
        all_fkg_paths = discover_bundles(results_root, "FKG", T)
        fkg_science, fkg_runs = ensemble_average_with_timing(all_fkg_paths)

        summary["FKG"][T] = fkg_science
        preloaded["FKG"][T] = fkg_runs

        print(
            f"Loaded FKG, T={T}: scientific quantities averaged over "
            f"{len(fkg_runs)} runs; "
            f"mean tumor mass={fkg_science['tumor_grid_mass']:.3f}, "
            f"mean immune mass={fkg_science['immune_grid_mass']:.3f}"
        )

        # ABM scientific quantities and convergence: all 100 realizations.
        abm_paths = discover_bundles(results_root, "ABM-edge", T)
        abm_avg, abm_runs = ensemble_average_with_timing(abm_paths)

        summary["ABM-edge"][T] = abm_avg
        preloaded["ABM-edge"][T] = abm_runs

        print(
            f"Loaded agent-based method, T={T}: {len(abm_runs)} runs; "
            f"mean tumor mass={abm_avg['tumor_grid_mass']:.3f}, "
            f"mean immune mass={abm_avg['immune_grid_mass']:.3f}"
        )

    distances = compute_distances(summary)
    conv = convergence_errors(preloaded)

    print_summary(summary)
    print_distances(distances)
    print_convergence(conv)

    plot_cpu(summary, distances)
    plot_marginals(summary, "marginal_x1", r"$x_1$")
    plot_marginals(summary, "marginal_x2", r"$x_2$")
    plot_scalar_quantities(summary)

    # Main morphology validation.
    plot_tumor_relative_immune_profiles(summary)
    plot_band_fraction(summary)

    plot_immune_label_distributions(summary)

    plot_distances(distances)
    plot_convergence(conv, summary)

    print("\nDisplaying figures. Close the windows to end the script.")
    plt.show()


if __name__ == "__main__":
    main()
