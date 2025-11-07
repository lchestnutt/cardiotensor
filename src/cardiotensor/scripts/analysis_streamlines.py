#!/usr/bin/env python3
"""
streamline_compare.py — histogram by default, KDE optional

Distributions:
- Length (arc length), Curvature, Tortuosity

Features:
- Histogram (counts by default), KDE optional
- --normalize rescales each curve so y ∈ [0,1] (relative shape only)
- Percentile clipping via --clip PLOW PHIGH
- Axis scales selectable via --xscale/--yscale (linear/log)
- Saves PNG + PDF
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    from tqdm import tqdm
except Exception:

    def tqdm(it, **kwargs):  # fallback
        return it


mpl.rcParams.update(
    {
        "savefig.dpi": 300,
        "savefig.transparent": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 120,
        "font.size": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": False,
    }
)

# ---------- helpers ----------


def load_streamlines(path: Path, key: str | None = None) -> list[np.ndarray]:
    """
    Load streamlines from .trk or .npz/.npy object arrays.
    Always returns list of streamlines in (x,y,z) coordinates.
    """
    suffix = path.suffix.lower()

    if suffix == ".trk":
        from cardiotensor.utils.streamlines_io_utils import load_trk_streamlines

        streamlines_xyz, attrs = load_trk_streamlines(path)
        return [np.asarray(sl, dtype=np.float32) for sl in streamlines_xyz]

    elif suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        k = key if key is not None else "streamlines"
        arr = data[k]

    elif suffix == ".npy":
        arr = np.load(path, allow_pickle=True)

    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    if isinstance(arr, np.ndarray) and arr.dtype == object:
        streamlines = [np.asarray(sl) for sl in arr.tolist()]
    else:
        streamlines = [np.asarray(sl) for sl in arr]

    return [sl for sl in streamlines if sl.ndim == 2 and sl.shape[1] == 3]


VoxelSize = float | tuple[float, float, float]


def _scale_phys(P: np.ndarray, voxel_size: VoxelSize) -> np.ndarray:
    Q = P.astype(np.float64).copy()
    if np.isscalar(voxel_size):
        Q *= float(voxel_size)
    else:
        arr = np.asarray(voxel_size, dtype=float)
        Q *= (arr.ravel()[:3])[None, :]
    return Q


def streamline_length(sl_xyz: np.ndarray, voxel_size: VoxelSize) -> float:
    P = _scale_phys(sl_xyz, voxel_size)
    diffs = np.diff(P, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def chord_length(sl_xyz: np.ndarray, voxel_size: VoxelSize) -> float:
    P = _scale_phys(sl_xyz, voxel_size)
    return float(np.linalg.norm(P[-1] - P[0]))


def curvature_discrete(sl_xyz: np.ndarray, voxel_size: VoxelSize) -> np.ndarray:
    if len(sl_xyz) < 3:
        return np.zeros((0,), dtype=np.float32)
    P = _scale_phys(sl_xyz, voxel_size)
    A, B, C = P[:-2], P[1:-1], P[2:]
    AB, AC = B - A, C - A
    cross = np.cross(AB, AC)
    area2 = np.linalg.norm(cross, axis=1)
    ab = np.linalg.norm(AB, axis=1)
    bc = np.linalg.norm(C - B, axis=1)
    ac = np.linalg.norm(AC, axis=1)
    denom = ab * bc * ac
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(denom > 0, (area2) / denom, 0.0)
    return np.nan_to_num(kappa).astype(np.float32)


def percentile_bounds(
    values_list: list[np.ndarray], p_lo: float, p_hi: float
) -> tuple[float, float]:
    vals = np.concatenate(
        [v[np.isfinite(v)] for v in values_list if v.size > 0], axis=0
    )
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(vals, p_lo))
    hi = float(np.percentile(vals, p_hi))
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    return lo, hi


def gaussian_kde_1d(
    samples: np.ndarray, grid: np.ndarray, bandwidth: float | None = None
) -> np.ndarray:
    x = samples[np.isfinite(samples)].astype(np.float64)
    n = len(x)
    if n < 2:
        return np.zeros_like(grid)
    std = x.std(ddof=1)
    if bandwidth is None:
        h = (
            std * (n ** (-1.0 / 5.0))
            if std > 0
            else (np.ptp(x) / 100.0 if np.ptp(x) > 0 else 1.0)
        )
    else:
        h = float(bandwidth)
    inv = 1.0 / (h * np.sqrt(2.0 * np.pi))
    Z = (grid[:, None] - x[None, :]) / h
    return inv * np.exp(-0.5 * Z * Z).mean(axis=1)


def make_grid(lo: float, hi: float, n_points: int = 512) -> np.ndarray:
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, n_points)


# ---------- plotting ----------


def normalize_curve(y: np.ndarray, do_norm: bool) -> np.ndarray:
    if not do_norm or y.size == 0:
        return y
    m = np.max(y)
    return y / m if m > 0 else y


def plot_hist(ax, data, bins, xlabel, normalize, label):
    counts, edges = np.histogram(data, bins=bins)
    y = normalize_curve(counts, normalize)

    # Use ax.hist with black borders
    ax.hist(
        data,
        bins=bins,
        weights=y / counts if normalize and counts.max() > 0 else None,
        alpha=0.4,
        label=label,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(
        "Frequency" if not normalize else "Normalized (0–1)", fontweight="bold"
    )


def plot_kde(ax, data, grid, xlabel, bandwidth_mult, normalize, label):
    n = len(data)
    if n == 0:
        return
    std = data.std(ddof=1) if n > 1 else 0.0
    h_scott = (
        std * (n ** (-1.0 / 5.0))
        if std > 0
        else (np.ptp(data) / 100.0 if np.ptp(data) > 0 else 1.0)
    )
    bw = h_scott * bandwidth_mult if bandwidth_mult != 1.0 else None
    dens = gaussian_kde_1d(data, grid, bandwidth=bw)
    y = dens
    y = normalize_curve(y, normalize)
    ax.plot(grid, y, label=label)
    ax.fill_between(grid, 0, y, alpha=0.25)
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel("Density" if not normalize else "Normalized (0–1)", fontweight="bold")


# ---------- compute metrics ----------


def compute_metrics(
    path: Path, key: str | None, voxel_size: VoxelSize, min_points: int
):
    print(f"Load {path}")
    sls = load_streamlines(path, key)

    # 🔹 Quick test: print size (min/max in pixels)
    if sls:
        all_pts = np.vstack(sls)
        min_vals = all_pts.min(axis=0)
        max_vals = all_pts.max(axis=0)
        size = max_vals - min_vals
        print(
            f"Space size (pixels): {size} (x={size[0]:.1f}, y={size[1]:.1f}, z={size[2]:.1f})"
        )

    lengths, mean_curv, torts = [], [], []
    for sl in sls:
        if sl.shape[0] < min_points:
            continue
        L = streamline_length(sl, voxel_size)
        D = chord_length(sl, voxel_size)
        K = curvature_discrete(sl, voxel_size)
        tort = (L / D - 1.0) if D > 0 else np.nan
        lengths.append(L)
        mean_curv.append(K.mean() if K.size else 0.0)
        torts.append(tort)
    return dict(
        length=np.array(lengths),
        mean_curvature=np.array(mean_curv),
        tortuosity=np.array(torts),
    )


# ---------- CLI ----------


def script():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--labels", nargs="+", type=str)
    ap.add_argument("--key", type=str, default=None)
    ap.add_argument("--voxel-size", type=float, default=1.0)
    ap.add_argument("--min-points", type=int, default=2)
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--kde-points", type=int, default=512)
    ap.add_argument("--bandwidth-mult", type=float, default=1.0)
    ap.add_argument("--clip", type=float, nargs=2, default=(0.1, 99.9))
    ap.add_argument("--kde", action="store_true")
    ap.add_argument(
        "--normalize", action="store_true", help="Rescale y-axis between 0 and 1"
    )
    ap.add_argument("--xscale", type=str, choices=["linear", "log"], default="linear")
    ap.add_argument("--yscale", type=str, choices=["linear", "log"], default="linear")
    ap.add_argument("--outdir", type=Path, default=Path("./compare_out"))
    args = ap.parse_args()

    labels = args.labels or [p.stem for p in args.inputs]

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("\nComparing:")
    for p, lab in zip(args.inputs, labels):
        print(f"  - {lab}: {p}")
    print("")

    results = []
    for path, label in zip(args.inputs, labels):
        metrics = compute_metrics(path, args.key, args.voxel_size, args.min_points)
        metrics["label"] = label
        results.append(metrics)

    # Percentile bounds
    arr_len = [m["length"] for m in results if m["length"].size > 0]
    arr_curv = [m["mean_curvature"] for m in results if m["mean_curvature"].size > 0]
    arr_tort = [m["tortuosity"] for m in results if m["tortuosity"].size > 0]

    lo_len, hi_len = percentile_bounds(arr_len, *args.clip)
    lo_curv, hi_curv = percentile_bounds(arr_curv, *args.clip)
    lo_tort, hi_tort = percentile_bounds(arr_tort, *args.clip)

    grid_len = make_grid(lo_len, hi_len, args.kde_points)
    grid_curv = make_grid(lo_curv, hi_curv, args.kde_points)
    grid_tort = make_grid(lo_tort, hi_tort, args.kde_points)

    bins_len = np.linspace(lo_len, hi_len, args.bins + 1)
    bins_curv = np.linspace(lo_curv, hi_curv, args.bins + 1)
    bins_tort = np.linspace(lo_tort, hi_tort, args.bins + 1)

    # --- plotting ---
    for metric, grid, bins, xlabel in [
        ("length", grid_len, bins_len, "Streamline length"),
        ("mean_curvature", grid_curv, bins_curv, "Mean curvature"),
        ("tortuosity", grid_tort, bins_tort, "Tortuosity"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for m in results:
            data = m[metric]
            data = data[np.isfinite(data)]
            label = m["label"]
            if args.kde:
                plot_kde(
                    ax, data, grid, xlabel, args.bandwidth_mult, args.normalize, label
                )
            else:
                plot_hist(ax, data, bins, xlabel, args.normalize, label)

        ax.set_xscale(args.xscale)
        ax.set_yscale(args.yscale)

        # Force non-negative lower limits if linear
        if args.xscale == "linear":
            ax.set_xlim(left=0)
        if args.yscale == "linear":
            ax.set_ylim(bottom=0)

        ax.legend()
        fig.tight_layout()
        fig.savefig(args.outdir / f"{metric}.png", bbox_inches="tight")
        fig.savefig(args.outdir / f"{metric}.pdf", bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    script()
