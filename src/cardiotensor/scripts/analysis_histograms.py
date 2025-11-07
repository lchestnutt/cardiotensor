#!/usr/bin/env python3
"""
volume_histograms_streaming.py
Streaming histograms for FA and angle volumes (HA, IA, AZ, EL) with optional mask.
Processes slice by slice to avoid high RAM use.

Key features
- Reads either a .conf (uses OUTPUT_PATH) or explicit dirs
- Auto-discovers angle dirs under OUTPUT_PATH
- Skips zeros created by mask before unit conversion
- Converts encodings:
    FA: 0..255 -> 0..1 if needed
    HA, IA, EL: 0..255 -> -90..90 deg
    AZ: 0..255 -> 0..360 deg or -180..180 via --az-range
- Quantization-aligned bin edges for byte-encoded data to remove comb-like dips
- Percentile clipping for non-quantized data using a streaming subsample
- Two-pass streaming: 1) estimate clip  2) fill histogram counts
- Optional moving-average smoothing of the displayed histogram
- Saves PNG and PDF and shows each figure

Examples
  volume_histograms_streaming.py params.conf
  volume_histograms_streaming.py --output-dir ./output
  volume_histograms_streaming.py --fa-dir ./output/FA --ha-dir ./output/HA
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from cardiotensor.utils.utils import read_conf_file
from cardiotensor.utils.DataReader import DataReader

mpl.rcParams.update({
    "savefig.dpi": 300,
    "savefig.transparent": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 120,
    "font.size": 13,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
})

# ---------- discovery and io ----------

def _discover_angle_dirs(base: Path) -> Dict[str, Path]:
    found = {}
    for key in ["HA", "IA", "AZ", "EL"]:
        p = base / key
        if p.exists() and p.is_dir():
            found[key] = p
    return found

def _reader(dir_path: Path) -> DataReader:
    rdr = DataReader(dir_path)
    if len(rdr.shape) != 3:
        raise ValueError(f"Expected 3D volume at {dir_path}, got shape {rdr.shape}")
    return rdr

# ---------- conversions ----------

def _fa_to_unit(vol: np.ndarray) -> np.ndarray:
    vmax = float(np.nanmax(vol)) if vol.size else 0.0
    if vmax > 1.5:
        return np.clip(vol / 255.0, 0.0, 1.0).astype(np.float32)
    return vol.astype(np.float32)

def _angle_to_deg(vol: np.ndarray, name: str, az_range: str) -> np.ndarray:
    vmax = float(np.nanmax(vol)) if vol.size else 0.0
    name = name.upper()
    if name in {"HA", "IA", "EL"}:
        deg = (vol / 255.0) * 180.0 - 90.0 if vmax > 1.5 else vol
    elif name == "AZ":
        if vmax > 2.0:
            if az_range == "0-360":
                deg = (vol / 255.0) * 360.0
            else:
                deg = (vol / 255.0) * 360.0 - 180.0
        else:
            deg = vol
    else:
        deg = vol
    return deg.astype(np.float32)

# ---------- quantization aligned edges ----------

def _quantized_edges(kind: str, az_range: str) -> np.ndarray:
    # 256 codes -> 257 edges, place edges at half steps
    k = np.arange(257, dtype=np.float32) - 0.5
    kind = kind.upper()
    if kind in {"HA", "IA", "EL"}:
        return -90.0 + 180.0 * k / 255.0
    if kind == "AZ":
        if az_range == "0-360":
            return 0.0 + 360.0 * k / 255.0
        else:
            return -180.0 + 360.0 * k / 255.0
    if kind == "FA":
        return (k / 255.0).clip(0.0, 1.0)
    raise ValueError(f"Unknown quantized kind {kind}")

# ---------- streaming utilities ----------

def _slice_values_raw_then_convert(
    rdr: DataReader,
    z: int,
    convert_fn: Callable[[np.ndarray], np.ndarray],
    mask_rdr: Optional[DataReader],
) -> np.ndarray:
    """
    Load one Z slice raw, apply mask if provided, drop zeros on the RAW values,
    then convert units, return a 1D float32 array.
    """
    raw = rdr.load_volume(start_index=z, end_index=z + 1)[0]  # shape (Y, X)

    # Apply mask if present
    if mask_rdr is not None:
        m = mask_rdr.load_volume(start_index=z, end_index=z + 1)[0]
        raw = raw[m > 0]

    # Drop zeros before any conversion to avoid mask artifacts
    if raw.size == 0:
        return raw.astype(np.float32)
    raw = raw.astype(np.float32)
    raw = raw[raw != 0.0]

    if raw.size == 0:
        return raw
    converted = convert_fn(raw)
    return converted.astype(np.float32)

def _estimate_clip_percentiles(
    rdr: DataReader,
    convert_fn: Callable[[np.ndarray], np.ndarray],
    mask_rdr: Optional[DataReader],
    p_lo: float,
    p_hi: float,
    sample_budget: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(12345)
    Z = rdr.shape[0]
    per_slice_quota = max(1, sample_budget // max(1, Z))
    samples = []
    for z in range(Z):
        vals = _slice_values_raw_then_convert(rdr, z, convert_fn, mask_rdr)
        if vals.size == 0:
            continue
        take = min(vals.size, per_slice_quota)
        idx = rng.choice(vals.size, size=take, replace=False)
        samples.append(vals[idx])
    if not samples:
        return 0.0, 1.0
    samp = np.concatenate(samples)
    lo = float(np.percentile(samp, p_lo))
    hi = float(np.percentile(samp, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float(np.nanmin(samp)), float(np.nanmax(samp))
    return lo, hi

def _stream_histogram(
    rdr: DataReader,
    convert_fn: Callable[[np.ndarray], np.ndarray],
    mask_rdr: Optional[DataReader],
    bin_edges: np.ndarray,
) -> np.ndarray:
    counts = np.zeros(len(bin_edges) - 1, dtype=np.int64)
    Z = rdr.shape[0]
    for z in range(Z):
        vals = _slice_values_raw_then_convert(rdr, z, convert_fn, mask_rdr)
        if vals.size == 0:
            continue
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        # keep only values within the histogram range
        vals = vals[(vals >= bin_edges[0]) & (vals <= bin_edges[-1])]
        if vals.size == 0:
            continue
        h, _ = np.histogram(vals, bins=bin_edges)
        counts += h
    return counts

# ---------- small smoothing utility ----------

def _smooth_counts(counts: np.ndarray, win: int) -> np.ndarray:
    if win is None or win < 2:
        return counts
    win = int(win)
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(counts.astype(np.float32), kernel, mode="same")

# ---------- plotting ----------

def _save_and_show_hist(counts: np.ndarray, edges: np.ndarray,
                        xlabel: str, title: str, out_png: Path, out_pdf: Path,
                        smooth: int = 0):
    y = _smooth_counts(counts, smooth)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(edges[:-1], y, where="post")
    ax.fill_between(edges[:-1], 0, y, step="post", alpha=0.3)
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# ---------- CLI ----------

def script():
    ap = argparse.ArgumentParser(
        description="Streaming histograms for FA and angles with optional mask.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", type=Path,
                    help="A .conf file or a directory containing FA, HA, IA, AZ, EL subfolders.")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Explicit OUTPUT_PATH if not using .conf.")
    ap.add_argument("--fa-dir", type=Path, default=None)
    ap.add_argument("--ha-dir", type=Path, default=None)
    ap.add_argument("--ia-dir", type=Path, default=None)
    ap.add_argument("--az-dir", type=Path, default=None)
    ap.add_argument("--el-dir", type=Path, default=None)
    ap.add_argument("--mask", type=Path, default=None,
                    help="Optional mask volume dir, voxels > 0 kept. Masked zeros are excluded by default.")
    ap.add_argument("--bins-angle", type=int, default=361,
                    help="Bins for non-quantized angles. Quantized data uses 256 aligned bins automatically.")
    ap.add_argument("--bins-fa", type=int, default=200,
                    help="Bins for non-quantized FA. Quantized data uses 256 aligned bins automatically.")
    ap.add_argument("--clip", type=float, nargs=2, default=(0.5, 99.5),
                    metavar=("PLOW", "PHIGH"),
                    help="Percentile clip for non-quantized data.")
    ap.add_argument("--az-range", choices=["0-360", "-180-180"], default="0-360")
    ap.add_argument("--sample-for-clip", type=int, default=1_000_000,
                    help="Target number of samples across slices to estimate percentiles.")
    ap.add_argument("--smooth", type=int, default=0,
                    help="Optional moving-average window in bins for display only.")
    ap.add_argument("--outdir", type=Path, default=Path("./volume_hist_streaming"))
    args = ap.parse_args()

    # Resolve OUTPUT_PATH or base dir
    if args.input.suffix.lower() == ".conf":
        params = read_conf_file(args.input)
        base_out = Path(params.get("OUTPUT_PATH", "./output"))
    elif args.input.is_dir():
        base_out = args.input
    else:
        raise ValueError("Input must be a .conf or a directory.")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Assemble dirs
    dirs: Dict[str, Path] = {}
    dirs["FA"] = args.fa_dir if args.fa_dir else (args.output_dir or base_out) / "FA"
    angles_explicit = {"HA": args.ha_dir, "IA": args.ia_dir, "AZ": args.az_dir, "EL": args.el_dir}
    for k, p in angles_explicit.items():
        if p is not None:
            dirs[k] = p
    if not any(angles_explicit.values()):
        dirs.update(_discover_angle_dirs(args.output_dir or base_out))

    # Keep only existing dirs
    dirs = {k: v for k, v in dirs.items() if v is not None and v.exists()}

    # Optional mask
    mask_rdr = _reader(args.mask) if args.mask is not None else None

    p_lo, p_hi = args.clip

    # ---------- FA ----------
    if "FA" in dirs and dirs["FA"].exists():
        print(f"[FA] reading from {dirs['FA']}")
        fa_r = _reader(dirs["FA"])

        # Peek a small sample before conversion to detect byte encoding
        peek = fa_r.load_volume(start_index=0, end_index=1)[0].astype(np.float32).ravel()
        if mask_rdr is not None:
            m0 = mask_rdr.load_volume(start_index=0, end_index=1)[0]
            peek = peek[m0 > 0]
        peek = peek[peek != 0.0]
        is_byte = peek.size > 0 and np.nanmax(peek) > 1.5

        if is_byte:
            edges = _quantized_edges("FA", args.az_range)
        else:
            lo, hi = _estimate_clip_percentiles(fa_r, _fa_to_unit, mask_rdr, p_lo, p_hi, args.sample_for_clip)
            edges = np.linspace(lo, hi, args.bins_fa + 1)

        counts = _stream_histogram(fa_r, _fa_to_unit, mask_rdr, edges)
        _save_and_show_hist(counts, edges, "FA", "FA histogram",
                            outdir / "hist_FA.png", outdir / "hist_FA.pdf",
                            smooth=args.smooth)
    else:
        print("FA directory not found, skipping FA.")

    # ---------- Angles ----------
    for ang in ["HA", "IA", "AZ", "EL"]:
        if ang not in dirs:
            continue
        print(f"[{ang}] reading from {dirs[ang]}")
        ang_r = _reader(dirs[ang])

        # Peek raw slice to detect byte encoding
        peek = ang_r.load_volume(start_index=0, end_index=1)[0].astype(np.float32).ravel()
        if mask_rdr is not None:
            m0 = mask_rdr.load_volume(start_index=0, end_index=1)[0]
            peek = peek[m0 > 0]
        peek = peek[peek != 0.0]
        is_byte = peek.size > 0 and np.nanmax(peek) > 1.5

        convert = (lambda arr, a=ang: _angle_to_deg(arr, a, args.az_range))

        if is_byte:
            edges = _quantized_edges(ang, args.az_range)
        else:
            lo, hi = _estimate_clip_percentiles(ang_r, convert, mask_rdr, p_lo, p_hi, args.sample_for_clip)
            edges = np.linspace(lo, hi, args.bins_angle + 1)

        counts = _stream_histogram(ang_r, convert, mask_rdr, edges)
        xlabel = f"{ang} (degrees)"
        _save_and_show_hist(counts, edges, xlabel, f"{ang} histogram",
                            outdir / f"hist_{ang}.png", outdir / f"hist_{ang}.pdf",
                            smooth=args.smooth)

    print(f"Done, figures in {outdir}")

if __name__ == "__main__":
    script()
