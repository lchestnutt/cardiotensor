#!/usr/bin/env python3
"""
analysis_histograms_aha.py

Compute histograms of angle volumes (HA) per AHA 17-segment map.
Uses `create_aha_mask` and `resample` from `cardiotensor.analysis.analysis_utils`.

Example:
  analysis_histograms_aha.py params.conf --lv-mask /path/to/lv_mask --outdir ./aha_hists
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from cardiotensor.utils.DataReader import DataReader
from cardiotensor.utils.utils import read_conf_file
import cardiotensor.analysis.analysis_utils as utils
import csv


def script() -> None:
    ap = argparse.ArgumentParser(description="AHA-segmented histograms")
    ap.add_argument("input", type=Path, help=".conf file or directory to make histogram of")
    ap.add_argument("--hist-type", type=str, default="HA", help="Type of histogram to compute (e.g., 'HA', 'FA', 'IA)")
    ap.add_argument("--lv-mask", type=Path, required=True, help="LV mask directory (slices)")
    ap.add_argument("--axis-points", type=str, default=None, help="Override axis points as 'z1,y1,x1;z2,y2,x2'")
    ap.add_argument("--septum", type=str, default=None, help="Septum point 'z,y,x' (optional interactive if omitted)")
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--write_csv", type=bool, default=True, help="If true record histogram values in a CSV file in the output directory")
    ap.add_argument("--plot", type=bool, default=True, help="Whether to plot and save histograms")
    ap.add_argument("--divide-radial", type=bool, default=False, help="Whether to divide segments radially into 4 layers")
    args = ap.parse_args()

    # Resolve config/base
    if args.input.suffix.lower() == ".conf":
        params = read_conf_file(args.input)
        # When a .conf is provided, image data (histograms) live under OUTPUT_PATH/<hist_type>
        output_base = Path(params.get("OUTPUT_PATH"))
        images_path = str(output_base / args.hist_type)
        cfg_axis = params.get("AXIS_POINTS", [])
        # Place output in existing file structure
        output_base = Path(params.get("OUTPUT_PATH"))
        default_out = output_base / "Analysis"
    elif args.input.is_dir():
        images_path = str(args.input)
        cfg_axis = []
        default_out = Path("./aha_histograms")
    else:
        raise ValueError("Input must be a .conf or a directory")

    ## Use requested output directory or default
    outdir = args.outdir if args.outdir is not None else default_out
    outdir.mkdir(parents=True, exist_ok=True)

    # HA directory discovery
    ha_dir = Path(images_path)  # fallback
    ha_rdr = DataReader(ha_dir)

    # Axis points
    if args.axis_points:
        try:
            p1s, p2s = args.axis_points.split(";")
            axis_points = [tuple(map(float, p1s.split(","))), tuple(map(float, p2s.split(",")))]
        except Exception:
            raise ValueError("--axis-points must be 'z1,y1,x1;z2,y2,x2'")
    else:
        axis_points = cfg_axis if cfg_axis else None

    # Septum
    if args.septum:
        sept = tuple(map(float, args.septum.split(",")))
    else:
        # interactive picker (will open a matplotlib window)
        sept = utils.get_septum_point(str(args.lv_mask))

    if axis_points is None or len(axis_points) < 2:
        raise ValueError("Axis points must be provided either in .conf or via --axis-points")

    print("Creating AHA mask")
    # Made at the same scale as the LV mask
    # Radial map controls whether or not transmural depth is divided into 4
    if args.divide_radial:
        seg_map, seg_radial_map = utils.create_aha_mask(
            str(args.lv_mask), axis_points, sept, str(images_path), 1, return_radial_map=True
        )
        seg_map_to_use = seg_radial_map
    else:
        seg_map = utils.create_aha_mask(
            str(args.lv_mask), axis_points, sept, str(images_path), 1, return_radial_map=False
        )
        seg_map_to_use = seg_map

    # Resample segment map to HA reader shape
    target_shape = tuple(ha_rdr.shape)
    seg_resampled = utils.resample(seg_map_to_use, target_shape=target_shape, order=0, TwoD=False, is_mask=False)

    ## Find slices with data - only open these
    slice_has_data = np.any(seg_resampled != 0, axis=(1, 2))
    start= np.argmax(slice_has_data)
    end = len(slice_has_data) - 1 - np.argmax(slice_has_data[::-1])
    
    # Use histogram utility to compute per-segment 0-255 bins from the DataReader
    print("Computing per-segment histograms")
    bins, unique_segments = utils.histogram(rdr=ha_rdr, start=int(start), end=int(end), seg_map=seg_resampled, factor=1)
    
    if unique_segments is None:
        print("No segments found in segment map, exiting.")
        return

    # Write histogram data to a csv file for future analysis
    if args.write_csv:
        csv_filename = outdir / 'histogram.csv'
        fieldnames = (
            ['VALUE'] +
            [f'SEGMENT-{zone}'
            for zone in unique_segments]
        )
        entries = []
        for value in range(256):  # histogram bins
            row = {'VALUE': value}
            for i, seg in enumerate(unique_segments):  # segments
                row[f'SEGMENT-{seg}'] = bins[i, value]
            entries.append(row)
        
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(entries)

    # PLot per-segment histogram
    if args.plot:
        # Build an array for 17 AHA segments. If radial subdivision is present
        # (4 layers per segment) aggregate the four radial parts into the
        # corresponding base AHA segment.
        bins17 = np.zeros((17, bins.shape[1]), dtype=bins.dtype)
        if args.divide_radial:
            # unique_segments are expected to contain labels like 11,12,13,14
            # for segment 1, and 101..104 for segment 10. Map by arithmetic:
            # base_label = segment * 10, radial parts = base_label + 1..4
            unique_ints = [int(u) for u in unique_segments]
            for s in range(1, 18):
                base = s * 10
                for r in range(1, 5):
                    label = base + r
                    if label in unique_ints:
                        idx = unique_ints.index(label)
                        bins17[s - 1] += bins[idx]
        else:
            for i, s in enumerate(unique_segments):
                if 1 <= int(s) <= 17:
                    bins17[int(s) - 1] = bins[i]

        # Plot each segment histogram
        for seg_id in range(1, 18):
            fig = utils.plot_segment_histogram(
                bins17,
                segments=[seg_id],
                value_range=(-90, 90),
                smooth_sigma=0,
                normalize=False,
                show_mean=False,
                circular_mean=False,
                xlab='HA (deg)',
                ylab='Frequency',
                title=f'AHA Segment {seg_id} HA histogram'
            )

            out_png = outdir / f"hist_AHA_segment_{seg_id}.png"
            out_pdf = outdir / f"hist_AHA_segment_{seg_id}.pdf"
            fig.savefig(out_png, bbox_inches="tight")
            fig.savefig(out_pdf, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved segment {seg_id} histograms to {out_png} / {out_pdf}")

    print(f"Done. Figures in {outdir}")


if __name__ == "__main__":
    script()
