#!/usr/bin/env python3
"""
trk_to_am.py
Convert a .trk or .npz streamlines file to an Amira SpatialGraph .am.

Options
  --edge-scalar-source {none,ha,elevation} select EDGE scalar source
  --edge-reduce {mean,median} aggregation across points
  --edge-name name used in EDGE { float <name> } @6
"""

import argparse
from pathlib import Path

from cardiotensor.utils.am_utils import write_spatialgraph_am
from cardiotensor.utils.streamlines_io_utils import (
    load_trk_streamlines,
    load_npz_streamlines,
    ha_to_degrees_per_streamline,
    compute_elevation_angles,
    reduce_per_edge,
)


def script():
    parser = argparse.ArgumentParser(
        description="Convert .trk or .npz to Amira SpatialGraph .am",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Path to .trk or .npz")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output .am path")
    parser.add_argument("--edge-scalar-source", choices=["none", "ha", "elevation"], default="ha",
                        help="Compute one scalar per edge from HA or elevation")
    parser.add_argument("--edge-reduce", choices=["mean", "median"], default="mean",
                        help="Reduction across points per edge")
    parser.add_argument("--edge-name", type=str, default="EdgeHA",
                        help="Name for EDGE { float <name> } @6")
    args = parser.parse_args()

    inp = args.input
    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    if inp.suffix.lower() == ".npz":
        streamlines_xyz, ha_list = load_npz_streamlines(inp)
    elif inp.suffix.lower() == ".trk":
        streamlines_xyz, ha_list = load_trk_streamlines(inp)
    else:
        raise ValueError("Unsupported extension. Use .trk or .npz")

    # Build EDGE scalar if requested
    edge_vals = None
    edge_name = args.edge_name
    if args.edge_scalar_source == "ha":
        if ha_list is None:
            
            raise ValueError("No HA values available. For .trk save data_per_point['HA'], for .npz include ha_values")
        per_point = ha_to_degrees_per_streamline(ha_list)
        edge_vals = reduce_per_edge(per_point, args.edge_reduce)
        if args.edge_name == "EdgeHA":
            edge_name = "EdgeHA"
    elif args.edge_scalar_source == "elevation":
        elev_pp = compute_elevation_angles(streamlines_xyz)
        edge_vals = reduce_per_edge(elev_pp, args.edge_reduce)
        if args.edge_name == "EdgeHA":
            edge_name = "Elevation"

    out_path = args.output if args.output is not None else inp.with_suffix(".am")

    write_spatialgraph_am(
        out_path=out_path,
        streamlines_xyz=streamlines_xyz,
        point_thickness=None,   # writes 1.0 per point
        edge_scalar=edge_vals,  # None if source is none
        edge_scalar_name=edge_name,
    )
    print(f"Wrote Amira SpatialGraph: {out_path}")


