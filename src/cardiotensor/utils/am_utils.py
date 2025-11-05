from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union
import numpy as np


def write_spatialgraph_am(
    out_path: str | Path,
    streamlines_xyz: List[np.ndarray],
    point_thickness: Optional[Sequence[np.ndarray] | np.ndarray] = None,
    edge_scalar: Optional[Union[Sequence[float], np.ndarray]] = None,
    edge_scalar_name: str = "EdgeHA",
) -> None:
    """
    Minimal Amira SpatialGraph writer with optional EDGE scalar at @6.

    Writes blocks:
      @1 VERTEX float[3]
      @2 EDGE   int[2]
      @3 EDGE   int          NumEdgePoints
      @4 POINT  float[3]     EdgePointCoordinates
      @5 POINT  float        thickness
      @6 EDGE   float <name> optional per edge scalar

    streamlines_xyz is a list of polylines in (x, y, z)
    point_thickness can be a flat array of length sum(N_i) or a list aligned to streamlines
    edge_scalar is one value per edge
    """
    out_path = Path(out_path)

    if len(streamlines_xyz) == 0:
        raise ValueError("No streamlines to write")

    streamlines_xyz = [np.asarray(sl, dtype=float) for sl in streamlines_xyz]
    num_points_per_edge = [int(sl.shape[0]) for sl in streamlines_xyz]
    if any(n < 2 for n in num_points_per_edge):
        raise ValueError("Each streamline must contain at least 2 points")

    n_edges = len(streamlines_xyz)

    # Vertices, start and end for each edge
    vertices = np.vstack([np.vstack((sl[0][None, :], sl[-1][None, :])) for sl in streamlines_xyz])
    n_vertices = vertices.shape[0]

    # Connectivity, zero based
    edge_conn = np.column_stack((
        np.arange(0, 2 * n_edges, 2, dtype=int),
        np.arange(1, 2 * n_edges, 2, dtype=int),
    ))

    # Concatenated points
    points_concat = np.concatenate(streamlines_xyz, axis=0).astype(float)
    n_points_total = points_concat.shape[0]

    # Thickness for @5
    if point_thickness is None:
        thickness_concat = np.ones((n_points_total,), dtype=float)
    else:
        thickness_concat = _normalize_point_attribute(
            point_thickness, num_points_per_edge, n_points_total, "point_thickness"
        )

    # Optional per edge scalar for @6
    if edge_scalar is not None:
        edge_scalar_vals = _normalize_edge_attribute(edge_scalar, n_edges, "edge_scalar")
        write_edge_block = True
    else:
        edge_scalar_vals = None
        write_edge_block = False

    # Header
    header = []
    header.append("# AmiraMesh 3D ASCII 3.0")
    header.append(f"define VERTEX {n_vertices}")
    header.append(f"define EDGE {n_edges}")
    header.append(f"define POINT {n_points_total}")
    header.append("")
    header.append("Parameters {")
    header.append("  SpatialGraphUnitsVertex { }")
    header.append("  SpatialGraphUnitsEdge { }")
    header.append("  SpatialGraphUnitsPoint {")
    header.append("    thickness { Unit -1, Dimension -1 }")
    header.append("  }")
    header.append("  HistoryLogHead { }")
    header.append('  ContentType "HxSpatialGraph"')
    header.append("}")
    header.append("")
    header.append("VERTEX { float[3] VertexCoordinates } @1")
    header.append("EDGE { int[2] EdgeConnectivity } @2")
    header.append("EDGE { int NumEdgePoints } @3")
    header.append("POINT { float[3] EdgePointCoordinates } @4")
    header.append("POINT { float thickness } @5")
    if write_edge_block:
        header.append(f"EDGE {{ float {edge_scalar_name} }} @6")
    header.append("")

    # Data blocks
    lines = []

    # @1 vertices
    lines.append("@1")
    for v in vertices:
        lines.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

    # @2 connectivity
    lines.append("")
    lines.append("@2")
    for a, b in edge_conn:
        lines.append(f"{a} {b}")

    # @3 number of points per edge
    lines.append("")
    lines.append("@3")
    for n in num_points_per_edge:
        lines.append(str(int(n)))

    # @4 concatenated coordinates
    lines.append("")
    lines.append("@4")
    for p in points_concat:
        lines.append(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")

    # @5 per point thickness
    lines.append("")
    lines.append("@5")
    for t in thickness_concat:
        lines.append(f"{float(t):.6f}")

    # @6 optional per edge scalar
    if write_edge_block:
        lines.append("")
        lines.append("@6")
        for val in edge_scalar_vals:
            lines.append(f"{float(val):.6f}")

    out_path.write_text("\n".join(header + lines), encoding="utf-8")


def _normalize_point_attribute(
    attr: Sequence[np.ndarray] | np.ndarray,
    num_points_per_edge: List[int],
    n_points_total: int,
    name: str,
) -> np.ndarray:
    if isinstance(attr, np.ndarray):
        flat = attr.astype(float).ravel()
        if flat.shape[0] != n_points_total:
            raise ValueError(f"{name} length {flat.shape[0]} does not match total points {n_points_total}")
        return flat
    # list case
    if len(attr) != len(num_points_per_edge):
        raise ValueError(f"{name} list must have one array per streamline")
    for i, (a, n) in enumerate(zip(attr, num_points_per_edge)):
        if len(a) != n:
            raise ValueError(f"{name}[{i}] length {len(a)} does not match streamline length {n}")
    return np.concatenate([np.asarray(a, dtype=float).ravel() for a in attr], axis=0)


def _normalize_edge_attribute(
    attr: Union[Sequence[float], np.ndarray],
    n_edges: int,
    name: str,
) -> np.ndarray:
    arr = np.asarray(attr, dtype=float).ravel()
    if arr.shape[0] != n_edges:
        raise ValueError(f"{name} length {arr.shape[0]} does not match n_edges {n_edges}")
    return arr