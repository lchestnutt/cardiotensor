from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


def write_spatialgraph_am(
    out_path: str | Path,
    streamlines_xyz: list[np.ndarray],
    point_thickness: Sequence[np.ndarray] | np.ndarray | None = None,
    edge_scalar: Sequence[float]
    | np.ndarray
    | Mapping[str, Sequence[float] | np.ndarray]
    | None = None,
    edge_scalar_name: str | None = None,
) -> None:
    """
    Minimal Amira SpatialGraph writer with optional EDGE scalar blocks.

    Writes blocks:
      @1 VERTEX float[3]
      @2 EDGE   int[2]
      @3 EDGE   int          NumEdgePoints
      @4 POINT  float[3]     EdgePointCoordinates
      @5 POINT  float        thickness
      @6..     EDGE  float <name>   one or more per-edge scalars

    Parameters
    ----------
    out_path : str | Path
        Output .am path.
    streamlines_xyz : list[np.ndarray]
        List of polylines (x, y, z). Each array shape = (Ni, 3), Ni >= 2.
    point_thickness : np.ndarray | list[np.ndarray], optional
        Per-point thickness. Either a flat array of length sum(Ni) or a list
        aligned to streamlines with lengths Ni.
    edge_scalar : array-like | dict[str, array-like], optional
        - If a 1D array-like: one scalar per edge, use `edge_scalar_name`.
        - If a dict: multiple scalars, each value must be 1D, length = n_edges.
          Keys become field names.
    edge_scalar_name : str, optional
        Name for the single-scalar case. If `edge_scalar` is a dict, this is ignored.

    Notes
    -----
    - Field names are lightly validated for Amira compatibility.
    - Values are written as ASCII floats with 6 decimal places.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(streamlines_xyz) == 0:
        raise ValueError("No streamlines to write")

    streamlines_xyz = [np.asarray(sl, dtype=float) for sl in streamlines_xyz]
    num_points_per_edge = [int(sl.shape[0]) for sl in streamlines_xyz]
    if any(n < 2 for n in num_points_per_edge):
        raise ValueError("Each streamline must contain at least 2 points")

    n_edges = len(streamlines_xyz)

    # Build vertices: start and end of each edge
    vertices = np.vstack(
        [np.vstack((sl[0][None, :], sl[-1][None, :])) for sl in streamlines_xyz]
    )
    n_vertices = vertices.shape[0]

    # Connectivity (zero-based)
    edge_conn = np.column_stack(
        (
            np.arange(0, 2 * n_edges, 2, dtype=int),
            np.arange(1, 2 * n_edges, 2, dtype=int),
        )
    )

    # Concatenate all points
    points_concat = np.concatenate(streamlines_xyz, axis=0).astype(float)
    n_points_total = points_concat.shape[0]

    # Thickness block (@5)
    if point_thickness is None:
        thickness_concat = np.ones((n_points_total,), dtype=float)
    else:
        thickness_concat = _normalize_point_attribute(
            point_thickness, num_points_per_edge, n_points_total, "point_thickness"
        )

    # Edge scalar(s)
    edge_scalar_blocks: list[tuple[str, np.ndarray]] = []
    if edge_scalar is not None:
        if isinstance(edge_scalar, dict):
            # multiple fields
            for name, vals in edge_scalar.items():
                field_name = _sanitize_field_name(name, param="edge_scalar (dict key)")
                arr = _normalize_edge_attribute(
                    vals, n_edges, f"edge_scalar['{field_name}']"
                )
                edge_scalar_blocks.append((field_name, arr))
        else:
            # single field
            if not edge_scalar_name:
                raise ValueError(
                    "edge_scalar_name must be provided for single edge_scalar array"
                )
            field_name = _sanitize_field_name(
                edge_scalar_name, param="edge_scalar_name"
            )
            arr = _normalize_edge_attribute(edge_scalar, n_edges, "edge_scalar")
            edge_scalar_blocks.append((field_name, arr))

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

    # Edge scalar headers start at @6
    data_block_index = 6
    for name, _ in edge_scalar_blocks:
        header.append(f"EDGE {{ float {name} }} @{data_block_index}")
        data_block_index += 1
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

    # subsequent @k edge scalar blocks
    block_idx = 6
    for _, arr in edge_scalar_blocks:
        lines.append("")
        lines.append(f"@{block_idx}")
        for val in arr:
            lines.append(f"{float(val):.6f}")
        block_idx += 1

    out_path.write_text("\n".join(header + lines), encoding="utf-8")


def _sanitize_field_name(name: str, param: str) -> str:
    """
    Amira field names should be simple symbols without spaces or quotes.
    This is a light sanitization that preserves common names like HA, IA, AZ, EL.
    """
    if not isinstance(name, str) or len(name.strip()) == 0:
        raise ValueError(f"{param} must be a non-empty string")
    name = name.strip()
    # Replace spaces and forbidden chars
    bad = set(" \t\r\n\"'{}[]()@")
    if any(ch in bad for ch in name):
        name = "".join(ch if ch not in bad else "_" for ch in name)
    return name


def _normalize_point_attribute(
    attr: Sequence[np.ndarray] | np.ndarray,
    num_points_per_edge: list[int],
    n_points_total: int,
    name: str,
) -> np.ndarray:
    if isinstance(attr, np.ndarray):
        flat = attr.astype(float).ravel()
        if flat.shape[0] != n_points_total:
            raise ValueError(
                f"{name} length {flat.shape[0]} does not match total points {n_points_total}"
            )
        return flat
    # list case
    if len(attr) != len(num_points_per_edge):
        raise ValueError(f"{name} list must have one array per streamline")
    for i, (a, n) in enumerate(zip(attr, num_points_per_edge)):
        if len(a) != n:
            raise ValueError(
                f"{name}[{i}] length {len(a)} does not match streamline length {n}"
            )
    return np.concatenate([np.asarray(a, dtype=float).ravel() for a in attr], axis=0)


def _normalize_edge_attribute(
    attr: Sequence[float] | np.ndarray,
    n_edges: int,
    name: str,
) -> np.ndarray:
    arr = np.asarray(attr, dtype=float).ravel()
    if arr.shape[0] != n_edges:
        raise ValueError(
            f"{name} length {arr.shape[0]} does not match n_edges {n_edges}"
        )
    return arr
