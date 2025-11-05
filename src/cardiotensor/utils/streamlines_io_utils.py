from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import numpy as np


def load_npz_streamlines(p: Path) -> Tuple[List[np.ndarray], List[np.ndarray] | None]:
    data = np.load(p, allow_pickle=True)
    raw_streamlines = data.get("streamlines")
    if raw_streamlines is None:
        raise ValueError("'streamlines' array missing in .npz")

    # stored as (z, y, x) convert to (x, y, z)
    streamlines_xyz = [
        np.asarray([(pt[2], pt[1], pt[0]) for pt in sl], dtype=np.float32)
        for sl in raw_streamlines.tolist()
    ]

    ha_list = None
    if "ha_values" in data:
        ha_obj = data["ha_values"]
        ha_list = [np.asarray(a) for a in ha_obj.tolist()]
        if len(ha_list) != len(streamlines_xyz):
            raise ValueError(
                f"ha_values length {len(ha_list)} does not match streamlines {len(streamlines_xyz)}"
            )
    return streamlines_xyz, ha_list


def load_trk_streamlines(p: Path) -> Tuple[List[np.ndarray], List[np.ndarray] | None]:
    import nibabel as nib

    obj = nib.streamlines.load(str(p))
    tg = obj.tractogram

    # Prefer RAS mm if available
    try:
        tg = tg.to_rasmm()
    except AttributeError:
        try:
            tg = tg.to_world()
        except Exception:
            pass

    streamlines_xyz = [np.asarray(sl, dtype=np.float32) for sl in tg.streamlines]

    ha_list = None
    dpp = getattr(tg, "data_per_point", None)
    if dpp and "HA" in dpp:
        ha_list = [np.asarray(a).reshape(-1) for a in dpp["HA"]]
        if len(ha_list) != len(streamlines_xyz):
            raise ValueError("HA list length does not match number of streamlines in TRK")
    return streamlines_xyz, ha_list


def ha_to_degrees_per_streamline(ha_list: List[np.ndarray]) -> List[np.ndarray]:
    out = []
    for ha in ha_list:
        ha = np.asarray(ha)
        if ha.size > 0 and np.nanmax(ha) > 1.5:  # likely 0..255
            ha_deg = (ha.astype(np.float32) / 255.0) * 180.0 - 90.0
        else:
            ha_deg = ha.astype(np.float32)
        out.append(ha_deg)
    return out


def compute_elevation_angles(streamlines_xyz: List[np.ndarray]) -> List[np.ndarray]:
    all_angles = []
    for pts in streamlines_xyz:
        pts = np.asarray(pts, dtype=np.float32)
        n = len(pts)
        if n < 2:
            all_angles.append(np.zeros((n,), dtype=np.float32))
            continue
        vecs = np.diff(pts, axis=0)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        normalized = np.divide(vecs, norms, out=np.zeros_like(vecs), where=norms != 0)
        elev = np.arcsin(np.clip(normalized[:, 2], -1.0, 1.0)) * (180.0 / np.pi)
        elev = np.concatenate([elev, [elev[-1]]]).astype(np.float32)
        all_angles.append(elev)
    return all_angles


def reduce_per_edge(values_per_point: List[np.ndarray], how: str = "mean") -> np.ndarray:
    if how == "mean":
        return np.array([float(np.nanmean(v)) if v.size else np.nan for v in values_per_point], dtype=float)
    if how == "median":
        return np.array([float(np.nanmedian(v)) if v.size else np.nan for v in values_per_point], dtype=float)
    raise ValueError("how must be 'mean' or 'median'")
