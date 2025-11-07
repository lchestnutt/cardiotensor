import math
from pathlib import Path
import numpy as np
from alive_progress import alive_bar
from typing import Dict, List, Tuple, Optional

import nibabel as nib
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from dipy.io.streamline import save_trk


from cardiotensor.utils.DataReader import DataReader
from cardiotensor.utils.downsampling import downsample_vector_volume, downsample_volume
from cardiotensor.utils.am_utils import write_spatialgraph_am


def trilinear_interpolate_vector(
    vector_field: np.ndarray, pt: tuple[float, float, float]
) -> np.ndarray:
    """
    Given a fractional (z,y,x), returns the trilinearly‐interpolated 3‐vector
    from `vector_field` (shape = (3, Z, Y, X)). Clamps to nearest voxel if out‐of‐bounds.
    """
    zf, yf, xf = pt
    _, Z, Y, X = vector_field.shape

    # Clamp floor and ceil to valid ranges
    z0 = max(min(int(np.floor(zf)), Z - 1), 0)
    z1 = min(z0 + 1, Z - 1)
    y0 = max(min(int(np.floor(yf)), Y - 1), 0)
    y1 = min(y0 + 1, Y - 1)
    x0 = max(min(int(np.floor(xf)), X - 1), 0)
    x1 = min(x0 + 1, X - 1)

    dz = zf - z0
    dy = yf - y0
    dx = xf - x0

    # 8 corner vectors
    c000 = vector_field[:, z0, y0, x0]
    c001 = vector_field[:, z0, y0, x1]
    c010 = vector_field[:, z0, y1, x0]
    c011 = vector_field[:, z0, y1, x1]
    c100 = vector_field[:, z1, y0, x0]
    c101 = vector_field[:, z1, y0, x1]
    c110 = vector_field[:, z1, y1, x0]
    c111 = vector_field[:, z1, y1, x1]

    # Interpolate along X
    c00 = c000 * (1 - dx) + c001 * dx
    c01 = c010 * (1 - dx) + c011 * dx
    c10 = c100 * (1 - dx) + c101 * dx
    c11 = c110 * (1 - dx) + c111 * dx

    # Interpolate along Y
    c0 = c00 * (1 - dy) + c01 * dy
    c1 = c10 * (1 - dy) + c11 * dy

    # Interpolate along Z
    c = c0 * (1 - dz) + c1 * dz
    return c  # shape (3,)


def trilinear_interpolate_scalar(
    volume: np.ndarray, pt: tuple[float, float, float]
) -> float:
    """
    Trilinearly interpolate a scalar volume at fractional point (z, y, x).
    Clamps to valid range.
    """
    zf, yf, xf = pt
    Z, Y, X = volume.shape

    z0 = int(np.floor(zf))
    z1 = min(z0 + 1, Z - 1)
    y0 = int(np.floor(yf))
    y1 = min(y0 + 1, Y - 1)
    x0 = int(np.floor(xf))
    x1 = min(x0 + 1, X - 1)

    dz = zf - z0
    dy = yf - y0
    dx = xf - x0

    c000 = volume[z0, y0, x0]
    c001 = volume[z0, y0, x1]
    c010 = volume[z0, y1, x0]
    c011 = volume[z0, y1, x1]
    c100 = volume[z1, y0, x0]
    c101 = volume[z1, y0, x1]
    c110 = volume[z1, y1, x0]
    c111 = volume[z1, y1, x1]

    c00 = c000 * (1 - dx) + c001 * dx
    c01 = c010 * (1 - dx) + c011 * dx
    c10 = c100 * (1 - dx) + c101 * dx
    c11 = c110 * (1 - dx) + c111 * dx

    c0 = c00 * (1 - dy) + c01 * dy
    c1 = c10 * (1 - dy) + c11 * dy

    c = c0 * (1 - dz) + c1 * dz
    return float(c)


def trace_streamline(
    start_pt: tuple[float, float, float],
    vector_field: np.ndarray,
    fa_volume: np.ndarray | None = None,
    fa_threshold: float = 0.1,
    step_length: float = 0.5,
    max_steps: int | None = 1000,
    angle_threshold: float = 60.0,
    eps: float = 1e-10,
    direction: int = 1,
) -> list[tuple[float, float, float]]:
    """
    Trace one streamline from `start_pt` (z,y,x) in the continuous vector_field.
    - Interpolate & normalize each sub‐step
    - Move forward by `step_length` voxels each step using RK4
    - `direction` = +1 (default) or -1 to reverse integration direction
    """
    Z, Y, X = vector_field.shape[1:]
    coords: list[tuple[float, float, float]] = [
        (float(start_pt[0]), float(start_pt[1]), float(start_pt[2]))
    ]
    current_pt = np.array(start_pt, dtype=np.float64)
    prev_dir: np.ndarray | None = None  # previous unit vector

    def interp_unit(pt: np.ndarray) -> np.ndarray | None:
        """Return a normalized direction vector at fractional pt, or None if invalid."""
        vec = trilinear_interpolate_vector(vector_field, (pt[0], pt[1], pt[2]))
        if np.isnan(vec).any():
            return None
        norm = np.linalg.norm(vec)
        if norm < eps:
            return None
        return np.array([vec[2], vec[1], vec[0]]) / norm * direction  # flip to (z,y,x)

    step_count = 0
    while max_steps is None or step_count < max_steps:
        step_count += 1

        if fa_volume is not None:
            fa_value = trilinear_interpolate_scalar(fa_volume, tuple(current_pt))
            if fa_value < fa_threshold:
                break

        k1 = interp_unit(current_pt)
        if k1 is None:
            break
        if prev_dir is not None:
            angle = np.degrees(np.arccos(np.clip(np.dot(prev_dir, k1), -1.0, 1.0)))
            if angle > angle_threshold:
                break

        mid1 = current_pt + 0.5 * step_length * k1
        k2 = interp_unit(mid1)
        if k2 is None:
            break

        mid2 = current_pt + 0.5 * step_length * k2
        k3 = interp_unit(mid2)
        if k3 is None:
            break

        end_pt = current_pt + step_length * k3
        k4 = interp_unit(end_pt)
        if k4 is None:
            break

        angle4 = np.degrees(np.arccos(np.clip(np.dot(k1, k4), -1.0, 1.0)))
        if angle4 > angle_threshold:
            break

        increment = (step_length / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        next_pt = current_pt + increment
        next_dir = k1

        zn, yn, xn = next_pt
        if not (0 <= zn < Z and 0 <= yn < Y and 0 <= xn < X):
            break

        coords.append((float(zn), float(yn), float(xn)))
        current_pt = next_pt
        prev_dir = next_dir

    return coords


def generate_streamlines_from_vector_field(
    vector_field: np.ndarray,
    seed_points: np.ndarray,
    fa_volume: np.ndarray | None = None,
    fa_threshold: float = 0.1,
    step_length: float = 0.5,
    max_steps: int | None = None,
    angle_threshold: float = 60.0,
    min_length_pts: int = 10,
    bidirectional: bool = True,
) -> list[list[tuple[float, float, float]]]:
    """
    Given a 3D vector_field (shape = (3, Z, Y, X)) and a set of integer‐seed voxels,
    returns a list of streamlines (each streamline = a list of float (z,y,x) points),
    filtered so that only those longer than `min_length_pts` are kept.
    Optionally trace in both directions from each seed point.
    """
    all_streamlines: list[list[tuple[float, float, float]]] = []
    total_seeds = len(seed_points)

    with alive_bar(total_seeds, title="Tracing Streamlines") as bar:
        for zi, yi, xi in seed_points:
            start = (float(zi), float(yi), float(xi))

            # Forward tracing
            forward_pts = trace_streamline(
                start_pt=start,
                vector_field=vector_field,
                fa_volume=fa_volume,
                fa_threshold=fa_threshold,
                step_length=step_length,
                max_steps=max_steps,
                angle_threshold=angle_threshold,
                direction=1,
            )

            # Backward tracing if enabled
            if bidirectional:
                backward_pts = trace_streamline(
                    start_pt=start,
                    vector_field=vector_field,
                    fa_volume=fa_volume,
                    fa_threshold=fa_threshold,
                    step_length=step_length,
                    max_steps=max_steps,
                    angle_threshold=angle_threshold,
                    direction=-1,
                )
                # Remove duplicate seed point and reverse
                backward_pts = backward_pts[::-1][:-1] if len(backward_pts) > 1 else []
                full_streamline = backward_pts + forward_pts
            else:
                full_streamline = forward_pts

            if len(full_streamline) >= min_length_pts:
                all_streamlines.append(full_streamline)

            bar()

    return all_streamlines


def save_trk_dipy_from_vox_zyx(
    streamlines_zyx: list[list[tuple[float, float, float]]],
    out_path: str | Path,
    vol_shape_zyx: tuple[int, int, int],
    voxel_sizes_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    data_values: Optional[List[np.ndarray]] = None,
    data_name: Optional[str] = None,
):
    """
    Save streamlines given in voxel indices (z,y,x) as TrackVis .trk using DIPY.
    Optionally attach one per-point scalar list under `data_name`.
    """
    Z, Y, X = vol_shape_zyx
    vz, vy, vx = voxel_sizes_zyx

    affine = np.array([[vx, 0, 0, 0],
                       [0, vy, 0, 0],
                       [0, 0, vz, 0],
                       [0, 0, 0, 1]], dtype=np.float32)

    # DIPY expects NIfTI with shape (X, Y, Z)
    ref_img = nib.Nifti1Image(np.zeros((X, Y, Z), dtype=np.uint8), affine)
    
    # reorder each streamline from (z,y,x) to (x,y,z)
    sl_xyz_vox = [np.stack([np.asarray(sl)[:, 2], np.asarray(sl)[:, 1], np.asarray(sl)[:, 0]], axis=1).astype(np.float32)
                  for sl in streamlines_zyx]

    sft = StatefulTractogram(sl_xyz_vox, ref_img, Space.VOX, origin=Origin.TRACKVIS)

    if data_values is not None and data_name:
        if len(data_values) != len(sl_xyz_vox):
            raise ValueError("data_values length must equal number of streamlines")
        from nibabel.streamlines.array_sequence import ArraySequence
        sft.data_per_point = {
            data_name: ArraySequence([np.asarray(v, dtype=np.float32).reshape(-1, 1) for v in data_values])
        }

    save_trk(sft, str(out_path), bbox_valid_check=False)



# ---------- tracing ----------

def trace_streamline(
    start_pt: tuple[float, float, float],
    vector_field: np.ndarray,
    fa_volume: np.ndarray | None = None,
    fa_threshold: float = 0.1,
    step_length: float = 0.5,
    max_steps: int | None = 1000,
    angle_threshold: float = 60.0,
    eps: float = 1e-10,
    direction: int = 1,
) -> list[tuple[float, float, float]]:
    Z, Y, X = vector_field.shape[1:]
    coords: list[tuple[float, float, float]] = [(float(start_pt[0]), float(start_pt[1]), float(start_pt[2]))]
    current_pt = np.array(start_pt, dtype=np.float64)
    prev_dir: np.ndarray | None = None

    def interp_unit(pt: np.ndarray) -> np.ndarray | None:
        vec = trilinear_interpolate_vector(vector_field, (pt[0], pt[1], pt[2]))
        if np.isnan(vec).any():
            return None
        n = np.linalg.norm(vec)
        if n < eps:
            return None
        return np.array([vec[2], vec[1], vec[0]]) / n * direction  # to (z,y,x)

    step_count = 0
    while max_steps is None or step_count < max_steps:
        step_count += 1

        if fa_volume is not None:
            if trilinear_interpolate_scalar(fa_volume, tuple(current_pt)) < fa_threshold:
                break

        k1 = interp_unit(current_pt)
        if k1 is None:
            break
        if prev_dir is not None:
            ang = np.degrees(np.arccos(np.clip(np.dot(prev_dir, k1), -1.0, 1.0)))
            if ang > angle_threshold:
                break

        mid1 = current_pt + 0.5 * step_length * k1; k2 = interp_unit(mid1)
        if k2 is None:
            break
        mid2 = current_pt + 0.5 * step_length * k2; k3 = interp_unit(mid2)
        if k3 is None:
            break
        end_pt = current_pt + step_length * k3; k4 = interp_unit(end_pt)
        if k4 is None:
            break

        ang4 = np.degrees(np.arccos(np.clip(np.dot(k1, k4), -1.0, 1.0)))
        if ang4 > angle_threshold:
            break

        next_pt = current_pt + (step_length / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        zn, yn, xn = next_pt
        if not (0 <= zn < Z and 0 <= yn < Y and 0 <= xn < X):
            break

        coords.append((float(zn), float(yn), float(xn)))
        current_pt = next_pt
        prev_dir = k1

    return coords


def generate_streamlines_from_vector_field(
    vector_field: np.ndarray,
    seed_points: np.ndarray,
    fa_volume: np.ndarray | None = None,
    fa_threshold: float = 0.1,
    step_length: float = 0.5,
    max_steps: int | None = None,
    angle_threshold: float = 60.0,
    min_length_pts: int = 10,
    bidirectional: bool = True,
) -> list[list[tuple[float, float, float]]]:
    all_streamlines: list[list[tuple[float, float, float]]] = []
    with alive_bar(len(seed_points), title="Tracing Streamlines") as bar:
        for zi, yi, xi in seed_points:
            start = (float(zi), float(yi), float(xi))
            forward_pts = trace_streamline(start, vector_field, fa_volume, fa_threshold, step_length, max_steps, angle_threshold, direction=1)
            if bidirectional:
                backward_pts = trace_streamline(start, vector_field, fa_volume, fa_threshold, step_length, max_steps, angle_threshold, direction=-1)
                backward_pts = backward_pts[::-1][:-1] if len(backward_pts) > 1 else []
                full = backward_pts + forward_pts
            else:
                full = forward_pts
            if len(full) >= min_length_pts:
                all_streamlines.append(full)
            bar()
    return all_streamlines


# ---------- TRK writer, generalized data name ----------

def save_trk_dipy_from_vox_zyx_multi(
    streamlines_zyx: List[List[Tuple[float, float, float]]],
    out_path: str | Path,
    vol_shape_zyx: Tuple[int, int, int],
    voxel_sizes_zyx: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    data_per_point: Optional[Dict[str, List[np.ndarray]]] = None,
) -> None:
    """
    Save streamlines in voxel indices (z,y,x) as TrackVis .trk using DIPY.
    Accepts multiple per-point scalar lists via `data_per_point` dict,
    with keys like "HA", "IA", "AZ", "EL". Each list must align with streamlines.
    """
    Z, Y, X = vol_shape_zyx
    vz, vy, vx = voxel_sizes_zyx

    # Voxel-index -> RAS mm affine (X,Y,Z order for NIfTI reference)
    affine = np.array([
        [vx, 0,  0,  0],
        [0,  vy, 0,  0],
        [0,  0,  vz, 0],
        [0,  0,  0,  1],
    ], dtype=np.float32)

    # NIfTI reference has data shape (X, Y, Z)
    ref_img = nib.Nifti1Image(np.zeros((X, Y, Z), dtype=np.uint8), affine)

    # Convert each streamline from (z,y,x) to (x,y,z)
    sl_xyz_vox = [
        np.stack([np.asarray(sl)[:, 2],
                  np.asarray(sl)[:, 1],
                  np.asarray(sl)[:, 0]], axis=1).astype(np.float32)
        for sl in streamlines_zyx
    ]

    sft = StatefulTractogram(sl_xyz_vox, ref_img, Space.VOX, origin=Origin.TRACKVIS)

    if data_per_point:
        n = len(sl_xyz_vox)
        from nibabel.streamlines.array_sequence import ArraySequence
        sft.data_per_point = {}
        for name, lists in data_per_point.items():
            if len(lists) != n:
                raise ValueError(
                    f"data_per_point['{name}'] length {len(lists)} != number of streamlines {n}"
                )
            sft.data_per_point[name] = ArraySequence(
                [np.asarray(v, dtype=np.float32).reshape(-1, 1) for v in lists]
            )

    from dipy.io.streamline import save_trk
    save_trk(sft, str(out_path), bbox_valid_check=False)


# ---------- top-level generator, angle-agnostic ----------

def generate_streamlines_from_params(
    vector_field_dir: str | Path,
    output_dir: str | Path,
    fa_dir: str | Path,
    angle_dir: str | Path,                      # single angle folder or parent of HA IA AZ EL
    mask_path: str | Path | None = None,
    start_xyz: tuple[int, int, int] = (0, 0, 0),
    end_xyz: tuple[int | None, int | None, int | None] = (None, None, None),
    bin_factor: int = 1,
    num_seeds: int = 20000,
    fa_seed_min: float = 0.4,
    fa_threshold: float = 0.1,
    step_length: float = 0.5,
    max_steps: int | None = None,
    angle_threshold: float = 60.0,
    min_length_pts: int = 10,
    bidirectional: bool = True,
    voxel_sizes_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    save_trk_file: bool = True,
) -> None:
    """
    Generate streamlines from the eigenvector field, then export:
      - .trk with all discovered per-point angle fields
      - .am with all per-edge mean angle scalars

    Angle discovery:
      If `angle_dir` is one of HA, IA, AZ, EL, discover siblings with those names and include all that exist.
      If `angle_dir` is a parent, include all subfolders named HA, IA, AZ, EL that exist.
      If none are found, treat `angle_dir` as a single custom angle and include it.
    """
    vector_field_dir = Path(vector_field_dir)
    fa_dir = Path(fa_dir)
    angle_dir = Path(angle_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all angle folders
    KNOWN = ("HA", "IA", "AZ", "EL")
    discovered: dict[str, Path] = {}

    if angle_dir.name.upper() in KNOWN:
        parent = angle_dir.parent
        for name in KNOWN:
            p = parent / name
            if p.exists():
                discovered[name] = p
    else:
        for name in KNOWN:
            p = angle_dir / name
            if p.exists():
                discovered[name] = p

    if not discovered:
        lbl = angle_dir.name.upper()
        discovered[lbl] = angle_dir
        print(f"Warning: no standard angle subfolders found, using single angle '{lbl}'")

    print(f"Angles selected: {sorted(discovered.keys())}")

    # ROI and shape
    start_z, start_y, start_x = start_xyz
    end_z, end_y, end_x = end_xyz
    vec_probe = DataReader(vector_field_dir)
    full_shape = vec_probe.shape  # (3, Z, Y, X)
    end_z = full_shape[1] if end_z is None else end_z
    end_y = full_shape[2] if end_y is None else end_y
    end_x = full_shape[3] if end_x is None else end_x

    # Binning
    if bin_factor > 1:
        downsample_vector_volume(vector_field_dir, bin_factor, output_dir)
        vec_load_dir = output_dir / f"bin{bin_factor}" / vector_field_dir.name

        downsample_volume(fa_dir, bin_factor, output_dir, subfolder="FA", out_ext="tif")
        fa_load_dir = output_dir / f"bin{bin_factor}" / "FA"

        angle_load_dirs: dict[str, Path] = {}
        for name, p in discovered.items():
            downsample_volume(p, bin_factor, output_dir, subfolder=name, out_ext="tif")
            angle_load_dirs[name] = output_dir / f"bin{bin_factor}" / name

        start_z_b = start_z // bin_factor; end_z_b = math.ceil(end_z / bin_factor)
        start_y_b = start_y // bin_factor; end_y_b = math.ceil(end_y / bin_factor)
        start_x_b = start_x // bin_factor; end_x_b = math.ceil(end_x / bin_factor)
    else:
        vec_load_dir = vector_field_dir
        fa_load_dir = fa_dir
        angle_load_dirs = {name: p for name, p in discovered.items()}
        start_z_b, end_z_b = start_z, end_z
        start_y_b, end_y_b = start_y, end_y
        start_x_b, end_x_b = start_x, end_x

    # Load vector field
    print("Loading vector field")
    vec_reader = DataReader(vec_load_dir)
    vector_field = vec_reader.load_volume(
        start_index=start_z_b, end_index=end_z_b
    )[:, :, start_y_b:end_y_b, start_x_b:end_x_b]
    if vector_field.ndim == 4 and vector_field.shape[-1] == 3:
        print("Reordering vector field axes")
        vector_field = np.moveaxis(vector_field, -1, 0)

    # Consistent sign
    neg_mask = vector_field[0] < 0
    vector_field[:, neg_mask] *= -1

    # Mask
    if mask_path:
        print("Loading mask")
        mask_reader = DataReader(mask_path)
        mask = mask_reader.load_volume(
            start_index=start_z_b, end_index=end_z_b, unbinned_shape=vec_reader.shape[1:]
        )
        mask = mask[:, start_y_b:end_y_b, start_x_b:end_x_b]
        mask = (mask > 0).astype(np.uint8)
        vector_field[:, mask == 0] = np.nan

    # FA
    print("Loading FA")
    fa_volume = DataReader(fa_load_dir).load_volume(
        start_index=start_z_b, end_index=end_z_b
    )
    fa_volume = fa_volume[:, start_y_b:end_y_b, start_x_b:end_x_b]

    # Seeds
    print("Selecting seeds")
    seed_mask = fa_volume > (fa_seed_min * 255)
    valid_indices = np.argwhere(seed_mask)
    if valid_indices.size == 0:
        raise RuntimeError("No voxels above FA seed threshold")
    chosen = valid_indices if len(valid_indices) <= num_seeds else \
        valid_indices[np.random.choice(valid_indices.shape[0], num_seeds, replace=False)]

    # Streamlines
    streamlines = generate_streamlines_from_vector_field(
        vector_field=vector_field,
        seed_points=chosen,
        fa_volume=fa_volume,
        fa_threshold=fa_threshold,
        step_length=step_length,
        max_steps=max_steps,
        angle_threshold=angle_threshold,
        min_length_pts=min_length_pts,
        bidirectional=bidirectional,
    )

    # Sample all discovered angles per point
    print("Sampling angles along streamlines")
    def load_crop(vol_dir: Path) -> np.ndarray:
        v = DataReader(vol_dir).load_volume(start_index=start_z_b, end_index=end_z_b)
        return v[:, start_y_b:end_y_b, start_x_b:end_x_b]

    cropped_angles = {name: load_crop(path) for name, path in angle_load_dirs.items()}

    def sample_along(volume: np.ndarray, sl: list[tuple[float, float, float]]) -> np.ndarray:
        Zc, Yc, Xc = volume.shape
        vals = []
        for z, y, x in sl:
            zc = min(max(z, 0.0), Zc - 1.0)
            yc = min(max(y, 0.0), Yc - 1.0)
            xc = min(max(x, 0.0), Xc - 1.0)
            vals.append(trilinear_interpolate_scalar(volume, (zc, yc, xc)))
        return np.asarray(vals, dtype=np.float32)

    per_point_angles = {name: [sample_along(vol, sl) for sl in streamlines]
                        for name, vol in cropped_angles.items()}

    # Unbin coordinates if needed
    if bin_factor > 1:
        streamlines = [[(z * bin_factor, y * bin_factor, x * bin_factor) for (z, y, x) in sl]
                       for sl in streamlines]

    # TRK with all per-point fields
    if save_trk_file:
        Zc = end_z_b - start_z_b; Yc = end_y_b - start_y_b; Xc = end_x_b - start_x_b
        out_trk = output_dir / "streamlines.trk"
        save_trk_dipy_from_vox_zyx_multi(
            streamlines_zyx=streamlines,
            out_path=out_trk,
            vol_shape_zyx=(Zc, Yc, Xc),
            voxel_sizes_zyx=voxel_sizes_zyx,
            data_per_point=per_point_angles,
        )
        print(f"Saved TRK with fields {sorted(per_point_angles.keys())} to {out_trk}")

    # Amira SpatialGraph with per-edge means for every angle
    streamlines_xyz = [np.stack([sl[:, 2], sl[:, 1], sl[:, 0]], axis=1) for sl in map(np.asarray, streamlines)]
    edge_scalar_dict = {
        name: np.array([float(np.nanmean(vals)) for vals in lists], dtype=float)
        for name, lists in per_point_angles.items()
    }
    write_spatialgraph_am(
        out_path=output_dir / "streamlines_spatialgraph.am",
        streamlines_xyz=streamlines_xyz,
        point_thickness=None,
        edge_scalar=edge_scalar_dict,   # multiple EDGE fields
        edge_scalar_name=None,
    )
    print(f"Wrote Amira SpatialGraph with fields {sorted(edge_scalar_dict.keys())} to {output_dir / 'streamlines_spatialgraph.am'}")
