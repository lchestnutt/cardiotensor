from pathlib import Path

import numpy as np

from cardiotensor.colormaps.helix_angle import helix_angle_cmap
from cardiotensor.visualization.fury_plotting_streamlines import show_streamlines




def _ha_to_degrees_per_streamline(ha_list):
    """
    Convert HA values per streamline to degrees in [-90, 90].
    Accepts values already in degrees or 0–255 encoded (uint8-like).
    Returns list of float32 arrays aligned to streamlines.
    """
    out = []
    for ha in ha_list:
        ha = np.asarray(ha)
        # Heuristic: if values look like 0–255, map to degrees; else assume they are already degrees.
        if ha.size > 0 and np.nanmax(ha) > 1.5:  # e.g., 255
            ha_deg = (ha.astype(np.float32) / 255.0) * 180.0 - 90.0
        else:
            ha_deg = ha.astype(np.float32)
        out.append(ha_deg)
    return out


def compute_elevation_angles(streamlines_xyz):
    """
    Compute per-vertex elevation angle for each streamline.
    Returns a list of (N_i,) float32 arrays aligned to streamlines.
    """
    all_angles = []
    for pts in streamlines_xyz:
        pts = np.asarray(pts, dtype=np.float32)
        n = len(pts)
        if n < 2:
            all_angles.append(np.zeros((n,), dtype=np.float32))
            continue

        # Tangent vectors between successive points (x,y,z)
        vecs = np.diff(pts, axis=0)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        # Safe normalize (avoid divide by zero)
        normalized = np.divide(vecs, norms, out=np.zeros_like(vecs), where=norms != 0)

        # Elevation = arcsin(z_component) in degrees
        z_components = normalized[:, 2]
        elev = np.arcsin(np.clip(z_components, -1.0, 1.0)) * (180.0 / np.pi)

        # Repeat last angle so length matches number of points
        elev = np.concatenate([elev, [elev[-1]]]).astype(np.float32)
        all_angles.append(elev)
    return all_angles




def visualize_streamlines(
    streamlines_file: str | Path,
    color_by: str = "ha",
    mode: str = "tube",
    line_width: float = 4.0,
    subsample_factor: int = 1,
    filter_min_len: int | None = None,
    downsample_factor: int = 1,
    max_streamlines: int | None = None,
    crop_bounds: tuple | None = None,
    interactive: bool = True,
    screenshot_path: str | None = None,
    window_size: tuple[int, int] = (800, 800),
    colormap=None,
):
    """
    Visualize precomputed streamlines with optional coloring.

    Parameters
    ----------
    streamlines_file : str or Path
        Path to a .npz file containing streamlines and optional ha_values.
    color_by : {"ha", "elevation"}
        Color streamlines by Helix Angle (HA) or elevation angle.
    mode : {"tube", "fake_tube", "line"}
        Rendering mode for streamlines.
    line_width : float
        Tube/line thickness.
    subsample_factor : int
        Factor to subsample streamline points for rendering speed.
    filter_min_len : int, optional
        Minimum streamline length to display.
    downsample_factor : int
        Downsampling factor for streamlines (visual only).
    max_streamlines : int, optional
        Maximum number of streamlines to visualize.
    crop_bounds : tuple, optional
        ((x_min, x_max), (y_min, y_max), (z_min, z_max)) for cropping.
    interactive : bool
        If False, window will close immediately after rendering or screenshot.
    screenshot_path : str or Path, optional
        Save a screenshot instead of opening interactive window.
    window_size : (int, int)
        Window width and height in pixels.
    colormap : callable or None
        Colormap function mapping values to RGB. If None, defaults to helix_angle_cmap.
    """
    streamlines_file = Path(streamlines_file)
    if not streamlines_file.exists():
        raise FileNotFoundError(f"❌ Streamlines file not found: {streamlines_file}")

    print(f"Loading streamlines: {streamlines_file}")
    data = np.load(streamlines_file, allow_pickle=True)
    raw_streamlines = data.get("streamlines")
    if raw_streamlines is None:
        raise ValueError("'streamlines' array missing in .npz.")

    # Convert (z, y, x) to (x, y, z)
    print("Convert (z, y, x) to (x, y, z)")
    streamlines_xyz = [
        np.array([(pt[2], pt[1], pt[0]) for pt in sl], dtype=np.float32)
        for sl in raw_streamlines.tolist()
    ]

    # Build per-vertex color arrays aligned with streamlines
    print("Computing color values per streamline")
    if color_by == "elevation":
        color_values = compute_elevation_angles(streamlines_xyz)
    elif color_by == "ha":
        ha_obj = data.get("ha_values")
        if ha_obj is None:
            raise ValueError("'ha_values' array missing in .npz for color_by='ha'.")
        # Ensure list-of-arrays aligned to streamlines
        ha_list = [np.asarray(a) for a in ha_obj.tolist()]
        if len(ha_list) != len(streamlines_xyz):
            raise ValueError(
                f"ha_values length ({len(ha_list)}) does not match streamlines ({len(streamlines_xyz)})"
            )
        # Convert to degrees if needed
        color_values = _ha_to_degrees_per_streamline(ha_list)
    else:
        raise ValueError("color_by must be 'ha' or 'elevation'.")

    # Default to helix_angle_cmap if no colormap is provided
    if colormap is None:
        colormap = helix_angle_cmap
    

    # Render streamlines
    show_streamlines(
        streamlines_xyz=streamlines_xyz,
        color_values=color_values,
        mode=mode,
        line_width=line_width,
        interactive=interactive,
        screenshot_path=screenshot_path,
        window_size=window_size,
        downsample_factor=downsample_factor,
        max_streamlines=max_streamlines,
        filter_min_len=filter_min_len,
        subsample_factor=subsample_factor,
        crop_bounds=crop_bounds,
        colormap=colormap,  # <-- Pass to the FURY renderer
    )
