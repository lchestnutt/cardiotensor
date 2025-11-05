from pathlib import Path
import numpy as np
import nibabel as nib

from cardiotensor.colormaps.helix_angle import helix_angle_cmap
from cardiotensor.visualization.fury_plotting_streamlines import show_streamlines

from cardiotensor.utils.streamlines_io_utils import (
    load_trk_streamlines,
    load_npz_streamlines,
    ha_to_degrees_per_streamline,
    compute_elevation_angles,
)



# ---------- main ----------

def visualize_streamlines(
    streamlines_file: str | Path,
    color_by: str = "ha",              # {"ha","elevation"}
    mode: str = "tube",                # {"tube","fake_tube","line"}
    line_width: float = 4.0,
    subsample_factor: int = 1,
    filter_min_len: int | None = None,
    downsample_factor: int = 1,
    max_streamlines: int | None = None,
    crop_bounds: tuple | None = None,  # ((xmin,xmax),(ymin,ymax),(zmin,zmax))
    interactive: bool = True,
    screenshot_path: str | None = None,
    window_size: tuple[int, int] = (800, 800),
    colormap=None,                     # matplotlib name or cmap, else defaults to HA cmap
):
    p = Path(streamlines_file)
    if not p.exists():
        raise FileNotFoundError(f"❌ Streamlines file not found: {p}")

    print(f"Loading streamlines: {p}")
    ext = p.suffix.lower()
    if ext == ".npz":
        streamlines_xyz, ha_list = load_npz_streamlines(p)
    elif ext == ".trk":
        streamlines_xyz, ha_list = load_trk_streamlines(p)
    else:
        raise ValueError(f"Unsupported extension '{ext}'. Use .npz or .trk")


    # Build per-vertex color arrays aligned with streamlines
    print("Computing color values per streamline")
    if color_by == "elevation":
        color_values = compute_elevation_angles(streamlines_xyz)
    elif color_by == "ha":
        if ha_list is None:
            raise ValueError(
                "No HA values found. For .trk, make sure you saved data_per_point['HA'], "
                "or use color_by='elevation'."
            )
        color_values = ha_to_degrees_per_streamline(ha_list)  # your 0–255 to degrees helper
    else:
        raise ValueError("color_by must be 'ha' or 'elevation'.")


    # Default colormap
    if colormap is None:
        colormap = helix_angle_cmap

    # Render
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
        colormap=colormap,
    )
