import random

import fury
import matplotlib.pyplot as plt
import numpy as np
import vtk


def downsample_streamline(streamline, factor=2):
    return streamline if len(streamline) < 3 else streamline[::factor]


def matplotlib_cmap_to_fury_lut(cmap, value_range=(-1, 1), n_colors=256):
    """
    Convert a Matplotlib colormap to a VTK LookupTable for FURY.

    Parameters
    ----------
    cmap : str or matplotlib colormap
        Name or object of Matplotlib colormap.
    value_range : (float, float)
        Tuple specifying the (min, max) of scalar values.
    n_colors : int
        Number of LUT entries.

    Returns
    -------
    lut : vtk.vtkLookupTable
        A VTK-compatible lookup table.
    """
    import numpy as np

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    colors = cmap(np.linspace(0, 1, n_colors))  # RGBA in [0,1]
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(n_colors)
    lut.SetRange(*value_range)
    lut.Build()

    for i in range(n_colors):
        r, g, b, a = colors[i]
        lut.SetTableValue(i, r, g, b, a)

    return lut


def _split_streamline_by_bounds(sl: np.ndarray,
                                cl: np.ndarray,
                                x_min, x_max, y_min, y_max, z_min, z_max):
    """
    Returns lists of segments (points) and colors, one per contiguous in-bounds run.
    No straight 'bridging' lines will be drawn between disjoint segments.
    """
    # in-bounds mask
    within = (
        (sl[:, 0] >= x_min) & (sl[:, 0] <= x_max) &
        (sl[:, 1] >= y_min) & (sl[:, 1] <= y_max) &
        (sl[:, 2] >= z_min) & (sl[:, 2] <= z_max)
    )

    if not np.any(within):
        return [], []

    # find starts/ends of True runs
    w = within.astype(np.int8)
    # transitions: +1 = False->True (start), -1 = True->False (end+1)
    trans = np.diff(np.pad(w, (1, 1), constant_values=0))
    starts = np.where(trans == +1)[0]
    ends   = np.where(trans == -1)[0]  # each end is exclusive index

    segs, cols = [], []
    for s, e in zip(starts, ends):
        seg = sl[s:e]         # e is exclusive
        col = cl[s:e]
        if len(seg) > 0:
            segs.append(seg)
            cols.append(col)

    return segs, cols


def show_streamlines(
    streamlines_xyz: list[np.ndarray],
    color_values: list[np.ndarray],
    mode: str = "tube",
    line_width: float = 4,
    interactive: bool = True,
    screenshot_path: str | None = None,
    window_size: tuple[int, int] = (800, 800),
    downsample_factor: int = 2,
    max_streamlines: int | None = None,
    filter_min_len: int | None = None,
    subsample_factor: int = 1,
    crop_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    | None = None,
    colormap=None,
):
    """
    Render 3D streamlines in FURY with per-vertex coloring.

    Parameters
    ----------
    streamlines_xyz : list of np.ndarray
        Each streamline as an array of (x, y, z) points.
    color_values : list of np.ndarray
        Flattened per-vertex scalar values for coloring.
    mode : {"tube", "fake_tube", "line"}
        Rendering mode for streamlines.
    line_width : float
        Thickness of tubes/lines.
    interactive : bool
        Whether to open an interactive FURY window.
    screenshot_path : str, optional
        Save screenshot if not interactive.
    window_size : (int, int)
        Window size for display or screenshot.
    downsample_factor : int
        Step size to reduce points in each streamline.
    max_streamlines : int, optional
        Maximum number of streamlines to display.
    filter_min_len : int, optional
        Minimum streamline length (after downsampling) to keep.
    subsample_factor : int
        Randomly keep 1 of every N streamlines.
    crop_bounds : tuple of ((x_min,x_max),...), optional
        Crop streamlines within bounds.
    colormap : callable or str, optional
        Function or name of Matplotlib colormap to map scalar values to RGB.
        If None, defaults to FURY HSV mapping.
    """
    print(f"Initial number of streamlines: {len(streamlines_xyz)}")

    # --- Cropping
    print(
    f"Cropping streamlines within bounds: {crop_bounds}"
    if crop_bounds
    else "No cropping applied."
    )

    if crop_bounds is not None:
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = crop_bounds
        print(f"Cropping streamlines within bounds: {crop_bounds}")
        new_streamlines, new_colors = [], []
        for sl, cl in zip(streamlines_xyz, color_values):
            segs, cols = _split_streamline_by_bounds(
                sl, cl, x_min, x_max, y_min, y_max, z_min, z_max
            )
            if segs:
                new_streamlines.extend(segs)
                new_colors.extend(cols)
        streamlines_xyz, color_values = new_streamlines, new_colors
        if not streamlines_xyz:
            raise ValueError("❌ No streamlines intersect the crop box.")
        print("Cropping applied.")
    else:
        print("No cropping applied.")

      
    # --- Downsample and filter
    print(f"Downsampling points by factor {downsample_factor}")
    if filter_min_len is not None:
        print(f"Filtering out streamlines shorter than {filter_min_len} points")

    ds_streamlines, ds_colors = [], []
    for sl, cl in zip(streamlines_xyz, color_values):
        ds_sl = downsample_streamline(sl, downsample_factor)
        ds_cl = downsample_streamline(cl, downsample_factor)
        if filter_min_len is None or len(ds_sl) >= filter_min_len:
            ds_streamlines.append(ds_sl)
            ds_colors.append(ds_cl)

    streamlines_xyz, color_values = ds_streamlines, ds_colors
    if not streamlines_xyz:
        raise ValueError("❌ No streamlines left after downsampling/filtering.")

    # --- Subsample
    if subsample_factor > 1:
        print(f"Subsampling: keeping 1 in every {subsample_factor} streamlines")
        total = len(streamlines_xyz)
        keep_idx = sorted(random.sample(range(total), max(1, total // subsample_factor)))
        streamlines_xyz = [streamlines_xyz[i] for i in keep_idx]
        color_values    = [color_values[i]    for i in keep_idx]

    # --- Cap max
    if max_streamlines is not None and len(streamlines_xyz) > max_streamlines:
        print(f"Limiting to max {max_streamlines} streamlines")
        keep_idx = sorted(random.sample(range(len(streamlines_xyz)), max_streamlines))
        streamlines_xyz = [streamlines_xyz[i] for i in keep_idx]
        color_values    = [color_values[i]    for i in keep_idx]

    print(f"Final number of streamlines to render: {len(streamlines_xyz)}")
    if not color_values:
        raise ValueError("❌ No streamlines left after filtering/cropping.")

    # Flatten per-vertex scalars for FURY input
    flat_colors = np.concatenate([np.asarray(c).ravel() for c in color_values])
    min_val = float(np.nanmin(flat_colors))
    max_val = float(np.nanmax(flat_colors))
    print(f"Coloring range: min={min_val:.3f}, max={max_val:.3f}")
    print(f"Rendering mode: {mode}")

    # --- Lookup table
    if colormap is None:
        lut = fury.actor.colormap_lookup_table(
            scale_range=(min_val, max_val),
            hue_range=(0.7, 0.0),
            saturation_range=(0.5, 1.0),
        )
    else:
        lut = matplotlib_cmap_to_fury_lut(
            cmap=colormap,
            value_range=(min_val, max_val),  # auto-range to your data
            n_colors=256,
        )

    # --- Render
    scene = fury.window.Scene()
    colors = flat_colors  # per-vertex scalars

    # --- Render according to mode
    if mode == "tube":
        actor = fury.actor.streamtube(
            streamlines_xyz,
            colors=colors,
            linewidth=line_width,
            opacity=1.0,
            lookup_colormap=lut,
        )
    elif mode == "fake_tube":
        actor = fury.actor.line(
            streamlines_xyz,
            colors=colors,
            linewidth=line_width,
            fake_tube=True,
            depth_cue=True,
            lookup_colormap=lut,
        )
    elif mode == "line":
        actor = fury.actor.line(
            streamlines_xyz,
            colors=colors,
            linewidth=line_width,
            fake_tube=False,
            depth_cue=False,
            lookup_colormap=lut,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    scene.add(actor)
    if lut is not None:
        scene.add(fury.actor.scalar_bar(lookup_table=lut, title="Angle (deg)"))

    scene.reset_camera()

    # radii = 7
    # # Example coordinates of the point (x, y, z)
    # highlight_point = np.array([[6548-6000,8001-7500,17296-17041]])/2  # 3D coords in same space as streamlines
    # # Create a sphere actor for the point
    # sphere_actor = fury.actor.sphere(centers=highlight_point, colors=(1, 0, 0), radii=radii)
    # # Add the point to the scene
    # scene.add(sphere_actor)

    # # Example coordinates of the point (x, y, z)
    # highlight_point = np.array([[0,0,0]])  # 3D coords in same space as streamlines
    # # Create a sphere actor for the point
    # sphere_actor = fury.actor.sphere(centers=highlight_point, colors=(1, 0, 0), radii=radii)
    # # Add the point to the scene
    # scene.add(sphere_actor)

    # # Example coordinates of the point (x, y, z)
    # highlight_point = np.array([[6570-6000,7970-7500,17142-17041]])/2  # 3D coords in same space as streamlines
    # # Create a sphere actor for the point
    # sphere_actor = fury.actor.sphere(centers=highlight_point, colors=(1, 0, 0), radii=radii)
    # # Add the point to the scene
    # scene.add(sphere_actor)

    # # Example coordinates of the point (x, y, z)
    # highlight_point = np.array([[6548-6000,7997-7500,17394-17041]])/2  # 3D coords in same space as streamlines
    # # Create a sphere actor for the point
    # sphere_actor = fury.actor.sphere(centers=highlight_point, colors=(1, 0, 0), radii=radii)
    # # Add the point to the scene
    # scene.add(sphere_actor)

    # --- Display or screenshot
    if interactive:
        print("🕹️ Opening interactive window...")
        fury.window.show(scene, size=window_size, reset_camera=False)
    else:
        if not screenshot_path:
            raise ValueError("Must specify screenshot_path when interactive=False.")
        print(f"📸 Saving screenshot to: {screenshot_path}")
        fury.window.record(scene=scene, out_path=screenshot_path, size=window_size)
