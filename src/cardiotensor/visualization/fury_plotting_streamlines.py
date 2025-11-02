#!/usr/bin/env python3
from __future__ import annotations
import random
from typing import List, Tuple, Optional

import numpy as np
import vtk
import fury
from fury import window
import matplotlib.pyplot as plt


# ---------------------------
# small utilities (unchanged)
# ---------------------------

def parse_background_color(color) -> tuple[float, float, float]:
    """Convert a string or tuple into an RGB triple in [0,1]."""
    NAMED = {
        "white": (1.0, 1.0, 1.0),
        "black": (0.0, 0.0, 0.0),
        "gray":  (0.5, 0.5, 0.5),
        "lightgray": (0.9, 0.9, 0.9),
        "red":   (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue":  (0.0, 0.0, 1.0),
    }
    if isinstance(color, str):
        key = color.lower()
        if key not in NAMED:
            raise ValueError(f"Unknown background color '{color}'. "
                             f"Available: {list(NAMED.keys())}")
        return NAMED[key]
    elif isinstance(color, (tuple, list)) and len(color) == 3:
        return tuple(float(c) for c in color)
    else:
        raise TypeError("background_color must be str or 3-tuple")


def downsample_streamline(streamline: np.ndarray, factor: int = 2) -> np.ndarray:
    return streamline if len(streamline) < 3 or factor <= 1 else streamline[::factor]


def matplotlib_cmap_to_fury_lut(cmap, value_range=(-1, 1), n_colors=256) -> vtk.vtkLookupTable:
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


def _split_streamline_by_bounds(sl: np.ndarray, cl: np.ndarray,
                                x_min, x_max, y_min, y_max, z_min, z_max):
    within = (
        (sl[:, 0] >= x_min) & (sl[:, 0] <= x_max) &
        (sl[:, 1] >= y_min) & (sl[:, 1] <= y_max) &
        (sl[:, 2] >= z_min) & (sl[:, 2] <= z_max)
    )
    if not np.any(within):
        return [], []

    w = within.astype(np.int8)
    trans = np.diff(np.pad(w, (1, 1), constant_values=0))
    starts = np.where(trans == +1)[0]
    ends   = np.where(trans == -1)[0]

    segs, cols = [], []
    for s, e in zip(starts, ends):
        seg = sl[s:e]         # e exclusive
        col = cl[s:e]
        if len(seg) > 0:
            segs.append(seg); cols.append(col)
    return segs, cols


def _polydata_from_streamlines(streamlines_xyz: List[np.ndarray],
                               color_values: List[np.ndarray],
                               scalar_name: str = "Scalars") -> vtk.vtkPolyData:
    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    scalars = vtk.vtkFloatArray()
    scalars.SetName(scalar_name)

    pid = 0
    for sl, cv in zip(streamlines_xyz, color_values):
        n = len(sl)
        if n < 2:
            continue
        lines.InsertNextCell(n)
        for i in range(n):
            x, y, z = map(float, sl[i])
            pts.InsertNextPoint(x, y, z)
            lines.InsertCellPoint(pid)
            scalars.InsertNextValue(float(cv[i]))
            pid += 1

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetLines(lines)
    poly.GetPointData().SetScalars(scalars)
    return poly


# ===========================
# Class-based viewer
# ===========================
class StreamlineViewer:
    def __init__(self, streamlines_xyz, color_values, mode, line_width,
                 window_size, lut, background_color="black"):

        self.streamlines_xyz = streamlines_xyz
        self.color_values = color_values
        self.mode = mode
        self.window_size = window_size
        self.lut = lut

        # Scene
        self.scene = fury.window.Scene()      
        self.current_bg = parse_background_color(background_color)
        self.scene.SetBackground(*self.current_bg)
        
        self.auto_enable_on_invert = True


        # thickness state
        self.radius_scale = 0.15
        self.radius = max(0.0001, float(line_width) * self.radius_scale)
        self.linewidth = max(1.0, float(line_width))

        self.scale_bar = None
        self.scale_bar_on = False
        self._add_scale_bar()

        # VTK/FURY objects
        self.showm: Optional[window.ShowManager] = None

        self.poly = _polydata_from_streamlines(streamlines_xyz, color_values, "Angle")

        # unclipped branch
        self.tuber0 = None
        self.mapper0 = None
        self.actor0 = None

        # clipped branch
        self.plane_rep = None
        self.plane_fn = None
        self.plane_widget = None
        self.clipper = None
        self.tuber = None
        self.mapper = None
        self.actor = None
        self.clipped = False

        # bounds/center
        b = np.array(self.poly.GetBounds())
        self.center = np.array([(b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2], dtype=float)

        self._build_pipelines()
        self._add_scalar_bar()
        self._show_unclipped()

    def _build_pipelines(self):
        # baseline (unclipped)
        if self.mode == "tube":
            self.tuber0 = vtk.vtkTubeFilter()
            self.tuber0.SetInputData(self.poly)
            self.tuber0.SetNumberOfSides(12)
            self.tuber0.SetVaryRadiusToVaryRadiusOff()
            self.tuber0.CappingOn()

            mapper0_in = self.tuber0.GetOutputPort()
            self.mapper0 = vtk.vtkPolyDataMapper()
            self.mapper0.SetInputConnection(mapper0_in)
        else:
            self.mapper0 = vtk.vtkPolyDataMapper()
            self.mapper0.SetInputData(self.poly)

        self.mapper0.SetLookupTable(self.lut)
        self.mapper0.UseLookupTableScalarRangeOn()
        self.mapper0.SetScalarModeToUsePointData()
        self.mapper0.ScalarVisibilityOn()

        self.actor0 = vtk.vtkActor()
        self.actor0.SetMapper(self.mapper0)
        if self.mode in ("line", "fake_tube"):
            self.actor0.GetProperty().SetLineWidth(self.linewidth)

        # clipped (oblique)
        self.plane_rep = vtk.vtkImplicitPlaneRepresentation()
        self.plane_rep.SetPlaceFactor(1.25)
        self.plane_rep.PlaceWidget(self.poly.GetBounds())
        self.plane_rep.SetOrigin(*self.center)
        self.plane_rep.SetNormal(1, 0, 0)

        self.plane_fn = vtk.vtkPlane()
        # robust: copy origin/normal not GetPlane()
        origin = [0.0, 0.0, 0.0]; normal = [1.0, 0.0, 0.0]
        self.plane_rep.GetOrigin(origin); self.plane_rep.GetNormal(normal)
        self.plane_fn.SetOrigin(origin); self.plane_fn.SetNormal(normal)

        self.clipper = vtk.vtkClipPolyData()
        self.clipper.SetInputData(self.poly)
        self.clipper.SetClipFunction(self.plane_fn)
        self.clipper.SetInsideOut(True)  # keep side along normal

        if self.mode == "tube":
            self.tuber = vtk.vtkTubeFilter()
            self.tuber.SetInputConnection(self.clipper.GetOutputPort())
            self.tuber.SetNumberOfSides(12)
            self.tuber.SetVaryRadiusToVaryRadiusOff()
            self.tuber.CappingOn()
            mapper_in = self.tuber.GetOutputPort()
        else:
            mapper_in = self.clipper.GetOutputPort()

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(mapper_in)
        self.mapper.SetLookupTable(self.lut)
        self.mapper.UseLookupTableScalarRangeOn()
        self.mapper.SetScalarModeToUsePointData()
        self.mapper.ScalarVisibilityOn()

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        if self.mode in ("line", "fake_tube"):
            self.actor.GetProperty().SetLineWidth(self.linewidth)

        self._apply_thickness()

    def _apply_thickness(self):
        if self.mode == 'tube':
            if self.tuber0 is not None:
                self.tuber0.SetRadius(self.radius); self.tuber0.Update()
            if self.tuber is not None:
                self.tuber.SetRadius(self.radius); self.tuber.Update()
        else:
            self.actor0.GetProperty().SetLineWidth(self.linewidth)
            self.actor.GetProperty().SetLineWidth(self.linewidth)

    def _add_scalar_bar(self):
        self.scene.add(fury.actor.scalar_bar(lookup_table=self.lut, title="Angle (deg)"))

    def _show_unclipped(self):
        if self.actor in self.scene.GetActors():
            self.scene.rm(self.actor)
        if self.actor0 not in self.scene.GetActors():
            self.scene.add(self.actor0)
        self.clipped = False
        self._apply_thickness()
        self._render_now()           # <-- add

    def _show_clipped(self):
        if self.actor0 in self.scene.GetActors():
            self.scene.rm(self.actor0)
        if self.actor not in self.scene.GetActors():
            self.scene.add(self.actor)
        self.clipped = True
        self._apply_thickness()
        self._render_now()   

    def _sync_plane_from_widget(self, *_):
        origin = [0.0, 0.0, 0.0]; normal = [1.0, 0.0, 0.0]
        self.plane_rep.GetOrigin(origin); self.plane_rep.GetNormal(normal)
        self.plane_fn.SetOrigin(origin);  self.plane_fn.SetNormal(normal)
        self.plane_fn.Modified()
        if self.clipper is not None:
            self.clipper.Modified()
        self._render_now()           # <-- add


    def _set_gizmo_visible(self, visible: bool):
        """Show/hide the plane widget visuals without changing the active clip."""
        if not self.plane_widget:
            return
        if visible:
            self.plane_widget.EnabledOn()
        else:
            # keep the clip (plane_fn still feeds clipper), just hide the gizmo
            self.plane_widget.EnabledOff()
        self._render_now()


    def _add_scale_bar(self):
        # 2D overlay ruler that tracks camera zoom
        self.scale_bar = vtk.vtkLegendScaleActor()
        # Show only one axis for a clean look
        self.scale_bar.LeftAxisVisibilityOff()
        self.scale_bar.TopAxisVisibilityOff()
        self.scale_bar.RightAxisVisibilityOff()
        self.scale_bar.BottomAxisVisibilityOn()

        # Optional niceties (comment out if version differs)
        try:
            self.scale_bar.SetNumberOfLabels(5)
            self.scale_bar.SetCornerOffset(5)  # pixels off the corner
            # self.scale_bar.GetRightAxis().SetTitle("mm")  # if your coords are mm
            # self.scale_bar.GetBottomAxis().SetTitle("mm")
        except Exception:
            pass

        # Add as a 2D actor to the renderer (FURY scene is a vtkRenderer)
        self.scene.add(self.scale_bar)
        self.scale_bar_on = True
        



    def _render_now(self):
        """Force an immediate redraw after pipeline/state changes."""
        # Make sure filters/mappers see changes
        if self.clipper is not None:
            self.clipper.Modified()
        if self.mapper0 is not None:
            self.mapper0.Update()
        if self.mapper is not None:
            self.mapper.Update()
        if self.actor0 is not None:
            self.actor0.Modified()
        if self.actor is not None:
            self.actor.Modified()
        if self.plane_widget is not None:
            self.plane_widget.Modified()
        if self.plane_rep is not None:
            self.plane_rep.Modified()
        if self.plane_fn is not None:
            self.plane_fn.Modified()

        # Avoid near/far plane culling issues after clipping changes
        try:
            # fury.window.Scene is a vtkRenderer, so this works:
            self.scene.ResetCameraClippingRange()
        except Exception:
            pass

        if self.showm is not None:
            # Either call the convenience method or VTK directly
            self.showm.render()
            # self.showm.renwin.Render()   # equivalent


    # ---------------------------
    # key handling
    # ---------------------------
    def _on_keypress(self, obj, evt):
        key = obj.GetKeySym().lower()

        if key == 'o':
            if self.clipped:
                if self.plane_widget: self.plane_widget.EnabledOff()
                self._show_unclipped()
                print("Oblique plane OFF")
            else:
                if self.plane_widget: self.plane_widget.EnabledOn()
                self._show_clipped()
                self._sync_plane_from_widget()
                print("Oblique plane ON (drag handles to move/rotate)")
                
        elif key == 'h':
            # Toggle handle visibility while keeping the clip
            if self.plane_widget:
                currently_on = self.plane_widget.GetEnabled()
                self._set_gizmo_visible(not currently_on)
                print("Plane gizmo {}".format("shown" if not currently_on else "hidden"))


        elif key == 'i':
            if not self.clipped:
                if self.auto_enable_on_invert and self.plane_widget:
                    self.plane_widget.EnabledOn()
                    self._show_clipped()
                    self._sync_plane_from_widget()
                    print("Oblique plane auto-enabled to allow flip ('I').")
                else:
                    print("Press 'O' to enable the cutting plane before using 'I'.")
                    return
            inside = self.clipper.GetInsideOut()
            self.clipper.SetInsideOut(not inside)
            self.clipper.Update()
            self._render_now()           
                    
        elif key == 'b':
            # toggle between white and black
            if self.current_bg == (1.0, 1.0, 1.0):
                self.current_bg = (0.0, 0.0, 0.0)
            else:
                self.current_bg = (1.0, 1.0, 1.0)
            self.scene.SetBackground(*self.current_bg)
            self._render_now()
            print(f"Background set to {self.current_bg}")

        elif key == 's':
            if self.scale_bar is None:
                self._add_scale_bar()
            else:
                if self.scale_bar_on:
                    self.scene.rm(self.scale_bar)
                    self.scale_bar_on = False
                    print("Scale bar OFF")
                else:
                    self.scene.add(self.scale_bar)
                    self.scale_bar_on = True
                    print("Scale bar ON")
            self._render_now()

        elif key == 'p':
            # save current view as high-resolution PNG
            import datetime, os
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.abspath(f"view_{ts}_hires.png")
            try:
                highres_size = (2000, 2000)
                fury.window.record(
                    scene=self.scene,
                    out_path=out_path,
                    size=highres_size,
                    reset_camera=False,
                )
                print(f"Saved high-resolution screenshot to {out_path} ({highres_size[0]}×{highres_size[1]})")
            except Exception as e:
                print(f"Failed to save screenshot: {e}")

        elif key == 'r':
            self.plane_rep.SetOrigin(*self.center)
            self.plane_rep.SetNormal(1, 0, 0)
            self.plane_rep.UpdatePlacement()
            self._sync_plane_from_widget()
            self.clipper.Update()
            self._render_now() 
            print("Oblique plane reset to center/aligned.")         

        elif key in ('plus', 'equal', 'kp_add'):
            if self.mode == 'tube':
                self.radius *= 1.25
            else:
                self.linewidth = min(1000.0, self.linewidth * 1.25)
            self._apply_thickness()
            self._render_now()           

        elif key in ('minus', 'kp_subtract', 'underscore'):
            if self.mode == 'tube':
                self.radius = max(0.00005, self.radius * 0.8)
            else:
                self.linewidth = max(1.0, self.linewidth * 0.8)
            self._apply_thickness()
            self._render_now()           

            self.showm.render()
            print(f"Thickness ↓ → radius={self.radius:.5f} / lw={self.linewidth:.2f}")

    # ---------------------------
    # main entry
    # ---------------------------
    def run(self, interactive: bool, screenshot_path: Optional[str]):
        if interactive:
            self.showm = window.ShowManager(scene=self.scene, size=self.window_size, reset_camera=False)
            self.showm.initialize()

            # plane widget bound to the interactor; start disabled
            self.plane_widget = vtk.vtkImplicitPlaneWidget2()
            self.plane_widget.SetRepresentation(self.plane_rep)
            self.plane_widget.SetInteractor(self.showm.iren)
            self.plane_widget.EnabledOff()

            # robust sync on all interaction phases
            self.plane_widget.AddObserver(vtk.vtkCommand.StartInteractionEvent, self._sync_plane_from_widget)
            self.plane_widget.AddObserver(vtk.vtkCommand.InteractionEvent,      self._sync_plane_from_widget)
            self.plane_widget.AddObserver(vtk.vtkCommand.EndInteractionEvent,   self._sync_plane_from_widget)

            self.showm.iren.AddObserver('KeyPressEvent', self._on_keypress)

            self.scene.reset_camera()
            print("🕹️ Keys: 'O' toggle plane, 'I' flip side, 'R' reset cropping plane, '+/-' thickness, 'B' background, 'S' scale bar, 'P' save PNG.")
            self.showm.start()
        else:
            if not screenshot_path:
                raise ValueError("Must specify screenshot_path when interactive=False.")
            self.scene.reset_camera()
            fury.window.record(scene=self.scene, out_path=screenshot_path, size=self.window_size)


# ===========================
# Public API (unchanged)
# ===========================
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
    background_color: str | tuple[float,float,float] = "black", 
):
    # ---- preprocessing steps identical to your function ----
    print(f"Initial number of streamlines: {len(streamlines_xyz)}")

    if crop_bounds is not None:
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = crop_bounds
        print(f"Cropping streamlines within bounds: {crop_bounds}")
        new_streamlines, new_colors = [], []
        for sl, cl in zip(streamlines_xyz, color_values):
            segs, cols = _split_streamline_by_bounds(sl, cl, x_min, x_max, y_min, y_max, z_min, z_max)
            if segs:
                new_streamlines.extend(segs); new_colors.extend(cols)
        streamlines_xyz, color_values = new_streamlines, new_colors
        if not streamlines_xyz:
            raise ValueError("❌ No streamlines intersect the crop box.")
        print("Cropping applied.")
    else:
        print("No cropping applied.")

    print(f"Downsampling points by factor {downsample_factor}")
    if filter_min_len is not None:
        print(f"Filtering out streamlines shorter than {filter_min_len} points")

    ds_streamlines, ds_colors = [], []
    for sl, cl in zip(streamlines_xyz, color_values):
        ds_sl = downsample_streamline(sl, downsample_factor)
        ds_cl = downsample_streamline(cl, downsample_factor)
        if filter_min_len is None or len(ds_sl) >= filter_min_len:
            ds_streamlines.append(ds_sl); ds_colors.append(ds_cl)

    streamlines_xyz, color_values = ds_streamlines, ds_colors
    if not streamlines_xyz:
        raise ValueError("❌ No streamlines left after downsampling/filtering.")

    if subsample_factor > 1:
        print(f"Subsampling: keeping 1 in every {subsample_factor} streamlines")
        total = len(streamlines_xyz)
        keep_idx = sorted(random.sample(range(total), max(1, total // subsample_factor)))
        streamlines_xyz = [streamlines_xyz[i] for i in keep_idx]
        color_values    = [color_values[i]    for i in keep_idx]

    if max_streamlines is not None and len(streamlines_xyz) > max_streamlines:
        print(f"Limiting to max {max_streamlines} streamlines")
        keep_idx = sorted(random.sample(range(len(streamlines_xyz)), max_streamlines))
        streamlines_xyz = [streamlines_xyz[i] for i in keep_idx]
        color_values    = [color_values[i]    for i in keep_idx]

    print(f"Final number of streamlines to render: {len(streamlines_xyz)}")
    if not color_values:
        raise ValueError("❌ No streamlines left after filtering/cropping.")

    flat_colors = np.concatenate([np.asarray(c).ravel() for c in color_values])
    min_val = float(np.nanmin(flat_colors)); max_val = float(np.nanmax(flat_colors))
    print(f"Coloring range: min={min_val:.3f}, max={max_val:.3f}")
    print(f"Rendering mode: {mode}")

    if colormap is None:
        lut = fury.actor.colormap_lookup_table(
            scale_range=(min_val, max_val),
            hue_range=(0.7, 0.0),
            saturation_range=(0.5, 1.0),
        )
    else:
        lut = matplotlib_cmap_to_fury_lut(cmap=colormap, value_range=(min_val, max_val), n_colors=256)

    # ---- class-based viewer ----
    viewer = StreamlineViewer(streamlines_xyz, color_values, mode, line_width, window_size, lut, background_color=background_color)
    viewer.run(interactive=interactive, screenshot_path=screenshot_path)


# ---------------------------
# Smoke test
# ---------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sls, cols = [], []
    for k in range(50):
        t = np.linspace(0, 2*np.pi, 150)
        r = 20 + 2 * rng.normal()
        x = r * np.cos(t) + 5 * rng.normal()
        y = r * np.sin(t) + 5 * rng.normal()
        z = np.linspace(-30, 30, t.size) + 3 * rng.normal()
        sl = np.c_[x, y, z].astype(np.float32)
        c = (np.degrees(np.arctan2(y, x))).astype(np.float32)
        sls.append(sl); cols.append(c)

    show_streamlines(
        streamlines_xyz=sls,
        color_values=cols,
        mode="tube",
        line_width=4,
        interactive=True,
        window_size=(900, 900),
        downsample_factor=1,
        subsample_factor=1,
        max_streamlines=None,
        filter_min_len=None,
        crop_bounds=None,
        colormap="turbo",
    )
