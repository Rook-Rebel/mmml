"""Owns the PyVista plotter scene: reference frame, grid and workspace objects."""

from __future__ import annotations

import numpy as np
import pyvista as pv
import vtk

from ..core.math_object import MathObject
from .object_renderer import LabelSet, ObjectRenderer, RenderItem

AXIS_COLORS = {"x": "#ef5350", "y": "#66bb6a", "z": "#42a5f5"}
SELECT_COLOR = "#ffd54f"


class SceneManager:
    def __init__(self, plotter: pv.Plotter, axis_length: float = 6.0, grid_half: int = 10):
        self.plotter = plotter
        self.renderer_ = ObjectRenderer()
        self.objects: dict[str, MathObject] = {}
        self._actors: dict[str, dict[str, object]] = {}
        self._main_mesh: dict[str, pv.DataSet] = {}
        self._labels: dict[str, list[LabelSet]] = {}
        self._label_actors: dict[str, list[str]] = {}
        self.selected: str | None = None
        self._frame_actors: list = []
        self._grid_actors: list = []
        self.axis_length = axis_length
        self.grid_half = grid_half
        self._setup()

    # ---------------------------------------------------------------- setup
    def _setup(self) -> None:
        pl = self.plotter
        pl.set_background("#0b0f17", top="#1b2536")
        try:
            pl.enable_anti_aliasing("fxaa")
        except Exception:  # pragma: no cover - depends on GPU/driver
            pass
        self._build_frame()
        self._build_grid()
        try:
            pl.add_axes(color="white", line_width=2, labels_off=False)
        except Exception:  # pragma: no cover
            pass
        self.reset_camera("front")

    def _build_frame(self) -> None:
        L = self.axis_length
        for name, vec in (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1))):
            v = np.array(vec, float)
            a = self.plotter.add_mesh(pv.Line(-L * v, L * v), color=AXIS_COLORS[name], line_width=2,
                                      name=f"__axis_{name}", pickable=False, reset_camera=False)
            cone = pv.Cone(center=L * v, direction=v, height=0.35, radius=0.09, resolution=24)
            b = self.plotter.add_mesh(cone, color=AXIS_COLORS[name], name=f"__axis_tip_{name}",
                                      pickable=False, reset_camera=False)
            self._frame_actors += [a, b]
        lbl = self.plotter.add_point_labels(np.array([[L + 0.4, 0, 0], [0, L + 0.4, 0], [0, 0, L + 0.4]]),
                                            ["X", "Y", "Z"], font_size=16, text_color="white", shape=None,
                                            show_points=False, name="__axis_labels", reset_camera=False,
                                            always_visible=True)
        self._frame_actors.append(lbl)

    def _build_grid(self) -> None:
        n = self.grid_half
        pts, lines = [], []
        for i in range(-n, n + 1):
            for p, q in (((i, -n, 0), (i, n, 0)), ((-n, i, 0), (n, i, 0))):
                lines.append([2, len(pts), len(pts) + 1])
                pts += [p, q]
        grid = pv.PolyData(np.array(pts, float))
        grid.lines = np.array(lines).ravel()
        a = self.plotter.add_mesh(grid, color="#2c3a4f", line_width=1, opacity=0.55, name="__grid",
                                  pickable=False, reset_camera=False)
        self._grid_actors.append(a)

    # --------------------------------------------------------- toggles/views
    def set_grid_visible(self, visible: bool) -> None:
        for a in self._grid_actors:
            a.SetVisibility(visible)
        self.plotter.render()

    def set_axes_visible(self, visible: bool) -> None:
        for a in self._frame_actors:
            a.SetVisibility(visible)
        self.plotter.render()

    def set_parallel_projection(self, enabled: bool) -> None:
        if enabled:
            self.plotter.enable_parallel_projection()
        else:
            self.plotter.disable_parallel_projection()
        self.plotter.render()

    def reset_camera(self, view: str = "front") -> None:
        pl = self.plotter
        {"front": pl.view_xy, "top": pl.view_xz, "side": pl.view_yz, "iso": pl.view_isometric}.get(view, pl.view_xy)()
        bounds = self.objects_bounds()
        if bounds is not None:
            pl.reset_camera(bounds=bounds)
        else:
            pl.reset_camera()
        pl.render()

    def objects_bounds(self):
        meshes = [self._transformed_bounds(uid) for uid in self.objects]
        meshes = [b for b in meshes if b is not None]
        if not meshes:
            return None
        b = np.array(meshes)
        lo, hi = b[:, ::2].min(axis=0), b[:, 1::2].max(axis=0)
        pad = 0.15 * max(float((hi - lo).max()), 1.0)
        return [lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad, lo[2] - pad, hi[2] + pad]

    def _transformed_bounds(self, uid):
        mesh = self._main_mesh.get(uid)
        if mesh is None or mesh.n_points == 0:
            return None
        b = np.array(mesh.bounds).reshape(3, 2)
        corners = np.array([[x, y, z] for x in b[0] for y in b[1] for z in b[2]])
        w = self.objects[uid].transform.apply(corners)
        lo, hi = w.min(axis=0), w.max(axis=0)
        return [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]

    # -------------------------------------------------------------- objects
    def add_object(self, obj: MathObject) -> None:
        self.objects[obj.uid] = obj
        self.rebuild(obj.uid)

    def remove_object(self, uid: str) -> None:
        self._clear_actors(uid)
        self.objects.pop(uid, None)
        if self.selected == uid:
            self.selected = None
        self.plotter.render()

    def clear(self) -> None:
        for uid in list(self.objects):
            self.remove_object(uid)

    def _clear_actors(self, uid: str) -> None:
        for name in list(self._actors.get(uid, {})):
            self.plotter.remove_actor(name, render=False)
        for name in self._label_actors.get(uid, []):
            self.plotter.remove_actor(name, render=False)
        self._actors[uid] = {}
        self._label_actors[uid] = []
        self._main_mesh.pop(uid, None)

    def rebuild(self, uid: str) -> None:
        """Re-create all meshes of an object (display mode / options changed).

        Raises the builder's exception (e.g. an invalid user expression) *after*
        leaving the previous geometry untouched.
        """
        obj = self.objects[uid]
        items, labels = self.renderer_.build(obj)
        self._clear_actors(uid)
        m = obj.transform.matrix()
        for item in items:
            self._add_item(uid, item, m)
        self._labels[uid] = labels
        self._update_labels(uid)
        self._update_selection_outline()
        self.plotter.render()

    def _add_item(self, uid: str, item: RenderItem, m: np.ndarray) -> None:
        name = f"{uid}:{item.name}"
        kwargs = dict(item.kwargs)
        actor = self.plotter.add_mesh(item.mesh, name=name, reset_camera=False, render=False,
                                      pickable=True, **kwargs)
        actor.user_matrix = m
        self._actors[uid][name] = actor
        if item.main or uid not in self._main_mesh:
            self._main_mesh[uid] = item.mesh

    def update_transform(self, uid: str, render: bool = True) -> None:
        obj = self.objects.get(uid)
        if obj is None:
            return
        m = obj.transform.matrix()
        for actor in self._actors.get(uid, {}).values():
            actor.user_matrix = m
        self._update_labels(uid)
        self._update_selection_outline()
        if render:
            self.plotter.render()

    def _update_labels(self, uid: str) -> None:
        for name in self._label_actors.get(uid, []):
            self.plotter.remove_actor(name, render=False)
        self._label_actors[uid] = []
        obj = self.objects[uid]
        if not obj.render.show_labels:
            return
        for ls in self._labels.get(uid, []):
            if len(ls.points) == 0:
                continue
            name = f"{uid}:label:{ls.name}"
            self.plotter.add_point_labels(obj.transform.apply(ls.points), ls.texts, name=name, font_size=15,
                                          text_color=ls.color, shape_color="#101722", shape_opacity=0.55,
                                          show_points=False, always_visible=True, reset_camera=False,
                                          render=False, bold=False)
            self._label_actors[uid].append(name)

    # ------------------------------------------------------------ selection
    def set_selected(self, uid: str | None) -> None:
        self.selected = uid if uid in self.objects else None
        self._update_selection_outline()
        self.plotter.render()

    def _update_selection_outline(self) -> None:
        self.plotter.remove_actor("__selection", render=False)
        uid = self.selected
        mesh = self._main_mesh.get(uid) if uid else None
        if mesh is None or mesh.n_points == 0:
            return
        outline = mesh.outline()
        actor = self.plotter.add_mesh(outline, color=SELECT_COLOR, line_width=1.2, opacity=0.6,
                                      name="__selection", pickable=False, reset_camera=False, render=False)
        actor.user_matrix = self.objects[uid].transform.matrix()

    def pick(self, x: int, y: int) -> str | None:
        """Return the uid of the object under display position (x, y), if any."""
        # CPU ray casting: vtkPropPicker's GPU hardware selection can abort on some drivers
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.01)          # generous, so thin curves are clickable
        if not picker.Pick(x, y, 0, self.plotter.renderer):
            return None
        prop = picker.GetProp3D()
        if prop is None:
            return None
        # VTK's Python wrapper hands back the same Python object for a live C++ object
        for uid, actors in self._actors.items():
            if any(a is prop for a in actors.values()):
                return uid
        return None

    # --------------------------------------------------------------- export
    def export_object(self, uid: str, path: str) -> None:
        obj = self.objects[uid]
        items, _ = self.renderer_.build(obj)
        meshes = [it.mesh.extract_geometry() if not isinstance(it.mesh, pv.PolyData) else it.mesh
                  for it in items if it.mesh.n_points]
        merged = meshes[0].copy()
        for m in meshes[1:]:
            merged = merged.merge(m)
        merged = merged.transform(obj.transform.matrix(), inplace=False)
        if path.lower().endswith((".stl", ".obj", ".ply")):
            merged = merged.extract_surface(algorithm="dataset_surface").triangulate()
        merged.save(path)
