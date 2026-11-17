"""Turn a :class:`MathObject` into PyVista meshes for its current display mode.

Meshes are built in the object's *local* coordinates; the scene manager applies
the object's transform through each actor's user matrix, so rotation, translation
and scaling never require rebuilding geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv

from ..core.math_object import DisplayMode, MathObject, VERTEX_NAMES
from ..mathematics import geometry as geo
from ..mathematics.expressions import compile_expression
from ..mathematics.function_fitting import fmt
from ..vision.models import ShapeKind

ACCENT = "#ffd54f"
DERIVATIVE_COLOR = "#ff8a65"
TANGENT_COLOR = "#e040fb"
LABEL_COLOR = "#e6edf3"


@dataclass
class RenderItem:
    name: str
    mesh: pv.DataSet
    kwargs: dict = field(default_factory=dict)
    main: bool = False


@dataclass
class LabelSet:
    name: str
    points: np.ndarray          # (N, 3) local coordinates
    texts: list[str]
    color: str = LABEL_COLOR


def vtk_text(s: str) -> str:
    """VTK's built-in font lacks a few Unicode symbols; use ASCII look-alikes."""
    return s.replace("−", "-").replace("≈", "~").replace("∠", "")


def _xyz(pts2, z=0.0) -> np.ndarray:
    p = np.asarray(pts2, float)
    return np.column_stack([p[:, 0], p[:, 1], np.full(len(p), z)])


def polyline(pts3, closed: bool = False) -> pv.PolyData:
    pts3 = np.asarray(pts3, float)
    n = len(pts3)
    idx = list(range(n)) + ([0] if closed else [])
    mesh = pv.PolyData(pts3)
    mesh.lines = np.array([len(idx)] + idx)
    return mesh


def polygon_face(pts2) -> pv.PolyData:
    n = len(pts2)
    face = pv.PolyData(_xyz(pts2), faces=np.hstack([[n], np.arange(n)]))
    return face.triangulate()


class ObjectRenderer:
    def build(self, obj: MathObject) -> tuple[list[RenderItem], list[LabelSet]]:
        mode = obj.options.mode
        k = obj.kind
        if k.is_closed:
            return self._closed(obj, mode)
        if k == ShapeKind.LINE:
            return self._line(obj, mode)
        return self._curve(obj, mode)

    # ------------------------------------------------------------ styles
    def _surface_kwargs(self, obj, color=None, fill_opacity=1.0) -> dict:
        r = obj.render
        kw = dict(color=color or r.color, opacity=r.opacity * fill_opacity, smooth_shading=True,
                  specular=0.3, line_width=max(1.0, r.line_width / 2), point_size=r.line_width * 2.5)
        if r.style == "wireframe":
            kw["style"] = "wireframe"
        elif r.style == "points":
            kw["style"] = "points"
            kw["render_points_as_spheres"] = True
        elif r.style == "edges":
            kw["show_edges"] = True
            kw["edge_color"] = "#0d1117"
        return kw

    def _line_kwargs(self, obj, color=None, width=None) -> dict:
        r = obj.render
        return dict(color=color or r.color, line_width=width or r.line_width,
                    render_lines_as_tubes=True, opacity=r.opacity)

    def _vertices_item(self, obj, pts3) -> RenderItem:
        return RenderItem("vertices", pv.PolyData(np.asarray(pts3, float)),
                          dict(color="#ffffff", point_size=max(8.0, obj.render.line_width * 3),
                               render_points_as_spheres=True))

    # ----------------------------------------------------------- closed
    def _closed(self, obj: MathObject, mode: DisplayMode):
        outline = obj.outline()
        items: list[RenderItem] = []
        labels: list[LabelSet] = []
        face = polygon_face(outline)
        if mode == DisplayMode.EXTRUDE:
            d = max(obj.options.extrude_depth, 1e-3)
            prism = face.extrude((0, 0, d), capping=True).translate((0, 0, -d / 2))
            items.append(RenderItem("solid", prism, self._surface_kwargs(obj), main=True))
            edges = prism.extract_feature_edges(feature_angle=35, boundary_edges=True,
                                                non_manifold_edges=False, manifold_edges=False)
            if edges.n_points:
                items.append(RenderItem("edges", edges, self._line_kwargs(obj, ACCENT, max(1.5, obj.render.line_width / 2))))
        else:
            fill = 0.35 if obj.render.style != "wireframe" else 0.0
            if fill > 0:
                items.append(RenderItem("face", face, self._surface_kwargs(obj, fill_opacity=fill), main=True))
            items.append(RenderItem("outline", polyline(_xyz(outline), closed=True), self._line_kwargs(obj),
                                    main=fill == 0))
        if obj.kind.is_polygon and obj.render.show_vertices:
            z = (obj.options.extrude_depth / 2) if mode == DisplayMode.EXTRUDE else 0.0
            items.append(self._vertices_item(obj, _xyz(obj.points, z)))
        if mode == DisplayMode.ANALYSIS:
            labels.extend(self._closed_analysis(obj, items))
        return items, labels

    def _closed_analysis(self, obj: MathObject, items: list[RenderItem]) -> list[LabelSet]:
        labels: list[LabelSet] = []
        if obj.kind.is_polygon:
            v = obj.points
            c = geo.polygon_centroid(v)
            names = list(VERTEX_NAMES[: len(v)])
            span = float(np.ptp(v, axis=0).max())
            out = [p + (p - c) / (np.linalg.norm(p - c) + 1e-9) * 0.08 * span for p in v]
            labels.append(LabelSet("vertex_names", _xyz(out), names, ACCENT))
            sides = geo.side_lengths(v)
            mids = [(v[i] + v[(i + 1) % len(v)]) / 2 for i in range(len(v))]
            labels.append(LabelSet("sides", _xyz(mids), [vtk_text(fmt(s)) for s in sides]))
            angles = geo.interior_angles(v)
            inner = [p + (c - p) * 0.22 for p in v]
            labels.append(LabelSet("angles", _xyz(inner), [f"{a:.1f}°" for a in angles], "#80deea"))
        elif obj.kind.is_conic:
            center, (a, b), ang = obj.conic_local()
            ua = np.array([math.cos(ang), math.sin(ang)])
            ub = np.array([-ua[1], ua[0]])
            pa, pb = center + a * ua, center + b * ub
            items.append(RenderItem("radius_a", polyline(_xyz([center, pa])), self._line_kwargs(obj, ACCENT, 2.5)))
            items.append(RenderItem("center", pv.PolyData(_xyz([center])),
                                    dict(color=ACCENT, point_size=12, render_points_as_spheres=True)))
            if obj.kind == ShapeKind.CIRCLE:
                labels.append(LabelSet("radius", _xyz([(center + pa) / 2]), [vtk_text(f"r = {fmt(a)}")], ACCENT))
            else:
                items.append(RenderItem("radius_b", polyline(_xyz([center, pb])), self._line_kwargs(obj, ACCENT, 2.5)))
                labels.append(LabelSet("radius", _xyz([(center + pa) / 2, (center + pb) / 2]),
                                       [vtk_text(f"a = {fmt(a)}"), vtk_text(f"b = {fmt(b)}")], ACCENT))
            labels.append(LabelSet("center_name", _xyz([center]), ["O"], LABEL_COLOR))
        return labels

    # ------------------------------------------------------------- line
    def _line(self, obj: MathObject, mode: DisplayMode):
        p = obj.points
        items, labels = [], []
        if mode == DisplayMode.EXTRUDE:
            d = max(obj.options.extrude_depth, 1e-3)
            strip = polyline(_xyz(p)).extrude((0, 0, d), capping=False).translate((0, 0, -d / 2))
            items.append(RenderItem("strip", strip, self._surface_kwargs(obj), main=True))
        items.append(RenderItem("line", polyline(_xyz(p)), self._line_kwargs(obj), main=mode != DisplayMode.EXTRUDE))
        if obj.render.show_vertices:
            items.append(self._vertices_item(obj, _xyz(p)))
        if mode == DisplayMode.ANALYSIS:
            length = float(np.linalg.norm(p[-1] - p[0]))
            labels.append(LabelSet("ends", _xyz(p), ["P", "Q"], ACCENT))
            labels.append(LabelSet("length", _xyz([(p[0] + p[-1]) / 2]), [vtk_text(f"|PQ| = {fmt(length)}")]))
        return items, labels

    # ------------------------------------------------------------ curves
    def _curve(self, obj: MathObject, mode: DisplayMode):
        items: list[RenderItem] = []
        labels: list[LabelSet] = []
        o = obj.options
        pts = obj.curve_points()
        if mode == DisplayMode.SURFACE and obj.function() is not None:
            items.extend(self._surface(obj))
        elif mode == DisplayMode.SPACE_CURVE:
            z = self._space_z(obj, pts)
            items.append(RenderItem("space_curve", polyline(np.column_stack([pts, z])), self._line_kwargs(obj), main=True))
            items.append(RenderItem("shadow", polyline(_xyz(pts)), self._line_kwargs(obj, "#5c6b80", 1.5)))
        elif mode == DisplayMode.EXTRUDE:
            d = max(o.extrude_depth, 1e-3)
            ribbon = polyline(_xyz(pts)).extrude((0, 0, d), capping=False).translate((0, 0, -d / 2))
            items.append(RenderItem("ribbon", ribbon, self._surface_kwargs(obj), main=True))
            items.append(RenderItem("curve", polyline(_xyz(pts)), self._line_kwargs(obj, ACCENT, 2.0)))
        else:
            items.append(RenderItem("curve", polyline(_xyz(pts)), self._line_kwargs(obj), main=True))
            if o.show_derivative and obj.derivative() is not None:
                x = pts[:, 0]
                dy = obj.derivative()(x)
                ok = np.isfinite(dy)
                if ok.sum() > 2:
                    items.append(RenderItem("derivative", polyline(_xyz(np.column_stack([x[ok], dy[ok]]))),
                                            self._line_kwargs(obj, DERIVATIVE_COLOR, max(1.5, obj.render.line_width * 0.6))))
                    labels.append(LabelSet("derivative_name", _xyz([[x[ok][-1], dy[ok][-1]]]), ["f'(x)"],
                                           DERIVATIVE_COLOR))
            tangent = obj.tangent() if o.show_tangent else None
            if tangent is not None:
                x0, y0, m = tangent
                span = 0.25 * (pts[:, 0].max() - pts[:, 0].min())
                h = span / math.sqrt(1 + m * m)
                seg = np.array([[x0 - h, y0 - m * h], [x0 + h, y0 + m * h]])
                items.append(RenderItem("tangent", polyline(_xyz(seg)), self._line_kwargs(obj, TANGENT_COLOR, 2.5)))
                items.append(RenderItem("tangent_point", pv.PolyData(_xyz([[x0, y0]])),
                                        dict(color=TANGENT_COLOR, point_size=14, render_points_as_spheres=True)))
                labels.append(LabelSet("tangent_label", _xyz([[x0, y0]]),
                                       [vtk_text(f"f'({fmt(x0)}) = {fmt(m)}")], TANGENT_COLOR))
            if obj.kind == ShapeKind.ARC and mode == DisplayMode.ANALYSIS:
                c, r = obj.arc_local()
                ends = [obj.points[0], obj.points[-1]]
                for i, e in enumerate(ends):
                    items.append(RenderItem(f"arc_radius{i}", polyline(_xyz([c, e])), self._line_kwargs(obj, ACCENT, 2.0)))
                items.append(RenderItem("arc_center", pv.PolyData(_xyz([c])),
                                        dict(color=ACCENT, point_size=12, render_points_as_spheres=True)))
                labels.append(LabelSet("arc_labels", _xyz([c, (c + ends[0]) / 2]),
                                       ["O", vtk_text(f"r = {fmt(r)}")], ACCENT))
        if o.show_samples:
            items.append(RenderItem("samples", pv.PolyData(_xyz(obj.points)),
                                    dict(color="#b0bec5", point_size=5, render_points_as_spheres=True)))
        eq = obj.equation()
        if eq and obj.render.show_labels and mode in (DisplayMode.CURVE_2D, DisplayMode.SURFACE):
            top = pts[np.argmax(pts[:, 1])]
            labels.append(LabelSet("equation", np.array([[top[0], top[1] + 0.15 * np.ptp(pts[:, 1]) + 0.1, 0.0]]),
                                   [vtk_text(eq)], ACCENT))
        return items, labels

    def _space_z(self, obj: MathObject, pts) -> np.ndarray:
        f = compile_expression(obj.options.z_expression)
        t = np.linspace(0.0, 1.0, len(pts))
        z = f(x=pts[:, 0], y=pts[:, 1], t=t)
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def _surface(self, obj: MathObject) -> list[RenderItem]:
        o = obj.options
        f = obj.function()
        x0, x1 = obj.domain()
        n = int(np.clip(o.resolution, 10, 300))
        kind = o.surface_type
        if kind == "product":
            g = np.linspace(x0, x1, n)
            X, Y = np.meshgrid(g, g)
            Z = f(X.ravel()).reshape(X.shape) * f(Y.ravel()).reshape(Y.shape)
            curve = None
        elif kind == "radial":
            r = np.linspace(0.0, x1 - x0, n)
            th = np.linspace(0.0, 2 * np.pi, n)
            R, TH = np.meshgrid(r, th)
            X, Y = R * np.cos(TH), R * np.sin(TH)
            Z = f(x0 + R.ravel()).reshape(R.shape)
            curve = np.column_stack([r, np.zeros_like(r), f(x0 + r)])
        elif kind == "revolution":
            x = np.linspace(x0, x1, n)
            th = np.linspace(0.0, 2 * np.pi, n)
            XX, TH = np.meshgrid(x, th)
            FX = f(XX.ravel()).reshape(XX.shape)
            X, Y, Z = XX, FX * np.cos(TH), FX * np.sin(TH)
            curve = np.column_stack([x, f(x), np.zeros_like(x)])
        else:  # translational: z = f(x) swept along y
            x = np.linspace(x0, x1, n)
            y = np.linspace(min(o.y_min, o.y_max), max(o.y_min, o.y_max), max(n // 2, 2))
            X, Y = np.meshgrid(x, y)
            Z = f(X.ravel()).reshape(X.shape)
            curve = np.column_stack([x, np.zeros_like(x), f(x)])
        Z = np.nan_to_num(Z)
        grid = pv.StructuredGrid(X, Y, Z)
        grid["z"] = Z.ravel(order="F")
        kw = self._surface_kwargs(obj)
        kw.pop("color", None)
        kw.update(scalars="z", cmap="viridis", show_scalar_bar=False)
        items = [RenderItem("surface", grid, kw, main=True)]
        if curve is not None:
            items.append(RenderItem("generator", polyline(curve), self._line_kwargs(obj, ACCENT, 3.0)))
        return items
