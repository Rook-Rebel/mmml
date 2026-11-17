"""A drawing extracted into the 3D workspace, with its mathematical reconstruction.

The object keeps the original detection (pixel geometry in the rectified image)
and derives everything else from it through a :class:`CoordinateSystem`. Changing
the calibration therefore re-derives coordinates, measurements and the fitted
equation consistently.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np

from ..i18n import FAMILY_NAMES, kind_name, tr
from ..mathematics import geometry as geo
from ..mathematics.coordinates import CoordinateSystem
from ..mathematics.function_fitting import FunctionFitResult, fit_function, fmt, is_x_monotonic
from ..mathematics.function_fitting import _shift as shift_text, _signed as signed_text
from ..mathematics.transformations import Transform3D
from ..vision.models import AxesInfo, DetectedObject, ShapeKind


class DisplayMode(str, Enum):
    FLAT = "flat"
    EXTRUDE = "extrude"
    ANALYSIS = "analysis"
    CURVE_2D = "curve2d"
    SPACE_CURVE = "space"
    SURFACE = "surface"


PALETTE = ["#4fc3f7", "#ffb74d", "#81c784", "#f06292", "#ba68c8", "#fff176", "#4db6ac", "#e57373"]


@dataclass
class RenderSettings:
    style: str = "surface"          # surface | wireframe | points | edges
    color: str = "#4fc3f7"
    line_width: float = 4.0
    opacity: float = 1.0
    show_vertices: bool = True
    show_labels: bool = True


@dataclass
class ShapeOptions:
    mode: DisplayMode = DisplayMode.FLAT
    extrude_depth: float = 1.0
    z_expression: str = "0.5*x"
    surface_type: str = "translational"   # translational | product | radial | revolution
    y_min: float = -2.0
    y_max: float = 2.0
    show_derivative: bool = False
    show_tangent: bool = False
    tangent_x: float = 0.0
    show_samples: bool = False
    resolution: int = 90


@dataclass
class Measurement:
    label: str
    value: str


_ids = itertools.count(1)
VERTEX_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class MathObject:
    def __init__(self, detection: DetectedObject, axes: AxesInfo | None = None):
        n = next(_ids)
        self.uid = f"obj{n}"
        self.number = n
        self.detection = detection
        self.kind: ShapeKind = detection.kind
        self.axes = axes if self._uses_axes(detection.kind) else None
        self.x_start: float | None = None
        self.coords = self.default_coords()
        self.transform = Transform3D()
        self.options = ShapeOptions(mode=self.available_modes()[0])
        self.render = RenderSettings(color=PALETTE[(n - 1) % len(PALETTE)])
        self._initial_options = replace(self.options)
        self._initial_render = replace(self.render)
        self.fit: FunctionFitResult | None = None
        self.points = np.zeros((0, 2))
        self.recompute()

    # ------------------------------------------------------------ identity
    @staticmethod
    def _uses_axes(kind: ShapeKind) -> bool:
        return kind in (ShapeKind.FUNCTION, ShapeKind.FREEHAND)

    @property
    def is_function_graph(self) -> bool:
        return self.fit is not None and len(self.fit.x) >= 6

    @property
    def is_recognized_function(self) -> bool:
        return self.fit is not None and self.fit.recognized and self.fit.best is not None

    @property
    def origin_meaningful(self) -> bool:
        return self.coords.origin_from_axes or self.x_start is not None

    @property
    def title(self) -> str:
        if self.kind == ShapeKind.FUNCTION and self.is_recognized_function:
            fam = self.fit.best.family
            if fam == "sine":
                fam = self.fit.best.sine_form(self.origin_meaningful)
            return f"#{self.number} {FAMILY_NAMES.get(fam, fam)}"
        return f"#{self.number} {kind_name(self.kind)}"

    # --------------------------------------------------------- coordinates
    def default_coords(self) -> CoordinateSystem:
        det = self.detection
        extent = det.extent
        if self.axes is not None:
            return CoordinateSystem.normalized(self.axes.origin, extent, origin_from_axes=True)
        if self.kind.is_polygon:
            origin = geo.polygon_centroid(det.geometry)
        elif self.kind.is_conic:
            origin = np.asarray(det.params["center"], float)
        else:
            x, y, w, h = det.bbox
            origin = np.array([x + w / 2.0, y + h / 2.0])
        return CoordinateSystem.normalized(origin, extent)

    def reference_px(self) -> tuple[float, str]:
        """Pixel length used for calibration and the i18n key describing it."""
        g = self.detection.geometry
        if self.kind.is_polygon:
            return float(np.linalg.norm(g[1] - g[0])), "ref_side_ab"
        if self.kind == ShapeKind.CIRCLE:
            return 2.0 * float(self.detection.params["semi_axes"][0]), "ref_diameter"
        if self.kind == ShapeKind.ELLIPSE:
            return 2.0 * float(self.detection.params["semi_axes"][0]), "ref_major_axis"
        if self.kind == ShapeKind.LINE:
            return float(np.linalg.norm(g[1] - g[0])), "ref_length"
        if self.kind == ShapeKind.ARC:
            return float(np.linalg.norm(g[-1] - g[0])), "ref_chord"
        return float(np.ptp(g[:, 0])), "ref_x_span"

    def calibrate(self, real_length: float, unit: str = "", x_start: float | None = None) -> None:
        ref_px, _ = self.reference_px()
        if real_length <= 0 or ref_px <= 0:
            raise ValueError("reference length must be positive")
        s = ref_px / real_length
        origin = np.array(self.coords.origin_px, float)
        if x_start is not None and self.kind.is_curve:
            origin[0] = float(self.detection.geometry[:, 0].min()) - x_start * s
        self.x_start = x_start if self.kind.is_curve else None
        self.coords = CoordinateSystem(origin, s, s, True, unit, self.coords.origin_from_axes)
        self.recompute()

    def clear_calibration(self) -> None:
        self.x_start = None
        self.coords = self.default_coords()
        self.recompute()

    def recompute(self) -> None:
        det = self.detection
        self.points = self.coords.to_local(det.geometry)
        self.fit = None
        if self.kind in (ShapeKind.FUNCTION, ShapeKind.FREEHAND, ShapeKind.LINE) and is_x_monotonic(self.points):
            self.fit = fit_function(self.points[:, 0], self.points[:, 1])
        if self.kind == ShapeKind.FREEHAND and self.fit is not None and self.fit.recognized:
            self.fit.recognized = False   # the detector already judged it unreliable

    # -------------------------------------------------------------- geometry
    @property
    def unit_label(self) -> str:
        return self.coords.unit if self.coords.calibrated else tr("units_normalized")

    def scale(self) -> float:
        return self.coords.px_per_unit_x

    def conic_local(self) -> tuple[np.ndarray, tuple[float, float], float]:
        p = self.detection.params
        c = self.coords.to_local([p["center"]])[0]
        s = self.scale()
        a, b = p["semi_axes"]
        return c, (a / s, b / s), -float(p["angle"])

    def arc_local(self) -> tuple[np.ndarray, float]:
        p = self.detection.params
        return self.coords.to_local([p["center"]])[0], float(p["radius"]) / self.scale()

    def outline(self) -> np.ndarray:
        """Closed outline (local coords) for closed shapes."""
        if self.kind.is_conic:
            c, ax, ang = self.conic_local()
            return geo.ellipse_points(c, ax, ang, 160)
        return self.points

    def domain(self) -> tuple[float, float]:
        return float(self.points[:, 0].min()), float(self.points[:, 0].max())

    def curve_points(self, n: int = 500) -> np.ndarray:
        """Dense, smooth 2D samples of the reconstructed curve (local coords)."""
        if self.is_function_graph:
            x0, x1 = self.domain()
            x = np.linspace(x0, x1, n)
            return np.column_stack([x, self.fit.function()(x)])
        return self.points

    def function(self):
        return self.fit.function() if self.is_function_graph else None

    def derivative(self):
        return self.fit.derivative() if self.is_function_graph else None

    def equation(self) -> str | None:
        if self.is_recognized_function:
            return self.fit.best.equation(self.origin_meaningful)
        return None

    # -------------------------------------------------------------- modes
    def available_modes(self) -> list[DisplayMode]:
        k = self.kind
        if k.is_closed:
            return [DisplayMode.FLAT, DisplayMode.EXTRUDE, DisplayMode.ANALYSIS]
        if k == ShapeKind.LINE:
            return [DisplayMode.FLAT, DisplayMode.EXTRUDE, DisplayMode.ANALYSIS]
        if k == ShapeKind.ARC:
            return [DisplayMode.CURVE_2D, DisplayMode.ANALYSIS, DisplayMode.EXTRUDE]
        modes = [DisplayMode.CURVE_2D, DisplayMode.SPACE_CURVE]
        if self._uses_axes(k) and is_x_monotonic(self.coords.to_local(self.detection.geometry)):
            modes.append(DisplayMode.SURFACE)
        modes.append(DisplayMode.EXTRUDE)
        return modes

    def restore_original(self) -> None:
        self.transform = Transform3D()
        self.options = replace(self._initial_options)
        self.render = replace(self._initial_render)

    # ----------------------------------------------------------- analysis
    def _p(self, pt) -> str:
        return f"({fmt(pt[0])}, {fmt(pt[1])})"

    def measurements(self) -> list[Measurement]:
        u = self.unit_label
        m: list[Measurement] = []
        k = self.kind
        if k.is_polygon:
            v = self.points
            names = VERTEX_NAMES[: len(v)]
            sides = geo.side_lengths(v)
            angles = geo.interior_angles(v)
            m.append(Measurement(tr("vertices"), ", ".join(f"{n}{self._p(p)}" for n, p in zip(names, v))))
            m.append(Measurement(tr("sides"), ", ".join(
                f"{names[i]}{names[(i + 1) % len(v)]} = {fmt(s)} {u}" for i, s in enumerate(sides))))
            m.append(Measurement(tr("angles"), ", ".join(f"∠{n} = {a:.1f}°" for n, a in zip(names, angles))))
            m.append(Measurement(tr("angle_sum"), f"{angles.sum():.1f}°"))
            m.append(Measurement(tr("perimeter"), f"{fmt(sides.sum())} {u}"))
            m.append(Measurement(tr("area"), f"{fmt(geo.polygon_area(v))} {u}²"))
            if k == ShapeKind.TRIANGLE:
                by_s, by_a = geo.classify_triangle(sides, angles)
                m.append(Measurement(tr("triangle_type"), f"{tr(by_s)}, {tr(by_a)}"))
        elif k.is_conic:
            c, (a, b), ang = self.conic_local()
            met = geo.ellipse_metrics(a, b)
            m.append(Measurement(tr("center"), self._p(c)))
            if k == ShapeKind.CIRCLE:
                m.append(Measurement(tr("radius"), f"{fmt(a)} {u}"))
                m.append(Measurement(tr("circumference"), f"{fmt(2 * math.pi * a)} {u}"))
            else:
                m.append(Measurement(tr("semi_axes"), f"a = {fmt(a)}, b = {fmt(b)} {u}"))
                m.append(Measurement(tr("perimeter"), f"≈ {fmt(met['perimeter'])} {u}"))
                m.append(Measurement(tr("eccentricity"), fmt(met["eccentricity"])))
                m.append(Measurement(tr("orientation"), f"{math.degrees(ang) % 180:.1f}°"))
            m.append(Measurement(tr("area"), f"{fmt(met['area'])} {u}²"))
        elif k == ShapeKind.CLOSED_CURVE:
            m.append(Measurement(tr("perimeter"), f"{fmt(geo.polyline_length(np.vstack([self.points, self.points[:1]])))} {u}"))
            m.append(Measurement(tr("area"), f"{fmt(geo.polygon_area(self.points))} {u}²"))
        elif k == ShapeKind.LINE:
            p0, p1 = self.points[0], self.points[-1]
            d = p1 - p0
            m.append(Measurement(tr("endpoints"), f"{self._p(p0)} — {self._p(p1)}"))
            m.append(Measurement(tr("length"), f"{fmt(float(np.hypot(*d)))} {u}"))
            if abs(d[0]) > 1e-9:
                m.append(Measurement(tr("slope"), fmt(d[1] / d[0])))
            m.append(Measurement(tr("orientation"), f"{math.degrees(math.atan2(d[1], d[0])) % 180:.1f}°"))
        elif k == ShapeKind.ARC:
            c, r = self.arc_local()
            sweep = float(self.detection.params["sweep"])
            m.append(Measurement(tr("center"), self._p(c)))
            m.append(Measurement(tr("radius"), f"{fmt(r)} {u}"))
            m.append(Measurement(tr("central_angle"), f"{math.degrees(sweep):.1f}°"))
            m.append(Measurement(tr("arc_length"), f"{fmt(r * sweep)} {u}"))
        if k.is_curve and k != ShapeKind.ARC:
            m.extend(self._curve_measurements(u))
        return m

    def _curve_measurements(self, u: str) -> list[Measurement]:
        m: list[Measurement] = []
        pts = self.curve_points()
        if self.is_recognized_function:
            for name, val in self.fit.best.named_params().items():
                if name == "period":
                    m.append(Measurement(tr("period"), f"{fmt(val)}"))
                else:
                    m.append(Measurement(f"{tr('parameters')}: {name}", fmt(val)))
        if self.is_function_graph:
            x0, x1 = self.domain()
            m.append(Measurement(tr("domain"), f"[{fmt(x0)}, {fmt(x1)}]"))
            m.append(Measurement(tr("range"), f"[{fmt(pts[:, 1].min())}, {fmt(pts[:, 1].max())}]"))
        else:
            lo, hi = self.points.min(axis=0), self.points.max(axis=0)
            m.append(Measurement(tr("bbox"), f"x ∈ [{fmt(lo[0])}, {fmt(hi[0])}], y ∈ [{fmt(lo[1])}, {fmt(hi[1])}]"))
        m.append(Measurement(tr("curve_length"), f"≈ {fmt(geo.polyline_length(pts))} {u}"))
        m.append(Measurement(tr("samples"), str(len(self.points))))
        tangent = self.tangent()
        if self.options.show_tangent and tangent is not None:
            x0, y0, slope = tangent
            m.append(Measurement(tr("tangent_eq"), f"y = {fmt(slope)}({shift_text(x0)}){signed_text(y0) or ''}"))
        return m

    def tangent(self) -> tuple[float, float, float] | None:
        f, df = self.function(), self.derivative()
        if f is None or df is None:
            return None
        x0, x1 = self.domain()
        x = float(np.clip(self.options.tangent_x, x0, x1))
        slope = float(df(np.array([x]))[0])
        if not math.isfinite(slope):
            return None
        return x, float(f(np.array([x]))[0]), slope
