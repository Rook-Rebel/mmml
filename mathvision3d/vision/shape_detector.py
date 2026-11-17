"""Classify connected ink components into geometric shapes and curves.

Strategy
--------
* Closed strokes (component with a large hole) and filled blobs are classified by
  comparing an ellipse model against a polygon model of the outline.
* Open strokes are reduced to an ordered centre line (skeleton) and tested, in
  order, for: accidental closure, straight line, circular arc, y = f(x) graph,
  otherwise a freehand curve.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..mathematics import geometry as geo
from ..mathematics.function_fitting import fit_function, is_x_monotonic
from .curve_extractor import skeleton_path
from .models import DetectedObject, ShapeKind
from .preprocessing import PreprocessParams


def _ellipse_model(pts: np.ndarray):
    """Fit an ellipse; returns (center, (a, b), angle_rad, residual) with a >= b."""
    if len(pts) < 5:
        return None
    (cx, cy), (w, h), ang = cv2.fitEllipse(pts.astype(np.float32))
    if not all(math.isfinite(v) for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
        return None
    a, b, theta = w / 2.0, h / 2.0, math.radians(ang)
    if b > a:
        a, b, theta = b, a, theta + math.pi / 2
    res = geo.ellipse_residual(pts, (cx, cy), (a, b), theta)
    return np.array([cx, cy]), (a, b), theta, res


def _polygon_model(pts: np.ndarray, eps_frac: float):
    peri = cv2.arcLength(pts.astype(np.float32).reshape(-1, 1, 2), True)
    approx = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2), eps_frac * peri, True).reshape(-1, 2)
    if len(approx) >= 3:
        approx = geo.simplify_polygon(approx, peri)
    if len(approx) >= 3:
        approx = geo.refine_polygon_vertices(pts, approx)
    if len(approx) < 3:
        return approx, float("inf")
    r = math.sqrt(max(geo.polygon_area(pts), 1.0) / math.pi)
    res = float(geo.point_polygon_distances(pts, approx).mean() / r)
    return approx, res


def _average_vertices(outer_v: np.ndarray, inner_v: np.ndarray) -> np.ndarray:
    """Average matched outer/inner outline vertices -> stroke centre line."""
    out = outer_v.copy()
    for i, v in enumerate(outer_v):
        j = int(np.argmin(np.linalg.norm(inner_v - v, axis=1)))
        out[i] = (v + inner_v[j]) / 2.0
    return out


class ShapeDetector:
    def __init__(self, params: PreprocessParams | None = None):
        self.params = params or PreprocessParams()

    # ------------------------------------------------------------------ public
    def detect(self, binary: np.ndarray, stroke_width: float = 3.0) -> list[DetectedObject]:
        h, w = binary.shape[:2]
        diag = math.hypot(w, h)
        min_diag = self.params.min_object_size * diag
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            return []
        hier = hierarchy[0]
        objects: list[DetectedObject] = []
        for i, c in enumerate(contours):
            if hier[i][3] != -1:
                continue  # holes are handled with their parent
            x, y, bw, bh = cv2.boundingRect(c)
            if math.hypot(bw, bh) < min_diag:
                continue
            if self.params.ignore_border_objects and (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1):
                continue
            try:
                obj = self._classify_component(binary, contours, hier, i, stroke_width, diag)
            except (cv2.error, ValueError, np.linalg.LinAlgError):
                obj = None
            if obj is not None:
                objects.append(obj)
        objects.sort(key=lambda o: (o.bbox[0], o.bbox[1]))
        for k, o in enumerate(objects, start=1):
            o.id = k
        return objects

    # --------------------------------------------------------------- internals
    def _classify_component(self, binary, contours, hier, i, t, diag) -> DetectedObject | None:
        c = contours[i]
        outer = c.reshape(-1, 2).astype(float)
        outer_area = cv2.contourArea(c)
        inner, inner_area = None, 0.0
        child = hier[i][2]
        while child != -1:
            a = cv2.contourArea(contours[child])
            if a > inner_area:
                inner, inner_area = contours[child].reshape(-1, 2).astype(float), a
            child = hier[child][0]
        if inner is not None and inner_area > 0.2 * outer_area and len(inner) >= 8:
            return self._classify_closed(outer, inner)

        x, y, bw, bh = cv2.boundingRect(c)
        pad = 2
        x0, y0 = x - pad, y - pad
        mask = np.zeros((bh + 2 * pad, bw + 2 * pad), np.uint8)
        cv2.drawContours(mask, [c - [x0, y0]], -1, 255, thickness=cv2.FILLED)
        H, W = binary.shape[:2]
        ys0, xs0, ys1, xs1 = max(y0, 0), max(x0, 0), min(y + bh + pad, H), min(x + bw + pad, W)
        crop = np.zeros_like(mask)
        crop[ys0 - y0: ys1 - y0, xs0 - x0: xs1 - x0] = binary[ys0:ys1, xs0:xs1]
        mask &= crop  # keep only this component's ink (other strokes may share the bbox)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        if float(dist.max()) > max(6.0, 0.12 * min(bw, bh), 1.6 * t):
            return self._classify_closed(outer, None)       # filled solid shape
        return self._classify_open(mask, np.array([x - pad, y - pad], float), outer, t, diag)

    def _classify_closed(self, outer: np.ndarray, inner: np.ndarray | None, contour=None) -> DetectedObject | None:
        eps = self.params.poly_epsilon
        ell = _ellipse_model(outer)
        poly, poly_res = _polygon_model(outer, eps)
        n = len(poly)
        ell_res = ell[3] if ell else float("inf")
        contour = outer if contour is None else contour

        is_conic = ell is not None and (
            (n >= 5 and ell_res < 0.05) or (ell_res < 0.6 * poly_res and ell_res < 0.05))
        if is_conic:
            center, (a, b), theta, _ = ell
            if inner is not None:
                ell_in = _ellipse_model(inner)
                if ell_in is not None:
                    center = (center + ell_in[0]) / 2.0
                    a, b = (a + ell_in[1][0]) / 2.0, (b + ell_in[1][1]) / 2.0
            kind = ShapeKind.CIRCLE if b / a > 0.86 else ShapeKind.ELLIPSE
            if kind == ShapeKind.CIRCLE:
                a = b = (a + b) / 2.0
            conf = float(np.clip(1.0 - ell_res / 0.08, 0.05, 1.0))
            return DetectedObject(kind, contour, geo.ellipse_points(center, (a, b), theta, 128), conf,
                                  params={"center": center, "semi_axes": (a, b), "angle": theta,
                                          "residual": ell_res})

        if n < 3 or (poly_res > 0.05 and ell_res > 0.05):
            pts = geo.resample_polyline(np.vstack([outer, outer[:1]]), 3.0)
            return DetectedObject(ShapeKind.CLOSED_CURVE, contour, geo.smooth_polyline(pts, 7, closed=True),
                                  float(np.clip(0.5 - poly_res, 0.1, 0.5)), params={"residual": poly_res})

        if inner is not None:
            poly_in, _ = _polygon_model(inner, eps)
            if len(poly_in) == n:
                poly = _average_vertices(poly, poly_in)
        conf = float(np.clip(1.0 - poly_res / 0.08, 0.05, 1.0))
        angles = geo.interior_angles(poly)
        if n == 3:
            kind = ShapeKind.TRIANGLE
        elif n == 4:
            sides = geo.side_lengths(poly)
            if np.all(np.abs(angles - 90.0) < 12.0):
                ratio = (sides[0] + sides[2]) / max(sides[1] + sides[3], 1e-9)
                kind = ShapeKind.SQUARE if 0.88 < ratio < 1.12 else ShapeKind.RECTANGLE
            else:
                kind = ShapeKind.QUADRILATERAL
        else:
            kind = ShapeKind.POLYGON
        return DetectedObject(kind, contour, poly, conf, params={"residual": poly_res, "angles": angles})

    def _classify_open(self, mask, offset, outer, t, diag) -> DetectedObject | None:
        path = skeleton_path(mask, spacing=2.0)
        if len(path) < 8:
            return None
        path = path + offset
        length = geo.polyline_length(path)
        if length < self.params.min_object_size * diag:
            return None
        bbox_diag = float(np.hypot(*np.ptp(path, axis=0)))

        # 1) nearly closed loop with a small gap -> treat as a closed shape
        chord = float(np.linalg.norm(path[0] - path[-1]))
        if chord < max(4.0 * t, 0.08 * length) and length > 2.5 * bbox_diag:
            return self._classify_closed(path, None, contour=outer)

        # 2) straight line
        point, direction, rms = geo.fit_line_pca(path)
        if rms / length < 0.012:
            proj = (path - point) @ direction
            p0, p1 = point + proj.min() * direction, point + proj.max() * direction
            conf = float(np.clip(1.0 - rms / length / 0.02, 0.05, 1.0))
            fit = fit_function(path[:, 0], -path[:, 1]) if abs(direction[0]) > 0.2 else None
            return DetectedObject(ShapeKind.LINE, outer, np.array([p0, p1]), conf,
                                  params={"rms": rms}, fit=fit)

        # 3) circular arc vs graph of a function
        cx, cy, r, crms = geo.fit_circle(path)
        sweep = abs(geo.swept_angle(path, (cx, cy)))
        arc_err = crms / max(bbox_diag, 1e-9)
        func_fit = None
        func_err = float("inf")
        if is_x_monotonic(path):
            func_fit = fit_function(path[:, 0], -path[:, 1])
            if func_fit.best is not None and func_fit.recognized:
                func_err = func_fit.best.rmse / max(bbox_diag, 1e-9)
        if (arc_err < 0.009 and math.radians(35) < sweep < math.radians(340) and r < 3.0 * length
                and arc_err <= 0.8 * func_err):
            conf = float(np.clip(1.0 - arc_err / 0.015, 0.05, 1.0))
            return DetectedObject(ShapeKind.ARC, outer, path, conf,
                                  params={"center": np.array([cx, cy]), "radius": r, "sweep": sweep,
                                          "rms": crms})

        if func_fit is not None and func_fit.recognized:
            return DetectedObject(ShapeKind.FUNCTION, outer, path, func_fit.confidence,
                                  params={"family": func_fit.best.family}, fit=func_fit)

        conf = 0.35 if func_fit is None else float(min(func_fit.confidence, 0.4))
        return DetectedObject(ShapeKind.FREEHAND, outer, path, conf,
                              params={"x_monotonic": func_fit is not None}, fit=func_fit)
