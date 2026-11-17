"""Planar geometry helpers used by recognition and analysis.

All functions take ``(N, 2)`` float arrays. They are orientation-agnostic unless
noted, so they work both in image pixel coordinates (y down) and in
mathematical coordinates (y up).
"""

from __future__ import annotations

import math

import numpy as np


def as_points(pts) -> np.ndarray:
    arr = np.asarray(pts, dtype=float)
    if arr.ndim == 3:  # OpenCV contour shape (N, 1, 2)
        arr = arr.reshape(-1, 2)
    return arr


def signed_area(pts) -> float:
    p = as_points(pts)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_area(pts) -> float:
    return abs(signed_area(pts))


def side_lengths(pts) -> np.ndarray:
    p = as_points(pts)
    return np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)


def polygon_perimeter(pts) -> float:
    return float(side_lengths(pts).sum())


def polyline_length(pts) -> float:
    p = as_points(pts)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def polygon_centroid(pts) -> np.ndarray:
    p = as_points(pts)
    a = signed_area(p)
    if abs(a) < 1e-12:
        return p.mean(axis=0)
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    cx = float(((x + xn) * cross).sum() / (6.0 * a))
    cy = float(((y + yn) * cross).sum() / (6.0 * a))
    return np.array([cx, cy])


def interior_angles(pts) -> np.ndarray:
    """Interior angles in degrees for a simple polygon (convex or concave)."""
    p = as_points(pts)
    n = len(p)
    orient = np.sign(signed_area(p)) or 1.0
    angles = np.empty(n)
    for i in range(n):
        prev, cur, nxt = p[i - 1], p[i], p[(i + 1) % n]
        v1, v2 = prev - cur, nxt - cur
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom < 1e-12:
            angles[i] = 180.0
            continue
        ang = math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))))
        e1, e2 = cur - prev, nxt - cur
        turn = e1[0] * e2[1] - e1[1] * e2[0]
        if turn * orient < 0:
            ang = 360.0 - ang
        angles[i] = ang
    return angles


def classify_triangle(sides, angles, side_tol: float = 0.06, angle_tol: float = 4.0) -> tuple[str, str]:
    """Return (by_sides, by_angles) as keys: equilateral/isosceles/scalene, acute/right/obtuse."""
    s = np.sort(np.asarray(sides, dtype=float))
    rel = lambda a, b: abs(a - b) / max(a, b, 1e-12)  # noqa: E731
    if rel(s[0], s[2]) < side_tol:
        by_sides = "equilateral"
    elif rel(s[0], s[1]) < side_tol or rel(s[1], s[2]) < side_tol:
        by_sides = "isosceles"
    else:
        by_sides = "scalene"
    amax = float(np.max(angles))
    if abs(amax - 90.0) <= angle_tol:
        by_angles = "right"
    elif amax > 90.0:
        by_angles = "obtuse"
    else:
        by_angles = "acute"
    return by_sides, by_angles


def fit_circle(pts) -> tuple[float, float, float, float]:
    """Algebraic (Kasa) least-squares circle fit. Returns (cx, cy, r, rms_residual)."""
    p = as_points(pts)
    x, y = p[:, 0], p[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))
    resid = np.hypot(x - cx, y - cy) - r
    return float(cx), float(cy), float(r), float(np.sqrt(np.mean(resid**2)))


def ellipse_points(center, semi_axes, angle: float, n: int = 128) -> np.ndarray:
    """Sample an ellipse. ``angle`` (radians) is the direction of the first semi-axis."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    a, b = semi_axes
    c, s = math.cos(angle), math.sin(angle)
    u, v = a * np.cos(t), b * np.sin(t)
    return np.column_stack([center[0] + u * c - v * s, center[1] + u * s + v * c])


def ellipse_residual(pts, center, semi_axes, angle: float) -> float:
    """Mean approximate point-to-ellipse distance, normalised by sqrt(a*b)."""
    p = as_points(pts) - np.asarray(center, dtype=float)
    a, b = max(semi_axes[0], 1e-9), max(semi_axes[1], 1e-9)
    c, s = math.cos(angle), math.sin(angle)
    u = p[:, 0] * c + p[:, 1] * s
    v = -p[:, 0] * s + p[:, 1] * c
    rho = np.sqrt((u / a) ** 2 + (v / b) ** 2)
    r = np.hypot(u, v)
    dist = np.abs(1.0 - 1.0 / np.maximum(rho, 1e-9)) * r
    return float(dist.mean() / math.sqrt(a * b))


def ellipse_metrics(a: float, b: float) -> dict:
    a, b = max(a, b), min(a, b)
    h = ((a - b) / (a + b)) ** 2 if a + b > 0 else 0.0
    perimeter = math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))  # Ramanujan II
    ecc = math.sqrt(max(0.0, 1 - (b * b) / (a * a))) if a > 0 else 0.0
    return {"area": math.pi * a * b, "perimeter": perimeter, "eccentricity": ecc}


def point_segment_distances(pts, a, b) -> np.ndarray:
    p = as_points(pts)
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return np.linalg.norm(p - a, axis=1)
    t = np.clip(((p - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(p - proj, axis=1)


def point_polygon_distances(pts, poly) -> np.ndarray:
    """Distance from each point to the closest edge of a closed polygon."""
    q = as_points(poly)
    d = np.full(len(as_points(pts)), np.inf)
    for i in range(len(q)):
        d = np.minimum(d, point_segment_distances(pts, q[i], q[(i + 1) % len(q)]))
    return d


def simplify_polygon(poly, perimeter: float, angle_tol_deg: float = 155.0, min_edge_frac: float = 0.05) -> np.ndarray:
    """Remove nearly collinear vertices and merge vertices joined by very short edges."""
    p = [np.asarray(v, float) for v in as_points(poly)]
    changed = True
    while changed and len(p) > 3:
        changed = False
        n = len(p)
        # merge the shortest edge if it is tiny
        lengths = [np.linalg.norm(p[(i + 1) % n] - p[i]) for i in range(n)]
        i = int(np.argmin(lengths))
        if lengths[i] < min_edge_frac * perimeter:
            j = (i + 1) % n
            mid = (p[i] + p[j]) / 2.0
            p[i] = mid
            del p[j]
            changed = True
            continue
        for i in range(n):
            prev, cur, nxt = p[i - 1], p[i], p[(i + 1) % n]
            v1, v2 = prev - cur, nxt - cur
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            if denom < 1e-12:
                continue
            ang = math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / denom, -1, 1))))
            if ang > angle_tol_deg:
                del p[i]
                changed = True
                break
    return np.array(p)


def line_intersection(p1, d1, p2, d2):
    """Intersect two infinite lines given point + direction. Returns None when parallel."""
    p1, d1, p2, d2 = (np.asarray(v, float) for v in (p1, d1, p2, d2))
    m = np.column_stack([d1, -d2])
    if abs(np.linalg.det(m)) < 1e-9:
        return None
    t = np.linalg.solve(m, p2 - p1)
    return p1 + t[0] * d1


def fit_line_pca(pts) -> tuple[np.ndarray, np.ndarray, float]:
    """Total least squares line. Returns (point, unit_direction, rms_perpendicular_residual)."""
    p = as_points(pts)
    c = p.mean(axis=0)
    _, s, vt = np.linalg.svd(p - c, full_matrices=False)
    direction = vt[0]
    rms = float(s[1] / math.sqrt(len(p))) if len(s) > 1 else 0.0
    return c, direction, rms


def refine_polygon_vertices(contour, approx, trim: float = 0.18) -> np.ndarray:
    """Re-estimate polygon vertices by fitting a line to each side and intersecting
    neighbouring sides. Much more stable than raw approxPolyDP corners for hand drawings."""
    c = as_points(contour)
    v = as_points(approx)
    n = len(v)
    if n < 3 or len(c) < 4 * n:
        return v
    idx = [int(np.argmin(np.linalg.norm(c - q, axis=1))) for q in v]
    # Keep indices in contour order
    order = np.argsort(idx)
    idx = [idx[i] for i in order]
    v = v[order]
    lines = []
    for k in range(n):
        i0, i1 = idx[k], idx[(k + 1) % n]
        seg = c[i0:i1 + 1] if i1 > i0 else np.vstack([c[i0:], c[: i1 + 1]])
        m = len(seg)
        lo, hi = int(m * trim), int(m * (1 - trim))
        core = seg[lo:hi] if hi - lo >= 3 else seg
        if len(core) < 2:
            lines.append(None)
            continue
        point, direction, _ = fit_line_pca(core)
        lines.append((point, direction))
    perimeter = polygon_perimeter(v)
    out = v.copy()
    for k in range(n):
        la, lb = lines[k - 1], lines[k]
        if la is None or lb is None:
            continue
        x = line_intersection(la[0], la[1], lb[0], lb[1])
        if x is not None and np.linalg.norm(x - v[k]) < 0.12 * perimeter:
            out[k] = x
    return out


def resample_polyline(pts, spacing: float) -> np.ndarray:
    p = as_points(pts)
    if len(p) < 2:
        return p
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total <= 0:
        return p[:1]
    n = max(2, int(total / spacing) + 1)
    t = np.linspace(0.0, total, n)
    return np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])


def smooth_polyline(pts, window: int = 5, closed: bool = False) -> np.ndarray:
    p = as_points(pts)
    if window < 2 or len(p) < window:
        return p
    k = np.ones(window) / window
    if closed:
        pad = np.vstack([p[-window:], p, p[:window]])
        out = np.column_stack([np.convolve(pad[:, i], k, mode="same") for i in range(2)])
        return out[window:-window]
    half = window // 2
    pad = np.vstack([np.repeat(p[:1], half, axis=0), p, np.repeat(p[-1:], half, axis=0)])
    out = np.column_stack([np.convolve(pad[:, i], k, mode="valid") for i in range(2)])
    out[0], out[-1] = p[0], p[-1]
    return out[: len(p)]


def swept_angle(pts, center) -> float:
    """Total signed angle (radians) swept by a path around ``center``."""
    p = as_points(pts) - np.asarray(center, float)
    ang = np.unwrap(np.arctan2(p[:, 1], p[:, 0]))
    return float(ang[-1] - ang[0])
