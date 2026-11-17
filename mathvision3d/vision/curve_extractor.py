"""Centre-line extraction for open strokes and coordinate-axis detection/removal."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np
from skimage.morphology import skeletonize

from ..mathematics.geometry import resample_polyline, smooth_polyline
from .models import AxesInfo

_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _bfs(adj: list[np.ndarray], start: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(adj)
    dist = np.full(n, -1, np.int64)
    parent = np.full(n, -1, np.int64)
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        du = dist[u] + 1
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = du
                parent[v] = u
                q.append(v)
    return dist, parent


def skeleton_path(mask: np.ndarray, spacing: float = 2.0) -> np.ndarray:
    """Longest geodesic path through the skeleton of a stroke mask, as ordered (x, y) points."""
    sk = skeletonize(mask > 0)
    ys, xs = np.nonzero(sk)
    n = len(xs)
    if n < 2:
        return np.column_stack([xs, ys]).astype(float)
    index = -np.ones((sk.shape[0] + 2, sk.shape[1] + 2), np.int64)
    index[ys + 1, xs + 1] = np.arange(n)
    neigh = np.stack([index[ys + 1 + dy, xs + 1 + dx] for dy, dx in _OFFSETS], axis=1)
    adj = [row[row >= 0] for row in neigh]
    # diameter of the skeleton graph (two BFS passes); restart from the largest piece
    dist, _ = _bfs(adj, 0)
    if (dist < 0).any():
        # skeleton is fragmented: keep the biggest connected piece
        best, best_size = 0, int((dist >= 0).sum())
        seen = dist >= 0
        for s in range(n):
            if not seen[s]:
                d, _ = _bfs(adj, s)
                comp = d >= 0
                seen |= comp
                if comp.sum() > best_size:
                    best, best_size = s, int(comp.sum())
        dist, _ = _bfs(adj, best)
    a = int(np.argmax(dist))
    dist, parent = _bfs(adj, a)
    b = int(np.argmax(dist))
    path = []
    v = b
    while v >= 0:
        path.append(v)
        v = int(parent[v])
    pts = np.column_stack([xs[path], ys[path]]).astype(float)
    pts = smooth_polyline(pts, window=5)
    return resample_polyline(pts, spacing)


# --------------------------------------------------------------------------- axes
def _angle_deg(seg) -> float:
    x1, y1, x2, y2 = seg
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)


def _merge_collinear(segs: list, horizontal: bool, tol: float) -> list[np.ndarray]:
    """Group near-collinear segments and return the full extent of each group."""
    groups: list[list] = []
    for s in sorted(segs, key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1])):
        key = (s[1] + s[3]) / 2 if horizontal else (s[0] + s[2]) / 2
        for g in groups:
            if abs(g[0] - key) <= tol:
                g[1].append(s)
                break
        else:
            groups.append([key, [s]])
    out = []
    for key, members in groups:
        arr = np.array(members, float)
        if horizontal:
            lo, hi = arr[:, [0, 2]].min(), arr[:, [0, 2]].max()
            out.append(np.array([[lo, key], [hi, key]]))
        else:
            lo, hi = arr[:, [1, 3]].min(), arr[:, [1, 3]].max()
            out.append(np.array([[key, lo], [key, hi]]))
    return out


def detect_axes(binary: np.ndarray, stroke_width: float, angle_tol: float = 6.0) -> AxesInfo | None:
    """Detect a pair of long, crossing horizontal/vertical lines (a '+' shaped coordinate frame)."""
    h, w = binary.shape
    min_len = 0.35 * min(h, w)
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=int(min_len * 0.5),
                            minLineLength=int(min_len * 0.6), maxLineGap=int(max(6, 2 * stroke_width)))
    if lines is None:
        return None
    horiz, vert = [], []
    for seg in np.asarray(lines).reshape(-1, 4):  # OpenCV 4: (N,1,4), OpenCV 5: (N,4)
        a = _angle_deg(seg)
        if a < angle_tol or a > 180 - angle_tol:
            horiz.append(seg)
        elif abs(a - 90) < angle_tol:
            vert.append(seg)
    if not horiz or not vert:
        return None
    tol = 2 * stroke_width + 3
    hs = [s for s in _merge_collinear(horiz, True, tol) if np.ptp(s[:, 0]) >= min_len]
    vs = [s for s in _merge_collinear(vert, False, tol) if np.ptp(s[:, 1]) >= min_len]
    best = None
    for hseg in hs:
        for vseg in vs:
            ox, oy = vseg[0, 0], hseg[0, 1]
            hlen, vlen = np.ptp(hseg[:, 0]), np.ptp(vseg[:, 1])
            # the crossing must be interior to both segments (rejects rectangle corners)
            fx = (ox - hseg[:, 0].min()) / hlen
            fy = (oy - vseg[:, 1].min()) / vlen
            if 0.08 < fx < 0.92 and 0.08 < fy < 0.92:
                score = hlen + vlen
                if best is None or score > best[0]:
                    best = (score, hseg, vseg, np.array([ox, oy]))
    if best is None:
        return None
    _, hseg, vseg, origin = best
    return AxesInfo(origin=origin, x_segment=hseg, y_segment=vseg, thickness=stroke_width)


def remove_axes(binary: np.ndarray, axes: AxesInfo) -> np.ndarray:
    """Erase axis lines and re-bridge strokes (curves) that crossed them."""
    band = int(round(axes.thickness * 1.6 + 4))
    removal = np.zeros_like(binary)
    hband = np.zeros_like(binary)
    vband = np.zeros_like(binary)
    (x0, yh), (x1, _) = axes.x_segment
    (xv, y0), (_, y1) = axes.y_segment
    ext_x = 0.03 * (x1 - x0)
    ext_y = 0.03 * (y1 - y0)
    cv2.line(hband, (int(x0 - ext_x), int(yh)), (int(x1 + ext_x), int(yh)), 255, band)
    cv2.line(vband, (int(xv), int(y0 - ext_y)), (int(xv), int(y1 + ext_y)), 255, band)
    removal = cv2.bitwise_or(hband, vband)
    out = cv2.bitwise_and(binary, cv2.bitwise_not(removal))
    # Re-connect strokes across the erased band: grow the remaining ink only *inside*
    # the band, so pieces on both sides of an axis meet again. Ink is needed on both
    # sides (an erosion with the same kernel removes one-sided stubs such as ticks).
    # (where both axes cross, the erased gap is about band * sqrt(2) wide)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 5, 2 * band + 5))
    grown = cv2.dilate(out, k) & removal
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
    _, ink_labels = cv2.connectedComponents(out, connectivity=8)
    H, W = out.shape
    bridges = np.zeros_like(out)
    for lbl in range(1, count):
        x, y, w, h = stats[lbl, :4]
        x0, y0, x1, y1 = max(x - 3, 0), max(y - 3, 0), min(x + w + 3, W), min(y + h + 3, H)
        blob = (labels[y0:y1, x0:x1] == lbl).astype(np.uint8)
        touch = cv2.dilate(blob, np.ones((5, 5), np.uint8)) > 0
        touched = np.unique(ink_labels[y0:y1, x0:x1][touch])
        # a real crossing re-joins at least two separate ink pieces; one-sided stubs
        # (ticks, a curve merely touching the axis) are left alone
        if np.count_nonzero(touched) >= 2:
            bridges[y0:y1, x0:x1][blob > 0] = 255
    return out | bridges
