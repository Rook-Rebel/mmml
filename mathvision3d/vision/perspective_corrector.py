"""Paper sheet detection and perspective rectification."""

from __future__ import annotations

import cv2
import numpy as np


def order_quad(pts) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    p = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = p.sum(axis=1)
    d = p[:, 1] - p[:, 0]
    return np.array([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]], np.float32)


def find_paper_quad(bgr: np.ndarray, min_area_frac: float = 0.15, max_area_frac: float = 0.97):
    """Find the largest bright convex quadrilateral (the paper sheet). Returns (4,2) or None."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    if float(blur.std()) < 8.0:
        return None
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    k = max(5, (min(h, w) // 60) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if not (min_area_frac * h * w <= area <= max_area_frac * h * w):
        return None
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = order_quad(approx)
            # reject quads that are mostly the image border (paper fills the frame)
            margin = 0.01 * max(h, w)
            on_border = np.sum((quad[:, 0] < margin) | (quad[:, 0] > w - margin) |
                               (quad[:, 1] < margin) | (quad[:, 1] > h - margin))
            if on_border >= 3:
                return None
            return quad
    return None


def rectify(bgr: np.ndarray, quad, max_dim: int = 960) -> tuple[np.ndarray, np.ndarray]:
    """Warp the quad to an upright rectangle. Returns (warped, H) where H maps source -> warped."""
    q = order_quad(quad)
    tl, tr, br, bl = q
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    s = max_dim / max(width, height, 1.0)
    w_out, h_out = max(int(round(width * s)), 2), max(int(round(height * s)), 2)
    dst = np.array([[0, 0], [w_out - 1, 0], [w_out - 1, h_out - 1], [0, h_out - 1]], np.float32)
    h_mat = cv2.getPerspectiveTransform(q, dst)
    warped = cv2.warpPerspective(bgr, h_mat, (w_out, h_out), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    return warped, h_mat
