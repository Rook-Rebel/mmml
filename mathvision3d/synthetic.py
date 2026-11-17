"""Synthetic "hand-drawn" test images with known ground truth.

Used by the automated tests and by ``tools/generate_samples.py`` to produce the
sample images that can be loaded in the app instead of a webcam.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

INK = (40, 35, 30)


def blank_paper(w: int = 1000, h: int = 750, shade: int = 244) -> np.ndarray:
    return np.full((h, w, 3), shade, np.uint8)


def _wobble(pts: np.ndarray, rng: np.random.Generator, amp: float) -> np.ndarray:
    """Add smooth low-frequency perpendicular noise, imitating an unsteady hand."""
    if amp <= 0 or len(pts) < 3:
        return pts
    n = len(pts)
    noise = rng.normal(0, 1, n + 20)
    k = np.ones(15) / 15
    noise = np.convolve(noise, k, mode="same")[10:10 + n]
    noise = noise / (np.abs(noise).max() + 1e-9) * amp
    d = np.gradient(pts, axis=0)
    nrm = np.column_stack([-d[:, 1], d[:, 0]])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    return pts + nrm * noise[:, None]


def _densify(poly: np.ndarray, closed: bool, step: float = 3.0) -> np.ndarray:
    pts = np.vstack([poly, poly[:1]]) if closed else poly
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(2, int(np.linalg.norm(b - a) / step))
        out.append(np.linspace(a, b, n, endpoint=False))
    out.append(pts[-1:])
    return np.vstack(out)


def draw_path(img, pts, thickness: int = 4, closed: bool = False, color=INK) -> None:
    p = np.round(np.asarray(pts)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [p], closed, color, thickness, lineType=cv2.LINE_AA)


def draw_polygon(img, vertices, rng=None, wobble: float = 1.5, thickness: int = 4) -> None:
    rng = rng or np.random.default_rng(0)
    v = np.asarray(vertices, float)
    pts = _wobble(_densify(v, True), rng, wobble)
    draw_path(img, pts, thickness, closed=True)


def draw_ellipse(img, center, axes, angle_deg=0.0, rng=None, wobble: float = 1.5, thickness: int = 4) -> None:
    rng = rng or np.random.default_rng(0)
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    a, b = axes
    th = math.radians(angle_deg)
    u, v = a * np.cos(t), b * np.sin(t)
    pts = np.column_stack([center[0] + u * math.cos(th) - v * math.sin(th),
                           center[1] + u * math.sin(th) + v * math.cos(th)])
    draw_path(img, _wobble(pts, rng, wobble), thickness, closed=True)


def draw_function(img, f, x0_px, x1_px, origin, scale, rng=None, wobble: float = 1.0,
                  thickness: int = 4, xmin=None, xmax=None) -> np.ndarray:
    """Draw y = f(x). ``origin`` is the pixel of (0, 0), ``scale`` is px per unit."""
    rng = rng or np.random.default_rng(0)
    xs_px = np.linspace(x0_px, x1_px, 500)
    xs = (xs_px - origin[0]) / scale
    ys_px = origin[1] - f(xs) * scale
    pts = _wobble(np.column_stack([xs_px, ys_px]), rng, wobble)
    draw_path(img, pts, thickness)
    return pts


def draw_axes(img, origin, half_w, half_h, thickness: int = 3, ticks: bool = True) -> None:
    ox, oy = origin
    cv2.line(img, (int(ox - half_w), int(oy)), (int(ox + half_w), int(oy)), INK, thickness, cv2.LINE_AA)
    cv2.line(img, (int(ox), int(oy + half_h)), (int(ox), int(oy - half_h)), INK, thickness, cv2.LINE_AA)
    if ticks:
        for k in range(-int(half_w // 60), int(half_w // 60) + 1):
            x = int(ox + k * 60)
            cv2.line(img, (x, int(oy - 6)), (x, int(oy + 6)), INK, 2, cv2.LINE_AA)


def add_noise(img, rng, sigma: float = 6.0, specks: int = 0) -> np.ndarray:
    out = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    for _ in range(specks):
        x, y = int(rng.integers(0, img.shape[1])), int(rng.integers(0, img.shape[0]))
        cv2.circle(out, (x, y), int(rng.integers(1, 3)), (90, 90, 90), -1)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_shadow(img, strength: float = 0.35) -> np.ndarray:
    h, w = img.shape[:2]
    gx = np.linspace(1.0 - strength, 1.0, w)[None, :]
    gy = np.linspace(1.0, 1.0 - strength * 0.5, h)[:, None]
    return np.clip(img.astype(np.float32) * (gx * gy)[..., None], 0, 255).astype(np.uint8)


def photograph(paper: np.ndarray, out_size=(1280, 720), corners=None, background: int = 55) -> tuple[np.ndarray, np.ndarray]:
    """Project the paper into a camera frame with perspective distortion."""
    w, h = out_size
    ph, pw = paper.shape[:2]
    if corners is None:
        corners = np.array([[260, 110], [1010, 70], [1120, 650], [180, 620]], np.float32)
    src = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], np.float32)
    m = cv2.getPerspectiveTransform(src, np.asarray(corners, np.float32))
    frame = np.full((h, w, 3), background, np.uint8)
    frame[:] = (background, background + 5, background + 12)
    warped = cv2.warpPerspective(paper, m, (w, h))
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), m, (w, h))
    frame[mask > 0] = warped[mask > 0]
    return frame, m


# ------------------------------------------------------------------ scenes
def scene_triangle(seed: int = 1):
    rng = np.random.default_rng(seed)
    img = blank_paper()
    v = np.array([[300, 560], [720, 590], [480, 180]], float)
    draw_polygon(img, v, rng)
    return add_noise(img, rng), {"vertices": v}


def scene_rotated_triangle(seed: int = 2):
    rng = np.random.default_rng(seed)
    img = blank_paper()
    base = np.array([[-200, 120], [220, 140], [-20, -230]], float)
    th = math.radians(37)
    r = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    v = base @ r.T + [500, 380]
    draw_polygon(img, v, rng)
    return add_noise(img, rng), {"vertices": v}


def scene_circle(seed: int = 3):
    rng = np.random.default_rng(seed)
    img = blank_paper()
    draw_ellipse(img, (500, 375), (210, 210), 0, rng, wobble=2.0)
    return add_noise(img, rng), {"center": (500, 375), "radius": 210}


def scene_ellipse(seed: int = 4):
    rng = np.random.default_rng(seed)
    img = blank_paper()
    draw_ellipse(img, (500, 375), (300, 150), 25, rng)
    return add_noise(img, rng), {"center": (500, 375), "axes": (300, 150), "angle": 25}


def scene_rectangle(seed: int = 5):
    rng = np.random.default_rng(seed)
    img = blank_paper()
    v = np.array([[250, 220], [760, 220], [760, 530], [250, 530]], float)
    draw_polygon(img, v, rng)
    return add_noise(img, rng), {"vertices": v}


def scene_sine(seed: int = 6, noisy: bool = False):
    rng = np.random.default_rng(seed)
    img = blank_paper()
    origin, scale = (500, 375), 70.0
    f = lambda x: 2.0 * np.sin(x)  # noqa: E731
    draw_function(img, f, 120, 880, origin, scale, rng, wobble=3.5 if noisy else 1.0)
    if noisy:
        img = add_noise(img, rng, sigma=14, specks=60)
        img = add_shadow(img, 0.4)
    else:
        img = add_noise(img, rng)
    span = (880 - 120) / scale
    return img, {"periods": span / (2 * math.pi), "amp_ratio": 2.0 * scale}


def scene_sine_with_axes(seed: int = 7):
    rng = np.random.default_rng(seed)
    img = blank_paper()
    origin, scale = (500, 375), 70.0
    draw_axes(img, origin, 440, 300)
    draw_function(img, lambda x: 2.0 * np.sin(x), 100, 900, origin, scale, rng)
    return add_noise(img, rng), {"origin": origin, "scale": scale}


def scene_parabola(seed: int = 8):
    rng = np.random.default_rng(seed)
    img = blank_paper()
    draw_function(img, lambda x: 0.5 * x**2, 260, 740, (500, 620), 60.0, rng)
    return add_noise(img, rng), {}


def scene_multiple(seed: int = 9):
    rng = np.random.default_rng(seed)
    img = blank_paper(1200, 800)
    draw_polygon(img, np.array([[90, 330], [330, 340], [200, 110]], float), rng)
    draw_ellipse(img, (600, 220), (130, 130), 0, rng)
    draw_polygon(img, np.array([[880, 110], [1110, 110], [1110, 330], [880, 330]], float), rng)
    draw_function(img, lambda x: 1.2 * np.sin(1.5 * x), 120, 1080, (600, 600), 60.0, rng)
    return add_noise(img, rng), {"kinds": {"triangle", "circle", "square", "function"}}


def scene_perspective_sine(seed: int = 10):
    rng = np.random.default_rng(seed)
    paper = blank_paper(1000, 700)
    draw_function(paper, lambda x: 1.8 * np.sin(x), 110, 890, (500, 350), 62.0, rng)
    frame, _ = photograph(add_noise(paper, rng, 4))
    frame = add_noise(frame, rng, 5)
    return frame, {"periods": (890 - 110) / 62.0 / (2 * math.pi)}


def scene_perspective_triangle(seed: int = 11):
    rng = np.random.default_rng(seed)
    paper = blank_paper(1000, 700)
    draw_polygon(paper, np.array([[250, 560], [760, 560], [505, 120]], float), rng)
    frame, _ = photograph(add_noise(paper, rng, 4))
    return add_noise(frame, rng, 5), {}


def scene_line_and_arc(seed: int = 12):
    rng = np.random.default_rng(seed)
    img = blank_paper()
    pts = _wobble(_densify(np.array([[120, 600], [880, 470]], float), False), rng, 0.8)
    draw_path(img, pts)
    t = np.linspace(math.radians(200), math.radians(340), 300)
    arc = np.column_stack([500 + 260 * np.cos(t), 360 + 260 * np.sin(t)])
    draw_path(img, _wobble(arc, rng, 0.8))
    return add_noise(img, rng), {}


def make_demo_video(path: str, seconds: float = 6.0, fps: int = 20, seed: int = 21) -> str:
    """A simulated hand-held webcam clip: a sheet with a triangle and a sine wave that
    sways and tilts in front of the camera. Used to test the live pipeline offline."""
    rng = np.random.default_rng(seed)
    paper = blank_paper(1000, 700)
    draw_polygon(paper, np.array([[90, 300], [330, 310], [210, 90]], float), rng)
    draw_function(paper, lambda x: 1.3 * np.sin(1.4 * x), 400, 940, (670, 300), 55.0, rng)
    draw_ellipse(paper, (500, 560), (110, 70), 10, rng)
    paper = add_noise(paper, rng, 3)
    base = np.array([[260, 110], [1010, 70], [1120, 650], [180, 620]], np.float32)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError(f"cannot write video {path}")
    n = int(seconds * fps)
    for i in range(n):
        t = i / fps
        sway = np.array([[18 * math.sin(1.3 * t), 10 * math.sin(0.9 * t)]], np.float32)
        tilt = 25 * math.sin(0.7 * t)
        corners = base + sway + np.array([[tilt, 0], [-tilt, 0], [tilt * 0.5, 0], [-tilt * 0.5, 0]], np.float32)
        frame, _ = photograph(paper, corners=corners)
        writer.write(add_shadow(add_noise(frame, rng, 4), 0.25))
    writer.release()
    return path


SCENES = {
    "triangle": scene_triangle,
    "rotated_triangle": scene_rotated_triangle,
    "circle": scene_circle,
    "ellipse": scene_ellipse,
    "rectangle": scene_rectangle,
    "sine": scene_sine,
    "noisy_sine": lambda: scene_sine(seed=16, noisy=True),
    "sine_with_axes": scene_sine_with_axes,
    "parabola": scene_parabola,
    "multiple_objects": scene_multiple,
    "perspective_sine": scene_perspective_sine,
    "perspective_triangle": scene_perspective_triangle,
    "line_and_arc": scene_line_and_arc,
}
