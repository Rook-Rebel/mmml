"""Fit hand-drawn curve samples to known function families.

Families (all closed under affine rescaling of x and y, so recognition does not
depend on the unknown drawing scale):

    linear     y = m x + b
    abs        y = a |x - h| + k
    quadratic  y = a x^2 + b x + c          (shown in vertex form)
    cubic      y = a x^3 + b x^2 + c x + d
    sine       y = A sin(B x + C) + D       (also reported as cos when the phase fits)

Model selection prefers the simplest family whose error is within a small margin
of the best one. When no family fits well the curve is kept as a smoothing
spline and *no equation is invented*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares

FAMILY_COMPLEXITY = {"linear": 2, "abs": 3, "quadratic": 3, "cubic": 4, "sine": 4}
# Families that contain another one as a special case: they never count as competing evidence.
SUPERSETS = {"linear": {"quadratic", "cubic", "abs", "sine"}, "quadratic": {"cubic"}}

RECOGNITION_THRESHOLD = 0.05   # max NRMSE (fraction of y-range) to accept a family
MIN_SINE_PERIODS = 0.6          # a sine must show at least this many periods
MAX_SINE_PERIODS = 14.0


def fmt(v: float, digits: int = 3) -> str:
    if not math.isfinite(v):
        return "?"
    if abs(v) < 0.5 * 10 ** (-digits):
        v = 0.0
    s = f"{v:.{digits}g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.2e}"
    return s.replace("-", "−")


def _signed(v: float, digits: int = 3) -> str:
    """' + 1.2' / ' − 1.2' for appending a term; empty for ~0."""
    if abs(v) < 0.5 * 10 ** (-digits):
        return ""
    return (" + " if v >= 0 else " − ") + fmt(abs(v), digits)


def _shift(h: float) -> str:
    """'x − 1.2' / 'x + 1.2' / 'x'."""
    if abs(h) < 5e-4:
        return "x"
    return f"x − {fmt(h)}" if h > 0 else f"x + {fmt(-h)}"


@dataclass
class ModelFit:
    family: str
    params: tuple
    rmse: float
    nrmse: float
    n_params: int

    # ---- evaluation ------------------------------------------------------
    def evaluate(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        p = self.params
        if self.family == "linear":
            return p[0] * x + p[1]
        if self.family == "quadratic":
            return np.polyval(p, x)
        if self.family == "cubic":
            return np.polyval(p, x)
        if self.family == "abs":
            return p[0] * np.abs(x - p[1]) + p[2]
        if self.family == "sine":
            return p[0] * np.sin(p[1] * x + p[2]) + p[3]
        raise ValueError(self.family)

    def derivative(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        p = self.params
        if self.family == "linear":
            return np.full_like(x, p[0])
        if self.family in ("quadratic", "cubic"):
            return np.polyval(np.polyder(p), x)
        if self.family == "abs":
            d = p[0] * np.sign(x - p[1])
            return np.where(np.abs(x - p[1]) < 1e-9, np.nan, d)
        if self.family == "sine":
            return p[0] * p[1] * np.cos(p[1] * x + p[2])
        raise ValueError(self.family)

    # ---- presentation ----------------------------------------------------
    def sine_form(self, origin_meaningful: bool) -> str:
        """'sin', 'cos' or 'sinusoid' depending on the phase relative to the origin."""
        if self.family != "sine" or not origin_meaningful:
            return "sinusoid"
        c = self.params[2]
        if abs(math.remainder(c, 2 * math.pi)) < 0.3:
            return "sin"
        if abs(math.remainder(c - math.pi / 2, 2 * math.pi)) < 0.3:
            return "cos"
        return "sinusoid"

    def equation(self, origin_meaningful: bool = True) -> str:
        p = self.params
        if self.family == "linear":
            return f"y = {fmt(p[0])}x{_signed(p[1])}"
        if self.family == "quadratic":
            a, b, c = p
            h = -b / (2 * a) if abs(a) > 1e-12 else 0.0
            k = c - b * b / (4 * a) if abs(a) > 1e-12 else c
            return f"y = {fmt(a)}({_shift(h)})²{_signed(k)}"
        if self.family == "cubic":
            a, b, c, d = p
            return f"y = {fmt(a)}x³{_signed(b)}x²{_signed(c)}x{_signed(d)}"
        if self.family == "abs":
            return f"y = {fmt(p[0])}|{_shift(p[1])}|{_signed(p[2])}"
        if self.family == "sine":
            a, b, c, d = p
            form = self.sine_form(origin_meaningful)
            if form == "cos":
                c2 = math.remainder(c - math.pi / 2, 2 * math.pi)
                return f"y = {fmt(a)}·cos({fmt(b)}x{_signed(c2)}){_signed(d)}"
            return f"y = {fmt(a)}·sin({fmt(b)}x{_signed(c)}){_signed(d)}"
        return "?"

    def named_params(self) -> dict:
        p = self.params
        if self.family == "sine":
            return {"A": p[0], "B": p[1], "C": p[2], "D": p[3], "period": 2 * math.pi / p[1]}
        if self.family == "linear":
            return {"m": p[0], "b": p[1]}
        if self.family == "abs":
            return {"a": p[0], "h": p[1], "k": p[2]}
        if self.family == "quadratic":
            a, b, c = p
            return {"a": a, "b": b, "c": c, "vertex_x": -b / (2 * a) if a else float("nan")}
        if self.family == "cubic":
            return dict(zip("abcd", p))
        return {}


@dataclass
class FunctionFitResult:
    x: np.ndarray                      # resampled, sorted samples used for fitting
    y: np.ndarray
    candidates: list[ModelFit] = field(default_factory=list)
    best: ModelFit | None = None
    recognized: bool = False
    confidence: float = 0.0
    spline: UnivariateSpline | None = None

    @property
    def domain(self) -> tuple[float, float]:
        return float(self.x.min()), float(self.x.max())

    @property
    def nrmse(self) -> float:
        if self.best is not None:
            return self.best.nrmse
        if self.spline is not None:
            r = self.spline(self.x) - self.y
            return float(np.sqrt(np.mean(r**2)) / _yscale(self.x, self.y))
        return float("nan")

    def function(self) -> Callable[[np.ndarray], np.ndarray]:
        if self.recognized and self.best is not None:
            return self.best.evaluate
        return lambda x: self.spline(np.asarray(x, float))  # type: ignore[misc]

    def derivative(self) -> Callable[[np.ndarray], np.ndarray]:
        if self.recognized and self.best is not None:
            return self.best.derivative
        d = self.spline.derivative()  # type: ignore[union-attr]
        return lambda x: d(np.asarray(x, float))


def _yscale(x, y) -> float:
    yr = float(np.ptp(y))
    xr = float(np.ptp(x))
    return max(yr, 0.05 * xr, 1e-9)


def prepare_samples(x, y, n_bins: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Sort by x and average y inside uniform x bins (removes stroke-thickness duplicates)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    if len(x) < 4 or np.ptp(x) <= 0:
        return x, y
    n_bins = int(min(n_bins, max(8, len(x))))
    edges = np.linspace(x[0], x[-1], n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    sx = np.bincount(idx, weights=x, minlength=n_bins)
    sy = np.bincount(idx, weights=y, minlength=n_bins)
    keep = counts > 0
    return sx[keep] / counts[keep], sy[keep] / counts[keep]


def _make_fit(family, params, x, y) -> ModelFit:
    m = ModelFit(family, tuple(float(v) for v in params), 0.0, 0.0, FAMILY_COMPLEXITY[family])
    r = m.evaluate(x) - y
    m.rmse = float(np.sqrt(np.mean(r**2)))
    m.nrmse = m.rmse / _yscale(x, y)
    return m


def fit_linear(x, y) -> ModelFit:
    return _make_fit("linear", np.polyfit(x, y, 1), x, y)


def fit_quadratic(x, y) -> ModelFit:
    return _make_fit("quadratic", np.polyfit(x, y, 2), x, y)


def fit_cubic(x, y) -> ModelFit:
    return _make_fit("cubic", np.polyfit(x, y, 3), x, y)


def fit_abs(x, y) -> ModelFit:
    best = None
    for h in np.quantile(x, np.linspace(0.08, 0.92, 48)):
        a_mat = np.column_stack([np.abs(x - h), np.ones_like(x)])
        coef, *_ = np.linalg.lstsq(a_mat, y, rcond=None)
        rss = float(np.sum((a_mat @ coef - y) ** 2))
        if best is None or rss < best[0]:
            best = (rss, coef[0], h, coef[1])
    _, a, h, k = best
    res = least_squares(lambda p: p[0] * np.abs(x - p[1]) + p[2] - y, [a, h, k],
                        bounds=([-np.inf, x.min(), -np.inf], [np.inf, x.max(), np.inf]))
    return _make_fit("abs", res.x, x, y)


def _normalize_sine(p) -> tuple:
    a, b, c, d = p
    if b < 0:            # sin(-Bx + C) = sin(Bx - C + pi)
        b, c = -b, -c + math.pi
    if a < 0:            # -A sin(t) = A sin(t + pi)
        a, c = -a, c + math.pi
    return a, b, math.remainder(c, 2 * math.pi), d


def fit_sine(x, y) -> ModelFit | None:
    span = float(np.ptp(x))
    if span <= 0 or len(x) < 12:
        return None
    b_min = 2 * math.pi * MIN_SINE_PERIODS / span
    b_max = 2 * math.pi * min(MAX_SINE_PERIODS, len(x) / 8.0) / span
    if b_max <= b_min:
        return None
    n = 256
    xu = np.linspace(x.min(), x.max(), n)
    yu = np.interp(xu, x, y)
    yc = yu - yu.mean()
    spec = np.abs(np.fft.rfft(yc))
    freqs = np.fft.rfftfreq(n, d=xu[1] - xu[0])
    cands: set[float] = set()
    if len(spec) > 2:
        k = int(np.argmax(spec[1:]) + 1)
        for kk in (k - 1, k, k + 1):
            if 1 <= kk < len(freqs):
                cands.add(2 * math.pi * freqs[kk])
    # zero-crossing estimate, robust when fewer than ~2 periods are visible
    sign = np.sign(yc)
    sign = sign[sign != 0]
    crossings = int(np.sum(sign[1:] != sign[:-1])) if len(sign) > 1 else 0
    for z in (crossings, crossings + 1, max(crossings - 1, 1)):
        cands.add(math.pi * z / span)
    cands.update(2 * math.pi * k / span for k in (0.75, 1.0, 1.5, 2.0))

    def resid(p):
        return p[0] * np.sin(p[1] * x + p[2]) + p[3] - y

    def jac(p):
        arg = p[1] * x + p[2]
        c = np.cos(arg)
        return np.column_stack([np.sin(arg), p[0] * x * c, p[0] * c, np.ones_like(x)])

    # Cheap pre-screen: for a fixed frequency the model is linear in (a sin, b cos, D).
    screened = []
    for b0 in cands:
        b0 = float(np.clip(b0, b_min * 1.001, b_max * 0.999))
        design = np.column_stack([np.sin(b0 * x), np.cos(b0 * x), np.ones_like(x)])
        (a1, a2, d0), *_ = np.linalg.lstsq(design, y, rcond=None)
        cost = float(np.sum((design @ [a1, a2, d0] - y) ** 2))
        screened.append((cost, [math.hypot(a1, a2), b0, math.atan2(a2, a1), d0]))
    screened.sort(key=lambda s: s[0])

    best = None
    for _, p0 in screened[:3]:
        try:
            res = least_squares(resid, p0, jac=jac, bounds=([-np.inf, b_min, -np.inf, -np.inf],
                                                            [np.inf, b_max, np.inf, np.inf]), max_nfev=200)
        except ValueError:
            continue
        cost = float(np.sum(res.fun**2))
        if best is None or cost < best[0]:
            best = (cost, res.x)
    if best is None:
        return None
    return _make_fit("sine", _normalize_sine(best[1]), x, y)


def fit_function(x, y, n_bins: int = 200) -> FunctionFitResult:
    """Fit all families, choose the most plausible and keep a spline fallback."""
    xs, ys = prepare_samples(x, y, n_bins)
    result = FunctionFitResult(xs, ys)
    if len(xs) < 6 or np.ptp(xs) <= 0:
        return result
    fits = [fit_linear(xs, ys), fit_quadratic(xs, ys), fit_cubic(xs, ys), fit_abs(xs, ys)]
    s = fit_sine(xs, ys)
    if s is not None:
        fits.append(s)
    fits.sort(key=lambda m: m.nrmse)
    result.candidates = fits

    best_err = fits[0].nrmse
    margin = max(best_err * 1.15, best_err + 0.004)
    eligible = [m for m in fits if m.nrmse <= margin]
    chosen = min(eligible, key=lambda m: (m.n_params, m.nrmse))
    result.best = chosen
    result.recognized = chosen.nrmse <= RECOGNITION_THRESHOLD

    # Confidence: absolute fit quality x separation from competing (non-superset) families.
    quality = float(np.clip(1.0 - chosen.nrmse / (2 * RECOGNITION_THRESHOLD), 0.0, 1.0))
    rivals = [m for m in fits if m is not chosen and m.family not in SUPERSETS.get(chosen.family, set())
              and chosen.family not in SUPERSETS.get(m.family, set())]
    if rivals:
        ratio = min(m.nrmse for m in rivals) / max(chosen.nrmse, 1e-6)
        separation = float(np.clip(0.5 + 0.25 * (ratio - 1.0), 0.5, 1.0))
    else:
        separation = 1.0
    result.confidence = quality * separation if result.recognized else quality * 0.5

    # Spline fallback is always available (freehand curves, derivatives of unknown shapes)
    smooth = len(xs) * (0.01 * _yscale(xs, ys)) ** 2
    try:
        result.spline = UnivariateSpline(xs, ys, k=3, s=smooth)
    except Exception:  # pragma: no cover - degenerate data
        result.spline = UnivariateSpline(xs, ys, k=1, s=smooth)
    return result


def is_x_monotonic(path, tol: float = 0.94) -> bool:
    """True when a path progresses consistently in one x direction (i.e. is a graph y(x))."""
    p = np.asarray(path, float)
    if len(p) < 3:
        return False
    dx = np.diff(p[:, 0])
    total = np.abs(dx).sum()
    if total <= 0:
        return False
    forward = dx[dx > 0].sum()
    frac = max(forward, total - forward) / total
    return frac >= tol
