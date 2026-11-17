import math

import numpy as np
import pytest

from mathvision3d.mathematics.function_fitting import fit_function, is_x_monotonic, prepare_samples

RNG = np.random.default_rng(42)


def noisy(y, sigma):
    return y + RNG.normal(0, sigma, len(y))


def test_sine_parameters_recovered():
    x = np.linspace(-6, 6, 400)
    y = 1.7 * np.sin(1.3 * x + 0.4) + 0.5
    res = fit_function(x, y)
    assert res.recognized and res.best.family == "sine"
    a, b, c, d = res.best.params
    assert (a, b, d) == pytest.approx((1.7, 1.3, 0.5), rel=0.01)
    assert math.remainder(c - 0.4, 2 * math.pi) == pytest.approx(0, abs=0.02)
    assert res.best.nrmse < 0.005
    assert res.confidence > 0.8


def test_noisy_sine_acceptable_error():
    x = np.linspace(0, 4 * np.pi, 500)
    y = noisy(2.0 * np.sin(x), 0.08)
    res = fit_function(x, y)
    assert res.best.family == "sine"
    assert res.best.params[1] == pytest.approx(1.0, rel=0.02)
    assert res.best.params[0] == pytest.approx(2.0, rel=0.03)
    assert res.best.nrmse < 0.03


def test_sine_and_cosine_naming_relative_to_origin():
    x = np.linspace(-7, 7, 300)
    sin_fit = fit_function(x, np.sin(x)).best
    cos_fit = fit_function(x, np.cos(x)).best
    assert sin_fit.sine_form(True) == "sin"
    assert cos_fit.sine_form(True) == "cos"
    assert "cos" in cos_fit.equation(True)
    assert cos_fit.sine_form(False) == "sinusoid"      # no meaningful origin -> generic form


@pytest.mark.parametrize("family,f", [
    ("quadratic", lambda x: 0.7 * x**2 - 1),
    ("cubic", lambda x: 0.3 * x**3 - x),
    ("abs", lambda x: 1.5 * np.abs(x - 0.3) + 0.2),
    ("linear", lambda x: -0.8 * x + 2),
])
def test_family_selection(family, f):
    x = np.linspace(-3, 3, 300)
    res = fit_function(x, noisy(f(x), 0.02))
    assert res.recognized
    assert res.best.family == family


def test_quadratic_is_not_mistaken_for_a_degenerate_sine():
    x = np.linspace(-1, 1, 200)
    assert fit_function(x, x**2).best.family == "quadratic"


def test_random_curve_is_not_given_an_invented_equation():
    x = np.linspace(0, 10, 400)
    y = np.cumsum(RNG.normal(0, 1, 400))   # random walk
    y = np.convolve(y, np.ones(9) / 9, mode="same")
    res = fit_function(x, y)
    assert not res.recognized
    assert res.spline is not None
    assert np.isfinite(res.function()(np.array([5.0]))).all()


def test_derivatives_match_numerical():
    x = np.linspace(-3, 3, 300)
    for f in (np.sin, lambda t: t**2, lambda t: 0.2 * t**3):
        res = fit_function(x, f(x))
        xs = np.array([-1.0, 0.5, 2.0])
        h = 1e-5
        num = (res.function()(xs + h) - res.function()(xs - h)) / (2 * h)
        assert res.derivative()(xs) == pytest.approx(num, rel=1e-3, abs=1e-6)


def test_prepare_samples_merges_duplicates():
    x = np.repeat(np.linspace(0, 1, 50), 4)
    y = np.repeat(np.linspace(0, 2, 50), 4) + np.tile([-0.1, 0.1, -0.1, 0.1], 50)
    xs, ys = prepare_samples(x, y)
    assert len(xs) <= 50
    assert np.allclose(ys, 2 * xs, atol=1e-6)


def test_x_monotonic():
    t = np.linspace(0, 1, 50)
    assert is_x_monotonic(np.column_stack([t, np.sin(5 * t)]))
    th = np.linspace(0, 1.8 * np.pi, 80)
    assert not is_x_monotonic(np.column_stack([np.cos(th), np.sin(th)]))
