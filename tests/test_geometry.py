import math

import numpy as np
import pytest

from mathvision3d.mathematics import geometry as geo
from mathvision3d.mathematics.expressions import ExpressionError, compile_expression
from mathvision3d.mathematics.transformations import Transform3D


def test_triangle_metrics():
    tri = np.array([[0, 0], [4, 0], [0, 3]], float)
    assert geo.polygon_area(tri) == pytest.approx(6.0)
    assert geo.polygon_perimeter(tri) == pytest.approx(12.0)
    angles = geo.interior_angles(tri)
    assert angles.sum() == pytest.approx(180.0)
    assert sorted(angles)[-1] == pytest.approx(90.0)
    assert geo.classify_triangle(geo.side_lengths(tri), angles) == ("scalene", "right")


def test_interior_angles_independent_of_orientation_and_concave():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    assert np.allclose(geo.interior_angles(square), 90)
    assert np.allclose(geo.interior_angles(square[::-1]), 90)
    arrow = np.array([[0, 0], [2, 1], [0, 2], [1, 1]], float)   # concave at (1, 1)
    a = geo.interior_angles(arrow)
    assert a.sum() == pytest.approx(360.0)
    assert a[3] > 180


def test_equilateral_classification():
    tri = np.array([[0, 0], [1, 0], [0.5, math.sqrt(3) / 2]])
    assert geo.classify_triangle(geo.side_lengths(tri), geo.interior_angles(tri)) == ("equilateral", "acute")


def test_circle_fit_recovers_parameters():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 2 * np.pi, 200)
    pts = np.column_stack([3 + 5 * np.cos(t), -2 + 5 * np.sin(t)]) + rng.normal(0, 0.02, (200, 2))
    cx, cy, r, rms = geo.fit_circle(pts)
    assert (cx, cy, r) == pytest.approx((3, -2, 5), abs=0.02)
    assert rms < 0.05


def test_ellipse_residual_and_metrics():
    pts = geo.ellipse_points((0, 0), (4, 2), 0.3, 200)
    assert geo.ellipse_residual(pts, (0, 0), (4, 2), 0.3) < 1e-6
    m = geo.ellipse_metrics(4, 2)
    assert m["area"] == pytest.approx(8 * math.pi)
    assert m["perimeter"] == pytest.approx(19.3768, rel=1e-4)
    assert geo.ellipse_metrics(3, 3)["eccentricity"] == pytest.approx(0)


def test_refine_polygon_vertices_from_noisy_contour():
    rng = np.random.default_rng(1)
    v = np.array([[0, 0], [100, 0], [50, 80]], float)
    contour = np.vstack([np.linspace(v[i], v[(i + 1) % 3], 80, endpoint=False) for i in range(3)])
    contour += rng.normal(0, 0.8, contour.shape)
    approx = v + rng.normal(0, 4, v.shape)          # poor initial corners
    refined = geo.refine_polygon_vertices(contour, approx)
    assert np.abs(refined - v).max() < 2.5


def test_transform_roundtrip_and_incremental_rotation():
    t = Transform3D()
    t.set_euler_deg([10, 20, 30])
    assert t.euler_deg() == pytest.approx([10, 20, 30])
    t.reset_orientation()
    t.rotate_world((0, 0, 1), 90)
    assert t.apply([[1, 0, 0]])[0] == pytest.approx([0, 1, 0], abs=1e-12)
    t.translate((1, 2, 3))
    t.scale_by(2)
    assert t.apply([[1, 0]])[0] == pytest.approx([1, 4, 3])
    m = t.matrix()
    assert m[:3, 3] == pytest.approx([1, 2, 3])
    t.scale_by(1e6)
    assert t.scale == Transform3D.MAX_SCALE


def test_expression_sandbox():
    f = compile_expression("0.5*sin(x) + y^2")
    x = np.array([0.0, np.pi / 2])
    assert f(x=x, y=np.array([1.0, 2.0])) == pytest.approx([1.0, 4.5])
    assert compile_expression("2")(x=x) == pytest.approx([2, 2])
    for bad in ("__import__('os')", "x.__class__", "open('f')", "lambda: 1", "[1,2]", "q + 1"):
        with pytest.raises(ExpressionError):
            compile_expression(bad)
