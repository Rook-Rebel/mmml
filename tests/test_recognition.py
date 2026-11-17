"""Shape detection on synthetic hand-drawn images (see mathvision3d/synthetic.py)."""

import math

import numpy as np
import pytest

from mathvision3d import synthetic as syn
from mathvision3d.core.math_object import MathObject
from mathvision3d.vision.models import ShapeKind


def only(result, kind):
    found = [o for o in result.objects if o.kind == kind]
    assert len(found) == 1, [o.kind for o in result.objects]
    return found[0]


def scale_of(result, img):
    # detection runs on an image resized to `working_size`
    return result.work_image.shape[1] / img.shape[1]


def match_vertices(found, truth):
    return max(min(np.linalg.norm(found - t, axis=1)) for t in truth)


@pytest.mark.parametrize("scene", ["triangle", "rotated_triangle"])
def test_triangle_vertices(pipeline, scene):
    img, gt = syn.SCENES[scene]()
    res = pipeline.process(img)
    tri = only(res, ShapeKind.TRIANGLE)
    s = scale_of(res, img)
    err = match_vertices(tri.geometry, gt["vertices"] * s)
    assert err < 6.0, f"vertex error {err:.1f}px"
    assert tri.confidence > 0.7
    assert len(res.objects) == 1


def test_circle(pipeline):
    img, gt = syn.scene_circle()
    res = pipeline.process(img)
    c = only(res, ShapeKind.CIRCLE)
    s = scale_of(res, img)
    assert np.linalg.norm(c.params["center"] - np.array(gt["center"]) * s) < 3
    assert c.params["semi_axes"][0] == pytest.approx(gt["radius"] * s, rel=0.02)


def test_ellipse(pipeline):
    img, gt = syn.scene_ellipse()
    res = pipeline.process(img)
    e = only(res, ShapeKind.ELLIPSE)
    s = scale_of(res, img)
    a, b = e.params["semi_axes"]
    assert (a, b) == pytest.approx((gt["axes"][0] * s, gt["axes"][1] * s), rel=0.03)
    assert math.degrees(e.params["angle"]) % 180 == pytest.approx(gt["angle"], abs=3)


def test_rectangle(pipeline):
    img, gt = syn.scene_rectangle()
    res = pipeline.process(img)
    r = only(res, ShapeKind.RECTANGLE)
    assert len(r.geometry) == 4
    assert match_vertices(r.geometry, gt["vertices"] * scale_of(res, img)) < 6


@pytest.mark.parametrize("noisy", [False, True])
def test_sine_wave(pipeline, noisy):
    img, gt = syn.scene_sine(seed=16 if noisy else 6, noisy=noisy)
    res = pipeline.process(img)
    f = only(res, ShapeKind.FUNCTION)
    assert f.fit.best.family == "sine"
    span = np.ptp(f.fit.x)
    periods = f.fit.best.params[1] * span / (2 * math.pi)
    assert periods == pytest.approx(gt["periods"], rel=0.06)
    assert f.fit.best.nrmse < (0.03 if noisy else 0.01)


def test_sine_with_axes_is_one_curve_with_origin(pipeline):
    img, gt = syn.scene_sine_with_axes()
    res = pipeline.process(img)
    assert res.axes is not None
    s = scale_of(res, img)
    assert np.linalg.norm(res.axes.origin - np.array(gt["origin"]) * s) < 4
    f = only(res, ShapeKind.FUNCTION)
    obj = MathObject(f, res.axes)
    assert obj.origin_meaningful
    # calibrate with the true x-span: the drawn function is y = 2 sin(x)
    true_span = np.ptp(f.geometry[:, 0]) / (gt["scale"] * s)
    obj.calibrate(true_span, "")
    a, b, c, d = obj.fit.best.params
    assert b == pytest.approx(1.0, rel=0.04)
    assert a == pytest.approx(2.0, rel=0.05)
    assert abs(d) < 0.1
    assert obj.fit.best.sine_form(obj.origin_meaningful) == "sin"


def test_parabola(pipeline):
    img, _ = syn.scene_parabola()
    res = pipeline.process(img)
    f = only(res, ShapeKind.FUNCTION)
    assert f.fit.best.family == "quadratic"
    obj = MathObject(f)
    assert obj.fit.best.params[0] > 0  # in local (y-up) coordinates the parabola opens upward


def test_multiple_objects_are_separated(pipeline):
    img, gt = syn.scene_multiple()
    res = pipeline.process(img)
    kinds = {o.kind.value for o in res.objects}
    assert kinds == gt["kinds"]
    assert sorted(o.id for o in res.objects) == list(range(1, len(res.objects) + 1))


@pytest.mark.parametrize("scene,kind", [("perspective_sine", ShapeKind.FUNCTION),
                                        ("perspective_triangle", ShapeKind.TRIANGLE)])
def test_perspective_correction(pipeline, scene, kind):
    img, gt = syn.SCENES[scene]()
    res = pipeline.process(img)
    assert res.paper_quad is not None
    obj = only(res, kind)
    if kind == ShapeKind.FUNCTION:
        periods = obj.fit.best.params[1] * np.ptp(obj.fit.x) / (2 * math.pi)
        assert periods == pytest.approx(gt["periods"], rel=0.08)
    else:
        # the drawn triangle is isosceles (base 510, height 440 -> base angles 59.9°);
        # rectification must restore the equal base angles despite the perspective
        angles = np.sort(obj.params["angles"])
        assert abs(angles.sum() - 180) < 1
        base = angles[:2] if angles[2] - angles[1] > angles[1] - angles[0] else angles[1:]
        assert abs(base[0] - base[1]) < 3.0
    # overlays map back into the camera frame, inside the paper
    cf = obj.contour_frame
    q = res.paper_quad
    assert cf[:, 0].min() > q[:, 0].min() and cf[:, 0].max() < q[:, 0].max()


def test_line_and_arc(pipeline):
    img, _ = syn.scene_line_and_arc()
    res = pipeline.process(img)
    only(res, ShapeKind.LINE)
    arc = only(res, ShapeKind.ARC)
    s = scale_of(res, img)
    assert arc.params["radius"] == pytest.approx(260 * s, rel=0.04)
    assert math.degrees(arc.params["sweep"]) == pytest.approx(140, abs=8)


@pytest.mark.parametrize("img", [
    np.full((480, 640, 3), 240, np.uint8),          # blank paper
    np.zeros((480, 640, 3), np.uint8),              # lens covered / dark
    np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8),  # pure noise
    np.zeros((0, 0, 3), np.uint8),                   # empty frame
])
def test_degenerate_images_do_not_crash(pipeline, img):
    res = pipeline.process(img)
    assert isinstance(res.objects, list)
    if img.size and img.std() < 1:
        assert res.objects == []


def test_roi_restricts_detection(pipeline):
    img, _ = syn.scene_multiple()
    res = pipeline.process(img, roi=(40, 60, 340, 320))     # only the triangle
    assert [o.kind for o in res.objects] == [ShapeKind.TRIANGLE]
    assert res.objects[0].contour_frame[:, 0].min() > 40


def test_math_object_calibrated_measurements(pipeline):
    img, gt = syn.scene_triangle()
    res = pipeline.process(img)
    obj = MathObject(res.objects[0])
    assert not obj.coords.calibrated
    ref_px, key = obj.reference_px()
    assert key == "ref_side_ab"
    obj.calibrate(5.0, "cm")
    sides = np.linalg.norm(np.roll(obj.points, -1, axis=0) - obj.points, axis=1)
    assert sides[0] == pytest.approx(5.0)
    labels = {m.label: m.value for m in obj.measurements()}
    assert any("cm" in v for v in labels.values())
    obj.clear_calibration()
    assert not obj.coords.calibrated
