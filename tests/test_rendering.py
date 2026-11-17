import numpy as np
import pytest
import pyvista as pv

from mathvision3d import synthetic as syn
from mathvision3d.core.math_object import DisplayMode, MathObject
from mathvision3d.rendering.object_renderer import ObjectRenderer
from mathvision3d.rendering.scene_manager import SceneManager


def all_objects(pipeline):
    objs = []
    for name in ("multiple_objects", "sine_with_axes", "ellipse", "line_and_arc", "parabola"):
        img, _ = syn.SCENES[name]()
        res = pipeline.process(img)
        objs += [MathObject(o, res.axes) for o in res.objects]
    return objs


def test_every_mode_builds_meshes(pipeline):
    renderer = ObjectRenderer()
    kinds = set()
    for obj in all_objects(pipeline):
        kinds.add(obj.kind.value)
        for mode in obj.available_modes():
            obj.options.mode = mode
            for surface in ("translational", "product", "radial", "revolution"):
                obj.options.surface_type = surface
                obj.options.show_derivative = obj.options.show_tangent = obj.options.show_samples = True
                items, labels = renderer.build(obj)
                assert any(it.main for it in items), (obj.kind, mode)
                for it in items:
                    assert it.mesh.n_points > 0
                    assert np.isfinite(np.asarray(it.mesh.points)).all()
                if mode != DisplayMode.SURFACE:
                    break
    assert {"triangle", "circle", "square", "function", "ellipse", "line", "arc"} <= kinds


def test_extrusion_is_a_closed_prism(pipeline):
    img, _ = syn.scene_triangle()
    obj = MathObject(pipeline.process(img).objects[0])
    obj.options.mode = DisplayMode.EXTRUDE
    obj.options.extrude_depth = 2.0
    items, _ = ObjectRenderer().build(obj)
    solid = next(it.mesh for it in items if it.main)
    zmin, zmax = solid.bounds[4], solid.bounds[5]
    assert (zmin, zmax) == pytest.approx((-1.0, 1.0))
    assert solid.extract_surface(algorithm="dataset_surface").triangulate().n_open_edges == 0


def test_scene_transform_pick_and_export(pipeline, tmp_path):
    pl = pv.Plotter(off_screen=True, window_size=(400, 300))
    scene = SceneManager(pl)
    img, _ = syn.scene_triangle()
    obj = MathObject(pipeline.process(img).objects[0])
    scene.add_object(obj)
    scene.set_selected(obj.uid)
    scene.reset_camera("front")
    pl.render()
    w, h = pl.window_size
    # the triangle's centroid sits at the local origin, i.e. the screen centre in the front view
    assert scene.pick(w // 2, h // 2) == obj.uid
    obj.transform.translate((50, 0, 0))
    scene.update_transform(obj.uid)
    assert scene.pick(w // 2, h // 2) is None
    for ext in ("vtk", "stl"):
        p = tmp_path / f"tri.{ext}"
        obj.options.mode = DisplayMode.EXTRUDE
        scene.export_object(obj.uid, str(p))
        mesh = pv.read(p)
        assert mesh.n_points > 0 and mesh.bounds[0] > 40       # exported with its transform
    scene.remove_object(obj.uid)
    assert not scene.objects
    pl.close()
