import numpy as np

from dimratio.camera import VIEW_ORDER, fixed_projection_scales
from dimratio.daaw import transform_by_axis_scales
from dimratio.demo_mesh import make_demo_chair, normalize_longest_extent


def test_daaw_is_extent_exact_and_preserves_other_axes():
    mesh = normalize_longest_extent(make_demo_chair())
    source_extent = np.asarray(mesh.extents)
    warped, diagnostics = transform_by_axis_scales(
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces),
        {"width": 1.2, "depth": 1.0, "height": 1.0},
        normals=np.asarray(mesh.vertex_normals),
    )
    target_extent = np.ptp(warped, axis=0)
    assert np.allclose(target_extent[0], source_extent[0] * 1.2, atol=1e-8)
    assert np.allclose(target_extent[1:], source_extent[1:], atol=1e-8)
    assert diagnostics["monotone"]


def test_global_camera_is_shared_and_covers_twenty_percent_variants():
    conditions = [
        {"width": w, "depth": d, "height": h}
        for w, d, h in [
            (1.0, 1.0, 1.0),
            (0.8, 1.0, 1.0), (1.2, 1.0, 1.0),
            (1.0, 0.8, 1.0), (1.0, 1.2, 1.0),
            (1.0, 1.0, 0.8), (1.0, 1.0, 1.2),
        ]
    ]
    scales = fixed_projection_scales(
        np.asarray([0.9, 0.94, 1.0]), conditions, margin=1.16, global_across_views=True
    )
    assert tuple(scales) == VIEW_ORDER
    assert len(set(scales.values())) == 1
    assert np.isclose(next(iter(scales.values())), 1.392)

