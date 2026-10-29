import numpy as np
import pytest

from napari._vispy.overlays.coordinate_grid import VispyCoordinateGridOverlay
from napari._vispy.utils.qt_font import FontInfo
from napari.components import ViewerModel
from napari.components.overlays import CoordinateGridOverlay
from napari.utils._units import compact_unit, compute_nice_ticks


@pytest.mark.parametrize('y_scale', [1, 0.3, 0.05])
def test_scene_axes_dimensions_properly_detected(y_scale):
    viewer = ViewerModel()
    coordinate_grid_model = CoordinateGridOverlay()
    coordinate_grid_vispy = VispyCoordinateGridOverlay(
        viewer=viewer, overlay=coordinate_grid_model, font_info=FontInfo()
    )
    viewer.add_image(
        np.zeros((10, 10, 10)),
        scale=(1234, y_scale, 0.002),
        translate=(-30, 2, 1),
    )

    viewer.dims.ndisplay = 2
    assert not coordinate_grid_vispy.node.axis_labels[-1].visible

    viewer.dims.ndisplay = 3
    assert coordinate_grid_vispy.node.axis_labels[-1].visible

    for axis in range(3):
        ticks = coordinate_grid_vispy.node.tick_labels[axis]
        range_ = viewer.dims.range[2 - axis]  # zyx -> xyz
        # labels are in prefixed pixels, e.g. mpx for the 0.002 scale
        _, factor = compact_unit(
            viewer.layers.units[2 - axis], range_.stop - range_.start
        )
        np.testing.assert_array_equal(
            [float(t) for t in ticks.text],
            compute_nice_ticks(
                range_.start * factor,
                range_.stop * factor,
                coordinate_grid_model.n_ticks,
            )[0],
        )


def test_ticks_use_compact_units():
    viewer = ViewerModel()
    coordinate_grid_model = CoordinateGridOverlay()
    coordinate_grid_vispy = VispyCoordinateGridOverlay(
        viewer=viewer, overlay=coordinate_grid_model, font_info=FontInfo()
    )
    # y spans 9 nm and x spans 9 mm, both in m
    viewer.add_image(np.zeros((10, 10)), scale=(1e-9, 1e-3), units='m')
    # units are pushed by the canvas in the app
    coordinate_grid_vispy._on_unit_change()

    # xyz order
    labels = [lab.text for lab in coordinate_grid_vispy.node.axis_labels]
    assert labels[0].endswith('(mm)')
    assert labels[1].endswith('(nm)')

    for axis, scale in enumerate([1e3, 1e9]):
        ticks = coordinate_grid_vispy.node.tick_labels[axis]
        np.testing.assert_array_equal(
            [float(t) for t in ticks.text], [0, 2, 4, 6, 8]
        )
        # ticks are still placed in world coordinates
        np.testing.assert_allclose(
            ticks.pos[:, 0], np.array([0, 2, 4, 6, 8]) / scale
        )
