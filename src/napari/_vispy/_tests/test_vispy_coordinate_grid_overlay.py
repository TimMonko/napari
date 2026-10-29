import numpy as np
import pytest

from napari._vispy.overlays.coordinate_grid import VispyCoordinateGridOverlay
from napari._vispy.utils.qt_font import FontInfo
from napari.components import ViewerModel
from napari.components.overlays import CoordinateGridOverlay
from napari.utils._units import compute_nice_ticks


@pytest.mark.usefixtures('qapp')
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
        np.testing.assert_array_equal(
            [float(t) for t in ticks.text],
            compute_nice_ticks(
                range_.start, range_.stop, coordinate_grid_model.n_ticks
            )[0],
        )


@pytest.mark.usefixtures('qapp')
def test_2d_axis_labels_clear_tick_labels():
    viewer = ViewerModel()
    coordinate_grid_vispy = VispyCoordinateGridOverlay(
        viewer=viewer, overlay=CoordinateGridOverlay(), font_info=FontInfo()
    )
    # long tick labels (e.g. 2650000) on the vertical axis
    viewer.add_image(
        np.zeros((10, 10)), scale=(30, 30), translate=(2_650_000, 101_000)
    )
    viewer.dims.axis_labels = ('northing', 'easting')
    node = coordinate_grid_vispy.node
    zoom = viewer.scene.camera.zoom
    # vispy is xyz, napari is yx
    edges = [viewer.dims.range[0].start, viewer.dims.range[1].start]

    # x ticks are stacked above the data, y ticks side by side left of it
    for axis, size_index in [(0, 1), (1, 0)]:
        offsets = []
        for text in (node.tick_labels[axis], node.axis_labels[axis]):
            grid = text.parent
            pos = np.atleast_2d(text.pos)[0]
            world = grid.transform.map(text.transform.map(pos))
            # distance from the data edge to the text center, in pixels
            other = 1 - axis
            offsets.append((edges[other ^ 1] - world[other]) * zoom)
        tick_offset, label_offset = offsets
        ticks = node.tick_labels[axis]
        tick_size = (
            ticks.get_width_height()[0]
            if size_index == 0
            else ticks.get_line_height()
        )
        label_size = node.axis_labels[axis].get_width_height()[size_index]
        # text is centered, so half of it is on each side of its position
        assert tick_offset - tick_size / 2 > 0
        assert label_offset - label_size / 2 > tick_offset + tick_size / 2
