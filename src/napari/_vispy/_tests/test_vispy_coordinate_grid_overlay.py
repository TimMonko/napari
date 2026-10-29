import numpy as np
import pytest

from napari._vispy.overlays.coordinate_grid import VispyCoordinateGridOverlay
from napari._vispy.utils.qt_font import FontInfo
from napari.components import ViewerModel
from napari.components.overlays import CoordinateGridOverlay
from napari.utils._units import compact_unit, compute_nice_ticks


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


@pytest.mark.usefixtures('qapp')
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
            [float(t) for t in ticks.text], [0, 2.5, 5, 7.5]
        )
        # ticks are still placed in world coordinates
        np.testing.assert_allclose(
            ticks.pos[:, 0], np.array([0, 2.5, 5, 7.5]) / scale
        )


@pytest.mark.usefixtures('qapp')
def test_2d_ticks_follow_zoom_and_pan():
    viewer = ViewerModel()
    coordinate_grid_vispy = VispyCoordinateGridOverlay(
        viewer=viewer, overlay=CoordinateGridOverlay(), font_info=FontInfo()
    )
    viewer.add_image(np.zeros((1000, 1000)), units='m')
    node = coordinate_grid_vispy.node

    # zoom so that about 0.5 m of the 1000 m image fills the canvas width
    viewer.scene.camera.center = (0, 500, 500)
    viewer.scene.camera.zoom = viewer.canvas.size[1] / 0.5

    # ticks only cover the visible part, in a unit that fits it
    assert node.axis_labels[0].text.endswith('(mm)')
    values = np.array([float(t) for t in node.tick_labels[0].text]) / 1000
    assert values.min() > 499.5
    assert values.max() < 500.5
    # but grid lines still cover the whole data
    x0, x1, _, _ = node.grids[0].grid_bounds
    assert x1 - x0 == 999

    # pan the image out of view: no ticks are left for x
    viewer.scene.camera.center = (0, 500, 5000)
    assert not node.tick_labels[0].visible


@pytest.mark.usefixtures('qapp')
def test_3d_ticks_cover_whole_data():
    viewer = ViewerModel()
    coordinate_grid_vispy = VispyCoordinateGridOverlay(
        viewer=viewer, overlay=CoordinateGridOverlay(), font_info=FontInfo()
    )
    viewer.add_image(np.zeros((10, 1000, 1000)), units='m')
    viewer.dims.ndisplay = 3
    viewer.scene.camera.zoom = 1000

    ticks = coordinate_grid_vispy.node.tick_labels[0]
    values = [float(t) for t in ticks.text]
    assert values[0] < 250
    assert values[-1] > 750


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
    # labels are next to the start of the data, or pinned inside the
    # canvas when that's off screen (xyz order)
    edges = [r.start for r in coordinate_grid_vispy._get_label_ranges()]

    # x ticks are stacked above the data, y ticks side by side left of it
    for axis, size_index in [(0, 1), (1, 0)]:
        offsets = []
        for text in (node.tick_labels[axis], node.axis_labels[axis]):
            grid = text.parent
            pos = np.atleast_2d(text.pos)[0]
            world = grid.transform.map(text.transform.map(pos))
            # distance from the data edge to the text center, in pixels
            other = 1 - axis
            offsets.append((edges[other] - world[other]) * zoom)
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


@pytest.mark.usefixtures('qapp')
def test_2d_ticks_keep_their_step_when_panning():
    viewer = ViewerModel()
    coordinate_grid_vispy = VispyCoordinateGridOverlay(
        viewer=viewer, overlay=CoordinateGridOverlay(), font_info=FontInfo()
    )
    viewer.add_image(np.zeros((1000, 1000)))
    node = coordinate_grid_vispy.node
    viewer.scene.camera.center = (0, 500, 500)
    viewer.scene.camera.zoom = 10

    def x_ticks():
        return np.array([float(t) for t in node.tick_labels[0].text])

    ticks = x_ticks()
    step = np.diff(ticks)[0]
    label = node.axis_labels[0].text
    for shift in (0.3, 3.7, 17.2):
        viewer.scene.camera.center = (0, 500, 500 + shift)
        np.testing.assert_allclose(np.diff(x_ticks()), step)
        assert node.axis_labels[0].text == label

    # a small pan moves labels with their transform, without new text
    shown = node._ticks[0][0]
    viewer.scene.camera.center = (0, 500, 500 + shift + 0.01)
    assert node._ticks[0][0] is shown
