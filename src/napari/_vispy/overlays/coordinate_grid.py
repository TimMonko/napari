from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from napari._vispy.overlays.base import ViewerOverlayMixin, VispySceneOverlay
from napari._vispy.visuals.coordinate_grid import CoordinateGrid
from napari.components.dims import RangeTuple
from napari.settings import get_settings
from napari.utils._units import compact_unit

if TYPE_CHECKING:
    from napari._vispy.utils.qt_font import FontInfo
    from napari.components.overlays import CoordinateGridOverlay


class VispyCoordinateGridOverlay(ViewerOverlayMixin, VispySceneOverlay):
    overlay: CoordinateGridOverlay
    node: CoordinateGrid

    def __init__(
        self,
        *,
        font_info: FontInfo,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node=CoordinateGrid(
                font_info=font_info,
            ),
            font_info=font_info,
            **kwargs,
        )

        self.overlay.events.color.connect(self._rebuild_all)
        self.overlay.events.axis_labels.connect(self._on_axis_labels_change)
        self.overlay.events.tick_labels.connect(self._on_ticks_change)
        self.overlay.events.font_size.connect(self._on_font_size_change)
        self.overlay.events.n_ticks.connect(self._on_ticks_change)
        self.viewer.dims.events.order.connect(self._rebuild_all)
        self.viewer.dims.events.range.connect(self._on_extent_change)
        self.viewer.dims.events.ndisplay.connect(self._rebuild_all)
        self.viewer.dims.events.axis_labels.connect(
            self._on_axis_labels_change
        )

        # would be nice to fire this less often to save performance
        self.viewer.scene.camera.events.angles.connect(
            self._on_view_direction_change
        )
        self.viewer.scene.camera.events.orientation.connect(
            self._on_camera_change
        )
        # in 2D, ticks follow the visible part of the data
        self.viewer.scene.camera.events.zoom.connect(self._on_camera_change)
        self.viewer.scene.camera.events.center.connect(self._on_camera_change)
        self.viewer.canvas.events.size.connect(self._on_camera_change)
        self._label_ranges: list[RangeTuple] = []
        get_settings().appearance.events.theme.connect(self._rebuild_all)
        self.viewer.events.theme.connect(self._rebuild_all)

        self.reset()

    def _get_data_ranges(self) -> list[RangeTuple]:
        # napari dims are zyx, but vispy uses xyz
        displayed = self.viewer.dims.displayed[::-1]
        return [self.viewer.dims.range[i] for i in displayed]

    def _get_label_ranges(self) -> list[RangeTuple]:
        """Get the ranges that ticks and labels are placed in.

        In 2D, this is the part of the data that is visible on the canvas,
        so that ticks adapt to zoom and labels stay on screen. In 3D, the
        visible area depends on the view angle, so it's the whole data.
        """
        ranges = self._get_data_ranges()
        if self.viewer.dims.ndisplay != 2:
            return ranges

        displayed = self.viewer.dims.displayed[::-1]
        corners = np.array(
            [
                self.viewer.canvas_to_world((0, 0)),
                self.viewer.canvas_to_world(
                    self.viewer.canvas.viewbox_size(self.viewer.layers)
                ),
            ]
        )
        # labels sit outside the start of the other axis, so keep that far
        # from the canvas edge to stay visible (axis labels are offset the
        # most, see CoordinateGrid.set_view_direction)
        margin = self.node.font_size * 7 / self.viewer.scene.camera.zoom
        clipped = []
        for dim, range_ in zip(displayed, ranges, strict=True):
            low = min(corners[:, dim])
            high = max(corners[:, dim])
            start = max(range_.start, low + margin)
            stop = min(range_.stop, high)
            # nothing visible: an empty range gives no ticks
            clipped.append(RangeTuple(start, max(start, stop), range_.step))
        return clipped

    def _get_ranges_and_axis_labels(
        self,
    ) -> tuple[list[RangeTuple], list[str], list[float]]:
        displayed = self.viewer.dims.displayed[::-1]
        ranges = self._get_label_ranges()
        axis_labels = [self.viewer.dims.axis_labels[i] for i in displayed]

        units = ['' for _ in displayed]
        scales = [1.0 for _ in displayed]
        if self.viewer.layers.units is not None:
            for axis, (dim, range_) in enumerate(
                zip(displayed, ranges, strict=True)
            ):
                # prefix the unit like the scale bar does (e.g. m -> nm)
                unit, scales[axis] = compact_unit(
                    self.viewer.layers.units[dim], range_.stop - range_.start
                )
                # see https://pint.readthedocs.io/en/stable/user/formatting.html
                units[axis] = f' ({unit:~#P})'

        axis_labels_with_units = [
            f'{lab}{unit}'
            for lab, unit in zip(axis_labels, units, strict=True)
        ]
        return ranges, axis_labels_with_units, scales

    def _on_unit_change(self) -> None:
        # NOTE: this is also called by VispyCanvas when layer units are updated
        #       so it doesn't need to be connected to events for that
        self._on_axis_labels_change()
        self._on_ticks_change()

    def _on_font_size_change(self) -> None:
        # called by vispy canvas when global fonts are updated
        font_size = (
            self.overlay.font_size
            if self.overlay.font_size is not None
            else self._default_font_size
        )

        self.node.font_size = font_size
        self._on_axis_labels_change()
        self._on_ticks_change()

    def _on_axis_labels_change(self) -> None:
        ranges, axis_labels, _ = self._get_ranges_and_axis_labels()
        self.node.set_axis_labels(
            self.overlay.axis_labels, ranges, axis_labels
        )
        # needed to ensure new grids/ticks are up to date
        self._on_blending_change()
        self._on_view_direction_change()

    def _on_ticks_change(self) -> None:
        ranges, _, scales = self._get_ranges_and_axis_labels()
        self._label_ranges = ranges
        self.node.set_ticks(
            self.overlay.tick_labels, self.overlay.n_ticks, ranges, scales
        )
        # needed to ensure new grids/ticks are up to date
        self._on_blending_change()
        self._on_view_direction_change()

    def _on_extent_change(self) -> None:
        self.node.set_extents(self._get_data_ranges())
        self._on_ticks_change()
        self._on_axis_labels_change()

    def _rebuild_all(self) -> None:
        self.node.color = (
            self.overlay.color
            if self.overlay.color is not None
            else self._get_fgcolor()
        )
        self._on_extent_change()

        self._on_view_direction_change()

    def _on_camera_change(self) -> None:
        if self.viewer.dims.ndisplay == 2:
            ranges, axis_labels, scales = self._get_ranges_and_axis_labels()
            if ranges != self._label_ranges:
                self._label_ranges = ranges
                self.node.set_ticks(
                    self.overlay.tick_labels,
                    self.overlay.n_ticks,
                    ranges,
                    scales,
                )
                self.node.set_axis_labels(
                    self.overlay.axis_labels, ranges, axis_labels
                )
        self._on_view_direction_change()

    def _on_view_direction_change(self) -> None:
        ranges = tuple(self._get_label_ranges())

        if len(ranges) == 3:
            view_direction = np.sign(self.viewer.scene.camera.view_direction)
            view_is_flipped = tuple(view_direction >= 0)[::-1]
        elif len(ranges) == 2:
            view_is_flipped = (False, False, False)
            ranges = ranges + (RangeTuple(0, 0, 1),)
        else:
            raise RuntimeError('unreachable')

        self.node.set_view_direction(
            ranges,
            view_is_flipped,
            zoom=self.viewer.scene.camera.zoom,
            ndisplay=self.viewer.dims.ndisplay,
        )

    def reset(self) -> None:
        super().reset()
        self._rebuild_all()
