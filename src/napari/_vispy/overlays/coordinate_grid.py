from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from napari._vispy.overlays.base import ViewerOverlayMixin, VispySceneOverlay
from napari._vispy.visuals.coordinate_grid import CoordinateGrid
from napari.components.dims import RangeTuple
from napari.settings import get_settings
from napari.utils._units import (
    compact_unit,
    compute_nice_step,
    compute_nice_ticks,
    compute_ticks_with_step,
)

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
        # 2D tick step of each axis and the scale it's in, kept while
        # zooming until it's far from ideal (see compute_nice_step)
        self._tick_steps: list[tuple[float, float] | None] = [None] * 3
        get_settings().appearance.events.theme.connect(self._rebuild_all)
        self.viewer.events.theme.connect(self._rebuild_all)

        self.reset()

    def _get_data_ranges(self) -> list[RangeTuple]:
        # napari dims are zyx, but vispy uses xyz
        displayed = self.viewer.dims.displayed[::-1]
        return [self.viewer.dims.range[i] for i in displayed]

    def _get_2d_view(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the part of the canvas that ticks can be in, in 2D.

        Returns the low and high world bounds of the canvas for each axis
        (in xy order). Labels sit outside the start of the other axis, so
        the low bounds leave room for them to stay on the canvas.
        """
        displayed = list(self.viewer.dims.displayed[::-1])
        corners = np.array(
            [
                self.viewer.canvas_to_world((0, 0)),
                self.viewer.canvas_to_world(
                    self.viewer.canvas.viewbox_size(self.viewer.layers)
                ),
            ]
        )[:, displayed]
        extents = self.node.get_2d_label_extents()
        # the x axis' labels are above the start of y, and y's are left of
        # the start of x
        margins = np.array([extents[1], extents[0]])
        low = corners.min(axis=0) + margins / self.viewer.scene.camera.zoom
        high = corners.max(axis=0)
        return low, high

    def _get_label_ranges(self) -> list[RangeTuple]:
        """Get the ranges that ticks and labels are placed in.

        In 2D, this is the part of the data that is visible on the canvas,
        so that ticks adapt to zoom and labels stay on screen. In 3D, the
        visible area depends on the view angle, so it's the whole data.
        """
        ranges = self._get_data_ranges()
        if self.viewer.dims.ndisplay != 2:
            return ranges

        low, high = self._get_2d_view()
        clipped = []
        for axis, range_ in enumerate(ranges):
            start = max(range_.start, low[axis])
            stop = min(range_.stop, high[axis])
            # nothing visible: an empty range gives no ticks
            clipped.append(RangeTuple(start, max(start, stop), range_.step))
        return clipped

    def _get_tick_spans(self) -> list[float]:
        """Get the length that ticks of each axis are spread over.

        In 2D, when the data is larger than the canvas, that's the canvas
        size rather than the visible part of the data: it doesn't change
        when panning, so neither do the tick step and unit.
        """
        spans = [r.stop - r.start for r in self._get_data_ranges()]
        if self.viewer.dims.ndisplay != 2:
            return spans
        low, high = self._get_2d_view()
        return [
            min(span, view_span)
            for span, view_span in zip(spans, high - low, strict=True)
        ]

    def _get_ranges_and_axis_labels(
        self,
    ) -> tuple[list[RangeTuple], list[str], list[float], list[float]]:
        displayed = self.viewer.dims.displayed[::-1]
        ranges = self._get_label_ranges()
        spans = self._get_tick_spans()
        axis_labels = [self.viewer.dims.axis_labels[i] for i in displayed]

        units = ['' for _ in displayed]
        scales = [1.0 for _ in displayed]
        if self.viewer.layers.units is not None:
            for axis, (dim, span) in enumerate(
                zip(displayed, spans, strict=True)
            ):
                # prefix the unit like the scale bar does (e.g. m -> nm)
                unit, scales[axis] = compact_unit(
                    self.viewer.layers.units[dim], span
                )
                # see https://pint.readthedocs.io/en/stable/user/formatting.html
                units[axis] = f' ({unit:~#P})'

        axis_labels_with_units = [
            f'{lab}{unit}'
            for lab, unit in zip(axis_labels, units, strict=True)
        ]
        return ranges, axis_labels_with_units, scales, spans

    def _get_ticks(
        self,
        ranges: list[RangeTuple],
        scales: list[float],
        spans: list[float],
    ) -> tuple[list[np.ndarray], list[int]]:
        """Get tick values in the label units, and their decimals."""
        n_ticks = self.overlay.n_ticks
        values = []
        decimals = []
        for axis, (range_, scale, span) in enumerate(
            zip(ranges, scales, spans, strict=True)
        ):
            if self.viewer.dims.ndisplay != 2:
                ticks = compute_nice_ticks(
                    range_.start * scale, range_.stop * scale, n_ticks
                )
            else:
                # ticks are multiples of a step that only depends on zoom,
                # so panning only adds or removes ticks at the edges
                kept = self._tick_steps[axis]
                previous = kept[1] if kept and kept[0] == scale else None
                step = compute_nice_step(span * scale, n_ticks, previous)
                self._tick_steps[axis] = (scale, step)
                ticks = compute_ticks_with_step(
                    range_.start * scale, range_.stop * scale, step
                )
            values.append(ticks[0])
            decimals.append(ticks[1])
        return values, decimals

    def _update_ticks_and_labels(self) -> None:
        ranges, axis_labels, scales, spans = self._get_ranges_and_axis_labels()
        values, decimals = self._get_ticks(ranges, scales, spans)
        self.node.set_ticks(self.overlay.tick_labels, values, decimals, scales)
        self.node.set_axis_labels(self.overlay.axis_labels, axis_labels)
        self._on_view_direction_change()

    def _on_unit_change(self) -> None:
        # NOTE: this is also called by VispyCanvas when layer units are updated
        #       so it doesn't need to be connected to events for that
        self._on_ticks_change()

    def _on_font_size_change(self) -> None:
        # called by vispy canvas when global fonts are updated
        font_size = (
            self.overlay.font_size
            if self.overlay.font_size is not None
            else self._default_font_size
        )

        self.node.font_size = font_size
        self._on_ticks_change()

    def _on_axis_labels_change(self) -> None:
        self._on_ticks_change()

    def _on_ticks_change(self) -> None:
        # pick new steps, e.g. for a new number of ticks
        self._tick_steps = [None] * 3
        self._update_ticks_and_labels()
        # needed to ensure new grids/ticks are up to date
        self._on_blending_change()

    def _on_extent_change(self) -> None:
        self.node.set_extents(self._get_data_ranges())
        self._on_ticks_change()

    def _rebuild_all(self) -> None:
        self.node.color = (
            self.overlay.color
            if self.overlay.color is not None
            else self._get_fgcolor()
        )
        self._on_extent_change()

    def _on_camera_change(self) -> None:
        if self.viewer.dims.ndisplay == 2:
            # texts are only updated when they change, so this is cheap
            self._update_ticks_and_labels()
        else:
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
