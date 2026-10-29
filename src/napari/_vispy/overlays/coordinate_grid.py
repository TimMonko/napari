from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from napari._vispy.overlays.base import ViewerOverlayMixin, VispySceneOverlay
from napari._vispy.visuals.coordinate_grid import CoordinateGrid
from napari.components.dims import RangeTuple
from napari.settings import get_settings
from napari.utils._units import (
    compact_unit,
    format_unit,
    get_unit_registry,
)

if TYPE_CHECKING:
    from napari._vispy.utils.qt_font import FontInfo
    from napari.components.overlays import CoordinateGridOverlay


# map conventions for axes named after geographic coordinates
_HEMISPHERES = {
    'latitude': ('N', 'S'),
    'lat': ('N', 'S'),
    'longitude': ('E', 'W'),
    'lon': ('E', 'W'),
    'long': ('E', 'W'),
}
_PROJECTED = {'easting', 'northing'}

TickFormatter = Callable[[float, int], str]


def _format_tick(value: float, decimals: int) -> str:
    return np.format_float_positional(value, decimals, trim='-')


def _hemisphere_formatter(positive: str, negative: str) -> TickFormatter:
    """Format degrees with a hemisphere letter, e.g. 78°W, not -78."""

    def format_tick(value: float, decimals: int) -> str:
        suffix = positive if value > 0 else negative if value < 0 else ''
        return f'{_format_tick(abs(value), decimals)}°{suffix}'

    return format_tick


def _format_projected_tick(value: float, decimals: int) -> str:
    """Format a projected coordinate in full, e.g. 2,640,000."""
    return f'{value:,.{decimals}f}'


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
            self._on_view_direction_change
        )
        self.viewer.scene.camera.events.zoom.connect(
            self._on_view_direction_change
        )
        get_settings().appearance.events.theme.connect(self._rebuild_all)
        self.viewer.events.theme.connect(self._rebuild_all)

        self.reset()

    def _get_ranges_and_axis_labels(
        self,
    ) -> tuple[list[RangeTuple], list[str], list[float], list[TickFormatter]]:
        # napari dims are zyx, but vispy uses xyz
        displayed = self.viewer.dims.displayed[::-1]
        ranges = [self.viewer.dims.range[i] for i in displayed]
        axis_labels = [self.viewer.dims.axis_labels[i] for i in displayed]

        units = ['' for _ in displayed]
        scales = [1.0 for _ in displayed]
        formatters: list[TickFormatter] = [_format_tick for _ in displayed]
        if self.viewer.layers.units is not None:
            reg = get_unit_registry()
            for axis, (dim, range_) in enumerate(
                zip(displayed, ranges, strict=True)
            ):
                unit = self.viewer.layers.units[dim]
                name = axis_labels[axis].lower()
                if name in _HEMISPHERES and unit == reg.degree:
                    # like a map: 78°W, and the ticks already say degrees
                    formatters[axis] = _hemisphere_formatter(
                        *_HEMISPHERES[name]
                    )
                    continue
                if name in _PROJECTED and unit.dimensionality == (
                    reg.meter.dimensionality
                ):
                    # like GIS grids: full coordinates in the CRS unit
                    formatters[axis] = _format_projected_tick
                else:
                    # prefix the unit like the scale bar does (e.g. m -> nm)
                    unit, scales[axis] = compact_unit(
                        unit, range_.stop - range_.start
                    )
                units[axis] = f' ({format_unit(unit)})'

        axis_labels_with_units = [
            f'{lab}{unit}'
            for lab, unit in zip(axis_labels, units, strict=True)
        ]
        return ranges, axis_labels_with_units, scales, formatters

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
        ranges, axis_labels, _, _ = self._get_ranges_and_axis_labels()
        self.node.set_axis_labels(
            self.overlay.axis_labels, ranges, axis_labels
        )
        # needed to ensure new grids/ticks are up to date
        self._on_blending_change()
        self._on_view_direction_change()

    def _on_ticks_change(self) -> None:
        ranges, _, scales, formatters = self._get_ranges_and_axis_labels()
        self.node.set_ticks(
            self.overlay.tick_labels,
            self.overlay.n_ticks,
            ranges,
            scales,
            formatters,
        )
        # needed to ensure new grids/ticks are up to date
        self._on_blending_change()
        self._on_view_direction_change()

    def _on_extent_change(self) -> None:
        ranges, _, _, _ = self._get_ranges_and_axis_labels()
        self.node.set_extents(ranges)
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

    def _on_view_direction_change(self) -> None:
        # all is flipped from zyx to xyz for vispy
        displayed = self.viewer.dims.displayed[::-1]
        ranges = tuple(self.viewer.dims.range[i] for i in displayed)

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
        )

    def reset(self) -> None:
        super().reset()
        self._rebuild_all()
