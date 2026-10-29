from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from vispy.color import Color
from vispy.scene import MatrixTransform, STTransform
from vispy.scene.visuals import GridLines, Node

from napari._vispy.visuals.text import Text
from napari.components.dims import RangeTuple

if TYPE_CHECKING:
    from vispy.visuals.filters.clipper import Clipper

    from napari._vispy.utils.qt_font import FontInfo
    from napari.utils.color import ColorValue


class CoordinateGrid(Node):
    def __init__(
        self,
        font_info: FontInfo,
    ):
        super().__init__()
        self.font_info = font_info
        self.font_size = 9.0

        # compound does not play well with sub-transforms for some reason
        # so we use a simple empty node with children instead
        # one Text visual per axis holds all its ticks: each visual has its
        # own shader program, and building those is slow, notably when
        # overlays are created or moved between views (e.g: grid mode)
        self.tick_labels: list[Text] = [
            Text(font_info=font_info) for _ in range(3)
        ]
        self.axis_labels: list[Text] = [
            Text(font_info=font_info) for _ in range(3)
        ]
        # ticks shown by each axis (values, scale and decimals), to skip
        # updates that change nothing
        self._ticks: list[tuple[np.ndarray, float, int]] = [
            (np.empty(0), 1.0, 0) for _ in range(3)
        ]
        # largest tick label size (width, line height) of each axis since
        # its scale, decimals or font size changed (the key)
        self._tick_sizes: list[tuple[tuple, tuple[float, float]]] = [
            ((), (0.0, 0.0)) for _ in range(3)
        ]
        self._color: ColorValue | str = 'white'
        self._opacity = 1.0

        # grids are only created once and then updated in place: creating
        # visuals means building new shader programs, which is slow
        self.grids: list[GridLines] = []
        for axis in range(3):
            grid = GridLines(parent=self, border_width=0, color=self._color)
            grid.transform = MatrixTransform()
            self.grids.append(grid)
            self.axis_labels[axis].transform = STTransform()
            self.axis_labels[axis].parent = grid
            self.tick_labels[axis].transform = STTransform()
            self.tick_labels[axis].parent = grid

    @property
    def color(self) -> ColorValue | str:
        return self._color

    @color.setter
    def color(self, color: ColorValue | str) -> None:
        self._color = color
        # GridLines only takes color on init, but it's stored as a uniform
        # in its shader function, so it can be updated without a rebuild
        for grid in self.grids:
            grid._grid_color_fn['color'] = Color(color).rgba
            grid.update()

    def _set_clipper(self, node: Node, clipper: Clipper | None) -> None:
        """Propagate the viewbox clipper to all child visuals.

        This is just a Node (not VisualNode), so it would otherwise
        ignore the clipper and draw outside its viewbox (like ColorBar).

        Unlike ColorBar, which is a child of the ViewBox, this is a child of
        its scene, which doesn't remove the clipper when the grid moves to
        another viewbox. The base Node does, but only for the clippers it
        finds in `_clippers`, so they need to be stored there.
        """
        if clipper is None:
            self._clippers.pop(node, None)
        else:
            self._clippers[node] = clipper
        # labels are children of the grids, not of this node. We need to explicity
        # set the clipper for all of them otherwise they can get out of sync.
        for grid in self.grids:
            for visual in (grid, *grid.children):
                visual._set_clipper(node, clipper)

    def set_gl_state(self, *args: Any, **kwargs: Any) -> None:
        for grid in self.grids:
            grid.set_gl_state(*args, **kwargs)

    @property
    def opacity(self) -> float:
        return self.grids[0].opacity

    @opacity.setter
    def opacity(self, opacity: float) -> None:
        self._opacity = opacity
        for grid in self.grids:
            grid.opacity = opacity
        for ticks in self.tick_labels:
            ticks.opacity = opacity
        for axis in self.axis_labels:
            axis.opacity = opacity

    def set_extents(self, ranges: list[RangeTuple]) -> None:
        ndisplay = len(ranges)
        bounds = []
        for i in range(ndisplay):
            rng0 = ranges[i]
            rng1 = ranges[(i + 1) % ndisplay]
            bounds.append((rng0.start, rng0.stop, rng1.start, rng1.stop))

        for grid, bound in zip(self.grids, bounds, strict=False):
            grid.grid_bounds = bound

        is_3d = len(bounds) == 3
        self.grids[2].visible = is_3d
        self.tick_labels[2].visible = is_3d
        self.axis_labels[2].visible = is_3d

        self.update()

    def set_view_direction(
        self,
        ranges: tuple[RangeTuple, RangeTuple, RangeTuple],
        view_is_flipped: tuple[bool, bool, bool],
        zoom: float,
        ndisplay: int,
    ) -> None:

        # translate everything with the grid on xy plane, we transpose later
        far_bounds = []
        for axis in range(3):
            self.grids[axis].transform.reset()
            self.tick_labels[axis].transform.translate = (0, 0, 0)
            self.axis_labels[axis].transform.translate = (0, 0, 0)

            # get the translation necessary to bring the grid to the back
            # and also the position of the near bound for later
            far_bound_idx = int(view_is_flipped[axis])
            farthest_bound = ranges[axis][far_bound_idx]
            far_bounds.append(farthest_bound)

        # the 3D camera can rotate, even for 2D data
        tick_offsets, axis_label_offsets = self._get_text_offsets(
            is_3d=ndisplay == 3
        )

        # offset ticks for each axis so they are positioned nicely for readability,
        # putting them all on the outside of the volume (over the background)
        for axis in range(3):
            prev_axis = (axis - 1) % 3
            # move grid to the back of the whole volume (far bound)
            self.grids[axis].transform.translate((0, 0, far_bounds[prev_axis]))

            # shift according to view angle to maximize visibility and have consistent positioning.
            # These branches were found by trial and error with the goal to reproduce the
            # tick positioning by plotly (e.g: https://plotly.com/python/3d-scatter-plots/)

            self._translate_text_based_on_camera(
                self.tick_labels[axis],
                axis,
                far_bounds,
                ranges,
                tick_offsets[axis] / zoom,
                view_is_flipped,
            )
            # axis labels are centered on the axis
            self._translate_text_based_on_camera(
                self.axis_labels[axis],
                axis,
                far_bounds,
                ranges,
                axis_label_offsets[axis] / zoom,
                view_is_flipped,
                along=(ranges[axis].start + ranges[axis].stop) / 2,
            )

        # rotate grids onto the right axes
        for axis in range(3):
            self.grids[axis].transform.rotate(angle=120 * axis, axis=(1, 1, 1))

    def _get_text_offsets(
        self, is_3d: bool
    ) -> tuple[list[float], list[float]]:
        """Get the distance of tick and axis labels from the grid, in pixels.

        Text is centered on its position, so in 2D labels are moved by half
        their size, plus padding, to clear the grid. Axis labels then clear
        the widest tick label, which can be much longer than `font_size`
        (e.g. 2,800,000). The x axis' labels are stacked vertically (so
        their height matters), the y axis' ones side by side (their width).

        In 3D, the direction labels are moved in depends on the view, so
        they use fixed offsets, which work ok at a wide range of zooms.
        """
        padding = self.font_size
        if is_3d:
            return [padding * 2] * 3, [padding * 6] * 3

        tick_offsets = []
        axis_label_offsets = []
        for axis in range(3):
            tick_size, label_size = self._get_2d_text_sizes(axis)
            tick_offsets.append(padding + tick_size / 2)
            axis_label_offsets.append(
                padding + tick_size + padding + label_size / 2
            )
        return tick_offsets, axis_label_offsets

    def _get_2d_text_sizes(self, axis: int) -> tuple[float, float]:
        """Get the size of an axis' tick and axis labels away from the grid.

        That's the width for the y axis, and the height for the x one.
        """
        # 0 is width, 1 is height
        size_index = 0 if axis == 1 else 1
        tick_size = 0.0
        if self.tick_labels[axis].visible:
            # the widest tick, but only one line high
            tick_size = self._tick_sizes[axis][1][size_index]
        label = self.axis_labels[axis]
        label_size = 0.0
        if label.visible and label.text:
            label_size = label.get_width_height()[size_index]
        return tick_size, label_size

    def get_2d_label_extents(self) -> list[float]:
        """Get how far the labels of each axis reach from the grid in 2D.

        In pixels, including padding past the outer side of the axis label.
        """
        padding = self.font_size
        extents = []
        for axis in range(3):
            tick_size, label_size = self._get_2d_text_sizes(axis)
            extents.append(
                padding + tick_size + padding + label_size + padding
            )
        return extents

    def _translate_text_based_on_camera(
        self,
        text: Text,
        axis: int,
        far_bounds: list[float],
        ranges: tuple[RangeTuple, ...],
        offset: float,
        view_is_flipped: tuple[bool, ...],
        along: float = 0,
    ) -> None:
        """Move text from the grid's origin to the edge of its axis.

        Texts are only positioned along their axis (ticks) or not at all
        (axis labels), and moved to their edge here. In 2D, that edge follows
        the camera so labels stay on the canvas, and moving a transform is
        much cheaper than updating text positions.
        """
        prev_axis = (axis - 1) % 3
        next_axis = (axis + 1) % 3

        start_next = ranges[next_axis][0]
        stop_next = ranges[next_axis][1]

        start_prev = ranges[prev_axis][0]
        stop_prev = ranges[prev_axis][1]

        # undo shifts caused by grid transform so we're back to the
        # reference frame of the axes (since ticks are not necessarily
        # on the same side as the grid lines depending on view)
        text.transform.move((along, start_next, -far_bounds[prev_axis]))

        if view_is_flipped[next_axis] ^ view_is_flipped[prev_axis]:
            text.transform.move((0, -offset, start_prev - offset))
        else:
            if axis == 0:
                # special case one axis so it's visually nicer (all axes are
                # on the "outside" and all are on the "same side")
                text.transform.move((0, -offset, stop_prev + offset))
            else:
                text.transform.move(
                    (0, stop_next - start_next + offset, start_prev - offset)
                )

    def set_ticks(
        self,
        show_ticks: bool,
        values: list[np.ndarray],
        decimals: list[int],
        scales: list[float],
    ) -> None:
        """Update tick labels.

        `values` are the ticks of each axis, in the units shown in the labels,
        and `decimals` the precision they need. `scales` converts each axis
        from world coordinates to those units (e.g. 1e9 to label meters in nm).
        """
        for axis, ticks in enumerate(self.tick_labels):
            if not show_ticks or axis >= len(values):
                ticks.visible = False
                continue
            axis_values = values[axis]
            ticks.color = self.color
            ticks.opacity = self._opacity
            ticks.font_size = self.font_size
            ticks.visible = axis_values.size > 0
            # this runs on every camera move in 2D, and updating text
            # visuals is costly, so skip it when nothing changed. Text
            # can't take empty positions, so leave it as is when hidden
            if not axis_values.size:
                continue
            shown_values, *shown_format = self._ticks[axis]
            tick_format = [scales[axis], decimals[axis]]
            text_changed = not (
                tick_format == shown_format
                and np.array_equal(axis_values, shown_values)
            )
            if text_changed:
                self._ticks[axis] = (axis_values, *tick_format)
                ticks.text = [
                    np.format_float_positional(val, decimals[axis], trim='-')
                    for val in axis_values
                ]
                # ticks are moved to the edge of the grid by their transform
                ticks.pos = [(val / scales[axis], 0, 0) for val in axis_values]
            # measuring is slow too, so only do it when sizes can change
            key = (*tick_format, self.font_size)
            if text_changed or key != self._tick_sizes[axis][0]:
                self._update_tick_size(axis, key)

    def _update_tick_size(self, axis: int, key: tuple) -> None:
        """Measure tick labels, keeping the largest size until `key` changes.

        Labels are spaced by this size, and keeping the largest one stops
        them from moving back and forth while panning shows ticks of
        different widths (e.g. 2653.5 and 2654.25).
        """
        ticks = self.tick_labels[axis]
        size = (ticks.get_width_height()[0], ticks.get_line_height())
        shown_key, shown_size = self._tick_sizes[axis]
        if key == shown_key:
            size = (max(size[0], shown_size[0]), max(size[1], shown_size[1]))
        self._tick_sizes[axis] = (key, size)

    def set_axis_labels(self, show_labels: bool, labels: list[str]) -> None:
        for axis, visual in enumerate(self.axis_labels):
            if axis >= len(labels):
                continue
            # setting text lays it out again, even if it's the same
            if visual.text != labels[axis]:
                visual.text = labels[axis]
            visual.color = self.color
            visual.opacity = self._opacity
            visual.font_size = self.font_size * 1.5
            visual.visible = show_labels
