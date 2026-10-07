"""Decide how much height the dock widgets stacked in one area should get.

Docks in the left and right areas share the height of the window, so they
cannot all have the height they ask for once a plugin adds a few. The rules
live here, in one place, and are testable without a window:

* a dock asks for the height at which its widget is fully visible;
* while the window can grow to fit them, it does;
* a dock taller than its share of the area is squeezed to that share, and
  its widget scrolls;
* a dock that still does not fit becomes a tab of the last dock that did,
  so nothing already on screen moves or hides.

The docks are grouped in the order they are stacked, top to bottom. Nothing
here touches Qt, and one call decides everything, so applying a decision
cannot change it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

#: A dock is squeezed to this share of the area before it is made a tab.
MIN_AREA_FRACTION = 1 / 4
#: A dock is never squeezed below this height, however short the area is.
MIN_DOCK_HEIGHT = 200


@dataclass(frozen=True)
class Dock:
    """What the layout needs to know about one dock widget."""

    #: the height at which the widget is fully visible
    natural: int
    #: whether the dock was added or moved just now
    new: bool = False


@dataclass(frozen=True)
class Plan:
    """How the docks in one area should be arranged.

    ``slots`` holds the indices of the input groups that share a place in
    the stack, in that order; a slot with more than one group has the later
    ones tabbed onto the first. ``sizes`` is the height of each slot, in the
    form ``resizeDocks`` wants.
    """

    slots: tuple[tuple[int, ...], ...]
    sizes: tuple[int, ...]


def share_space(
    minimums: Sequence[int],
    wants: Sequence[int],
    priority: Sequence[bool],
    space: int,
) -> list[int]:
    """Split ``space`` between docks stacked in an area.

    Every dock gets its minimum. Docks with ``priority`` (newly added ones)
    then grow towards the size they want, and whatever is left is shared by
    the others in proportion to how much more they want.
    """
    sizes = list(minimums)
    missing = [max(w - m, 0) for w, m in zip(wants, minimums, strict=True)]
    spare = space - sum(sizes)
    for i, first in enumerate(priority):
        if first and spare > 0:
            extra = min(spare, missing[i])
            sizes[i] += extra
            missing[i] -= extra
            spare -= extra
    total_missing = sum(missing)
    if spare > 0 and total_missing:
        for i, m in enumerate(missing):
            sizes[i] += spare * m // total_missing
    return sizes


def plan_area(
    groups: Sequence[Sequence[Dock]],
    *,
    space: int,
    separator: int,
) -> Plan:
    """Fit ``groups`` into ``space``, tabbing the ones that no longer fit.

    ``groups`` is in the order the docks are stacked, top to bottom. A dock
    that cannot keep ``MIN_DOCK_HEIGHT``, or a share of the area, becomes a
    tab of the last dock that could.
    """
    # a quarter of the area each, less the separator it also needs, so that
    # four docks of that height fit exactly
    cap = max(MIN_DOCK_HEIGHT, int(space * MIN_AREA_FRACTION) - separator)

    def comfortable(slot: Sequence[int]) -> int:
        return max(min(dock.natural, cap) for i in slot for dock in groups[i])

    def natural(slot: Sequence[int]) -> int:
        return max(dock.natural for i in slot for dock in groups[i])

    slots: list[list[int]] = []
    for index in range(len(groups)):
        if not slots:
            slots.append([index])
        elif (
            sum(comfortable(slot) + separator for slot in slots)
            + comfortable([index])
            + separator
            > space
        ):
            # no room left at the bottom of the stack: this dock, and every
            # one after it, becomes a tab of the last dock that fits
            slots[-1].append(index)
        else:
            slots.append([index])

    sizes = share_space(
        minimums=[comfortable(slot) for slot in slots],
        wants=[natural(slot) for slot in slots],
        priority=[
            any(dock.new for i in slot for dock in groups[i]) for slot in slots
        ],
        space=space - separator * len(slots),
    )
    return Plan(tuple(tuple(slot) for slot in slots), tuple(sizes))
