"""Tests for the dock layout rules, which need no window."""

import pytest

from napari._qt._dock_layout import Dock, plan_area, share_space


@pytest.mark.parametrize(
    ('minimums', 'wants', 'priority', 'space', 'expected'),
    [
        # everyone gets their minimum, the new dock grows first
        (
            [100, 100, 100],
            [500, 500, 500],
            [False, False, True],
            500,
            [100, 100, 300],
        ),
        # leftovers go to the others, by how much more they want
        (
            [100, 100, 100],
            [300, 200, 150],
            [False, False, True],
            500,
            [200, 150, 150],
        ),
        # not even the minimums fit: minimums are kept
        ([200, 200], [400, 400], [True, False], 300, [200, 200]),
    ],
)
def test_share_space(minimums, wants, priority, space, expected):
    assert share_space(minimums, wants, priority, space) == expected


def _group(*heights, new=False):
    """One dock per height, the last one newly added if ``new``."""
    docks = [Dock(natural=h) for h in heights]
    if new:
        docks[-1] = Dock(natural=heights[-1], new=True)
    return docks


def test_docks_that_fit_keep_their_height():
    plan = plan_area(
        [_group(100), _group(100), _group(100)], space=1000, separator=10
    )
    assert plan.slots == ((0,), (1,), (2,))
    assert plan.sizes == (100, 100, 100)


def test_tall_docks_each_keep_a_quarter_of_the_area():
    plan = plan_area(
        [_group(1000) for _ in range(6)], space=1000, separator=10
    )
    cap = int(1000 * 1 / 4) - 10
    # four docks of that height fit, so the rest become tabs of the fourth
    assert plan.slots == ((0,), (1,), (2,), (3, 4, 5))
    assert plan.sizes == (cap, cap, cap, cap)


def test_short_docks_are_not_squeezed():
    plan = plan_area([_group(2000), _group(50)], space=600, separator=10)
    assert plan.sizes[1] == 50
    assert plan.sizes[0] > 200


def test_no_dock_is_squeezed_below_the_floor():
    # 500 is not four times the floor, so the floor decides the squeeze
    plan = plan_area([_group(5000) for _ in range(5)], space=500, separator=10)
    assert plan.slots == ((0,), (1, 2, 3, 4))
    assert all(size >= 200 for size in plan.sizes)


def test_a_new_dock_grows_towards_what_it_asks_for():
    plan = plan_area(
        [_group(600), _group(600, new=True)], space=1000, separator=10
    )
    assert plan.sizes[1] > plan.sizes[0]
