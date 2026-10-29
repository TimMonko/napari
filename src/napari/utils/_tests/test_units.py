import numpy as np
import pytest

from napari.utils._units import (
    compact_unit,
    compute_nice_ticks,
    get_unit_registry,
)


@pytest.mark.parametrize(
    ('start', 'stop', 'n_ticks', 'expected_ticks', 'expected_decimals'),
    [
        (0, 3, 5, [0, 0.75, 1.5, 2.25, 3], 2),
        (0, 5, 5, [0, 1.5, 3, 4.5], 1),
        (0, 3, 10, [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3], 1),
        (-1, 1, 5, [-1, -0.5, 0, 0.5, 1], 1),
        (0, 0.03, 5, [0, 0.0075, 0.015, 0.0225, 0.03], 4),
        (101985, 339015, 5, [150000, 200000, 250000, 300000], 0),
        (1, 1, 5, [], 0),
    ],
)
def test_compute_nice_ticks(
    start, stop, n_ticks, expected_ticks, expected_decimals
):
    ticks, decimals = compute_nice_ticks(start, stop, n_ticks)
    np.testing.assert_allclose(ticks, expected_ticks)
    assert decimals == expected_decimals


@pytest.mark.parametrize(
    ('unit', 'span', 'expected_unit', 'expected_factor'),
    [
        ('m', 2e-9, 'nm', 1e9),
        ('m', -2e-9, 'nm', 1e9),
        ('um', 3000, 'mm', 1e-3),
        ('um', 30, 'um', 1),
        ('nm', 1e-3, 'pm', 1e3),
        ('pixel', 2000, 'kilopixel', 1e-3),
        ('dimensionless', 1e-9, 'dimensionless', 1),
        # angles are dimensionless in pint but still take prefixes
        ('degree', 2e-3, 'millidegree', 1e3),
        ('m', 0, 'm', 1),
        ('m', float('inf'), 'm', 1),
    ],
)
def test_compact_unit(unit, span, expected_unit, expected_factor):
    reg = get_unit_registry()
    compact, factor = compact_unit(reg(unit).units, span)
    assert compact == reg(expected_unit).units
    # exact, so that scaled tick values stay round
    assert factor == expected_factor
