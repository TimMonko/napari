"""Units utilities."""

from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pint

# define preferred scale bar values
PREFERRED_VALUES = [
    1,
    2,
    5,
    10,
    15,
    20,
    25,
    50,
    75,
    100,
    125,
    150,
    200,
    500,
    750,
]


def get_unit_registry() -> 'pint.UnitRegistry':
    """Get pint's UnitRegistry.

    Pint greedily imports many libraries, (including dask, xarray, pandas, and
    babel) to check for compatibility.  Some of those libraries may be slow to
    import.  This accessor function should be used (and only when units are
    actually necessary) to avoid incurring a large import time penalty.

    See comment for details:
    https://github.com/napari/napari/pull/2617#issuecomment-827747792
    """
    import pint

    return pint.get_application_registry()


PREFERRED_TICK_VALUES = [1, 1.5, 2, 2.5, 3, 4, 5, 7.5]


def _generate_ticks(
    base: Decimal, exp: int, min_value: Decimal, max_value: Decimal
) -> np.ndarray:
    step = base * Decimal(10) ** exp

    # ignore typing because checkers are not happy with numpy and Decimal
    tick_min = np.ceil(min_value / step) * step  # pyrefly: ignore [no-matching-overload]
    tick_max = np.floor(max_value / step) * step  # pyrefly: ignore [no-matching-overload]

    # actually generate ticks with the given step
    return np.arange(tick_min, tick_max + step, step).astype(float)  # pyrefly: ignore [no-matching-overload]


def compute_nice_ticks(
    min_value: float, max_value: float, target_ticks: int = 5
) -> tuple[np.ndarray, int]:
    """Compute round tick values spanning a range.

    Returns
    -------
    ticks : np.ndarray
        The tick values.
    decimals : int
        Decimal places needed to show the ticks exactly.
    """
    best_ticks = np.empty(0, float)
    best_decimals = 0

    if min_value == max_value:
        return best_ticks, best_decimals

    # Decimal needed for small values float imprecision
    # we need to convert to str first, otherwise Decimal just takes the
    # literal float value, imprecision and all
    minv = Decimal(str(min_value))
    maxv = Decimal(str(max_value))

    span = maxv - minv
    ideal_step = span / (target_ticks - 1)

    # ignore typing because checkers are not happy with numpy and Decimal
    ideal_exponent = int(np.floor(np.log10(ideal_step)))  # pyrefly: ignore [no-matching-overload]
    exp_candidates = range(ideal_exponent - 1, ideal_exponent + 2)

    for exp in exp_candidates:
        for base in PREFERRED_TICK_VALUES:
            ticks = _generate_ticks(Decimal(base), exp, minv, maxv)

            # get number of ticks closer to target_ticks
            if abs(len(ticks) - target_ticks) <= abs(
                len(best_ticks) - target_ticks
            ):
                best_ticks = ticks
                # ticks are multiples of the step, so the step's last
                # significant digit is the precision every tick needs
                # (e.g. 0.75 -> 2)
                step = Decimal(base) * Decimal(10) ** exp
                exponent = step.normalize().as_tuple().exponent
                best_decimals = max(0, -int(exponent))

    return best_ticks, best_decimals


# a step is kept until the ideal one is this many times larger or smaller,
# so that zooming around a threshold does not make ticks flicker. Preferred
# steps are at most 1.5 times apart, so the closest one is always kept
_STEP_HYSTERESIS = 1.5


def compute_nice_step(
    span: float, target_ticks: int = 5, previous: float | None = None
) -> float:
    """Get a round step that divides a span into about `target_ticks` ticks.

    Unlike `compute_nice_ticks`, this only depends on the length of the span,
    not on where it starts, so the step doesn't change when panning.

    Parameters
    ----------
    span : float
        Length to divide.
    target_ticks : int
        Number of ticks to aim for.
    previous : float, optional
        Step used so far, which is kept unless it's far from the ideal one.

    Returns
    -------
    float
        The step, or nan if `span` is not positive and finite.
    """
    if not (span > 0 and np.isfinite(span)):
        return float('nan')
    ideal = span / max(target_ticks - 1, 1)
    if (
        previous is not None
        and ideal / _STEP_HYSTERESIS <= previous <= ideal * _STEP_HYSTERESIS
    ):
        return previous
    exponent = int(np.floor(np.log10(ideal)))
    candidates = [
        base * 10.0**exp
        for exp in (exponent - 1, exponent, exponent + 1)
        for base in PREFERRED_TICK_VALUES
    ]
    step = min(candidates, key=lambda c: abs(np.log(c / ideal)))
    # drop float noise (e.g. 0.30000000000000004)
    return float(f'{step:.12g}')


def compute_ticks_with_step(
    min_value: float, max_value: float, step: float
) -> tuple[np.ndarray, int]:
    """Get the multiples of a step in a range.

    Returns
    -------
    ticks : np.ndarray
        The tick values.
    decimals : int
        Decimal places needed to show the ticks exactly.
    """
    if not (max_value > min_value and step > 0):
        return np.empty(0, float), 0
    # see compute_nice_ticks for why Decimal and str
    step_decimal = Decimal(str(step))
    ticks = _generate_ticks(
        step_decimal, 0, Decimal(str(min_value)), Decimal(str(max_value))
    )
    exponent = step_decimal.normalize().as_tuple().exponent
    return ticks, max(0, -int(exponent))


def compact_unit(unit: 'pint.Unit', span: float) -> tuple['pint.Unit', float]:
    """Get a prefixed unit that shows a span with a readable magnitude.

    Parameters
    ----------
    unit : pint.Unit
        Unit of the span.
    span : float
        Extent to be shown, in `unit`.

    Returns
    -------
    compact : pint.Unit
        Prefixed unit (e.g. nm for 2e-9 m). Same as `unit` if
        `span` is zero or not finite.
    factor : float
        Factor converting values in `unit` to `compact`.
    """
    if span == 0 or not np.isfinite(span):
        return unit, 1.0
    compact = (abs(span) * unit).to_compact().units
    factor = (1 * unit).to(compact).magnitude
    # prefixes are powers of 10, so drop float noise (999999999.9999999)
    return compact, float(f'{factor:.15g}')
