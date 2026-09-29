"""Single source of truth for the discrete ``resolution`` validation rules.

The same rules are applied by every pricing entry point (synchronous HTTP,
asynchronous jobs and per-leg batch pricing):

* ``None`` (field omitted) -> auto-convergence mode, accepted everywhere;
* an exact ``int`` in [MIN_RESOLUTION, MAX_RESOLUTION] (boundaries included)
  -> fixed-resolution mode, accepted;
* booleans, floats (even integral-valued ones like ``1.0``), numeric strings
  (``"100"``) and any other non-int type -> rejected, NEVER coerced into a
  step count;
* ints outside the range (0, negatives, above the cap) -> rejected.
"""
from __future__ import annotations

from .errors import InvalidParameterError

# Discrete-resolution bounds for the deterministic binomial pricer. Every
# integer in [MIN_RESOLUTION, MAX_RESOLUTION] is accepted and computed
# exactly; values outside are rejected (never silently substituted).
MIN_RESOLUTION = 1
MAX_RESOLUTION = 2000

RANGE_TEXT = f"an exact integer in [{MIN_RESOLUTION}, {MAX_RESOLUTION}]"


def validate_resolution(value) -> int | None:
    """Validate a raw ``resolution`` value exactly once for every entry.

    Returns a clean ``int`` (or ``None``); raises ``InvalidParameterError``
    with a self-describing, distinguishable reason otherwise. No accepted
    value is ever substituted, and no rejected value is ever converted.
    """
    if value is None:
        return None

    # bool is a subclass of int in Python; it must be checked first so that
    # JSON ``true``/``false`` are never silently read as steps 1/0.
    if isinstance(value, bool):
        raise InvalidParameterError(
            "resolution must be " + RANGE_TEXT +
            f" (the number of binomial steps); got boolean {str(value).lower()}"
            " — booleans are not an accepted type and are never converted to "
            "a step count"
        )

    if isinstance(value, int):
        steps = value
    elif isinstance(value, float):
        raise InvalidParameterError(
            "resolution must be " + RANGE_TEXT +
            f"; got the float {value!r} — fractional/decimal values are not "
            "accepted (integral-valued floats are not converted to int either)"
        )
    elif isinstance(value, str):
        raise InvalidParameterError(
            "resolution must be " + RANGE_TEXT +
            f"; got the string {value!r} — numeric strings are not accepted "
            "and are never parsed into a step count"
        )
    else:
        raise InvalidParameterError(
            "resolution must be " + RANGE_TEXT +
            f"; got value of unsupported type {type(value).__name__}: "
            f"{value!r}"
        )

    if steps < MIN_RESOLUTION or steps > MAX_RESOLUTION:
        raise InvalidParameterError(
            f"resolution must be within [{MIN_RESOLUTION}, "
            f"{MAX_RESOLUTION}] (boundaries included); got {steps}"
        )
    return steps
