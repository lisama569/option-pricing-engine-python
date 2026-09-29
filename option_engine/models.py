"""Request/response models and enums for the pricing service."""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from pydantic.functional_validators import BeforeValidator

from .errors import InvalidParameterError
from .resolution import (
    MAX_RESOLUTION,
    MIN_RESOLUTION,
    RANGE_TEXT,
    validate_resolution,
)


def _strict_resolution(value):
    """Before-validator enforcing the shared resolution rules at the model
    layer for the single/async entry points.

    Booleans, floats and strings are rejected here (Pydantic's lax int
    coercion would otherwise turn ``true``/``1.0``/``"100"`` into steps),
    and the explanatory reason is surfaced verbatim in the 422 body.
    """
    try:
        return validate_resolution(value)
    except InvalidParameterError as exc:
        # ValueError inside a validator becomes a field-scoped
        # RequestValidationError carrying our exact, distinguishable message.
        raise ValueError(str(exc)) from exc


# Exact ints (or omitted) only; never a coerced bool/float/string.
StrictResolution = Annotated[
    int | None,
    BeforeValidator(_strict_resolution),
]

# Discrete-resolution bounds live in option_engine.resolution so the single
# validation rule can be shared by all entry points without an import cycle.
# MIN_RESOLUTION / MAX_RESOLUTION are imported above and remain available as
# ``option_engine.models.MIN_RESOLUTION`` (backward-compatible re-export).


class OptionType(str, Enum):
    VANILLA_CALL = "vanilla_call"
    VANILLA_PUT = "vanilla_put"
    DIGITAL_CALL = "digital_call"
    DIGITAL_PUT = "digital_put"
    AMERICAN_PUT = "american_put"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PricingRequest(BaseModel):
    """A single pricing request.

    Market inputs (spot/rate/dividend) default to the values carried by the
    bound surface version; per-request overrides are allowed for scenario
    analysis. Volatility always comes from the bound surface version.
    """

    surface_id: str = Field(default="default", description="Surface identifier")
    version: int | None = Field(
        default=None, description="Bind to a specific published version; "
        "None means the latest version at submission time"
    )
    option_type: OptionType
    expiry: float = Field(description="Time to expiry in years")
    strike: float = Field(description="Strike price")

    spot: float | None = Field(default=None, gt=0)
    rate: float | None = Field(default=None)
    dividend: float | None = Field(default=None, ge=0)

    payout: float = Field(default=1.0, gt=0, description="Digital option rebate")

    # Fixed discrete resolution (American put only): when set, the CRR tree
    # is built with exactly this many steps and no convergence search is
    # performed. The result is deterministic for a bound surface version.
    #
    # Type rules are deliberately strict and identical across every entry
    # point: only an exact int in [MIN_RESOLUTION, MAX_RESOLUTION] (both
    # boundaries included) is accepted; booleans, floats (even 1.0) and
    # numeric strings are rejected, never coerced into a step count.
    resolution: StrictResolution = Field(
        default=None,
        ge=MIN_RESOLUTION,
        le=MAX_RESOLUTION,
        description=(
            f"Exact binomial step count; {RANGE_TEXT} only; bypasses "
            "convergence search. Booleans, floats and numeric strings are "
            "rejected, never converted to another step count."
        ),
    )

    # Extrapolation is opt-in per request and is honoured only inside the
    # extrapolation range declared by the bound surface version.
    allow_extrapolation: bool = Field(
        default=False,
        description="Request linear extrapolation up to the surface's "
                    "declared extrapolation limits",
    )

    # Binomial American solver controls used only in the auto-converge mode
    # (resolution unset).
    steps: int = Field(default=500, gt=0, le=20000)
    convergence_tol: float = Field(default=1e-4, gt=0)
    max_steps: int = Field(default=4000, gt=0, le=200000)

    # When set, solve for implied volatility matching this market price
    # (European vanilla only) instead of pricing from the surface vol.
    target_price: float | None = Field(default=None, gt=0)


class ExtrapolationLimits(BaseModel):
    """Extrapolation range declared at surface publish time.

    Coordinates are absolute expiry/strike values. The bounds must enclose
    the grid support; requests beyond these bounds are rejected even when
    the caller opts into extrapolation.
    """

    min_expiry: float
    max_expiry: float
    min_strike: float
    max_strike: float


class ExtrapolationInfo(BaseModel):
    """Audit trail for a vol lookup that actually left the grid support."""

    expiry_outside: bool
    strike_outside: bool
    # Absolute distance beyond the grid support (0 for in-support axes).
    expiry_overhang: float
    strike_overhang: float
    # Overhang as a fraction of the declared extrapolation band width.
    expiry_overhang_fraction: float
    strike_overhang_fraction: float


class PriceResponse(BaseModel):
    price: float
    vol_used: float
    surface_version: int
    method: str
    detail: dict = Field(default_factory=dict)
    cached: bool = False
    # Which precision mode produced the price:
    #   fixed_resolution / auto_converged (American), closed_form (European).
    pricing_mode: Literal[
        "closed_form", "fixed_resolution", "auto_converged"
    ] = "closed_form"
    steps_used: int | None = None
    extrapolated: bool = False
    extrapolation: ExtrapolationInfo | None = None


class ErrorBody(BaseModel):
    error: str
    message: str
    detail: dict = Field(default_factory=dict)


class SurfaceGridPayload(BaseModel):
    """Upload payload for a volatility surface grid.

    ``vols[i][j]`` is the IV for ``expiries[i]`` and ``strikes[j]``.
    """

    expiries: list[float]
    strikes: list[float]
    vols: list[list[float]]
    spot: float = Field(gt=0)
    rate: float = 0.0
    dividend: float = Field(default=0.0, ge=0)
    # Declared, version-bound extrapolation range. None means the surface
    # forbids extrapolation outright; callers cannot opt in for this version.
    extrapolation_limits: ExtrapolationLimits | None = None


class SurfaceVersionInfo(BaseModel):
    surface_id: str
    version: int
    created_at: str
    spot: float
    rate: float
    dividend: float
    extrapolation_limits: ExtrapolationLimits | None = None
    expiries: list[float]
    strikes: list[float]


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    surface_version: int | None = None
    result: PriceResponse | None = None
    error_code: str | None = None
    error_message: str | None = None


class BatchPricingItem(BaseModel):
    """One leg of a batch. Surface binding lives on the enclosing batch."""

    option_type: OptionType
    expiry: float
    strike: float

    spot: float | None = Field(default=None, gt=0)
    rate: float | None = Field(default=None)
    dividend: float | None = Field(default=None, ge=0)
    payout: float = Field(default=1.0, gt=0)

    # Kept as the raw JSON value on purpose: a bad resolution on one leg
    # must fail that leg only (per-leg invalid_parameter) rather than fail
    # the whole batch at request-body validation. The exact shared rules in
    # option_engine.resolution are applied per leg when it is executed.
    resolution: Any = Field(
        default=None,
        description=(
            f"Exact binomial step count; {RANGE_TEXT} only. Booleans, "
            "floats and numeric strings are rejected for this leg, never "
            "converted to another step count."
        ),
    )
    allow_extrapolation: bool = False

    steps: int = Field(default=500, gt=0, le=20000)
    convergence_tol: float = Field(default=1e-4, gt=0)
    max_steps: int = Field(default=4000, gt=0, le=200000)

    target_price: float | None = Field(default=None, gt=0)


class BatchPricingRequest(BaseModel):
    """A whole book of pricing legs bound to ONE surface version."""

    surface_id: str = "default"
    version: int | None = None
    items: list[BatchPricingItem] = Field(min_length=1)


class BatchPricingResult(BaseModel):
    index: int
    status: Literal["succeeded", "failed"]
    result: PriceResponse | None = None
    error_code: str | None = None
    error_message: str | None = None


class BatchPricingResponse(BaseModel):
    surface_id: str
    surface_version: int
    results: list[BatchPricingResult]
