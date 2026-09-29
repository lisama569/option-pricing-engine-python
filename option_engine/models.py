"""Request/response models and enums for the pricing service."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_core import PydanticCustomError

# Discrete-resolution bounds for the deterministic binomial pricer. Every
# integer in [MIN_RESOLUTION, MAX_RESOLUTION] is accepted and computed
# exactly; values outside are rejected (never silently substituted).
MIN_RESOLUTION = 1
MAX_RESOLUTION = 2000


def _describe_resolution_value(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, float):
        return "float"
    return type(value).__name__


def validate_resolution(value: Any) -> int | None:
    """Strictly validate the discrete ``resolution`` step count.

    Only ``None`` (unset) and genuine integers within
    ``[MIN_RESOLUTION, MAX_RESOLUTION]`` are accepted. Pydantic's lax mode
    would otherwise coerce ``True``->1, ``5.0``->5 and ``"5"``->5; those
    coercions are explicitly forbidden so an illegal value can never be
    silently priced as some other step count.

    Raises ``PydanticCustomError`` with a distinguishable type:
      * ``resolution_bad_type``  — not an exact JSON integer (bool/float/str)
      * ``resolution_out_of_range`` — integer outside the allowed interval
    """
    if value is None:
        return None
    # bool is a subclass of int, so it must be rejected before the int check.
    if isinstance(value, bool):
        raise PydanticCustomError(
            "resolution_bad_type",
            "resolution must be an exact integer step count in "
            "[{min_resolution}, {max_resolution}]; got a boolean "
            "({value_literal}), which is never coerced to a step count",
            {"min_resolution": MIN_RESOLUTION, "max_resolution": MAX_RESOLUTION,
             "value_literal": str(value).lower(), "actual_type": "boolean"},
        )
    if not isinstance(value, int):
        raise PydanticCustomError(
            "resolution_bad_type",
            "resolution must be an exact integer step count in "
            "[{min_resolution}, {max_resolution}]; got {actual_type} "
            "{value_repr} — send a JSON integer without quotes or decimal "
            "point; floats and numeric strings are not coerced",
            {"min_resolution": MIN_RESOLUTION, "max_resolution": MAX_RESOLUTION,
             "actual_type": _describe_resolution_value(value),
             "value_repr": str(value)},
        )
    if not MIN_RESOLUTION <= value <= MAX_RESOLUTION:
        raise PydanticCustomError(
            "resolution_out_of_range",
            "resolution must be an integer in [{min_resolution}, "
            "{max_resolution}]; got {value}, which is outside the allowed "
            "range and is never substituted by another step count",
            {"min_resolution": MIN_RESOLUTION, "max_resolution": MAX_RESOLUTION,
             "value": value},
        )
    return value


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
    # The strict before-validator rejects bools/floats/strings outright;
    # no value is ever coerced into another step count.
    resolution: int | None = Field(
        default=None,
        ge=MIN_RESOLUTION,
        le=MAX_RESOLUTION,
        description="Exact binomial step count; JSON integer in "
                    f"[{MIN_RESOLUTION}, {MAX_RESOLUTION}] only; bypasses "
                    "convergence search",
    )

    @field_validator("resolution", mode="before")
    @classmethod
    def _resolution_is_strict_integer(cls, value: Any) -> int | None:
        return validate_resolution(value)

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

    # NOTE: deliberately untyped/unconstrained at the batch-envelope layer.
    # An illegal resolution on one leg must fail ONLY that leg (the leg is
    # rebuilt into PricingRequest, where the strict validator runs); letting
    # the envelope reject it would fail the whole batch.
    resolution: Any = None
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
