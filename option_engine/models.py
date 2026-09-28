"""Request/response models and enums for the pricing service."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, StrictInt, model_validator


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


class ResolutionMode(str, Enum):
    """How the discrete resolution of the American binomial solver is set.

    * ``converged`` (default): the tree doubles its step count until the
      relative price change drops below ``convergence_tol`` or the work
      budget ``max_steps`` is exhausted (then the request is rejected).
    * ``fixed``: exactly ``resolution`` binomial steps are run; no
      convergence loop, no silent substitution of any other step count.
    """

    CONVERGED = "converged"
    FIXED = "fixed"


# Shared, documented resolution bounds. Values at either boundary MUST be
# accepted and priced; anything outside MUST be rejected (no clamping).
# The cap is the largest resolution the O(n**2) stdlib tree serves in
# bounded time (~seconds); it is advertised rather than silently enforced.
MIN_RESOLUTION = 1
MAX_RESOLUTION = 10_000
MAX_BATCH_ITEMS = 1_000


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

    # Binomial American solver controls.
    steps: StrictInt = Field(default=500, gt=0, le=20000)
    convergence_tol: float = Field(default=1e-4, gt=0)
    max_steps: StrictInt = Field(default=4000, gt=0, le=200000)

    # Explicit discrete-resolution control for reproducible "fixed number of
    # steps" prices (e.g. backtests reproducing a counterparty's run).
    resolution: StrictInt | None = Field(
        default=None,
        ge=MIN_RESOLUTION,
        le=MAX_RESOLUTION,
        description="Pin the binomial tree to exactly this many steps "
                    f"[{MIN_RESOLUTION}, {MAX_RESOLUTION}]; accepted only by "
                    "american_put and mutually exclusive with the "
                    "steps/convergence_tol/max_steps controls",
    )

    # When set, solve for implied volatility matching this market price
    # (European vanilla only) instead of pricing from the surface vol.
    target_price: float | None = Field(default=None, gt=0)

    # Explicit per-request authorisation to consume a surface version's
    # declared extrapolation region. Default stays "reject out of range".
    allow_extrapolation: bool = Field(
        default=False,
        description="Consume vol points inside the bound surface version's "
                    "declared extrapolation limits; ignored/irrelevant for "
                    "in-range points and rejected when the version declares "
                    "no limits",
    )

    # Optional caller correlation id (e.g. portfolio line id); never part of
    # the pricing identity / cache key.
    request_id: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def _validate_resolution_controls(self) -> "PricingRequest":
        # StrictInt already rejects booleans/floats; here we only enforce
        # semantic applicability and mutual exclusivity of the two modes.
        if self.resolution is not None:
            if self.option_type is not OptionType.AMERICAN_PUT:
                raise ValueError(
                    "resolution is supported only for option_type "
                    "american_put (use it for fixed-step binomial pricing)"
                )
            if (self.steps != 500 or self.convergence_tol != 1e-4
                    or self.max_steps != 4000):
                raise ValueError(
                    "resolution (fixed mode) is mutually exclusive with "
                    "steps/convergence_tol/max_steps (converged mode)"
                )
        return self


class PriceResponse(BaseModel):
    price: float
    vol_used: float
    surface_version: int
    method: str
    detail: dict = Field(default_factory=dict)
    cached: bool = False
    # Discrete-resolution audit: which mode ran and how many tree steps were
    # actually used (fixed mode reports exactly the requested resolution).
    resolution_mode: ResolutionMode = ResolutionMode.CONVERGED
    resolution_used: int | None = None
    # Extrapolation audit: ``extrapolated`` is True iff the vol point was
    # actually served from outside the grid support; ``extrapolation`` then
    # records each axis's overshoot relative to the grid and to the version's
    # declared limits (fraction of the declared headroom consumed).
    extrapolated: bool = False
    extrapolation: dict | None = None


class ErrorBody(BaseModel):
    error: str
    message: str
    detail: dict = Field(default_factory=dict)


class ExtrapolationLimits(BaseModel):
    """Region, declared at publish time, inside which a version may serve
    vol points outside its grid support via linear extrapolation.

    The region must contain the whole grid support
    (``min_expiry <= grid min`` etc.). A version without limits forbids
    extrapolation unconditionally; the limits are immutable for the life of
    the version, so historical replays honour exactly what was declared.
    """

    min_expiry: float = Field(description="Lowest servable expiry (>= 0)")
    max_expiry: float = Field(description="Highest servable expiry")
    min_strike: float = Field(gt=0, description="Lowest servable strike (> 0)")
    max_strike: float = Field(gt=0, description="Highest servable strike")


class SurfaceGridPayload(BaseModel):
    """Upload payload for a volatility surface grid.

    ``vols[i][j]`` is the IV for ``expiries[i]`` and ``strikes[j]``.
    Extrapolation is authorised per version only via ``extrapolation_limits``;
    omitting it means the version forbids every out-of-grid query.
    """

    expiries: list[float]
    strikes: list[float]
    vols: list[list[float]]
    spot: float = Field(gt=0)
    rate: float = 0.0
    dividend: float = Field(default=0.0, ge=0)
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


class BatchItem(BaseModel):
    """One entry of a batch: pricing fields only plus an optional caller
    correlation id. Surface binding comes from the enclosing batch."""

    model_config = {"extra": "forbid"}

    id: str | None = Field(default=None, max_length=128)
    option_type: OptionType
    expiry: float
    strike: float
    spot: float | None = Field(default=None, gt=0)
    rate: float | None = Field(default=None)
    dividend: float | None = Field(default=None, ge=0)
    payout: float = Field(default=1.0, gt=0)
    steps: StrictInt = Field(default=500, gt=0, le=20000)
    convergence_tol: float = Field(default=1e-4, gt=0)
    max_steps: StrictInt = Field(default=4000, gt=0, le=200000)
    resolution: StrictInt | None = Field(
        default=None, ge=MIN_RESOLUTION, le=MAX_RESOLUTION)
    target_price: float | None = Field(default=None, gt=0)
    allow_extrapolation: bool = False

    def to_request(self, surface_id: str, version: int) -> PricingRequest:
        return PricingRequest(
            surface_id=surface_id, version=version,
            request_id=self.id, **self.model_dump(exclude={"id"}))

    @model_validator(mode="after")
    def _validate_controls(self) -> "BatchItem":
        if self.resolution is not None:
            if self.option_type is not OptionType.AMERICAN_PUT:
                raise ValueError(
                    "resolution is supported only for option_type "
                    "american_put")
            if (self.steps != 500 or self.convergence_tol != 1e-4
                    or self.max_steps != 4000):
                raise ValueError(
                    "resolution (fixed mode) is mutually exclusive with "
                    "steps/convergence_tol/max_steps (converged mode)")
        return self


class BatchPricingRequest(BaseModel):
    """A whole book's worth of pricing requests bound to ONE surface
    version, evaluated independently and concurrently."""

    surface_id: str = "default"
    version: int | None = Field(
        default=None,
        description="Version shared by every item; None pins the latest "
                    "version once, at submission time, for the whole batch",
    )
    items: list[BatchItem] = Field(min_length=1, max_length=MAX_BATCH_ITEMS)

    @model_validator(mode="after")
    def _unique_item_ids(self) -> "BatchPricingRequest":
        ids = [it.id for it in self.items if it.id is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("item ids must be unique within a batch")
        return self


class BatchItemError(BaseModel):
    code: str
    message: str


class BatchItemResult(BaseModel):
    """One item's outcome. Exactly one of ``result`` / ``error`` is set;
    index/id let callers correlate outcomes with the submitted entries."""

    index: int
    id: str | None = None
    result: PriceResponse | None = None
    error: BatchItemError | None = None


class BatchPricingResponse(BaseModel):
    surface_id: str
    surface_version: int
    items: list[BatchItemResult]
