"""Request/response models and enums for the pricing service."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


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

    # Binomial American solver controls.
    steps: int = Field(default=500, gt=0, le=20000)
    convergence_tol: float = Field(default=1e-4, gt=0)
    max_steps: int = Field(default=4000, gt=0, le=200000)

    # When set, solve for implied volatility matching this market price
    # (European vanilla only) instead of pricing from the surface vol.
    target_price: float | None = Field(default=None, gt=0)


class PriceResponse(BaseModel):
    price: float
    vol_used: float
    surface_version: int
    method: str
    detail: dict = Field(default_factory=dict)
    cached: bool = False


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
    allow_extrapolation: bool = False


class SurfaceVersionInfo(BaseModel):
    surface_id: str
    version: int
    created_at: str
    spot: float
    rate: float
    dividend: float
    allow_extrapolation: bool
    expiries: list[float]
    strikes: list[float]


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    surface_version: int | None = None
    result: PriceResponse | None = None
    error_code: str | None = None
    error_message: str | None = None
