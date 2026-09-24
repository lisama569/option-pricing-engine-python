"""跨层共享的数据契约（Pydantic 模型）。"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class OptionType(str, Enum):
    EUROPEAN_CALL = "european_call"
    EUROPEAN_PUT = "european_put"
    CASH_OR_NOTHING_CALL = "cash_or_nothing_call"
    CASH_OR_NOTHING_PUT = "cash_or_nothing_put"
    AMERICAN_CALL = "american_call"
    AMERICAN_PUT = "american_put"


class VolSurfaceData(BaseModel):
    """一次曲面发布的不可变内容。"""

    label: str
    as_of: str
    maturities: list[float]
    strikes: list[float]
    vols: list[list[float]]
    risk_free_rate: float = 0.0
    dividend_yield: float = 0.0


class SurfaceVersionInfo(BaseModel):
    version: int
    label: str
    as_of: str
    published_at: datetime
    num_maturities: int
    num_strikes: int


class PricingRequest(BaseModel):
    # 数值合法性（正数/非负）由服务层统一校验并返回 INVALID_MODEL_PARAMS，
    # 因此契约层不施加 Field 约束，避免在 HTTP 层产生无 error_code 的 422。
    option_type: OptionType
    spot: float
    strike: float
    maturity: float
    cash: float | None = None
    tree_steps: int | None = None
    surface_version: int | None = None
    request_id: str | None = None


class Greeks(BaseModel):
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class PricingResult(BaseModel):
    price: float
    greeks: Greeks
    implied_vol: float
    option_type: OptionType
    spot: float
    strike: float
    maturity: float
    surface_version: int
    surface_label: str
    model: str
    request_id: str | None = None
    computed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class IVSolveRequest(BaseModel):
    option_type: OptionType
    spot: float
    strike: float
    maturity: float
    target_price: float
    surface_version: int | None = None
    request_id: str | None = None


class IVSolveResult(BaseModel):
    implied_vol: float
    iterations: int
    surface_version: int
    request_id: str | None = None


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus
    request: PricingRequest
    result: PricingResult | None = None
    error: dict | None = None
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
