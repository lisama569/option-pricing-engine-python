"""可区分的失败原因：错误码 + 自定义异常。"""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    INVALID_MODEL_PARAMS = "INVALID_MODEL_PARAMS"
    SURFACE_OUT_OF_RANGE = "SURFACE_OUT_OF_RANGE"
    SOLVER_DID_NOT_CONVERGE = "SOLVER_DID_NOT_CONVERGE"
    UNKNOWN_OPTION_TYPE = "UNKNOWN_OPTION_TYPE"
    VERSION_NOT_FOUND = "VERSION_NOT_FOUND"
    SURFACE_DATA_INVALID = "SURFACE_DATA_INVALID"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    JOB_ALREADY_DONE = "JOB_ALREADY_DONE"
    JOB_CANCELLED = "JOB_CANCELLED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class PricingEngineError(Exception):
    """业务错误：携带稳定可判定的错误码。"""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict:
        return {"error_code": self.code.value, "message": self.message}
