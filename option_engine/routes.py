"""HTTP 路由：曲面查询/更新、同步定价、IV 反解、异步任务。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .models import (
    IVSolveRequest,
    JobRecord,
    PricingRequest,
    PricingResult,
    SurfaceVersionInfo,
    VolSurfaceData,
)
from .errors import ErrorCode, PricingEngineError
from .jobs.manager import JobManager
from .pricing.service import PricingService
from .surface.store import SurfaceStore

# 业务错误码 -> HTTP 状态码
_HTTP_STATUS = {
    ErrorCode.INVALID_MODEL_PARAMS: 422,
    ErrorCode.SURFACE_OUT_OF_RANGE: 422,
    ErrorCode.SOLVER_DID_NOT_CONVERGE: 422,
    ErrorCode.VERSION_NOT_FOUND: 404,
    ErrorCode.SURFACE_DATA_INVALID: 422,
    ErrorCode.JOB_NOT_FOUND: 404,
    ErrorCode.JOB_ALREADY_DONE: 409,
    ErrorCode.JOB_CANCELLED: 409,
    ErrorCode.UNKNOWN_OPTION_TYPE: 422,
    ErrorCode.INTERNAL_ERROR: 500,
}


def _raise_http(exc: PricingEngineError) -> None:
    raise HTTPException(
        status_code=_HTTP_STATUS.get(exc.code, 500),
        detail=exc.to_dict(),
    )


def build_router(
    store: SurfaceStore, service: PricingService, jobs: JobManager
) -> APIRouter:
    router = APIRouter(prefix="/api/v1")

    @router.get("/health")
    def health() -> dict:
        latest = store.latest_version()
        return {
            "status": "ok",
            "latest_surface_version": latest,
            "num_versions": len(store.list_versions()),
        }

    @router.post("/surfaces", response_model=SurfaceVersionInfo, status_code=201)
    def publish_surface(data: VolSurfaceData) -> SurfaceVersionInfo:
        try:
            snapshot = store.publish(data)
        except PricingEngineError as exc:
            _raise_http(exc)
        return snapshot.to_info()  # type: ignore[return-value]

    @router.get("/surfaces", response_model=list[SurfaceVersionInfo])
    def list_surfaces() -> list[SurfaceVersionInfo]:
        return store.list_versions()

    @router.get("/surfaces/{version}", response_model=SurfaceVersionInfo)
    def get_surface(version: int) -> SurfaceVersionInfo:
        try:
            return store.get(version).to_info()
        except PricingEngineError as exc:
            _raise_http(exc)

    @router.post("/pricing/price", response_model=PricingResult)
    def price(request: PricingRequest) -> PricingResult:
        try:
            return service.price(request)
        except PricingEngineError as exc:
            _raise_http(exc)

    @router.post("/pricing/implied-vol")
    def implied_vol(request: IVSolveRequest) -> dict:
        try:
            result = service.solve_iv(request)
        except PricingEngineError as exc:
            _raise_http(exc)
        return result.model_dump()

    @router.post("/jobs", response_model=JobRecord, status_code=202)
    def submit_job(request: PricingRequest) -> JobRecord:
        return jobs.submit(request)

    @router.get("/jobs", response_model=list[JobRecord])
    def list_jobs() -> list[JobRecord]:
        return jobs.list_jobs()

    @router.get("/jobs/{job_id}", response_model=JobRecord)
    def get_job(job_id: str) -> JobRecord:
        try:
            return jobs.get_job(job_id)
        except PricingEngineError as exc:
            _raise_http(exc)

    @router.post("/jobs/{job_id}/cancel", response_model=JobRecord)
    def cancel_job(job_id: str) -> JobRecord:
        try:
            return jobs.cancel(job_id)
        except PricingEngineError as exc:
            _raise_http(exc)

    return router
