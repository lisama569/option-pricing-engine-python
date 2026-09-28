"""FastAPI application: pricing, surface versioning and async job routes.

All state lives in module-level singletons initialised from bundled local
data, so ``uvicorn main:app`` runs the service fully offline.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .data import load_bundled_surfaces
from .jobs import JobManager, JobNotFoundError
from .errors import (
    ExtrapolationNotAuthorizedError,
    InvalidParameterError,
    PricingError,
    SurfaceNotFoundError,
    SurfaceOutOfRangeError,
    SolverNonConvergenceError,
)
from .models import (
    BatchPricingRequest,
    BatchPricingResponse,
    ErrorBody,
    ExtrapolationLimits,
    JobResponse,
    PriceResponse,
    PricingRequest,
    SurfaceGridPayload,
    SurfaceVersionInfo,
)
from .service import PricingService
from .surface import SurfaceStore

ERROR_STATUS = {
    InvalidParameterError.code: 422,
    SurfaceOutOfRangeError.code: 422,
    ExtrapolationNotAuthorizedError.code: 422,
    SurfaceNotFoundError.code: 404,
    SolverNonConvergenceError.code: 422,
    JobNotFoundError.code: 404,
    "internal_error": 500,
}


def _error_response(exc: PricingError) -> JSONResponse:
    status = ERROR_STATUS.get(exc.code, 500)
    return JSONResponse(
        status_code=status,
        content=ErrorBody(error=exc.code, message=str(exc)).model_dump(),
    )


def create_app() -> tuple[FastAPI, SurfaceStore, PricingService, JobManager]:
    """Build a fully wired application instance from bundled local data.

    The factory keeps construction explicit and lets tests spin up an
    isolated instance; uvicorn uses the module-level default instance.
    """
    surface_store = load_bundled_surfaces()
    pricing_service = PricingService(surface_store)
    job_manager = JobManager(pricing_service)
    application = FastAPI(title="Option Pricing Engine", version="0.1.0")

    @application.exception_handler(PricingError)
    def pricing_error_handler(request, exc: PricingError):  # noqa: ANN001
        return _error_response(exc)

    @application.exception_handler(RequestValidationError)
    def validation_error_handler(request, exc: RequestValidationError):  # noqa: ANN001
        # Unified, distinguishable failure body for illegal request parameters.
        # Strip non-JSON-serialisable error context (e.g. ValueError ctx).
        errors = []
        for err in exc.errors():
            clean = {k: v for k, v in err.items() if k != "ctx"}
            if "ctx" in err:
                clean["ctx"] = {k: str(v) for k, v in err["ctx"].items()}
            errors.append(clean)
        return JSONResponse(
            status_code=422,
            content=ErrorBody(
                error=InvalidParameterError.code,
                message="request parameter validation failed",
                detail={"errors": errors},
            ).model_dump(mode="json"),
        )

    @application.get("/health")
    def health():
        return {"status": "ok"}

    @application.post("/pricing/price", response_model=PriceResponse)
    def price_sync(req: PricingRequest):
        return pricing_service.price(req)

    @application.post("/pricing/batch", response_model=BatchPricingResponse)
    def price_batch(req: BatchPricingRequest):
        # One version for the whole batch; per-item failures never fail the
        # HTTP request itself -- they live in each item's error slot.
        return pricing_service.price_batch(req)

    @application.post("/surfaces/{surface_id}/versions",
                      response_model=SurfaceVersionInfo)
    def publish_surface(surface_id: str, payload: SurfaceGridPayload):
        snap = surface_store.publish(surface_id, payload)
        return _version_info(snap)

    @application.get("/surfaces/{surface_id}/versions",
                     response_model=list[SurfaceVersionInfo])
    def list_surface_versions(surface_id: str):
        return [_version_info(s) for s in surface_store.list_versions(surface_id)]

    @application.get("/surfaces/{surface_id}/versions/{version}",
                     response_model=SurfaceVersionInfo)
    def get_surface_version(surface_id: str, version: int):
        return _version_info(surface_store.get(surface_id, version))

    @application.get("/surfaces/{surface_id}/versions/{version}/vol")
    def query_vol(surface_id: str, version: int, expiry: float, strike: float,
                  allow_extrapolation: bool = False):
        snap = surface_store.get(surface_id, version)
        query = snap.query_vol(expiry, strike,
                               allow_extrapolation=allow_extrapolation)
        return {"surface_id": surface_id, "version": version,
                "expiry": expiry, "strike": strike,
                "vol": query.vol,
                "extrapolated": query.audit is not None,
                "extrapolation": query.audit}

    @application.post("/jobs", response_model=JobResponse, status_code=202)
    def submit_job(req: PricingRequest):
        return JobManager.to_response(job_manager.submit(req))

    @application.get("/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str):
        return JobManager.to_response(job_manager.status(job_id))

    @application.post("/jobs/{job_id}/cancel", response_model=JobResponse)
    def cancel_job(job_id: str):
        return JobManager.to_response(job_manager.cancel(job_id))

    return application, surface_store, pricing_service, job_manager


def _version_info(snap) -> SurfaceVersionInfo:
    limits = snap.extrapolation_limits
    return SurfaceVersionInfo(
        surface_id=snap.surface_id, version=snap.version,
        created_at=snap.created_at, spot=snap.spot, rate=snap.rate,
        dividend=snap.dividend,
        extrapolation_limits=ExtrapolationLimits(**limits.as_dict())
        if limits is not None else None,
        expiries=list(snap.expiries), strikes=list(snap.strikes),
    )


app, store, service, jobs = create_app()
