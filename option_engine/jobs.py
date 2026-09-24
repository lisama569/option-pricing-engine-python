"""Asynchronous pricing job manager.

Long computations run on a bounded ThreadPoolExecutor so they never block
the event loop or other requests. Job state transitions are guarded by a
lock; a failed/cancelled job leaves the manager fully usable. Cancellation
is cooperative: pricers poll a threading.Event at iteration boundaries.
"""
from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .errors import JobCancelledError, PricingError
from .models import JobResponse, JobStatus, PriceResponse, PricingRequest
from .service import PricingService


@dataclass
class _Job:
    job_id: str
    status: JobStatus = JobStatus.PENDING
    surface_version: int | None = None
    result: PriceResponse | None = None
    error_code: str | None = None
    error_message: str | None = None
    request: PricingRequest | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    bound_version: int | None = None


class JobNotFoundError(PricingError):
    code = "job_not_found"


class JobManager:
    def __init__(self, service: PricingService, max_workers: int = 4) -> None:
        self._service = service
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="pricing"
        )
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()

    def submit(self, req: PricingRequest) -> _Job:
        job = _Job(job_id=uuid.uuid4().hex, request=req)
        with self._lock:
            self._jobs[job.job_id] = job
        self._pool.submit(self._run, job)
        return job

    def _run(self, job: _Job) -> None:
        with job.lock:
            if job.status != JobStatus.PENDING:
                return  # cancelled before it started
            job.status = JobStatus.RUNNING
        try:
            # Pin the surface version before any heavy work so the bound
            # version survives even if pricing fails afterwards.
            snap = self._service.store.get(job.request.surface_id,
                                           job.request.version)
            with job.lock:
                job.bound_version = snap.version
                job.surface_version = snap.version
            resp = self._service.price(job.request, cancel_event=job.cancel_event)
            with job.lock:
                if job.cancel_event.is_set():
                    job.status = JobStatus.CANCELLED
                    job.error_code = JobCancelledError.code
                    job.error_message = "job cancelled"
                    return
                job.result = resp
                job.surface_version = resp.surface_version
                job.status = JobStatus.SUCCEEDED
        except PricingError as exc:
            with job.lock:
                if job.cancel_event.is_set() or isinstance(exc, JobCancelledError):
                    job.status = JobStatus.CANCELLED
                    job.error_code = JobCancelledError.code
                    job.error_message = str(exc) or "job cancelled"
                else:
                    job.status = JobStatus.FAILED
                    job.error_code = exc.code
                    job.error_message = str(exc)
        except Exception as exc:  # defensive: never kill the worker thread
            with job.lock:
                job.status = JobStatus.FAILED
                job.error_code = "internal_error"
                job.error_message = repr(exc)

    def cancel(self, job_id: str) -> _Job:
        job = self._get(job_id)
        with job.lock:
            if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
                job.cancel_event.set()
                terminal = job.status == JobStatus.PENDING
                if terminal:
                    job.status = JobStatus.CANCELLED
                    job.error_code = JobCancelledError.code
                    job.error_message = "job cancelled before execution"
            return job

    def _get(self, job_id: str) -> _Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"job '{job_id}' does not exist")
        return job

    def status(self, job_id: str) -> _Job:
        return self._get(job_id)

    @staticmethod
    def to_response(job: _Job) -> JobResponse:
        with job.lock:
            return JobResponse(
                job_id=job.job_id,
                status=job.status,
                surface_version=job.surface_version,
                result=job.result,
                error_code=job.error_code,
                error_message=job.error_message,
            )

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
