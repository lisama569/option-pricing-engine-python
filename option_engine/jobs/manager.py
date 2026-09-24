"""并发任务管理器。

规则：
- 定价在独立线程池中执行，耗时计算不阻塞 HTTP 事件循环与其他任务；
- 任务状态表受锁保护，状态迁移只能是
  pending -> running -> succeeded/failed，或 pending -> cancelled；
- 取消仅对尚未开始的任务有效（线程无法安全强杀，运行中任务跑完为止，
  其失败不影响其他任务与服务整体）；
- 任何任务异常都被捕获并落成 failed + 结构化错误，不外泄、不拖垮服务。
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..errors import ErrorCode, PricingEngineError
from ..models import JobRecord, JobStatus, PricingRequest, PricingResult
from ..pricing.service import PricingService

logger = logging.getLogger("option_engine.jobs")


@dataclass
class _JobEntry:
    record: JobRecord
    cancel_event: threading.Event = field(default_factory=threading.Event)
    future: object = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobManager:
    def __init__(self, service: PricingService, max_workers: int = 4) -> None:
        self._service = service
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="pricing-worker"
        )
        self._lock = threading.RLock()
        self._entries: dict[str, _JobEntry] = {}

    def submit(self, request: PricingRequest) -> JobRecord:
        job_id = uuid.uuid4().hex
        record = JobRecord(
            job_id=job_id,
            status=JobStatus.PENDING,
            request=request,
            submitted_at=_now(),
        )
        entry = _JobEntry(record=record)
        with self._lock:
            self._entries[job_id] = entry
            entry.future = self._pool.submit(self._run, job_id)
        logger.info(
            "job submitted: job_id=%s option_type=%s surface_version=%s "
            "spot=%s strike=%s maturity=%s tree_steps=%s",
            job_id,
            request.option_type.value,
            request.surface_version,
            request.spot,
            request.strike,
            request.maturity,
            request.tree_steps,
        )
        return record.model_copy(deep=True)

    def _run(self, job_id: str) -> None:
        with self._lock:
            entry = self._entries[job_id]
            if entry.cancel_event.is_set():
                entry.record.status = JobStatus.CANCELLED
                entry.record.finished_at = _now()
                logger.info("job cancelled before start: job_id=%s", job_id)
                return
            entry.record.status = JobStatus.RUNNING
            entry.record.started_at = _now()
            request = entry.record.request

        try:
            result: PricingResult = self._service.price(request)
            with self._lock:
                entry = self._entries[job_id]
                entry.record.status = JobStatus.SUCCEEDED
                entry.record.result = result
                entry.record.finished_at = _now()
            logger.info(
                "job succeeded: job_id=%s surface_version=%s price=%.10f",
                job_id,
                result.surface_version,
                result.price,
            )
        except PricingEngineError as exc:
            self._fail(job_id, exc.to_dict(), exc)
        except Exception as exc:  # 防御性：未知异常同样落成结构化失败
            logger.exception("job failed with unexpected error: job_id=%s", job_id)
            self._fail(
                job_id,
                {"error_code": ErrorCode.INTERNAL_ERROR.value, "message": str(exc)},
                exc,
            )

    def _fail(self, job_id: str, error: dict, exc: Exception) -> None:
        with self._lock:
            entry = self._entries[job_id]
            entry.record.status = JobStatus.FAILED
            entry.record.error = error
            entry.record.finished_at = _now()
        logger.warning(
            "job failed: job_id=%s error_code=%s message=%s",
            job_id,
            error.get("error_code"),
            error.get("message"),
        )

    def get_job(self, job_id: str) -> JobRecord:
        with self._lock:
            entry = self._entries.get(job_id)
            if entry is None:
                raise PricingEngineError(
                    ErrorCode.JOB_NOT_FOUND, f"job not found: {job_id}"
                )
            return entry.record.model_copy(deep=True)

    def list_jobs(self) -> list[JobRecord]:
        with self._lock:
            return [e.record.model_copy(deep=True) for e in self._entries.values()]

    def cancel(self, job_id: str) -> JobRecord:
        with self._lock:
            entry = self._entries.get(job_id)
            if entry is None:
                raise PricingEngineError(
                    ErrorCode.JOB_NOT_FOUND, f"job not found: {job_id}"
                )
            status = entry.record.status
            if status is JobStatus.PENDING:
                entry.cancel_event.set()
                if entry.future is not None:
                    entry.future.cancel()
                entry.record.status = JobStatus.CANCELLED
                entry.record.finished_at = _now()
                logger.info("job cancelled: job_id=%s", job_id)
                return entry.record.model_copy(deep=True)
        raise PricingEngineError(
            ErrorCode.JOB_ALREADY_DONE,
            f"job {job_id} is in status '{status.value}' and cannot be cancelled; "
            "only pending jobs can be cancelled",
        )

    def shutdown(self, wait: bool = False) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=not wait)
