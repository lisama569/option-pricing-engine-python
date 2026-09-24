"""并发任务管理：状态机、慢任务不阻塞、失败隔离、取消。"""

from __future__ import annotations

import logging
import time

import pytest

from option_engine.models import JobStatus, OptionType as OT, PricingRequest
from option_engine.errors import ErrorCode, PricingEngineError

logger = logging.getLogger("test.jobs")


def _american(steps: int, version: int = 1) -> PricingRequest:
    return PricingRequest(option_type=OT.AMERICAN_PUT, spot=100, strike=100,
                          maturity=1.0, tree_steps=steps, surface_version=version)


def _wait(jm, job_id, timeout: float = 20.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        rec = jm.get_job(job_id)
        if rec.status in (JobStatus.SUCCEEDED, JobStatus.FAILED,
                          JobStatus.CANCELLED):
            return rec.status.value
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def test_concurrent_identical_jobs_single_deterministic_price(job_manager):
    jobs = [job_manager.submit(_american(500)) for _ in range(12)]
    statuses = [_wait(job_manager, j.job_id) for j in jobs]
    assert statuses == ["succeeded"] * 12
    prices = {job_manager.get_job(j.job_id).result.price for j in jobs}
    logger.info("12 个并发相同输入任务价格集合=%s（判定依据：集合大小必须为 1）",
                prices)
    assert len(prices) == 1


def test_slow_job_does_not_block_fast_job(job_manager):
    slow = job_manager.submit(_american(900))
    fast = job_manager.submit(PricingRequest(
        option_type=OT.EUROPEAN_CALL, spot=100, strike=100, maturity=0.5,
        surface_version=2))
    t0 = time.time()
    assert _wait(job_manager, fast.job_id, timeout=10) == "succeeded"
    elapsed = time.time() - t0
    logger.info("快速任务耗时 %.3fs；此时慢任务状态=%s",
                elapsed, job_manager.get_job(slow.job_id).status.value)
    assert elapsed < 5.0
    assert _wait(job_manager, slow.job_id) == "succeeded"


def test_failed_job_is_structured_and_service_continues(job_manager):
    bad = job_manager.submit(PricingRequest(
        option_type=OT.EUROPEAN_CALL, spot=100, strike=50, maturity=0.5,
        surface_version=1))
    assert _wait(job_manager, bad.job_id) == "failed"
    rec = job_manager.get_job(bad.job_id)
    logger.info("失败任务错误码=%s 信息=%s（与参数非法/版本缺失可区分）",
                rec.error["error_code"], rec.error["message"])
    assert rec.error["error_code"] == ErrorCode.SURFACE_OUT_OF_RANGE.value

    good = job_manager.submit(PricingRequest(
        option_type=OT.EUROPEAN_CALL, spot=100, strike=100, maturity=0.5,
        surface_version=1))
    assert _wait(job_manager, good.job_id) == "succeeded"
    assert job_manager.get_job(good.job_id).result.price > 0


def test_cancel_pending_job(job_manager):
    # 用重任务占满全部工作线程，制造确定排队中的 pending 任务
    blockers = [job_manager.submit(_american(1500)) for _ in range(8)]
    victim = job_manager.submit(PricingRequest(
        option_type=OT.EUROPEAN_CALL, spot=100, strike=100, maturity=0.5,
        surface_version=1))
    # 取消必须在受害者被调度前完成
    assert job_manager.cancel(victim.job_id).status is JobStatus.CANCELLED
    assert _wait(job_manager, victim.job_id, timeout=5) == "cancelled"
    with pytest.raises(PricingEngineError) as exc:
        job_manager.cancel(victim.job_id)
    assert exc.value.code is ErrorCode.JOB_ALREADY_DONE
    for b in blockers:
        assert _wait(job_manager, b.job_id) == "succeeded"


def test_job_not_found(job_manager):
    with pytest.raises(PricingEngineError) as exc:
        job_manager.get_job("missing")
    assert exc.value.code is ErrorCode.JOB_NOT_FOUND
