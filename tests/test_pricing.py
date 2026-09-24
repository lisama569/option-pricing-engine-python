"""定价编排服务：失败原因可区分、版本绑定、结论一致可复现。"""

from __future__ import annotations

import logging

import pytest

from option_engine.errors import ErrorCode, PricingEngineError
from option_engine.models import (
    IVSolveRequest,
    OptionType as OT,
    PricingRequest,
    VolSurfaceData,
)
from option_engine.pricing.service import PricingService
from option_engine.surface.store import SurfaceStore

logger = logging.getLogger("test.pricing")


def _req(**over) -> PricingRequest:
    base = dict(option_type=OT.EUROPEAN_CALL, spot=100.0, strike=100.0,
                maturity=0.5, surface_version=1)
    base.update(over)
    return PricingRequest(**base)


def test_price_matches_benchmark_on_flat_surface(flat_surface):
    """平坦 20% 曲面上的 ATM 1 年期看涨，必须等于 BSM 解析基准。"""
    store = SurfaceStore()
    store.publish(VolSurfaceData(**flat_surface))
    svc = PricingService(store)
    res = svc.price(_req(maturity=1.0))
    bench = 10.450583572185565
    logger.info(
        "输入 S=100 K=100 T=1 曲面版本=%s(%s) 插值vol=%.4f 模型=%s "
        "价格=%.12f 解析基准=%.12f 判定依据=平坦曲面取点严格等于0.2",
        res.surface_version, res.surface_label, res.implied_vol, res.model,
        res.price, bench,
    )
    assert res.price == pytest.approx(bench, abs=1e-10)
    assert res.model == "black_scholes_merton"


def test_invalid_params_out_of_range_missing_version_are_distinct(service):
    cases = [
        (_req(spot=0.0), ErrorCode.INVALID_MODEL_PARAMS),
        (_req(strike=80.0), ErrorCode.SURFACE_OUT_OF_RANGE),
        (_req(maturity=5.0), ErrorCode.SURFACE_OUT_OF_RANGE),
        (_req(surface_version=99), ErrorCode.VERSION_NOT_FOUND),
        (_req(option_type=OT.AMERICAN_PUT), ErrorCode.INVALID_MODEL_PARAMS),
        (_req(option_type=OT.AMERICAN_PUT, tree_steps=0),
         ErrorCode.INVALID_MODEL_PARAMS),
        (_req(option_type=OT.CASH_OR_NOTHING_CALL),
         ErrorCode.INVALID_MODEL_PARAMS),
        (_req(option_type=OT.CASH_OR_NOTHING_CALL, cash=-1.0),
         ErrorCode.INVALID_MODEL_PARAMS),
    ]
    for req, expected in cases:
        with pytest.raises(PricingEngineError) as exc:
            service.price(req)
        logger.info("输入 %s -> %s（期望 %s），依据=%s",
                    req.model_dump(), exc.value.code.value, expected.value,
                    exc.value.message)
        assert exc.value.code is expected


def test_same_inputs_same_version_give_identical_result(service):
    res1 = service.price(_req())
    res2 = service.price(_req())
    assert res1.price == res2.price
    assert res1.greeks == res2.greeks


def test_bound_versions_replay_and_latest_differ(service):
    pinned = service.price(_req(surface_version=1))
    latest = service.price(_req(surface_version=2))
    replayed = service.price(_req(surface_version=1))
    logger.info("历史版本 v1 价格=%.10f；当前 v2 价格=%.10f；v1 回放=%.10f",
                pinned.price, latest.price, replayed.price)
    assert pinned.price != latest.price
    assert replayed.price == pinned.price
    assert service.price(_req(surface_version=None)).surface_version == 2


def test_american_routed_to_tree(service):
    res = service.price(_req(option_type=OT.AMERICAN_PUT, maturity=1.0,
                             tree_steps=400))
    assert res.model == "crr_binomial_tree"
    again = service.price(_req(option_type=OT.AMERICAN_PUT, maturity=1.0,
                               tree_steps=400))
    assert res.price == again.price  # 确定性


def test_terminal_expiry_does_not_require_surface_point(service):
    res = service.price(_req(maturity=0.0, strike=100.0, spot=105.0))
    assert res.price == 5.0
    assert res.implied_vol == 0.0


def test_iv_solve_roundtrip_and_failures(service):
    ref = service.price(_req())
    ok = service.solve_iv(IVSolveRequest(
        option_type=OT.EUROPEAN_CALL, spot=100, strike=100, maturity=0.5,
        target_price=ref.price, surface_version=1))
    assert ok.implied_vol == pytest.approx(ref.implied_vol, abs=1e-8)

    bad_cases = [
        (IVSolveRequest(option_type=OT.CASH_OR_NOTHING_CALL, spot=100,
                        strike=100, maturity=0.5, target_price=5.0,
                        surface_version=1),
         ErrorCode.INVALID_MODEL_PARAMS),
        (IVSolveRequest(option_type=OT.EUROPEAN_CALL, spot=100, strike=100,
                        maturity=0.5, target_price=1e9, surface_version=1),
         ErrorCode.SOLVER_DID_NOT_CONVERGE),
        (IVSolveRequest(option_type=OT.EUROPEAN_CALL, spot=100, strike=100,
                        maturity=0.5, target_price=0.0, surface_version=1),
         ErrorCode.INVALID_MODEL_PARAMS),
        (IVSolveRequest(option_type=OT.EUROPEAN_CALL, spot=100, strike=100,
                        maturity=0.0, target_price=5.0, surface_version=1),
         ErrorCode.INVALID_MODEL_PARAMS),
    ]
    for req, expected in bad_cases:
        with pytest.raises(PricingEngineError) as exc:
            service.solve_iv(req)
        logger.info("IV 输入 target=%s -> %s（期望 %s）",
                    getattr(req, "target_price", None),
                    exc.value.code.value, expected.value)
        assert exc.value.code is expected
