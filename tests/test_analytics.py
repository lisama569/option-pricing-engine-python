"""分析层与解析基准的对照测试。"""

from __future__ import annotations

import logging
import math

import pytest

from option_engine.analytics.american import american_greeks, american_price
from option_engine.analytics.black_scholes import european_greeks, european_price
from option_engine.analytics.implied_vol import implied_volatility
from option_engine.analytics.normal import norm_cdf
from option_engine.errors import ErrorCode, PricingEngineError
from option_engine.models import OptionType as OT

logger = logging.getLogger("test.analytics")


def test_norm_cdf_tabulated_values():
    cases = {0.0: 0.5, 1.0: 0.8413447460685429, 1.96: 0.9750021048517795,
             -1.96: 0.024997895148220435, 3.0: 0.9986501019683699}
    for x, want in cases.items():
        got = norm_cdf(x)
        logger.info("Phi(%s)=%.15f vs 解析基准 %.15f (|err|=%.2e)",
                    x, got, want, abs(got - want))
        assert abs(got - want) < 1e-12


@pytest.mark.parametrize(
    "option_type,benchmark",
    [(OT.EUROPEAN_CALL, 10.450583572185565),
     (OT.EUROPEAN_PUT, 5.573526022256971)],
)
def test_bsm_matches_analytical_benchmark(option_type, benchmark):
    """经典基准：S=K=100,T=1,r=5%,q=0,sigma=20%（Hull 教材数值）。"""
    got = european_price(option_type, 100, 100, 1.0, 0.2, 0.05, 0.0)
    logger.info(
        "输入 option_type=%s S=100 K=100 T=1 r=0.05 q=0 sigma=0.2 "
        "模型=BSM 价格=%.12f 解析基准=%.12f",
        option_type.value, got, benchmark,
    )
    assert got == pytest.approx(benchmark, abs=1e-10)


def test_put_call_parity_with_dividend():
    S, K, T, v, r, q = 100, 95, 0.75, 0.28, 0.045, 0.018
    c = european_price(OT.EUROPEAN_CALL, S, K, T, v, r, q)
    p = european_price(OT.EUROPEAN_PUT, S, K, T, v, r, q)
    residual = c - p - (S * math.exp(-q * T) - K * math.exp(-r * T))
    logger.info("买卖权平价残差=%.3e（判定依据：理论应为 0）", residual)
    assert abs(residual) < 1e-10


def test_digital_call_put_sum():
    S, K, T, v, r, cash = 100, 100, 1.0, 0.2, 0.05, 10.0
    dc = european_price(OT.CASH_OR_NOTHING_CALL, S, K, T, v, r, 0.0, cash)
    dp = european_price(OT.CASH_OR_NOTHING_PUT, S, K, T, v, r, 0.0, cash)
    assert dc + dp == pytest.approx(cash * math.exp(-r * T), abs=1e-12)


@pytest.mark.parametrize(
    "option_type,kwargs",
    [(OT.EUROPEAN_CALL, {}), (OT.EUROPEAN_PUT, {}),
     (OT.CASH_OR_NOTHING_CALL, {"cash": 10.0}),
     (OT.CASH_OR_NOTHING_PUT, {"cash": 10.0})],
)
def test_greeks_match_finite_differences(option_type, kwargs):
    S, K, T, v, r, q = 101, 100, 0.6, 0.27, 0.04, 0.015

    def price(s=S, k=K, t=T, vol=v, rate=r, div=q):
        return european_price(option_type, s, k, t, vol, rate, div, **kwargs)

    g = european_greeks(option_type, S, K, T, v, r, q, **kwargs)
    hs, hv, ht, hr = 1e-3, 1e-5, 1e-5, 1e-5
    fds = {
        "delta": (price(s=S + hs) - price(s=S - hs)) / (2 * hs),
        "gamma": (price(s=S + hs) - 2 * price() + price(s=S - hs)) / hs**2,
        "vega": (price(vol=v + hv) - price(vol=v - hv)) / (2 * hv),
        "theta": -(price(t=T + ht) - price(t=T - ht)) / (2 * ht),
        "rho": (price(rate=r + hr) - price(rate=r - hr)) / (2 * hr),
    }
    for name, fd in fds.items():
        analytic = getattr(g, name)
        tol = 2e-3 * (1 + abs(fd))
        logger.info("%s %s: 解析=%.8f 差分=%.8f 判定依据|差|<%.1e",
                    option_type.value, name, analytic, fd, tol)
        assert analytic == pytest.approx(fd, abs=tol)


def test_american_no_dividend_call_equals_european():
    """无股息美式看涨不应提前行权，二叉树收敛到欧式价格。"""
    bench = european_price(OT.EUROPEAN_CALL, 100, 100, 1.0, 0.2, 0.05)
    got = american_price(True, 100, 100, 1.0, 0.2, 0.05, 0.0, 2000)
    assert got == pytest.approx(bench, abs=2e-3)


def test_american_early_exercise_premium_positive():
    ap = american_price(False, 100, 100, 1.0, 0.2, 0.05, 0.0, 1000)
    ep = european_price(OT.EUROPEAN_PUT, 100, 100, 1.0, 0.2, 0.05)
    logger.info("美式看跌 %.6f vs 欧式 %.6f，提前行权溢价=%.6f", ap, ep, ap - ep)
    assert ap > ep


def test_american_greeks_deterministic():
    a = american_greeks(False, 100, 100, 1.0, 0.2, 0.05, 0.01, 300)
    b = american_greeks(False, 100, 100, 1.0, 0.2, 0.05, 0.01, 300)
    assert a == b


@pytest.mark.parametrize("true_vol", [0.08, 0.2, 0.45, 1.2])
def test_implied_vol_roundtrip(true_vol):
    S, K, T, r = 100, 100, 0.8, 0.04
    price = european_price(OT.EUROPEAN_CALL, S, K, T, true_vol, r)
    iv, iters = implied_volatility(price, True, S, K, T, r)
    logger.info("IV 反解：真实=%.4f 解=%.10f 迭代=%d", true_vol, iv, iters)
    assert iv == pytest.approx(true_vol, abs=1e-7)


@pytest.mark.parametrize("bad_price", [0.0, -5.0])
def test_implied_vol_nonpositive_price_rejected(bad_price):
    with pytest.raises(PricingEngineError) as exc:
        implied_volatility(bad_price, True, 100, 100, 1.0, 0.05)
    assert exc.value.code is ErrorCode.SOLVER_DID_NOT_CONVERGE
