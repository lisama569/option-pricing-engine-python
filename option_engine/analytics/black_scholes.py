"""Black-Scholes-Merton 闭式解：欧式与现金或无（数字）期权。

约定：
- 标的支付连续股息率 q（Black-Scholes-Merton 推广）。
- vega 以"每 1.00 波动率"计（即未除以 100）。
- theta 以"每日历年"计（年化导数取负后直接返回，即 ∂V/∂T 的相反数，
  表示距离到期减少 1 年的价值变化；文档中给出说明）。
- T == 0 时返回内在价值；该点 Greeks 数学上退化，按 0.0 返回（文档明示）。
"""

from __future__ import annotations

import math

from ..models import Greeks, OptionType
from .normal import norm_cdf, norm_pdf


def _d1_d2(
    spot: float,
    strike: float,
    maturity: float,
    vol: float,
    rate: float,
    dividend: float,
) -> tuple[float, float]:
    sqrt_t = math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend + 0.5 * vol * vol) * maturity
    ) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return d1, d2


def _terminal_value(
    option_type: OptionType,
    spot: float,
    strike: float,
    cash: float | None,
) -> float:
    """到期日（T=0）的解析价值。"""
    if option_type in (OptionType.EUROPEAN_CALL, OptionType.AMERICAN_CALL):
        return max(spot - strike, 0.0)
    if option_type in (OptionType.EUROPEAN_PUT, OptionType.AMERICAN_PUT):
        return max(strike - spot, 0.0)
    assert cash is not None
    if spot == strike:
        # Phi(0) = 0.5 的到期极限：这是解析结论而非近似
        return 0.5 * cash
    if option_type is OptionType.CASH_OR_NOTHING_CALL:
        return cash if spot > strike else 0.0
    return cash if spot < strike else 0.0


def european_price(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    vol: float,
    rate: float,
    dividend: float = 0.0,
    cash: float | None = None,
) -> float:
    """欧式期权闭式价格。"""
    if maturity == 0.0:
        return _terminal_value(option_type, spot, strike, cash)

    d1, d2 = _d1_d2(spot, strike, maturity, vol, rate, dividend)
    disc_r = math.exp(-rate * maturity)
    disc_q = math.exp(-dividend * maturity)

    if option_type is OptionType.EUROPEAN_CALL:
        return spot * disc_q * norm_cdf(d1) - strike * disc_r * norm_cdf(d2)
    if option_type is OptionType.EUROPEAN_PUT:
        return strike * disc_r * norm_cdf(-d2) - spot * disc_q * norm_cdf(-d1)

    # 数字期权：现金或无
    assert cash is not None
    if option_type is OptionType.CASH_OR_NOTHING_CALL:
        return cash * disc_r * norm_cdf(d2)
    if option_type is OptionType.CASH_OR_NOTHING_PUT:
        return cash * disc_r * norm_cdf(-d2)
    raise ValueError(f"unsupported european option_type: {option_type}")


def _vanilla_greeks(
    is_call: bool,
    spot: float,
    strike: float,
    maturity: float,
    vol: float,
    rate: float,
    dividend: float,
) -> Greeks:
    d1, d2 = _d1_d2(spot, strike, maturity, vol, rate, dividend)
    disc_r = math.exp(-rate * maturity)
    disc_q = math.exp(-dividend * maturity)
    pdf_d1 = norm_pdf(d1)
    sqrt_t = math.sqrt(maturity)

    if is_call:
        delta = disc_q * norm_cdf(d1)
        theta = (
            -spot * disc_q * pdf_d1 * vol / (2.0 * sqrt_t)
            - rate * strike * disc_r * norm_cdf(d2)
            + dividend * spot * disc_q * norm_cdf(d1)
        )
        rho = strike * maturity * disc_r * norm_cdf(d2)
    else:
        delta = disc_q * (norm_cdf(d1) - 1.0)
        theta = (
            -spot * disc_q * pdf_d1 * vol / (2.0 * sqrt_t)
            + rate * strike * disc_r * norm_cdf(-d2)
            - dividend * spot * disc_q * norm_cdf(-d1)
        )
        rho = -strike * maturity * disc_r * norm_cdf(-d2)

    gamma = disc_q * pdf_d1 / (spot * vol * sqrt_t)
    vega = spot * disc_q * pdf_d1 * sqrt_t
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def _digital_greeks(
    is_call: bool,
    spot: float,
    strike: float,
    maturity: float,
    vol: float,
    rate: float,
    dividend: float,
    cash: float,
) -> Greeks:
    d1, d2 = _d1_d2(spot, strike, maturity, vol, rate, dividend)
    disc_r = math.exp(-rate * maturity)
    sqrt_t = math.sqrt(maturity)
    pdf_d2 = norm_pdf(d2)
    sign = 1.0 if is_call else -1.0

    # 统一形式：V = C * e^{-rT} * N(sign * d2)，每个希腊字母恰含一个 sign。
    value = cash * disc_r * norm_cdf(d2 if is_call else -d2)
    delta = sign * cash * disc_r * pdf_d2 / (spot * vol * sqrt_t)
    gamma = (
        -sign
        * cash
        * disc_r
        * pdf_d2
        * (d2 + vol * sqrt_t)
        / (spot * spot * vol * vol * maturity)
    )
    vega = -sign * cash * disc_r * pdf_d2 * d1 / vol
    dd2_dt = (
        -d2 / (2.0 * maturity)
        + (rate - dividend) / (vol * sqrt_t)
        - vol / (2.0 * sqrt_t)
    )
    theta = rate * value - sign * cash * disc_r * pdf_d2 * dd2_dt
    rho = (
        -maturity * value
        + sign * cash * disc_r * pdf_d2 * sqrt_t / vol
    )
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


_ZERO_GREEKS = Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)


def european_greeks(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    vol: float,
    rate: float,
    dividend: float = 0.0,
    cash: float | None = None,
) -> Greeks:
    """欧式期权解析 Greeks。T==0 时退化点按 0.0 返回。"""
    if maturity == 0.0:
        return _ZERO_GREEKS.model_copy(deep=True)

    if option_type is OptionType.EUROPEAN_CALL:
        return _vanilla_greeks(True, spot, strike, maturity, vol, rate, dividend)
    if option_type is OptionType.EUROPEAN_PUT:
        return _vanilla_greeks(False, spot, strike, maturity, vol, rate, dividend)
    if option_type is OptionType.CASH_OR_NOTHING_CALL:
        assert cash is not None
        return _digital_greeks(True, spot, strike, maturity, vol, rate, dividend, cash)
    if option_type is OptionType.CASH_OR_NOTHING_PUT:
        assert cash is not None
        return _digital_greeks(False, spot, strike, maturity, vol, rate, dividend, cash)
    raise ValueError(f"unsupported european option_type: {option_type}")
