"""隐含波动率反解：Brent 法为主、二分法回退；失败显式报错。

价格关于波动率在 (0, ∞) 上连续且严格单调（香草期权），因此：
- 目标价必须落在 (内在价值下界, σ_max 对应价格上界) 之间，否则无解；
- 反演在有根区间 [1e-8, 5.0]（即 0.00001% ~ 500%）内进行；
- 100 次迭代后残差仍大于容差则抛出 SOLVER_DID_NOT_CONVERGE。
"""

from __future__ import annotations

from collections.abc import Callable

from ..errors import ErrorCode, PricingEngineError

VOL_LOWER = 1e-8
VOL_UPPER = 5.0
MAX_ITERATIONS = 100
PRICE_TOL = 1e-10


def _brent_solve(f: Callable[[float], float], a: float, b: float) -> tuple[float, int]:
    """求 f(x)=0，f(a) 与 f(b) 异号。返回 (根, 迭代次数)。"""
    fa = f(a)
    fb = f(b)
    if fa == 0.0:
        return a, 0
    if fb == 0.0:
        return b, 0

    # 保证 fa < 0
    if fa > 0.0:
        a, b = b, a
        fa, fb = fb, fa

    c, fc = a, fa
    d = b - a
    mflag = True

    for iteration in range(1, MAX_ITERATIONS + 1):
        if fa != fc and fb != fc:
            # 反二次插值
            s = (
                a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
            )
        else:
            # 割线法
            s = b - fb * (b - a) / (fb - fa)

        # Brent 可接受性条件，不满足则回退到二分
        condition = (
            (s < (3.0 * a + b) / 4.0 or s > b)
            or (mflag and abs(s - b) >= abs(b - c) / 2.0)
            or (not mflag and abs(s - b) >= abs(c - d) / 2.0)
            or (mflag and abs(b - c) < 1e-14)
            or (not mflag and abs(c - d) < 1e-14)
        )
        if condition:
            s = 0.5 * (a + b)
            mflag = True
        else:
            mflag = False

        fs = f(s)
        if abs(fs) <= PRICE_TOL or abs(b - a) < 1e-12:
            return s, iteration

        d, c, fc = c, b, fb
        if fa * fs < 0.0:
            b, fb = s, fs
        else:
            a, fa = s, fs
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    raise PricingEngineError(
        ErrorCode.SOLVER_DID_NOT_CONVERGE,
        f"implied volatility solver did not converge within "
        f"{MAX_ITERATIONS} iterations (bracket=[{a:.6g},{b:.6g}])",
    )


def implied_volatility(
    price: float,
    is_call: bool,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float = 0.0,
    pricer: Callable[[float], float] | None = None,
) -> tuple[float, int]:
    """由目标价格反解香草期权隐含波动率。

    ``pricer(vol)`` 由调用方注入（默认使用欧式香草闭式解），
    从而同一求解器可服务于不同定价路径，保证行为一致。
    """
    if pricer is None:
        from .black_scholes import european_price
        from ..models import OptionType

        option_type = (
            OptionType.EUROPEAN_CALL if is_call else OptionType.EUROPEAN_PUT
        )

        def pricer(vol: float) -> float:
            return european_price(
                option_type, spot, strike, maturity, vol, rate, dividend
            )

    if price <= 0.0:
        raise PricingEngineError(
            ErrorCode.SOLVER_DID_NOT_CONVERGE,
            f"target price {price} must be strictly positive; "
            "no positive implied volatility can reproduce it",
        )

    f_low = pricer(VOL_LOWER) - price
    f_high = pricer(VOL_UPPER) - price
    if f_low > 0.0:
        raise PricingEngineError(
            ErrorCode.SOLVER_DID_NOT_CONVERGE,
            f"target price {price:.8g} is below the no-volatility lower bound "
            f"{f_low + price:.8g}; no solution in [{VOL_LOWER},{VOL_UPPER}]",
        )
    if f_high < 0.0:
        raise PricingEngineError(
            ErrorCode.SOLVER_DID_NOT_CONVERGE,
            f"target price {price:.8g} exceeds the price at vol={VOL_UPPER} "
            f"({f_high + price:.8g}); no solution in [{VOL_LOWER},{VOL_UPPER}]",
        )

    root, iterations = _brent_solve(
        lambda vol: pricer(vol) - price, VOL_LOWER, VOL_UPPER
    )
    return root, iterations
