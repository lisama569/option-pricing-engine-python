"""定价编排层。

处理顺序（任一步失败都用稳定错误码拒绝，绝不返回近似/默认值）：
1. 校验请求参数 -> INVALID_MODEL_PARAMS；
2. 绑定曲面版本（None=最新）-> VERSION_NOT_FOUND；
3. 从曲面分片双线性内插取波动率 -> SURFACE_OUT_OF_RANGE；
4. 分派定价模型（欧式/数字：BSM 闭式解；美式：CRR 二叉树）。

定价核心是纯函数：相同 (请求参数, 曲面版本) 永远得到逐位一致的结果。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..analytics.american import american_greeks, american_price
from ..analytics.black_scholes import european_greeks, european_price
from ..analytics.implied_vol import implied_volatility
from ..errors import ErrorCode, PricingEngineError
from ..models import (
    IVSolveRequest,
    IVSolveResult,
    OptionType,
    PricingRequest,
    PricingResult,
)
from ..surface.grid import interpolate_vol
from ..surface.store import SurfaceStore

logger = logging.getLogger("option_engine.pricing")

MAX_TREE_STEPS = 20000

_EUROPEAN_TYPES = (
    OptionType.EUROPEAN_CALL,
    OptionType.EUROPEAN_PUT,
    OptionType.CASH_OR_NOTHING_CALL,
    OptionType.CASH_OR_NOTHING_PUT,
)
_AMERICAN_TYPES = (OptionType.AMERICAN_CALL, OptionType.AMERICAN_PUT)
_DIGITAL_TYPES = (
    OptionType.CASH_OR_NOTHING_CALL,
    OptionType.CASH_OR_NOTHING_PUT,
)
MODEL_BSM = "black_scholes_merton"
MODEL_CRR = "crr_binomial_tree"


@dataclass(frozen=True)
class _Market:
    rate: float
    dividend: float


def _validate_common(
    spot: float,
    strike: float,
    maturity: float,
    option_type: OptionType,
    cash: float | None,
) -> None:
    if not (spot == spot) or spot <= 0.0:
        raise PricingEngineError(
            ErrorCode.INVALID_MODEL_PARAMS,
            f"spot must be a finite positive number, got {spot}",
        )
    if not (strike == strike) or strike <= 0.0:
        raise PricingEngineError(
            ErrorCode.INVALID_MODEL_PARAMS,
            f"strike must be a finite positive number, got {strike}",
        )
    if maturity != maturity or maturity < 0.0:
        raise PricingEngineError(
            ErrorCode.INVALID_MODEL_PARAMS,
            f"maturity must be a finite non-negative number (years), "
            f"got {maturity}",
        )
    if option_type in _DIGITAL_TYPES:
        if cash is None or cash != cash or cash <= 0.0:
            raise PricingEngineError(
                ErrorCode.INVALID_MODEL_PARAMS,
                f"cash_or_nothing options require a strictly positive "
                f"'cash' payout, got {cash}",
            )


def _validate_steps(option_type: OptionType, tree_steps: int | None) -> int:
    if option_type not in _AMERICAN_TYPES:
        return 0
    if tree_steps is None:
        raise PricingEngineError(
            ErrorCode.INVALID_MODEL_PARAMS,
            f"{option_type.value} requires 'tree_steps' (>=1)",
        )
    if not isinstance(tree_steps, int) or isinstance(tree_steps, bool):
        raise PricingEngineError(
            ErrorCode.INVALID_MODEL_PARAMS,
            f"'tree_steps' must be an integer, got {tree_steps!r}",
        )
    if tree_steps < 1:
        raise PricingEngineError(
            ErrorCode.INVALID_MODEL_PARAMS,
            f"'tree_steps' must be >= 1, got {tree_steps}",
        )
    if tree_steps > MAX_TREE_STEPS:
        raise PricingEngineError(
            ErrorCode.INVALID_MODEL_PARAMS,
            f"'tree_steps' {tree_steps} exceeds safety cap "
            f"{MAX_TREE_STEPS}",
        )
    return tree_steps


class PricingService:
    def __init__(self, store: SurfaceStore) -> None:
        self._store = store

    def price(self, request: PricingRequest) -> PricingResult:
        logger.info(
            "pricing request received: request_id=%s option_type=%s spot=%s "
            "strike=%s maturity=%s cash=%s tree_steps=%s surface_version=%s",
            request.request_id,
            request.option_type.value,
            request.spot,
            request.strike,
            request.maturity,
            request.cash,
            request.tree_steps,
            request.surface_version,
        )
        _validate_common(
            request.spot,
            request.strike,
            request.maturity,
            request.option_type,
            request.cash,
        )
        steps = _validate_steps(request.option_type, request.tree_steps)

        snapshot = self._store.get(request.surface_version)
        market = _Market(
            rate=snapshot.data.risk_free_rate,
            dividend=snapshot.data.dividend_yield,
        )
        if request.maturity == 0.0:
            vol = 0.0
        else:
            vol = interpolate_vol(
                snapshot.data, request.maturity, request.strike
            )

        if request.option_type in _EUROPEAN_TYPES:
            price = european_price(
                request.option_type,
                request.spot,
                request.strike,
                request.maturity,
                vol,
                market.rate,
                market.dividend,
                request.cash,
            )
            greeks = european_greeks(
                request.option_type,
                request.spot,
                request.strike,
                request.maturity,
                vol,
                market.rate,
                market.dividend,
                request.cash,
            )
            model = MODEL_BSM
        else:
            is_call = request.option_type is OptionType.AMERICAN_CALL
            price = american_price(
                is_call,
                request.spot,
                request.strike,
                request.maturity,
                vol,
                market.rate,
                market.dividend,
                steps,
            )
            greeks = american_greeks(
                is_call,
                request.spot,
                request.strike,
                request.maturity,
                vol,
                market.rate,
                market.dividend,
                steps,
            )
            model = MODEL_CRR

        logger.info(
            "pricing resolved: request_id=%s surface_version=%s "
            "surface_label=%s model=%s vol=%.10f price=%.10f "
            "basis='%s bilinear-vol@(T=%s,K=%s); %s'",
            request.request_id,
            snapshot.version,
            snapshot.data.label,
            model,
            vol,
            price,
            snapshot.data.label,
            request.maturity,
            request.strike,
            model,
        )
        return PricingResult(
            price=price,
            greeks=greeks,
            implied_vol=vol,
            option_type=request.option_type,
            spot=request.spot,
            strike=request.strike,
            maturity=request.maturity,
            surface_version=snapshot.version,
            surface_label=snapshot.data.label,
            model=model,
            request_id=request.request_id,
        )

    def solve_iv(self, request: IVSolveRequest) -> IVSolveResult:
        """反解欧式香草期权隐含波动率（不支持数字/美式，显式拒绝）。"""
        if request.option_type not in (
            OptionType.EUROPEAN_CALL,
            OptionType.EUROPEAN_PUT,
        ):
            raise PricingEngineError(
                ErrorCode.INVALID_MODEL_PARAMS,
                f"implied volatility inversion supports European vanilla "
                f"calls/puts only, got {request.option_type.value}",
            )
        _validate_common(
            request.spot,
            request.strike,
            request.maturity,
            request.option_type,
            cash=None,
        )
        if request.maturity <= 0.0:
            raise PricingEngineError(
                ErrorCode.INVALID_MODEL_PARAMS,
                "implied volatility inversion requires maturity > 0",
            )
        if not (request.target_price == request.target_price) or (
            request.target_price <= 0.0
        ):
            raise PricingEngineError(
                ErrorCode.INVALID_MODEL_PARAMS,
                f"target_price must be finite and positive, "
                f"got {request.target_price}",
            )

        snapshot = self._store.get(request.surface_version)
        is_call = request.option_type is OptionType.EUROPEAN_CALL

        def pricer(vol: float) -> float:
            return european_price(
                request.option_type,
                request.spot,
                request.strike,
                request.maturity,
                vol,
                snapshot.data.risk_free_rate,
                snapshot.data.dividend_yield,
            )

        vol, iterations = implied_volatility(
            request.target_price,
            is_call,
            request.spot,
            request.strike,
            request.maturity,
            snapshot.data.risk_free_rate,
            snapshot.data.dividend_yield,
            pricer=pricer,
        )
        logger.info(
            "iv resolved: request_id=%s surface_version=%s iterations=%s iv=%.10f",
            request.request_id,
            snapshot.version,
            iterations,
            vol,
        )
        return IVSolveResult(
            implied_vol=vol,
            iterations=iterations,
            surface_version=snapshot.version,
            request_id=request.request_id,
        )
