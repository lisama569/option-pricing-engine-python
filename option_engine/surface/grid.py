"""曲面网格：数据校验 + 分片双线性插值。

规则：
- 到期日、行权价必须严格递增，各轴至少 2 个点；
- 波动率矩阵必须为矩形，所有波动率严格为正（0 或负数的曲面不予发布）；
- 查询点严格落在网格包围盒内（含边界）才给出结果；
  落在网格之外一律抛出 SURFACE_OUT_OF_RANGE，绝不外插、绝不回退默认值；
- 盒内采用分片双线性插值：先对两个相邻到期日切片做行权价线性插值，
  再对到期日做线性插值。该映射确定且唯一，可复现。
"""

from __future__ import annotations

from ..errors import ErrorCode, PricingEngineError
from ..models import VolSurfaceData

_BOUNDARY_TOL = 1e-10


def validate_surface(data: VolSurfaceData) -> None:
    """校验曲面数据的语义合法性，不合法则抛 SURFACE_DATA_INVALID。"""
    if len(data.maturities) < 2:
        raise PricingEngineError(
            ErrorCode.SURFACE_DATA_INVALID,
            f"need at least 2 maturities, got {len(data.maturities)}",
        )
    if len(data.strikes) < 2:
        raise PricingEngineError(
            ErrorCode.SURFACE_DATA_INVALID,
            f"need at least 2 strikes, got {len(data.strikes)}",
        )
    for axis_name, axis in (("maturities", data.maturities), ("strikes", data.strikes)):
        if any(x <= 0 for x in axis):
            raise PricingEngineError(
                ErrorCode.SURFACE_DATA_INVALID,
                f"all {axis_name} must be strictly positive",
            )
        if any(b <= a for a, b in zip(axis, axis[1:])):
            raise PricingEngineError(
                ErrorCode.SURFACE_DATA_INVALID,
                f"{axis_name} must be strictly increasing",
            )

    n_m, n_k = len(data.maturities), len(data.strikes)
    if len(data.vols) != n_m:
        raise PricingEngineError(
            ErrorCode.SURFACE_DATA_INVALID,
            f"vols has {len(data.vols)} rows, expected {n_m}",
        )
    for i, row in enumerate(data.vols):
        if len(row) != n_k:
            raise PricingEngineError(
                ErrorCode.SURFACE_DATA_INVALID,
                f"vols row {i} has {len(row)} entries, expected {n_k}",
            )
        if any(not (v == v) or v <= 0.0 for v in row):  # NaN 或非正
            raise PricingEngineError(
                ErrorCode.SURFACE_DATA_INVALID,
                f"vols row {i} contains non-positive or NaN volatility",
            )

    if data.risk_free_rate < 0.0 or data.dividend_yield < 0.0:
        raise PricingEngineError(
            ErrorCode.SURFACE_DATA_INVALID,
            "risk_free_rate and dividend_yield must be non-negative",
        )


def _find_interval(axis: list[float], x: float) -> tuple[int, int, float]:
    """返回 (低位下标, 高位下标, 低段权重)。x 必须在轴范围内。"""
    n = len(axis)
    lo, hi = axis[0], axis[-1]
    span = hi - lo
    if x < lo - _BOUNDARY_TOL * (1.0 + abs(lo)) or x > hi + _BOUNDARY_TOL * (
        1.0 + abs(hi)
    ):
        raise LookupError("out of axis range")
    x = min(max(x, lo), hi)
    if x == hi:
        return n - 2, n - 1, 0.0
    i = 0
    while axis[i + 1] < x:
        i += 1
    w = (x - axis[i]) / (axis[i + 1] - axis[i]) if span > 0 else 0.0
    return i, i + 1, 1.0 - w


def interpolate_vol(
    surface: VolSurfaceData, maturity: float, strike: float
) -> float:
    """分片双线性插值取波动率；越界抛 SURFACE_OUT_OF_RANGE。"""
    try:
        m_lo, m_hi, w_m = _find_interval(surface.maturities, maturity)
    except LookupError:
        raise PricingEngineError(
            ErrorCode.SURFACE_OUT_OF_RANGE,
            f"maturity {maturity} is outside surface "
            f"{surface.label} maturity range "
            f"[{surface.maturities[0]}, {surface.maturities[-1]}]; "
            "extrapolation is not permitted",
        ) from None
    try:
        k_lo, k_hi, w_k = _find_interval(surface.strikes, strike)
    except LookupError:
        raise PricingEngineError(
            ErrorCode.SURFACE_OUT_OF_RANGE,
            f"strike {strike} is outside surface "
            f"{surface.label} strike range "
            f"[{surface.strikes[0]}, {surface.strikes[-1]}]; "
            "extrapolation is not permitted",
        ) from None

    def slice_vol(row: int) -> float:
        v0 = surface.vols[row][k_lo]
        v1 = surface.vols[row][k_hi]
        return w_k * v0 + (1.0 - w_k) * v1

    v_at_lo_maturity = slice_vol(m_lo)
    v_at_hi_maturity = slice_vol(m_hi)
    return w_m * v_at_lo_maturity + (1.0 - w_m) * v_at_hi_maturity
