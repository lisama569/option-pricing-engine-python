"""标准正态分布函数（标准库实现，避免第三方数值依赖）。

CDF 直接由标准库 ``math.erfc`` 计算（Phi(x)=0.5*erfc(-x/sqrt2)），
达到双精度机器精度，且结果完全确定、可复现。
"""

from __future__ import annotations

import math

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)


def norm_pdf(x: float) -> float:
    """标准正态密度函数。"""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def norm_cdf(x: float) -> float:
    """标准正态累积分布函数，全实数域稳定。"""
    return 0.5 * math.erfc(-x / _SQRT_2)
