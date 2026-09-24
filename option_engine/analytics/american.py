"""美式期权：Cox-Ross-Rubinstein 二叉树（后向归纳，含提前行权）。

- 参数：步数 steps >= 1（步数由调用方负责校验）。
- 含连续股息率 q：u=e^{σ√Δt}, d=1/u, p=(e^{(r-q)Δt}-d)/(u-d)。
- 每个节点取“继续持有”与“立即行权”的较大者。
- Greeks：同一棵树上用标准有限差分（步长固定以保证可复现）。
"""

from __future__ import annotations

import math

from ..models import Greeks


def american_price(
    is_call: bool,
    spot: float,
    strike: float,
    maturity: float,
    vol: float,
    rate: float,
    dividend: float,
    steps: int,
) -> float:
    """CRR 二叉树美式期权价格。"""
    if maturity == 0.0:
        return max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)

    dt = maturity / steps
    up = math.exp(vol * math.sqrt(dt))
    down = 1.0 / up
    down_sq = down * down
    growth = math.exp((rate - dividend) * dt)
    prob = (growth - down) / (up - down)
    disc = math.exp(-rate * dt)

    # 到期日各节点的期权价值（一维数组，向后滚动）
    values = [0.0] * (steps + 1)
    for j in range(steps + 1):
        terminal_spot = spot * up ** (steps - 2 * j)  # u^(n-j) d^j = u^(n-2j)
        values[j] = (
            terminal_spot - strike
            if is_call
            else strike - terminal_spot
        )
        if values[j] < 0.0:
            values[j] = 0.0

    disc_prob = disc * prob
    disc_1mp = disc * (1.0 - prob)
    for i in range(steps - 1, -1, -1):
        # 该层节点价：S * u^(i-2j)，随 j 递减，滚动更新
        node_spot = spot * up**i
        for j in range(i + 1):
            cont = disc_prob * values[j] + disc_1mp * values[j + 1]
            ex = node_spot - strike if is_call else strike - node_spot
            values[j] = cont if cont > ex else ex
            node_spot *= down_sq
    return values[0]


_DELTA_SHIFT = 0.01
_VOL_SHIFT = 1e-4
_RATE_SHIFT = 1e-4
_TIME_SHIFT = 1e-4


def american_greeks(
    is_call: bool,
    spot: float,
    strike: float,
    maturity: float,
    vol: float,
    rate: float,
    dividend: float,
    steps: int,
) -> Greeks:
    """美式期权 Greeks：固定步长中心差分（确定性、可复现）。"""
    if maturity == 0.0:
        return Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    def p(s=spot, k=strike, t=maturity, v=vol, r=rate, q=dividend):
        return american_price(is_call, s, k, t, v, r, q, steps)

    hs = _DELTA_SHIFT * spot
    delta = (p(s=spot + hs) - p(s=spot - hs)) / (2.0 * hs)
    gamma = (p(s=spot + hs) - 2.0 * p() + p(s=spot - hs)) / (hs * hs)
    vega = (p(v=vol + _VOL_SHIFT) - p(v=vol - _VOL_SHIFT)) / (2.0 * _VOL_SHIFT)
    theta = -(p(t=maturity + _TIME_SHIFT) - p(t=maturity - _TIME_SHIFT)) / (
        2.0 * _TIME_SHIFT
    )
    rho = (p(r=rate + _RATE_SHIFT) - p(r=rate - _RATE_SHIFT)) / (2.0 * _RATE_SHIFT)
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
