"""Pricing models and numerical solvers (standard library only).

* European vanilla call/put and digital (cash-or-nothing) call/put:
  closed-form Black-Scholes-Merton.
* American put: Cox-Ross-Rubinstein binomial tree with Richardson-style
  convergence check; raises instead of returning an unconverged value.
* Implied volatility: guarded bisection; raises when no bracketed root exists
  or the tolerance is not reached.
"""
from __future__ import annotations

import math

from .errors import InvalidParameterError, SolverNonConvergenceError

SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _validate_european(spot: float, strike: float, expiry: float, vol: float,
                        rate: float, dividend: float) -> None:
    if spot <= 0:
        raise InvalidParameterError(f"spot must be > 0, got {spot}")
    if strike <= 0:
        raise InvalidParameterError(f"strike must be > 0, got {strike}")
    if expiry <= 0:
        raise InvalidParameterError(f"expiry must be > 0, got {expiry}")
    if vol <= 0:
        raise InvalidParameterError(f"vol must be > 0, got {vol}")
    if dividend < 0:
        raise InvalidParameterError(f"dividend must be >= 0, got {dividend}")
    if not math.isfinite(rate):
        raise InvalidParameterError("rate must be finite")


def _bsm_terms(spot, strike, expiry, vol, rate, dividend):
    dt = rate - dividend
    sig_t = vol * math.sqrt(expiry)
    d1 = (math.log(spot / strike) + (dt + 0.5 * vol * vol) * expiry) / sig_t
    d2 = d1 - sig_t
    disc = math.exp(-rate * expiry)
    fwd_disc = math.exp(-dividend * expiry)
    return d1, d2, disc, fwd_disc


def bs_price(option_type: str, spot: float, strike: float, expiry: float,
             vol: float, rate: float = 0.0, dividend: float = 0.0,
             payout: float = 1.0) -> float:
    _validate_european(spot, strike, expiry, vol, rate, dividend)
    if payout <= 0:
        raise InvalidParameterError(f"payout must be > 0, got {payout}")
    d1, d2, disc, fwd_disc = _bsm_terms(spot, strike, expiry, vol, rate,
                                        dividend)
    if option_type == "vanilla_call":
        return spot * fwd_disc * norm_cdf(d1) - strike * disc * norm_cdf(d2)
    if option_type == "vanilla_put":
        return strike * disc * norm_cdf(-d2) - spot * fwd_disc * norm_cdf(-d1)
    if option_type == "digital_call":
        return payout * disc * norm_cdf(d2)
    if option_type == "digital_put":
        return payout * disc * norm_cdf(-d2)
    raise InvalidParameterError(
        f"bs_price does not support option type '{option_type}'"
    )


def american_put_crr(spot: float, strike: float, expiry: float, vol: float,
                     rate: float, dividend: float, steps: int) -> float:
    """Single CRR binomial price for an American put."""
    _validate_european(spot, strike, expiry, vol, rate, dividend)
    if steps <= 0:
        raise InvalidParameterError(f"steps must be > 0, got {steps}")
    dt = expiry / steps
    up = math.exp(vol * math.sqrt(dt))
    down = 1.0 / up
    p_up = (math.exp((rate - dividend) * dt) - down) / (up - down)
    if not (0.0 < p_up < 1.0):
        raise InvalidParameterError(
            f"invalid CRR risk-neutral probability {p_up}; check inputs"
        )
    disc = math.exp(-rate * dt)
    values = [
        max(strike - spot * (up ** j) * (down ** (steps - j)), 0.0)
        for j in range(steps + 1)
    ]
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            cont = disc * (p_up * values[j + 1] + (1.0 - p_up) * values[j])
            s_ij = spot * (up ** j) * (down ** (i - j))
            values[j] = max(cont, strike - s_ij)
    return values[0]


def american_put_price(spot: float, strike: float, expiry: float, vol: float,
                       rate: float, dividend: float, start_steps: int,
                       tol: float, max_steps: int,
                       cancel_event=None) -> tuple[float, dict]:
    """American put via CRR with doubling until relative change < ``tol``.

    Never returns an approximate result on failure: if the price has not
    converged by ``max_steps`` a SolverNonConvergenceError is raised.
    """
    n = start_steps
    if max_steps < start_steps:
        raise InvalidParameterError(
            f"max_steps ({max_steps}) must be >= steps ({start_steps})"
        )
    prev = american_put_crr(spot, strike, expiry, vol, rate, dividend, n)
    if cancel_event is not None and cancel_event.is_set():
        from .errors import JobCancelledError
        raise JobCancelledError("cancelled during binomial solve")
    while n < max_steps:
        n = min(n * 2, max_steps)
        cur = american_put_crr(spot, strike, expiry, vol, rate, dividend, n)
        scale = max(abs(cur), 1e-12)
        if abs(cur - prev) / scale < tol:
            return cur, {"steps_used": n, "converged": True,
                         "last_diff": abs(cur - prev)}
        prev = cur
        if cancel_event is not None and cancel_event.is_set():
            from .errors import JobCancelledError
            raise JobCancelledError("cancelled during binomial solve")
    raise SolverNonConvergenceError(
        f"American put price not converged within {max_steps} steps "
        f"(tol={tol}, last price={prev})"
    )


def implied_vol(option_type: str, market_price: float, spot: float,
                strike: float, expiry: float, rate: float, dividend: float,
                vol_lo: float = 1e-6, vol_hi: float = 5.0,
                tol: float = 1e-7, max_iter: int = 200) -> float:
    """Bisection IV for European vanilla options.

    Fails distinguishably:
      * InvalidParameterError when the target price is outside the
        attainable no-arbitrage bracket (no bracketed root);
      * SolverNonConvergenceError when the bracket does not shrink to ``tol``.
    """
    _validate_european(spot, strike, expiry, vol_hi, rate, dividend)
    if market_price <= 0:
        raise InvalidParameterError(
            f"market_price must be > 0, got {market_price}"
        )
    if option_type not in ("vanilla_call", "vanilla_put"):
        raise InvalidParameterError(
            "implied volatility supported only for vanilla_call/vanilla_put"
        )

    def price_of(v: float) -> float:
        return bs_price(option_type, spot, strike, expiry, v, rate, dividend)

    f_lo = price_of(vol_lo) - market_price
    f_hi = price_of(vol_hi) - market_price
    if f_lo == 0.0:
        return vol_lo
    if f_lo * f_hi > 0:
        # Put-call boundary hint makes the failure reason self-explanatory.
        if option_type == "vanilla_call":
            lower = max(spot * math.exp(-dividend * expiry)
                        - strike * math.exp(-rate * expiry), 0.0)
            upper = spot * math.exp(-dividend * expiry)
        else:
            lower = max(strike * math.exp(-rate * expiry)
                        - spot * math.exp(-dividend * expiry), 0.0)
            upper = strike * math.exp(-rate * expiry)
        raise InvalidParameterError(
            f"market_price {market_price} outside no-arbitrage price range "
            f"[{lower:.6f}, {upper:.6f}] for {option_type}; no IV root"
        )
    lo, hi = vol_lo, vol_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = price_of(mid) - market_price
        if f_mid == 0.0 or (hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    raise SolverNonConvergenceError(
        f"IV bisection did not converge in {max_iter} iterations "
        f"(bracket [{lo}, {hi}])"
    )
