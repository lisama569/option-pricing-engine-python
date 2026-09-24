"""Cross-checks against published analytic/textbook benchmarks and
model-internal identities. Deterministic, no external services.
"""
import math

import pytest

from option_engine import pricing as P

TOL = 1e-4


def test_hull_vanilla_benchmark(audit):
    # Hull "Options, Futures and Other Derivatives" example:
    # S=42 K=40 T=0.5 r=0.10 sigma=0.20, q=0 -> c=4.7594 p=0.8086
    inputs = dict(S=42, K=40, T=0.5, r=0.10, q=0.0, sigma=0.20)
    audit("hull_benchmark", inputs, surface_version="n/a (direct model)",
          criterion="c≈4.7594, p≈0.8086 (Hull)")
    c = P.bs_price("vanilla_call", 42, 40, 0.5, 0.20, 0.10)
    p = P.bs_price("vanilla_put", 42, 40, 0.5, 0.20, 0.10)
    assert c == pytest.approx(4.7594, abs=TOL)
    assert p == pytest.approx(0.8086, abs=TOL)


def test_put_call_parity(audit):
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.25
    inputs = dict(S=S, K=K, T=T, r=r, q=q, sigma=sigma)
    audit("put_call_parity", inputs, surface_version="n/a (direct model)",
          criterion="c + K*e^-rT == p + S*e^-qT")
    c = P.bs_price("vanilla_call", S, K, T, sigma, r, q)
    p = P.bs_price("vanilla_put", S, K, T, sigma, r, q)
    lhs = c + K * math.exp(-r * T)
    rhs = p + S * math.exp(-q * T)
    assert lhs == pytest.approx(rhs, abs=1e-12)


def test_digital_call_put_partition(audit):
    S, K, T, r, sigma = 100.0, 105.0, 0.75, 0.03, 0.3
    inputs = dict(S=S, K=K, T=T, r=r, sigma=sigma, payout=1.0)
    audit("digital_partition", inputs, surface_version="n/a (direct model)",
          criterion="digital_call + digital_put == e^-rT")
    dc = P.bs_price("digital_call", S, K, T, sigma, r)
    dp = P.bs_price("digital_put", S, K, T, sigma, r)
    assert dc + dp == pytest.approx(math.exp(-r * T), abs=1e-12)


def test_vanilla_limits(audit):
    # Deep ITM call ≈ discounted forward intrinsic; deep OTM put ≈ 0.
    S, K, T, r, sigma = 200.0, 50.0, 1.0, 0.02, 0.20
    inputs = dict(S=S, K=K, T=T, r=r, sigma=sigma)
    audit("vanilla_limits", inputs, surface_version="n/a",
          criterion="deep ITM call≈S-K*e^-rT; deep OTM put≈0")
    c = P.bs_price("vanilla_call", S, K, T, sigma, r)
    p = P.bs_price("vanilla_put", S, K, T, sigma, r)
    assert c == pytest.approx(S - K * math.exp(-r * T), abs=1e-2)
    assert p == pytest.approx(0.0, abs=1e-9)


def test_american_put_within_bounds(audit):
    # Hull CRR example: S=50 K=52 r=0.05 T=2 sigma=0.2263
    # 500-step tree ≈ 5.869; converged answer must sit between intrinsic 2
    # and the European put.
    inputs = dict(S=50, K=52, T=2, r=0.05, sigma=0.2263, start_steps=64,
                  max_steps=4096, tol=1e-3)
    audit("american_put_bounds", inputs, surface_version="n/a",
          criterion="intrinsic<Am<K; Am>=European; matches 500-step tree within"
                      " 1e-2 and reference 5.540")
    am, info = P.american_put_price(50, 52, 2, 0.2263, 0.05, 0,
                                    64, 1e-3, 4096)
    eu = P.bs_price("vanilla_put", 50, 52, 2, 0.2263, 0.05)
    assert 2.0 < am < 52.0
    assert am >= eu
    am_ref = P.american_put_crr(50, 52, 2, 0.2263, 0.05, 0, 500)
    assert am == pytest.approx(am_ref, abs=1e-2)
    assert am == pytest.approx(5.540, abs=3e-3)
    assert info["converged"] is True


def test_iv_roundtrip_and_known_value(audit):
    # Invert the model price computed at sigma=0.20 (analytic round trip);
    # the 4.7594 agreement with Hull is covered by test_hull_vanilla_benchmark.
    target = P.bs_price("vanilla_call", 42, 40, 0.5, 0.20, 0.10)
    inputs = dict(target=target, S=42, K=40, T=0.5, r=0.10)
    audit("iv_roundtrip", inputs, surface_version="n/a",
          criterion="IV(model call price)≈0.20 and re-price matches target")
    iv = P.implied_vol("vanilla_call", target, 42, 40, 0.5, 0.10, 0, tol=1e-9)
    assert iv == pytest.approx(0.20, abs=1e-7)
    reprice = P.bs_price("vanilla_call", 42, 40, 0.5, iv, 0.10)
    assert reprice == pytest.approx(target, abs=1e-8)
