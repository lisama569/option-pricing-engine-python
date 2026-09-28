"""Illegal parameters, out-of-support surface queries and non-converging
solvers must all be rejected with distinguishable error codes.
"""
import pytest
from pydantic import ValidationError

from option_engine.errors import (
    InvalidParameterError,
    SurfaceNotFoundError,
    SurfaceOutOfRangeError,
    SolverNonConvergenceError,
)
from option_engine.models import (
    OptionType,
    PricingRequest,
    SurfaceGridPayload,
)
from option_engine.service import PricingService
from option_engine.surface import SurfaceStore


def svc():
    store = SurfaceStore()
    store.publish("default", SurfaceGridPayload(
        expiries=[0.25, 1.0], strikes=[90, 100, 110],
        vols=[[0.22, 0.20, 0.19], [0.23, 0.205, 0.195]],
        spot=100, rate=0.05))
    return PricingService(store), store


def test_invalid_model_parameters(audit):
    service, _ = svc()
    bad_cases = [
        ("nonpositive expiry", dict(option_type="vanilla_call",
                                    expiry=0.0, strike=100)),
        ("negative strike", dict(option_type="vanilla_put",
                                 expiry=1.0, strike=-5)),
        ("bad spot override (model layer)", dict(option_type="vanilla_call",
                                   expiry=1.0, strike=100, spot=0)),
        ("negative dividend (model layer)", dict(option_type="digital_call",
                                   expiry=1.0, strike=100, dividend=-0.1)),
    ]
    for label, kw in bad_cases:
        audit("invalid_param", {"case": label, **kw},
              surface_version="n/a",
              criterion="rejected at model or engine layer, no price")
        with pytest.raises((InvalidParameterError, ValidationError)):
            service.price(PricingRequest(**kw))

    # Engine-layer guard when a request bypasses model validation.
    req = PricingRequest.model_construct(
        surface_id="default", version=None, option_type=OptionType.VANILLA_CALL,
        expiry=-1.0, strike=100, spot=None, rate=None, dividend=None,
        payout=1.0, steps=500, convergence_tol=1e-4, max_steps=4000,
        target_price=None,
    )
    audit("invalid_param", {"case": "negative expiry past model layer",
                            "expiry": -1.0},
          surface_version="n/a",
          criterion="surface.vol rejects with InvalidParameterError")
    with pytest.raises(InvalidParameterError):
        service.price(req)


def test_surface_out_of_range_distinguishable(audit):
    service, _ = svc()
    for label, kw in [
        ("expiry above", dict(option_type="vanilla_call",
                              expiry=3.0, strike=100)),
        ("expiry below", dict(option_type="vanilla_call",
                              expiry=0.01, strike=100)),
        ("strike above", dict(option_type="vanilla_call",
                              expiry=1.0, strike=500)),
        ("strike below", dict(option_type="vanilla_call",
                              expiry=1.0, strike=10)),
    ]:
        audit("out_of_range", {"case": label, **kw},
              surface_version=1,
              criterion="SurfaceOutOfRangeError (not invalid_parameter)")
        with pytest.raises(SurfaceOutOfRangeError) as ei:
            service.price(PricingRequest(**kw))
        assert ei.value.code == "surface_out_of_range"


def test_extrapolation_allowed_surface(audit):
    service, store = svc()
    store.publish("default", SurfaceGridPayload(
        expiries=[0.25, 1.0], strikes=[90, 100, 110],
        vols=[[0.22, 0.20, 0.19], [0.23, 0.205, 0.195]],
        spot=100, rate=0.05,
        extrapolation_limits=dict(min_expiry=0.1, max_expiry=3.0,
                                  min_strike=80, max_strike=150)))
    # Even a surface that declares an extrapolation range rejects unless the
    # caller explicitly opts in.
    audit("extrapolation_default_denied",
          {"expiry": 3.0, "strike": 150, "version": 2, "allow": False},
          surface_version=2,
          criterion="opt-in is required: SurfaceOutOfRangeError")
    with pytest.raises(SurfaceOutOfRangeError):
        service.price(PricingRequest(
            option_type="vanilla_call", expiry=3.0, strike=150, version=2))

    audit("extrapolation_allowed",
          {"expiry": 3.0, "strike": 150, "version": 2, "allow": True},
          surface_version=2,
          criterion="price computed with linear-extrapolated vol, audited")
    resp = service.price(PricingRequest(
        option_type="vanilla_call", expiry=3.0, strike=150, version=2,
        allow_extrapolation=True))
    assert resp.vol_used > 0
    assert resp.extrapolated is True
    assert resp.extrapolation is not None
    assert resp.extrapolation.expiry_outside is True
    assert resp.extrapolation.strike_outside is True
    # Beyond the declared range: still rejected even with opt-in.
    with pytest.raises(SurfaceOutOfRangeError):
        service.price(PricingRequest(
            option_type="vanilla_call", expiry=9.0, strike=150, version=2,
            allow_extrapolation=True))
    # While v1 (no declared range) rejects the same query even with opt-in.
    with pytest.raises(SurfaceOutOfRangeError):
        service.price(PricingRequest(
            option_type="vanilla_call", expiry=3.0, strike=150, version=1,
            allow_extrapolation=True))


def test_missing_surface_and_version(audit):
    service, _ = svc()
    audit("surface_not_found", {"surface_id": "nope", "version": None},
          surface_version="n/a",
          criterion="SurfaceNotFoundError / 404-equivalent code")
    with pytest.raises(SurfaceNotFoundError) as ei:
        service.price(PricingRequest(
            surface_id="nope", option_type="vanilla_call",
            expiry=1.0, strike=100))
    assert ei.value.code == "surface_not_found"
    with pytest.raises(SurfaceNotFoundError):
        service.price(PricingRequest(
            version=99, option_type="vanilla_call",
            expiry=1.0, strike=100))


def test_american_nonconvergence_rejected(audit):
    service, store = svc()
    # Wide grid so the vol lookup succeeds; force the tree solver budget
    # far below what its oscillatory convergence needs.
    store.publish("wide", SurfaceGridPayload(
        expiries=[0.1, 5.0], strikes=[10, 500],
        vols=[[0.2, 0.2], [0.2, 0.2]],
        spot=50, rate=0.05,
        extrapolation_limits=dict(min_expiry=0.01, max_expiry=10.0,
                                  min_strike=1.0, max_strike=1000.0)))
    inputs = dict(option_type="american_put", surface_id="wide",
                  expiry=2.0, strike=52, spot=50,
                  steps=4, max_steps=32, convergence_tol=1e-12)
    audit("solver_non_convergence", inputs, surface_version=1,
          criterion="SolverNonConvergenceError, no approximate price")
    with pytest.raises(SolverNonConvergenceError) as ei:
        service.price(PricingRequest(**inputs))
    assert ei.value.code == "solver_non_convergence"


def test_iv_impossible_target_rejected(audit):
    service, _ = svc()
    inputs = dict(option_type="vanilla_call", expiry=1.0, strike=100,
                  target_price=10_000.0)
    audit("iv_no_root", inputs, surface_version=1,
          criterion="InvalidParameterError: target beyond no-arb upper bound")
    with pytest.raises(InvalidParameterError) as ei:
        service.price(PricingRequest(**inputs))
    assert "no-arbitrage" in ei.value.args[0]


def test_bad_payload_publish_rejected(audit):
    store = SurfaceStore()
    audit("bad_payload", {"case": "non-increasing expiries"},
          surface_version="n/a",
          criterion="publish rejected, store stays empty")
    with pytest.raises(InvalidParameterError):
        store.publish("x", SurfaceGridPayload(
            expiries=[1.0, 0.5], strikes=[100], vols=[[0.2], [0.3]], spot=100))
    with pytest.raises(SurfaceNotFoundError):
        store.get("x")
