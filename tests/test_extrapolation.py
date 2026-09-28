"""Extrapolation authorization & audit:

* default = out-of-range rejected; opt-in is necessary but not sufficient;
* extrapolation only happens within the range declared by the bound version;
* actual extrapolation is always reported (never silent);
* historical replay follows the declaration stored on that version.
"""
import pytest

from option_engine.errors import SurfaceOutOfRangeError
from option_engine.models import (
    PricingRequest,
    SurfaceGridPayload,
    SurfaceVersionInfo,
)
from option_engine.service import PricingService
from option_engine.surface import SurfaceStore


def _svc_with_range():
    store = SurfaceStore()
    store.publish("ex", SurfaceGridPayload(
        expiries=[0.5, 1.0], strikes=[90, 100, 110],
        vols=[[0.22, 0.20, 0.19], [0.23, 0.205, 0.195]],
        spot=100, rate=0.05,
        extrapolation_limits=dict(min_expiry=0.1, max_expiry=2.0,
                                  min_strike=80, max_strike=130)))
    return PricingService(store), store


def test_out_of_range_rejected_without_explicit_request(audit):
    service, _ = _svc_with_range()
    audit("extrap_not_requested", {"expiry": 1.5, "strike": 125},
          surface_version=1,
          criterion="surface_out_of_range even though a range is declared")
    with pytest.raises(SurfaceOutOfRangeError):
        service.price(PricingRequest(
            surface_id="ex", option_type="vanilla_call",
            expiry=1.5, strike=125))


def test_declared_range_opt_in_extrapolates_and_reports(audit):
    service, _ = _svc_with_range()
    audit("extrap_used", {"expiry": 1.5, "strike": 125,
                          "allow_extrapolation": True},
          surface_version=1,
          criterion="price returned; extrapolated=true; overhang reported")
    resp = service.price(PricingRequest(
        surface_id="ex", option_type="vanilla_call",
        expiry=1.5, strike=125, allow_extrapolation=True))
    assert resp.extrapolated is True
    info = resp.extrapolation
    assert info is not None
    assert info.expiry_outside is True
    assert info.strike_outside is True
    assert info.expiry_overhang == pytest.approx(0.5)
    assert info.strike_overhang == pytest.approx(15.0)
    # Overhang fractions are measured against the declared band width.
    assert info.expiry_overhang_fraction == pytest.approx(0.5 / 1.9)
    assert info.strike_overhang_fraction == pytest.approx(15.0 / 50.0)


def test_in_support_lookup_never_marks_extrapolation(audit):
    service, _ = _svc_with_range()
    audit("extrap_in_support", {"expiry": 0.75, "strike": 100,
                                "allow_extrapolation": True},
          surface_version=1,
          criterion="interpolation only: extrapolated=false, no audit block")
    resp = service.price(PricingRequest(
        surface_id="ex", option_type="vanilla_call",
        expiry=0.75, strike=100, allow_extrapolation=True))
    assert resp.extrapolated is False
    assert resp.extrapolation is None


def test_beyond_declared_range_rejected_even_with_opt_in(audit):
    service, _ = _svc_with_range()
    audit("extrap_beyond_range", {"expiry": 2.5, "strike": 125},
          surface_version=1,
          criterion="outside declared limits -> surface_out_of_range")
    with pytest.raises(SurfaceOutOfRangeError) as ei:
        service.price(PricingRequest(
            surface_id="ex", option_type="vanilla_call",
            expiry=2.5, strike=125, allow_extrapolation=True))
    assert ei.value.code == "surface_out_of_range"


def test_surface_without_declared_range_cannot_be_opted_in(audit):
    store = SurfaceStore()
    store.publish("noext", SurfaceGridPayload(
        expiries=[0.5, 1.0], strikes=[90, 100, 110],
        vols=[[0.22, 0.20, 0.19], [0.23, 0.205, 0.195]],
        spot=100, rate=0.05))
    service = PricingService(store)
    audit("extrap_no_declaration", {"expiry": 1.5, "strike": 125,
                                    "allow_extrapolation": True},
          surface_version=1,
          criterion="no declared range -> surface_out_of_range")
    with pytest.raises(SurfaceOutOfRangeError):
        service.price(PricingRequest(
            surface_id="noext", option_type="vanilla_call",
            expiry=1.5, strike=125, allow_extrapolation=True))


def test_historical_version_keeps_its_own_extrapolation_declaration(audit):
    service, store = _svc_with_range()
    # v2: declares NO extrapolation range.
    store.publish("ex", SurfaceGridPayload(
        expiries=[0.5, 1.0], strikes=[90, 100, 110],
        vols=[[0.22, 0.20, 0.19], [0.23, 0.205, 0.195]],
        spot=100, rate=0.05))
    audit("extrap_history", {"versions": [1, 2], "point": [1.5, 125]},
          surface_version=[1, 2],
          criterion="v1 extrapolates within its declared range; v2 rejects; "
                    "v1 replay unchanged after v2 publish")
    v1_price = service.price(PricingRequest(
        surface_id="ex", version=1, option_type="vanilla_call",
        expiry=1.5, strike=125, allow_extrapolation=True))
    assert v1_price.extrapolated is True

    with pytest.raises(SurfaceOutOfRangeError):
        service.price(PricingRequest(
            surface_id="ex", version=2, option_type="vanilla_call",
            expiry=1.5, strike=125, allow_extrapolation=True))

    replayed = service.price(PricingRequest(
        surface_id="ex", version=1, option_type="vanilla_call",
        expiry=1.5, strike=125, allow_extrapolation=True))
    assert replayed.price == v1_price.price
    assert replayed.extrapolated is True


def test_boundary_coordinates_are_handled_predictably(audit):
    service, _ = _svc_with_range()
    # Exactly on the grid-support edge: interpolation (boundary value),
    # never reported as extrapolation.
    audit("extrap_edge", {"on_support": [1.0, 110]},
          surface_version=1,
          criterion="grid edge -> not extrapolated")
    edge = service.price(PricingRequest(
        surface_id="ex", option_type="vanilla_call",
        expiry=1.0, strike=110, allow_extrapolation=True))
    assert edge.extrapolated is False
    # Exactly on the declared-limit edge: accepted and reported.
    audit("extrap_limit_edge", {"on_limit": [2.0, 130]},
          surface_version=1,
          criterion="declared-limit boundary -> extrapolated, accepted")
    limit_edge = service.price(PricingRequest(
        surface_id="ex", option_type="vanilla_call",
        expiry=2.0, strike=130, allow_extrapolation=True))
    assert limit_edge.extrapolated is True
    assert limit_edge.extrapolation.expiry_overhang == pytest.approx(1.0)
    assert limit_edge.extrapolation.strike_overhang == pytest.approx(20.0)
    # Just beyond the declared limit: rejected.
    with pytest.raises(SurfaceOutOfRangeError):
        service.price(PricingRequest(
            surface_id="ex", option_type="vanilla_call",
            expiry=2.0001, strike=130, allow_extrapolation=True))


def test_publishing_invalid_limits_rejected(audit):
    store = SurfaceStore()
    audit("bad_limits", {"case": "range not enclosing support"},
          surface_version="n/a",
          criterion="InvalidParameterError; no version published")
    with pytest.raises(Exception):
        store.publish("ex", SurfaceGridPayload(
            expiries=[0.5, 1.0], strikes=[90, 100, 110],
            vols=[[0.22, 0.20, 0.19], [0.23, 0.205, 0.195]],
            spot=100, rate=0.05,
            extrapolation_limits=dict(min_expiry=0.6, max_expiry=0.9,
                                      min_strike=80, max_strike=130)))
