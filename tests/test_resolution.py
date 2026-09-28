"""Fixed discrete-resolution pricing: exact step counts, deterministic
results, clear distinction from the auto-convergence mode and predictable
handling of the whole accepted range (including its boundaries).
"""
import pytest
from pydantic import ValidationError

from option_engine.models import MAX_RESOLUTION, MIN_RESOLUTION, PricingRequest
from option_engine.pricing import american_put_crr
from option_engine.service import PricingService
from option_engine.errors import InvalidParameterError


def _svc() -> PricingService:
    from option_engine.data import load_bundled_surfaces
    from option_engine.surface import SurfaceStore
    return PricingService(load_bundled_surfaces(SurfaceStore()))


def _req(**kw):
    base = dict(surface_id="default", option_type="american_put",
                expiry=1.0, strike=100)
    base.update(kw)
    return PricingRequest(**base)


def test_minimum_resolution_boundary_is_computed(audit):
    service = _svc()
    audit("resolution_min", {"resolution": MIN_RESOLUTION},
          surface_version="latest",
          criterion=f"resolution={MIN_RESOLUTION} accepted; mode fixed; "
                    "exactly one tree step; equals raw CRR(1)")
    resp = service.price(_req(resolution=MIN_RESOLUTION, version=1))
    snap = service.store.get("default", 1)
    raw = american_put_crr(snap.spot, 100, 1.0, resp.vol_used, snap.rate,
                           snap.dividend, MIN_RESOLUTION)
    assert resp.pricing_mode == "fixed_resolution"
    assert resp.steps_used == MIN_RESOLUTION
    assert resp.price == raw
    assert resp.detail["converged"] is None


def test_maximum_resolution_boundary_is_computed(audit):
    service = _svc()
    audit("resolution_max", {"resolution": MAX_RESOLUTION},
          surface_version=1,
          criterion=f"resolution={MAX_RESOLUTION} (the declared upper "
                    "boundary) is actually computed, never rejected or "
                    "substituted")
    resp = service.price(_req(resolution=MAX_RESOLUTION, version=1))
    snap = service.store.get("default", 1)
    raw = american_put_crr(snap.spot, 100, 1.0, resp.vol_used, snap.rate,
                           snap.dividend, MAX_RESOLUTION)
    assert resp.pricing_mode == "fixed_resolution"
    assert resp.steps_used == MAX_RESOLUTION
    assert resp.price == raw


def test_resolution_just_beyond_upper_boundary_rejected(audit):
    audit("resolution_over", {"resolution": MAX_RESOLUTION + 1},
          surface_version="n/a",
          criterion=f"{MAX_RESOLUTION + 1} rejected at model layer; "
                    "no silent substitution")
    with pytest.raises(ValidationError):
        _req(resolution=MAX_RESOLUTION + 1)
    with pytest.raises(ValidationError):
        _req(resolution=0)
    with pytest.raises(ValidationError):
        _req(resolution=-5)


def test_fixed_modes_are_distinguishable_from_auto(audit):
    service = _svc()
    audit("modes", {"fixed": 300, "auto": "steps=500->4000"},
          surface_version=1,
          criterion="fixed_resolution vs auto_converged identifiable; "
                    "steps_used reports actual work")
    fixed = service.price(_req(resolution=300, version=1))
    auto = service.price(_req(version=1))
    assert fixed.pricing_mode == "fixed_resolution"
    assert fixed.steps_used == 300
    assert auto.pricing_mode == "auto_converged"
    assert auto.detail["converged"] is True
    assert isinstance(auto.steps_used, int) and auto.steps_used >= 500


def test_fixed_resolution_is_deterministic_and_version_bound(audit):
    service = _svc()
    audit("determinism", {"resolution": 137, "versions": [1, 2]},
          surface_version=[1, 2],
          criterion="repeat runs bit-identical; each version keeps its own "
                    "value; cache flag reported")
    r1 = service.price(_req(resolution=137, version=1))
    r2 = service.price(_req(resolution=137, version=1))
    r3 = service.price(_req(resolution=137, version=1))
    assert r1.price == r2.price == r3.price
    assert r2.cached is True
    other_version = service.price(_req(resolution=137, version=2))
    assert other_version.surface_version == 2
    assert other_version.price == service.price(
        _req(resolution=137, version=2)).price


def test_resolution_rejected_for_european_options(audit):
    service = _svc()
    audit("resolution_european", {"option_type": "vanilla_call",
                                  "resolution": 50},
          surface_version=1,
          criterion="InvalidParameterError, no price returned")
    with pytest.raises(InvalidParameterError) as ei:
        service.price(PricingRequest(
            surface_id="default", version=1,
            option_type="vanilla_call", expiry=1.0, strike=100,
            resolution=50))
    assert ei.value.code == "invalid_parameter"
