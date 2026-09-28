"""Fixed discrete resolution, authorized/bounded extrapolation auditing and
batch pricing semantics (partial failure, single-version binding).
"""
import threading
import time

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from main import app
from option_engine import pricing as P
from option_engine.errors import (
    ExtrapolationNotAuthorizedError,
    InvalidParameterError,
    SurfaceOutOfRangeError,
)
from option_engine.models import (
    BatchItem,
    BatchPricingRequest,
    MAX_RESOLUTION,
    MIN_RESOLUTION,
    PricingRequest,
    ResolutionMode,
    SurfaceGridPayload,
)
from option_engine.service import PricingService
from option_engine.surface import SurfaceStore

client = TestClient(app)


def _store_with_limits(limits=None, vol=0.22):
    store = SurfaceStore()
    store.publish("default", SurfaceGridPayload(
        expiries=[0.25, 1.0], strikes=[90, 100, 110],
        vols=[[vol + 0.02, vol, vol - 0.01],
              [vol + 0.03, vol + 0.005, vol - 0.005]],
        spot=100, rate=0.05,
        extrapolation_limits=limits))
    return store


LIMITS = dict(min_expiry=0.05, max_expiry=3.0,
              min_strike=60.0, max_strike=160.0)


# ---------------------------------------------------------------- resolution

def test_fixed_resolution_is_exact_and_deterministic(audit):
    store = _store_with_limits()
    svc = PricingService(store)
    for n in (1, 2, 3, 17, 100):
        audit("fixed_resolution", {"steps": n}, surface_version=1,
              criterion="price == direct n-step CRR; mode fixed; steps==n")
        resp = svc.price(PricingRequest(
            option_type="american_put", expiry=1.0, strike=100,
            resolution=n))
        direct = P.american_put_crr(100, 100, 1.0, resp.vol_used,
                                    0.05, 0.0, n)
        assert resp.resolution_mode is ResolutionMode.FIXED
        assert resp.resolution_used == n
        assert resp.detail["steps_used"] == n
        assert resp.detail["converged"] is False
        assert resp.price == direct
        # Repeated identical requests on the pinned version are identical.
        again = svc.price(PricingRequest(
            surface_id="default", version=1,
            option_type="american_put", expiry=1.0, strike=100,
            resolution=n))
        assert again.price == resp.price
        assert again.surface_version == 1


def test_fixed_mode_is_distinguishable_from_converged_mode(audit):
    store = _store_with_limits()
    svc = PricingService(store)
    audit("mode_distinction", {}, surface_version=1,
          criterion="fixed vs converged flags/steps differ and are explicit")
    fixed = svc.price(PricingRequest(
        option_type="american_put", expiry=1.0, strike=100, resolution=50))
    converged = svc.price(PricingRequest(
        option_type="american_put", expiry=1.0, strike=100))
    assert fixed.resolution_mode is ResolutionMode.FIXED
    assert fixed.resolution_used == 50
    assert converged.resolution_mode is ResolutionMode.CONVERGED
    assert converged.detail["converged"] is True
    assert converged.resolution_used >= 500
    # European closed-form responses keep the converged default, no steps.
    eu = svc.price(PricingRequest(
        option_type="vanilla_call", expiry=1.0, strike=100))
    assert eu.resolution_mode is ResolutionMode.CONVERGED
    assert eu.resolution_used is None


def test_resolution_boundary_values_accepted_or_rejected_predictably(audit):
    store = _store_with_limits()
    svc = PricingService(store)

    # Lower boundary (1) and upper boundary are accepted by the model and
    # actually priced. The upper bound is priced once (deterministic) to
    # prove it is not rejected before the advertised limit.
    audit("resolution_boundaries",
          {"min": MIN_RESOLUTION, "max": MAX_RESOLUTION},
          surface_version=1,
          criterion="both boundary values price; outside rejected")
    low = svc.price(PricingRequest(
        option_type="american_put", expiry=1.0, strike=100,
        resolution=MIN_RESOLUTION))
    assert low.resolution_used == MIN_RESOLUTION
    high = svc.price(PricingRequest(
        option_type="american_put", expiry=1.0, strike=100,
        resolution=MAX_RESOLUTION))
    assert high.resolution_used == MAX_RESOLUTION
    direct_hi = P.american_put_crr(100, 100, 1.0, high.vol_used,
                                   0.05, 0.0, MAX_RESOLUTION)
    assert high.price == direct_hi

    for bad in (0, -1, MAX_RESOLUTION + 1, 10 ** 9):
        audit("resolution_rejected", {"resolution": bad},
              surface_version="n/a",
              criterion=f"model validation rejects outside "
                        f"[{MIN_RESOLUTION}, {MAX_RESOLUTION}]")
        with pytest.raises(ValidationError):
            PricingRequest(option_type="american_put", expiry=1.0,
                           strike=100, resolution=bad)

    # Booleans and non-integer values must not be coerced into resolutions.
    for bad in (True, False, 50.5):
        with pytest.raises(ValidationError):
            PricingRequest(option_type="american_put", expiry=1.0,
                           strike=100, resolution=bad)


def test_resolution_engine_layer_guard(audit):
    # A request that bypasses pydantic (model_construct) is still rejected
    # by the engine with InvalidParameterError rather than mis-priced.
    store = _store_with_limits()
    svc = PricingService(store)
    audit("resolution_engine_guard", {"resolution": 999999},
          surface_version="n/a",
          criterion="engine rejects out-of-range resolution past model")
    req = PricingRequest.model_construct(
        surface_id="default", version=None,
        option_type=__import__("option_engine.models", fromlist=["OptionType"]
                               ).OptionType.AMERICAN_PUT,
        expiry=1.0, strike=100, spot=None, rate=None, dividend=None,
        payout=1.0, steps=500, convergence_tol=1e-4, max_steps=4000,
        resolution=999999, target_price=None, allow_extrapolation=False,
        request_id=None)
    with pytest.raises(InvalidParameterError):
        svc.price(req)


def test_resolution_only_for_american_and_mutex_with_converged_controls(audit):
    audit("resolution_type_guard", {}, surface_version="n/a",
          criterion="rejected for European; mutually exclusive with "
                    "steps/tol/max_steps")
    with pytest.raises(ValidationError):
        PricingRequest(option_type="vanilla_call", expiry=1.0, strike=100,
                       resolution=100)
    with pytest.raises(ValidationError):
        PricingRequest(option_type="american_put", expiry=1.0, strike=100,
                       resolution=100, steps=100)
    with pytest.raises(ValidationError):
        PricingRequest(option_type="american_put", expiry=1.0, strike=100,
                       resolution=100, max_steps=2000)
    with pytest.raises(ValidationError):
        PricingRequest(option_type="american_put", expiry=1.0, strike=100,
                       resolution=100, convergence_tol=1e-8)


# ------------------------------------------------------------ extrapolation

def test_extrapolation_requires_authorization_and_is_audited(audit):
    store = _store_with_limits(LIMITS)
    svc = PricingService(store)

    audit("extrap_default_denied", {"expiry": 2.0, "strike": 100},
          surface_version=1,
          criterion="out-of-grid without flag -> surface_out_of_range")
    with pytest.raises(SurfaceOutOfRangeError):
        svc.price(PricingRequest(option_type="vanilla_call",
                                 expiry=2.0, strike=100))

    audit("extrap_authorized", {"expiry": 2.0, "strike": 100},
          surface_version=1,
          criterion="authorized point served; audit shows one axis overshoot")
    resp = svc.price(PricingRequest(option_type="vanilla_call",
                                    expiry=2.0, strike=100,
                                    allow_extrapolation=True))
    assert resp.extrapolated is True
    axes = resp.extrapolation["axes"]
    assert [a["axis"] for a in axes] == ["expiry"]
    assert axes[0]["side"] == "above"
    assert axes[0]["grid_edge"] == 1.0
    assert axes[0]["limit_edge"] == 3.0
    assert axes[0]["headroom_fraction_used"] == pytest.approx(0.5)
    assert resp.extrapolation["surface_version"] == 1

    # In-grid points never flag extrapolation even with the flag set.
    inside = svc.price(PricingRequest(option_type="vanilla_call",
                                      expiry=1.0, strike=100,
                                      allow_extrapolation=True))
    assert inside.extrapolated is False
    assert inside.extrapolation is None

    # Exactly on a declared limit edge is still serviceable; beyond it,
    # authorization does not help (no silent clamp).
    edge = svc.price(PricingRequest(option_type="vanilla_call",
                                    expiry=3.0, strike=100,
                                    allow_extrapolation=True))
    assert edge.extrapolated is True
    with pytest.raises(SurfaceOutOfRangeError):
        svc.price(PricingRequest(option_type="vanilla_call",
                                 expiry=3.0001, strike=100,
                                 allow_extrapolation=True))


def test_extrapolation_undefined_on_version_without_limits(audit):
    store = _store_with_limits(None)
    svc = PricingService(store)
    audit("extrap_not_authorized", {"expiry": 2.0}, surface_version=1,
          criterion="explicit request vs version w/o limits -> "
                    "extrapolation_not_authorized")
    with pytest.raises(ExtrapolationNotAuthorizedError) as ei:
        svc.price(PricingRequest(option_type="vanilla_call",
                                 expiry=2.0, strike=100,
                                 allow_extrapolation=True))
    assert ei.value.code == "extrapolation_not_authorized"
    with pytest.raises(SurfaceOutOfRangeError):
        svc.price(PricingRequest(option_type="vanilla_call",
                                 expiry=2.0, strike=100))


def test_historical_versions_keep_their_own_extrapolation_policy(audit):
    store = _store_with_limits(None)
    # v2 adds limits; v1 must remain forbidden forever.
    payload = SurfaceGridPayload(
        expiries=[0.25, 1.0], strikes=[90, 100, 110],
        vols=[[0.24, 0.22, 0.21], [0.25, 0.225, 0.215]],
        spot=100, rate=0.05, extrapolation_limits=LIMITS)
    store.publish("default", payload)
    svc = PricingService(store)

    audit("version_limits", {"v1": "none", "v2": "limits"},
          surface_version=[1, 2],
          criterion="v1 forbids; v2 serves the same point; v2 beyond limits "
                    "refused")
    with pytest.raises(ExtrapolationNotAuthorizedError):
        svc.price(PricingRequest(version=1, option_type="vanilla_call",
                                 expiry=2.0, strike=100,
                                 allow_extrapolation=True))
    v2 = svc.price(PricingRequest(version=2, option_type="vanilla_call",
                                  expiry=2.0, strike=100,
                                  allow_extrapolation=True))
    assert v2.surface_version == 2 and v2.extrapolated is True
    with pytest.raises(SurfaceOutOfRangeError):
        svc.price(PricingRequest(version=2, option_type="vanilla_call",
                                 expiry=5.0, strike=100,
                                 allow_extrapolation=True))


def test_publish_rejects_limits_not_containing_grid(audit):
    audit("bad_limits", {}, surface_version="n/a",
          criterion="limits must contain grid support; publish rejected")
    store = SurfaceStore()
    with pytest.raises(InvalidParameterError):
        store.publish("x", SurfaceGridPayload(
            expiries=[0.25, 1.0], strikes=[90, 100, 110],
            vols=[[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]], spot=100,
            extrapolation_limits=dict(min_expiry=0.5, max_expiry=2.0,
                                      min_strike=90, max_strike=110)))
    # Strike limits inside the grid are rejected too.
    with pytest.raises(InvalidParameterError):
        store.publish("x", SurfaceGridPayload(
            expiries=[0.25, 1.0], strikes=[90, 100, 110],
            vols=[[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]], spot=100,
            extrapolation_limits=dict(min_expiry=0.1, max_expiry=2.0,
                                      min_strike=95, max_strike=105)))


# ------------------------------------------------------------------- batch

def _book():
    from pydantic import ValidationError as _VE  # noqa

    items = [
        BatchItem(id="call", option_type="vanilla_call",
                  expiry=1.0, strike=100),
        BatchItem(id="oob", option_type="vanilla_call",
                  expiry=99.0, strike=100),  # out of range
        BatchItem(id="am50", option_type="american_put",
                  expiry=1.0, strike=100, resolution=50),
        BatchItem(id="digital", option_type="digital_put",
                  expiry=0.5, strike=105),
    ]
    # An item whose parameters are structurally valid but is rejected at
    # pricing time (impossible IV target -> invalid_parameter), proving a
    # per-item engine failure is isolated like every other error.
    items.append(BatchItem(id="bad_iv", option_type="vanilla_call",
                           expiry=1.0, strike=100, target_price=10_000.0))
    return BatchPricingRequest(surface_id="default", items=items)


def test_invalid_batch_item_shape_is_rejected_at_request_layer(audit):
    audit("batch_item_shape", {}, surface_version="n/a",
          criterion="resolution on a European item -> 422 for the request")
    with pytest.raises(ValidationError):
        BatchPricingRequest(items=[
            BatchItem(option_type="vanilla_call", expiry=1.0, strike=100,
                      resolution=10)])


def test_batch_partial_failure_is_isolated_and_indexed(audit):
    store = _store_with_limits()
    svc = PricingService(store)
    audit("batch_partial", {"items": 5}, surface_version=1,
          criterion="successes and per-item errors coexist; order preserved")
    resp = svc.price_batch(_book())
    assert resp.surface_version == 1
    assert [it.index for it in resp.items] == [0, 1, 2, 3, 4]
    assert [it.id for it in resp.items] == [
        "call", "oob", "am50", "digital", "bad_iv"]

    ok = {it.id: it for it in resp.items if it.result}
    fails = {it.id: it for it in resp.items if it.error}
    assert set(ok) == {"call", "am50", "digital"}
    assert set(fails) == {"oob", "bad_iv"}
    assert fails["oob"].error.code == "surface_out_of_range"
    assert fails["bad_iv"].error.code == "invalid_parameter"
    assert ok["am50"].result.resolution_used == 50
    assert ok["am50"].result.resolution_mode is ResolutionMode.FIXED


def test_batch_items_equal_standalone_calls_on_same_version(audit):
    store = _store_with_limits()
    svc = PricingService(store)
    audit("batch_equals_solo", {"items": 3}, surface_version=1,
          criterion="each batch result equals a standalone pinned call")
    reqs = [
        PricingRequest(surface_id="default", version=1,
                       option_type="vanilla_call", expiry=1.0, strike=100),
        PricingRequest(surface_id="default", version=1,
                       option_type="american_put", expiry=1.0, strike=95,
                       resolution=123),
        PricingRequest(surface_id="default", version=1,
                       option_type="digital_put", expiry=0.5, strike=105),
    ]
    batch = BatchPricingRequest(surface_id="default", version=1, items=[
        BatchItem(**r.model_dump(exclude={"surface_id", "version",
                                          "request_id"}))
        for r in reqs])
    resp = svc.price_batch(batch)
    for item, solo_req in zip(resp.items, reqs):
        solo = svc.price(solo_req)
        assert item.result is not None
        assert item.result.price == solo.price
        assert item.result.vol_used == solo.vol_used
        assert item.result.surface_version == solo.surface_version == 1
        assert item.result.resolution_used == solo.resolution_used


def test_batch_pins_one_version_even_when_publish_races(audit):
    store = _store_with_limits()
    svc = PricingService(store)

    def grid(level):
        return SurfaceGridPayload(
            expiries=[0.25, 1.0], strikes=[90, 100, 110],
            vols=[[level] * 3 for _ in range(2)], spot=100, rate=0.05)

    stop = threading.Event()

    def publisher():
        while not stop.is_set():
            store.publish("default", grid(0.15 + 0.001 * (time.time_ns() % 50)))

    t = threading.Thread(target=publisher)
    t.start()
    try:
        versions = set()
        audit("batch_version_pin", {"runs": 10}, surface_version="pinned",
              criterion="every run reports one version; items share it")
        for _ in range(10):
            batch = BatchPricingRequest(items=[
                BatchItem(option_type="vanilla_call", expiry=1.0,
                          strike=100) for _ in range(8)])
            resp = svc.price_batch(batch)
            versions.add(resp.surface_version)
            assert {it.result.surface_version for it in resp.items} == {
                resp.surface_version}
    finally:
        stop.set()
        t.join()
    assert versions  # ran under possibly several different latest versions


def test_batch_slow_item_does_not_block_others(audit):
    # A deliberately slow, non-converging American item sits beside cheap
    # items; the cheap ones must all finish promptly with the heavy item
    # failing in isolation.
    batch = BatchPricingRequest(surface_id="default", items=[
        BatchItem(id="heavy", option_type="american_put",
                  expiry=1.0, strike=100, spot=100,
                  steps=2000, max_steps=4000, convergence_tol=1e-12),
    ] + [BatchItem(id=f"c{i}", option_type="vanilla_call",
                   expiry=1.0, strike=100) for i in range(8)])

    from option_engine.api import service as api_service
    audit("batch_isolation", {"cheap": 8, "heavy": 1},
          surface_version="latest",
          criterion="cheap items succeed; heavy fails on its own slot")
    start = time.time()
    resp = api_service.price_batch(batch)
    elapsed = time.time() - start
    by_id = {it.id: it for it in resp.items}
    assert all(by_id[f"c{i}"].result is not None for i in range(8))
    assert by_id["heavy"].error is not None
    assert by_id["heavy"].error.code == "solver_non_convergence"
    # Cheap items were served concurrently, far below the heavy item's
    # serial compute time.
    assert elapsed < 15


def test_batch_extrapolation_flag_is_per_item(audit):
    store = _store_with_limits(LIMITS)
    svc = PricingService(store)
    audit("batch_extrap", {"point": (2.0, 100)}, surface_version=1,
          criterion="identical point: authorized item served & audited, "
                    "unauthorized item rejected in its own slot")
    batch = BatchPricingRequest(items=[
        BatchItem(id="granted", option_type="vanilla_call",
                  expiry=2.0, strike=100, allow_extrapolation=True),
        BatchItem(id="denied", option_type="vanilla_call",
                  expiry=2.0, strike=100, allow_extrapolation=False),
    ])
    resp = svc.price_batch(batch)
    granted, denied = resp.items
    assert granted.result is not None
    assert granted.result.extrapolated is True
    assert granted.result.extrapolation["axes"][0]["axis"] == "expiry"
    assert denied.error is not None
    assert denied.error.code == "surface_out_of_range"


def test_batch_http_contracts(audit):
    audit("batch_http", {}, surface_version="default/1",
          criterion="200 envelope; empty/oversized -> 422")
    r = client.post("/pricing/batch", json={
        "surface_id": "default", "items": [
            {"id": "a", "option_type": "vanilla_call",
             "expiry": 1.0, "strike": 100},
            {"id": "b", "option_type": "vanilla_call",
             "expiry": 99.0, "strike": 100}]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["surface_id"] == "default"
    assert isinstance(body["surface_version"], int)
    assert body["items"][0]["result"]["price"] > 0
    assert body["items"][1]["error"]["code"] == "surface_out_of_range"

    assert client.post("/pricing/batch", json={"items": []}).status_code == 422
    dup = client.post("/pricing/batch", json={"items": [
        {"id": "x", "option_type": "vanilla_call", "expiry": 1,
         "strike": 100},
        {"id": "x", "option_type": "vanilla_call", "expiry": 1,
         "strike": 100}]})
    assert dup.status_code == 422
    # Items cannot smuggle their own version binding.
    smuggled = client.post("/pricing/batch", json={"items": [
        {"option_type": "vanilla_call", "expiry": 1, "strike": 100,
         "version": 1}]})
    assert smuggled.status_code == 422
