"""Strict typing for the fixed ``resolution`` field.

Regression coverage for the tightening: booleans, floats (even integral
ones like ``1.0``) and numeric strings must be rejected as *invalid types*
with distinguishable, field-specific reasons at every entry point
(synchronous pricing, asynchronous jobs and each batch leg), and never
silently coerced into another step count. Genuine integers inside the
allowed range, including both boundaries, are still accepted and computed
deterministically.
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from main import app
from option_engine.errors import InvalidParameterError
from option_engine.models import (
    BatchPricingItem,
    BatchPricingRequest,
    MAX_RESOLUTION,
    MIN_RESOLUTION,
    OptionType,
    PricingRequest,
)
from option_engine.pricing import american_put_crr

client = TestClient(app)

# (label, illegal raw value, marker that must appear in the reason)
ILLEGAL_TYPED = [
    ("boolean_true", True, "boolean"),
    ("boolean_false", False, "boolean"),
    ("float_integral", 1.0, "float"),
    ("float_fractional", 1.5, "float"),
    ("string_integer", "100", "string"),
    ("string_decimal", "1.0", "string"),
    ("string_bool", "true", "string"),
]
ILLEGAL_RANGED = [
    ("zero", 0, "within"),
    ("negative", -5, "within"),
    ("above_cap", MAX_RESOLUTION + 1, "within"),
]


def _american_body(**over):
    body = {"surface_id": "default", "version": 1,
            "option_type": "american_put", "expiry": 1.0, "strike": 100}
    body.update(over)
    return body


# ---------------------------------------------------------------------------
# Model layer: single-request typing rules
# ---------------------------------------------------------------------------
def test_illegal_resolution_types_rejected_at_model_layer(audit):
    for label, value, marker in ILLEGAL_TYPED + ILLEGAL_RANGED:
        audit("resolution_type_model", {"label": label, "resolution": value},
              surface_version="n/a",
              criterion=f"ValidationError naming field 'resolution', reason "
                        f"mentions {marker}; never coerced to a step count")
        with pytest.raises(ValidationError) as ei:
            PricingRequest(option_type="american_put", expiry=1.0,
                           strike=100, resolution=value)
        err = ei.value.errors()[0]
        assert err["loc"] == ("resolution",), err
        assert "resolution" in err["msg"]
        assert marker in err["msg"]


def test_rejected_resolution_is_never_coerced(audit):
    # The previously-buggy coercions must not produce a model at all.
    audit("resolution_no_coercion", {"true->1": True, "1.0->1": 1.0,
                                     "'100'->100": "100"},
          surface_version="n/a",
          criterion="none of the values constructs a PricingRequest")
    for value in (True, 1.0, "100"):
        with pytest.raises(ValidationError):
            PricingRequest(option_type="american_put", expiry=1.0,
                           strike=100, resolution=value)


def test_boundary_integers_accepted_and_others_rejected(audit):
    audit("resolution_boundaries",
          {"min": MIN_RESOLUTION, "max": MAX_RESOLUTION},
          surface_version=1,
          criterion="1 and 2000 accepted as exact ints; 0/2001 rejected")
    assert PricingRequest(option_type="american_put", expiry=1.0,
                          strike=100, resolution=MIN_RESOLUTION).resolution == 1
    assert PricingRequest(option_type="american_put", expiry=1.0,
                          strike=100,
                          resolution=MAX_RESOLUTION).resolution == 2000
    for bad in (0, MAX_RESOLUTION + 1):
        with pytest.raises(ValidationError):
            PricingRequest(option_type="american_put", expiry=1.0,
                           strike=100, resolution=bad)


# ---------------------------------------------------------------------------
# Synchronous HTTP entry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label,value,marker",
                         [pytest.param(l, v, m, id=l)
                          for l, v, m in ILLEGAL_TYPED + ILLEGAL_RANGED])
def test_sync_rejects_illegal_resolution_with_422(audit, label, value, marker):
    audit("resolution_type_sync", {"label": label, "resolution": value},
          surface_version="n/a",
          criterion="HTTP 422 invalid_parameter; field resolution; "
                    f"reason mentions {marker}; no price returned")
    r = client.post("/pricing/price", json=_american_body(resolution=value))
    assert r.status_code == 422, r.text
    body = r.json()
    assert body["error"] == "invalid_parameter"
    assert "resolution" in body["message"]
    assert marker in body["message"]
    loc = body["detail"]["errors"][0]["loc"]
    assert loc[-1] == "resolution"
    assert "price" not in body  # never a price envelope on failure


def test_sync_boundaries_are_computed_deterministically(audit, service):
    audit("resolution_sync_boundaries", {"min": 1, "max": 2000},
          surface_version=1,
          criterion="both boundaries priced fixed_resolution with exact "
                    "steps; repeat runs bit-identical")
    lo = client.post("/pricing/price",
                     json=_american_body(resolution=MIN_RESOLUTION)).json()
    hi = client.post("/pricing/price",
                     json=_american_body(resolution=MAX_RESOLUTION)).json()
    assert lo["pricing_mode"] == "fixed_resolution"
    assert lo["steps_used"] == MIN_RESOLUTION
    assert hi["pricing_mode"] == "fixed_resolution"
    assert hi["steps_used"] == MAX_RESOLUTION
    # Determinism: same request pinned to the same version is bit-identical.
    lo2 = client.post("/pricing/price",
                      json=_american_body(resolution=MIN_RESOLUTION)).json()
    assert lo2["price"] == lo["price"]
    # Cross-check the reported price really is CRR at that exact step count.
    snap = service.store.get("default", 1)
    raw_hi = american_put_crr(snap.spot, 100, 1.0, hi["vol_used"], snap.rate,
                              snap.dividend, MAX_RESOLUTION)
    assert hi["price"] == raw_hi


# ---------------------------------------------------------------------------
# Asynchronous job entry: illegal resolution is refused at submission
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label,value,marker",
                         [pytest.param(l, v, m, id=l)
                          for l, v, m in ILLEGAL_TYPED + ILLEGAL_RANGED])
def test_async_rejects_illegal_resolution_at_submission(
        audit, label, value, marker):
    audit("resolution_type_async", {"label": label, "resolution": value},
          surface_version="n/a",
          criterion="POST /jobs returns 422 invalid_parameter at submission "
                    "(no job created); field resolution named in reason")
    r = client.post("/jobs", json=_american_body(resolution=value))
    assert r.status_code == 422, r.text
    body = r.json()
    assert body["error"] == "invalid_parameter"
    assert "resolution" in body["message"]
    assert marker in body["message"]
    assert body["detail"]["errors"][0]["loc"][-1] == "resolution"


def test_async_valid_resolution_still_succeeds(audit):
    audit("resolution_async_valid", {"resolution": 41},
          surface_version="default/1",
          criterion="job SUCCEEDS with fixed_resolution / 41 steps")
    r = client.post("/jobs", json=_american_body(resolution=41))
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    final = {}
    for _ in range(300):
        final = client.get(f"/jobs/{job_id}").json()
        if final["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.01)
    assert final["status"] == "succeeded", final
    assert final["result"]["pricing_mode"] == "fixed_resolution"
    assert final["result"]["steps_used"] == 41


# ---------------------------------------------------------------------------
# Batch entry: per-leg rejection, full isolation
# ---------------------------------------------------------------------------
def test_batch_illegal_resolution_fails_only_that_leg(audit, service):
    items = [
        # 0: legal fixed leg
        BatchPricingItem(option_type="american_put", expiry=1.0, strike=100,
                         resolution=137),
        # 1..6: various illegal resolutions (typed + ranged)
        BatchPricingItem(option_type="american_put", expiry=1.0, strike=100,
                         resolution=True),
        BatchPricingItem(option_type="american_put", expiry=1.0, strike=100,
                         resolution=1.0),
        BatchPricingItem(option_type="american_put", expiry=1.0, strike=100,
                         resolution="100"),
        BatchPricingItem(option_type="american_put", expiry=1.0, strike=100,
                         resolution=0),
        BatchPricingItem(option_type="american_put", expiry=1.0, strike=100,
                         resolution=-5),
        BatchPricingItem(option_type="american_put", expiry=1.0, strike=100,
                         resolution=MAX_RESOLUTION + 1),
        # 7: legal closed-form leg keeps working
        BatchPricingItem(option_type="vanilla_call", expiry=1.0, strike=100),
        # 8: upper boundary leg still accepted
        BatchPricingItem(option_type="american_put", expiry=1.0, strike=100,
                         resolution=MAX_RESOLUTION),
    ]
    audit("resolution_batch_isolation",
          {"legs": len(items), "illegal_legs": [1, 2, 3, 4, 5, 6]},
          surface_version=1,
          criterion="HTTP 200; only illegal legs fail with their own "
                    "invalid_parameter reason; other legs priced normally")
    req = BatchPricingRequest(surface_id="default", version=1, items=items)
    from option_engine.api import batch
    resp = batch.price_batch(req)
    by_index = {x.index: x for x in resp.results}
    assert sorted(by_index) == list(range(len(items)))

    good = by_index[0]
    assert good.status == "succeeded"
    assert good.result.pricing_mode == "fixed_resolution"
    assert good.result.steps_used == 137
    assert by_index[7].status == "succeeded"
    assert by_index[7].result.pricing_mode == "closed_form"
    edge = by_index[8]
    assert edge.status == "succeeded"
    assert edge.result.steps_used == MAX_RESOLUTION

    for bad_idx, marker in [(1, "boolean"), (2, "float"), (3, "string"),
                            (4, "within"), (5, "within"), (6, "within")]:
        leg = by_index[bad_idx]
        assert leg.status == "failed", (bad_idx, leg)
        assert leg.error_code == "invalid_parameter"
        assert leg.result is None
        assert "resolution" in leg.error_message
        assert marker in leg.error_message


def test_batch_illegal_resolution_via_http_partial_200(audit):
    body = {"surface_id": "default", "version": 1, "items": [
        {"option_type": "american_put", "expiry": 1.0, "strike": 100,
         "resolution": 51},
        {"option_type": "american_put", "expiry": 1.0, "strike": 100,
         "resolution": True},
        {"option_type": "vanilla_put", "expiry": 1.0, "strike": 100},
        {"option_type": "american_put", "expiry": 1.0, "strike": 100,
         "resolution": "252"},
    ]}
    audit("resolution_batch_http", {"legs": 4, "illegal": [1, 3]},
          surface_version=1,
          criterion="batch returns 200; legs 1 and 3 fail independently "
                    "with invalid_parameter; legs 0 and 2 succeed")
    r = client.post("/pricing/price/batch", json=body)
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert [x["index"] for x in results] == [0, 1, 2, 3]
    assert results[0]["status"] == "succeeded"
    assert results[0]["result"]["steps_used"] == 51
    assert results[1]["status"] == "failed"
    assert results[1]["error_code"] == "invalid_parameter"
    assert "boolean" in results[1]["error_message"]
    assert results[2]["status"] == "succeeded"
    assert results[3]["status"] == "failed"
    assert results[3]["error_code"] == "invalid_parameter"
    assert "string" in results[3]["error_message"]


def test_batch_illegal_leg_matches_standalone_failure(audit):
    # Consistency: a standalone call with the same illegal value is a 422
    # invalid_parameter; in batch the same leg is failed with that same code
    # and reason rather than the whole request failing.
    audit("resolution_batch_consistency", {"resolution": True},
          surface_version="default/1",
          criterion="same invalid_parameter reason standalone vs batch leg")
    standalone = client.post(
        "/pricing/price", json=_american_body(resolution=True))
    assert standalone.status_code == 422
    sync_msg = standalone.json()["message"]
    assert "boolean" in sync_msg

    body = {"surface_id": "default", "version": 1, "items": [
        {"option_type": "american_put", "expiry": 1.0, "strike": 100,
         "resolution": True}]}
    r = client.post("/pricing/price/batch", json=body)
    assert r.status_code == 200
    leg = r.json()["results"][0]
    assert leg["status"] == "failed"
    assert leg["error_code"] == "invalid_parameter"
    assert "boolean" in leg["error_message"]


def test_batch_leg_with_bad_resolution_does_not_block_other_legs(audit):
    # Timing/isolation: a heavy valid leg next to a rejected leg still prices.
    from option_engine.api import batch
    req = BatchPricingRequest(surface_id="default", version=1, items=[
        BatchPricingItem(option_type="american_put", expiry=1.0, strike=100,
                         resolution=1500),
        BatchPricingItem(option_type="american_put", expiry=1.0, strike=100,
                         resolution="nope"),
        BatchPricingItem(option_type="vanilla_call", expiry=1.0, strike=100),
    ])
    audit("resolution_batch_no_block", {"heavy": 1500, "bad_string": 1},
          surface_version=1,
          criterion="heavy leg priced, bad leg isolated, cheap leg priced")
    resp = batch.price_batch(req)
    results = {x.index: x for x in resp.results}
    assert results[0].status == "succeeded"
    assert results[0].result.steps_used == 1500
    assert results[1].status == "failed"
    assert results[1].error_code == "invalid_parameter"
    assert results[2].status == "succeeded"


# ---------------------------------------------------------------------------
# Engine-layer guard: even a request that bypassed model validation is
# rejected deterministically.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value,marker",
                         [pytest.param(True, "boolean", id="bool"),
                          pytest.param(1.0, "float", id="float"),
                          pytest.param("100", "string", id="string"),
                          pytest.param(0, "within", id="zero")])
def test_engine_layer_rejects_bypassed_bad_resolution(audit, service,
                                                      value, marker):
    audit("resolution_engine_guard", {"resolution": value},
          surface_version="n/a",
          criterion="InvalidParameterError even when model layer is bypassed")
    req = PricingRequest.model_construct(
        surface_id="default", version=1, option_type=OptionType.AMERICAN_PUT,
        expiry=1.0, strike=100, spot=None, rate=None, dividend=None,
        payout=1.0, resolution=value, allow_extrapolation=False,
        steps=500, convergence_tol=1e-4, max_steps=4000, target_price=None)
    with pytest.raises(InvalidParameterError) as ei:
        service.price(req)
    assert ei.value.code == "invalid_parameter"
    assert marker in str(ei.value)
