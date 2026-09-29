"""Strict resolution validation: bool/float/string values must be rejected
(never coerced into another step count) identically at every entry point —
synchronous pricing, asynchronous jobs and every batch leg — while exact
integers on both range boundaries keep pricing deterministically.

Reproduces the backtest/reconciliation regression where ``true`` was
silently priced as a 1-step tree.
"""
import time

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from pydantic_core import PydanticCustomError

from main import app
from option_engine.errors import InvalidParameterError
from option_engine.models import (
    MAX_RESOLUTION,
    MIN_RESOLUTION,
    BatchPricingItem,
    BatchPricingRequest,
    PricingRequest,
    validate_resolution,
)

client = TestClient(app, raise_server_exceptions=False)

ILLEGAL_TYPED = [
    ("boolean_true", True, "resolution_bad_type"),
    ("boolean_false", False, "resolution_bad_type"),
    ("float_integral", 5.0, "resolution_bad_type"),
    ("float_fractional", 5.5, "resolution_bad_type"),
    ("float_boundary", float(MAX_RESOLUTION), "resolution_bad_type"),
    ("string_integer", "5", "resolution_bad_type"),
    ("string_decimal", "5.0", "resolution_bad_type"),
    ("string_bool", "true", "resolution_bad_type"),
    ("list", [5], "resolution_bad_type"),
]
ILLEGAL_OUT_OF_RANGE = [
    ("zero", 0),
    ("negative", -7),
    ("just_above_max", MAX_RESOLUTION + 1),
    ("far_above_max", MAX_RESOLUTION * 10),
]


def _american_body(**over):
    body = {
        "surface_id": "tech", "version": 1,
        "option_type": "american_put", "expiry": 0.75, "strike": 200,
        "spot": 200,
    }
    body.update(over)
    return body


# ---------------------------------------------------------------- model layer

def test_validate_resolution_helper_rules():
    assert validate_resolution(None) is None
    for n in (MIN_RESOLUTION, 137, MAX_RESOLUTION):
        assert validate_resolution(n) is n
    for label, value, etype in ILLEGAL_TYPED:
        with pytest.raises(PydanticCustomError) as ei:
            validate_resolution(value)
        assert ei.value.type == etype, label
    for label, value in ILLEGAL_OUT_OF_RANGE:
        with pytest.raises(PydanticCustomError) as ei:
            validate_resolution(value)
        assert ei.value.type == "resolution_out_of_range", label


def test_request_model_rejects_non_integer_resolution(audit):
    for label, value, etype in ILLEGAL_TYPED:
        audit("resolution_strict", {"case": label, "resolution": str(value)},
              surface_version="n/a",
              criterion=f"model rejects {label} with {etype}; never coerced")
        with pytest.raises(ValidationError) as ei:
            PricingRequest(option_type="american_put", expiry=1.0,
                           strike=100, resolution=value)
        err = ei.value.errors()[0]
        assert err["loc"] == ("resolution",)
        assert err["type"] == etype
        assert "resolution" in err["msg"]
    for label, value in ILLEGAL_OUT_OF_RANGE:
        with pytest.raises(ValidationError) as ei:
            PricingRequest(option_type="american_put", expiry=1.0,
                           strike=100, resolution=value)
        assert ei.value.errors()[0]["type"] == "resolution_out_of_range"


def test_request_model_accepts_boundary_integers():
    for n in (MIN_RESOLUTION, MAX_RESOLUTION):
        req = PricingRequest(option_type="american_put", expiry=1.0,
                             strike=100, resolution=n)
        assert req.resolution is n and type(req.resolution) is int


def test_engine_layer_rejects_bypassed_model(service):
    # model_construct skips pydantic validation; the engine must still
    # refuse an illegal resolution instead of pricing a coerced step count.
    for value, etype in [(True, "boolean"), (5.0, "float"), ("5", "string"),
                         (0, "range"), (MAX_RESOLUTION + 1, "range")]:
        req = PricingRequest.model_construct(
            surface_id="tech", version=1, option_type="american_put",
            expiry=0.75, strike=200, spot=None, rate=None, dividend=None,
            payout=1.0, resolution=value, allow_extrapolation=False,
            steps=500, convergence_tol=1e-4, max_steps=4000,
            target_price=None)
        with pytest.raises(InvalidParameterError) as ei:
            service.price(req)
        assert "resolution" in str(ei.value)
        if etype != "range":
            assert etype in str(ei.value)


# ------------------------------------------------------------------ sync entry

@pytest.mark.parametrize("label,value,etype",
                         ILLEGAL_TYPED, ids=[c[0] for c in ILLEGAL_TYPED])
def test_sync_rejects_illegal_resolution_types(audit, label, value, etype):
    audit("sync_resolution_illegal", {"case": label},
          surface_version="tech/1",
          criterion="422 invalid_parameter, field resolution named, "
                    f"type {etype}, no price returned")
    r = client.post("/pricing/price",
                    json=_american_body(resolution=value))
    assert r.status_code == 422, r.text
    body = r.json()
    assert body["error"] == "invalid_parameter"
    assert "resolution" in body["message"]
    assert etype in body["message"]
    err = body["detail"]["errors"][0]
    assert err["loc"] == ["body", "resolution"]
    assert err["type"] == etype


@pytest.mark.parametrize("label,value",
                         ILLEGAL_OUT_OF_RANGE,
                         ids=[c[0] for c in ILLEGAL_OUT_OF_RANGE])
def test_sync_rejects_out_of_range_resolution(audit, label, value):
    audit("sync_resolution_range", {"case": label, "value": value},
          surface_version="n/a",
          criterion="422 resolution_out_of_range; no substitution")
    r = client.post("/pricing/price",
                    json=_american_body(resolution=value))
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "invalid_parameter"
    assert body["detail"]["errors"][0]["type"] == "resolution_out_of_range"


def test_sync_legal_boundaries_compute_and_report_steps(audit):
    prices = {}
    for n in (MIN_RESOLUTION, MAX_RESOLUTION):
        audit("sync_resolution_boundary", {"resolution": n},
              surface_version="tech/1",
              criterion=f"200, fixed_resolution, steps_used={n}, "
                        "deterministic across repeats")
        r1 = client.post("/pricing/price",
                         json=_american_body(resolution=n))
        assert r1.status_code == 200, r1.text
        j = r1.json()
        assert j["pricing_mode"] == "fixed_resolution"
        assert j["steps_used"] == n
        assert j["surface_version"] == 1
        # Same request on the same pinned version: bit-identical repeats.
        r2 = client.post("/pricing/price",
                         json=_american_body(resolution=n))
        assert r2.json()["price"] == j["price"]
        prices[n] = j["price"]
    # Distinct step counts must not collapse into the same substituted value.
    assert prices[MIN_RESOLUTION] != prices[MAX_RESOLUTION]


# ----------------------------------------------------------------- async entry

@pytest.mark.parametrize("label,value,etype",
                         ILLEGAL_TYPED, ids=[c[0] for c in ILLEGAL_TYPED])
def test_async_rejects_illegal_resolution_at_submission(audit, label, value, etype):
    audit("async_resolution_illegal", {"case": label},
          surface_version="tech/1",
          criterion="POST /jobs returns 422 before any job is created")
    r = client.post("/jobs", json=_american_body(resolution=value))
    assert r.status_code == 422, r.text
    body = r.json()
    assert body["error"] == "invalid_parameter"
    err = body["detail"]["errors"][0]
    assert err["loc"] == ["body", "resolution"]
    assert err["type"] == etype
    assert "resolution" in body["message"]


def test_async_legal_resolution_succeeds_with_exact_steps():
    r = client.post("/jobs", json=_american_body(resolution=41))
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    final = {}
    for _ in range(200):
        final = client.get(f"/jobs/{job_id}").json()
        if final["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.01)
    assert final["status"] == "succeeded", final
    assert final["result"]["pricing_mode"] == "fixed_resolution"
    assert final["result"]["steps_used"] == 41


# ----------------------------------------------------------------- batch entry

def test_batch_item_envelope_does_not_pre_reject_resolution():
    # The envelope must carry the raw value through; per-leg rebuild applies
    # the identical strict rule, isolating the failure to that leg.
    item = BatchPricingItem(option_type="american_put", expiry=1.0,
                            strike=100, resolution=True)
    assert item.resolution is True


def test_batch_each_illegal_leg_fails_alone(audit):
    legs = [
        ("bool", True, "resolution_bad_type"),
        ("float", 7.0, "resolution_bad_type"),
        ("string", "7", "resolution_bad_type"),
        ("over", MAX_RESOLUTION + 1, "resolution_out_of_range"),
    ]
    items = [
        {"option_type": "american_put", "expiry": 0.75, "strike": 200,
         "spot": 200, "resolution": 17},
    ]
    for _, value, _ in legs:
        items.append({"option_type": "american_put", "expiry": 0.75,
                      "strike": 200, "spot": 200, "resolution": value})
    items.append({"option_type": "vanilla_call", "expiry": 1.0,
                  "strike": 200})

    audit("batch_resolution_isolation", {"legs": len(items),
                                         "bad": [1, 2, 3, 4]},
          surface_version="tech/1",
          criterion="batch 200; only illegal legs fail with their own "
                    "invalid_parameter reason; good legs priced")
    r = client.post("/pricing/price/batch",
                    json={"surface_id": "tech", "version": 1, "items": items})
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert [x["index"] for x in results] == list(range(len(items)))

    good0 = results[0]
    assert good0["status"] == "succeeded"
    assert good0["result"]["pricing_mode"] == "fixed_resolution"
    assert good0["result"]["steps_used"] == 17

    for offset, (_, _, etype) in enumerate(legs, start=1):
        leg = results[offset]
        assert leg["status"] == "failed"
        assert leg["result"] is None
        assert leg["error_code"] == "invalid_parameter"
        assert "resolution" in leg["error_message"]
        assert etype in leg["error_message"]

    good_last = results[-1]
    assert good_last["status"] == "succeeded"
    assert good_last["result"]["pricing_mode"] == "closed_form"


def test_batch_leg_result_matches_standalone_sync(audit):
    audit("batch_equals_sync", {"resolution": 17},
          surface_version="tech/1",
          criterion="batch leg price/mode/steps identical to standalone call")
    sync = client.post("/pricing/price",
                       json=_american_body(resolution=17)).json()
    batch = client.post("/pricing/price/batch", json={
        "surface_id": "tech", "version": 1, "items": [
            {"option_type": "american_put", "expiry": 0.75, "strike": 200,
             "spot": 200, "resolution": 17}]}).json()
    leg = batch["results"][0]
    assert leg["status"] == "succeeded"
    assert leg["result"]["price"] == sync["price"]
    assert leg["result"]["steps_used"] == sync["steps_used"] == 17
    assert leg["result"]["pricing_mode"] == sync["pricing_mode"]

    # And the illegal-type rejection matches between the two entries too.
    b = client.post("/pricing/price/batch", json={
        "surface_id": "tech", "version": 1, "items": [
            {"option_type": "american_put", "expiry": 0.75, "strike": 200,
             "spot": 200, "resolution": True}]}).json()
    assert b["results"][0]["error_code"] == "invalid_parameter"
    assert "resolution_bad_type" in b["results"][0]["error_message"]


def test_batch_illegal_leg_does_not_affect_other_surface_legs(service):
    # Service-level check: even an engine-level service built directly gets
    # per-leg isolation with identical PricingRequest materialisation.
    from option_engine.batch import BatchPricer
    pricer = BatchPricer(service)
    batch = BatchPricingRequest(surface_id="tech", version=1, items=[
        BatchPricingItem(option_type="american_put", expiry=0.75,
                         strike=200, spot=200, resolution=23),
        BatchPricingItem(option_type="american_put", expiry=0.75,
                         strike=200, spot=200, resolution=True),
        BatchPricingItem(option_type="american_put", expiry=0.75,
                         strike=200, spot=200, resolution="9"),
        BatchPricingItem(option_type="american_put", expiry=0.75,
                         strike=200, spot=200, resolution=0),
    ])
    resp = pricer.price_batch(batch)
    statuses = [(x.index, x.status, x.error_code) for x in resp.results]
    assert statuses == [
        (0, "succeeded", None),
        (1, "failed", "invalid_parameter"),
        (2, "failed", "invalid_parameter"),
        (3, "failed", "invalid_parameter"),
    ]
    assert resp.results[0].result.steps_used == 23
