"""Batch pricing: single version binding, per-leg isolation, per-leg failure
reasons, equivalence with standalone calls, and concurrency independence.
"""
import threading
import time

from fastapi.testclient import TestClient

from main import app
from option_engine.models import BatchPricingRequest, BatchPricingItem
from option_engine.pricing import american_put_crr

client = TestClient(app)


def test_batch_partial_failures_are_isolated(audit):
    body = {
        "surface_id": "default", "version": 1,
        "items": [
            {"option_type": "vanilla_call", "expiry": 1.0, "strike": 100},
            # Out of support on a surface that declares no range.
            {"option_type": "vanilla_call", "expiry": 99.0, "strike": 100},
            {"option_type": "vanilla_put", "expiry": 0.5, "strike": 95},
        ],
    }
    audit("batch_partial", {"legs": 3, "bad": [1]}, surface_version=1,
          criterion="HTTP 200; legs 0/2 succeed, leg 1 fails on its own "
                    "with surface_out_of_range")
    r = client.post("/pricing/price/batch", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["surface_version"] == 1
    assert [x["index"] for x in data["results"]] == [0, 1, 2]
    assert data["results"][0]["status"] == "succeeded"
    assert data["results"][2]["status"] == "succeeded"
    bad = data["results"][1]
    assert bad["status"] == "failed"
    assert bad["error_code"] == "surface_out_of_range"
    assert bad["result"] is None


def test_batch_each_leg_matches_standalone_call(audit):
    from option_engine.api import batch
    req = BatchPricingRequest(surface_id="default", version=1, items=[
        BatchPricingItem(option_type="vanilla_call", expiry=1.0, strike=100),
        BatchPricingItem(option_type="digital_put", expiry=0.5, strike=105),
        BatchPricingItem(option_type="american_put", expiry=1.0, strike=100,
                         resolution=211),
        BatchPricingItem(option_type="american_put", expiry=2.0, strike=100),
    ])
    audit("batch_equiv", {"legs": len(req.items)}, surface_version=1,
          criterion="each batch leg equals the standalone /pricing/price "
                    "result (price/vol/mode/steps)")
    resp = batch.price_batch(req)
    for leg in resp.results:
        item = req.items[leg.index]
        standalone = client.post("/pricing/price", json={
            "surface_id": "default", "version": 1,
            "option_type": item.option_type.value,
            "expiry": item.expiry, "strike": item.strike,
            "resolution": item.resolution,
        }).json()
        got = leg.result
        assert got.price == standalone["price"]
        assert got.vol_used == standalone["vol_used"]
        assert got.pricing_mode == standalone["pricing_mode"]
        assert got.steps_used == standalone["steps_used"]


def test_batch_fixed_resolution_leg_is_exact(audit):
    from option_engine.api import batch, store
    req = BatchPricingRequest(surface_id="default", version=1, items=[
        BatchPricingItem(option_type="american_put", expiry=1.0, strike=100,
                         resolution=333),
    ])
    audit("batch_fixed", {"resolution": 333}, surface_version=1,
          criterion="price == CRR(333) on the bound version's inputs")
    resp = batch.price_batch(req)
    got = resp.results[0].result
    snap = store.get("default", 1)
    expected = american_put_crr(snap.spot, 100, 1.0, got.vol_used,
                                snap.rate, snap.dividend, 333)
    assert got.pricing_mode == "fixed_resolution"
    assert got.steps_used == 333
    assert got.price == expected


def test_batch_pins_one_version_for_whole_book(audit):
    from option_engine.api import batch
    req = BatchPricingRequest(surface_id="default", items=[
        BatchPricingItem(option_type="vanilla_call", expiry=1.0, strike=100),
        BatchPricingItem(option_type="vanilla_call", expiry=0.5, strike=95),
    ])
    resp = batch.price_batch(req)
    audit("batch_version", {"requested": "latest",
                            "pinned": resp.surface_version},
          surface_version=resp.surface_version,
          criterion="every leg reports the same pinned version")
    assert all(leg.result.surface_version == resp.surface_version
               for leg in resp.results)


def test_batch_unknown_surface_fails_batch_as_a_whole(audit):
    r = client.post("/pricing/price/batch", json={
        "surface_id": "ghost", "items": [
            {"option_type": "vanilla_call", "expiry": 1.0, "strike": 100}]})
    audit("batch_404", {"surface_id": "ghost"}, surface_version="n/a",
          criterion="404 surface_not_found")
    assert r.status_code == 404
    assert r.json()["error"] == "surface_not_found"


def test_batch_slow_leg_does_not_block_others(audit):
    from option_engine.api import batch
    # A deliberately heavy fixed-resolution leg alongside cheap legs; the
    # cheap legs must finish independently and the heavy leg still succeeds.
    req = BatchPricingRequest(surface_id="default", version=1, items=[
        BatchPricingItem(option_type="american_put", expiry=1.0, strike=100,
                         resolution=1500),
        BatchPricingItem(option_type="vanilla_call", expiry=1.0, strike=100),
        BatchPricingItem(option_type="vanilla_put", expiry=1.0, strike=100),
        BatchPricingItem(option_type="vanilla_call", expiry=99, strike=100),
    ])
    done = threading.Event()
    box = {}

    def run():
        box["resp"] = batch.price_batch(req)
        done.set()

    audit("batch_slow_leg", {"heavy": 1500, "cheap": 2, "bad": 1},
          surface_version=1,
          criterion="batch completes; cheap legs and heavy leg all priced; "
                    "bad leg isolated")
    t = threading.Thread(target=run)
    t.start()
    assert done.wait(timeout=30), "batch did not complete in time"
    t.join()
    results = sorted(box["resp"].results, key=lambda x: x.index)
    assert results[0].status == "succeeded"
    assert results[0].result.steps_used == 1500
    assert results[1].status == "succeeded"
    assert results[2].status == "succeeded"
    assert results[3].status == "failed"
    assert results[3].error_code == "surface_out_of_range"


def test_batch_empty_rejected(audit):
    r = client.post("/pricing/price/batch",
                    json={"surface_id": "default", "items": []})
    audit("batch_empty", {}, surface_version="n/a",
          criterion="422 invalid_parameter envelope")
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_parameter"
