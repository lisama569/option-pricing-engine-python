"""Concurrency: atomic surface publication, historical version replay,
deterministic cross-path results and async job isolation.
"""
import threading
import time

from fastapi.testclient import TestClient

from main import app
from option_engine.models import (
    JobStatus,
    PricingRequest,
    SurfaceGridPayload,
)

client = TestClient(app)


def _grid(vol_level: float) -> SurfaceGridPayload:
    return SurfaceGridPayload(
        expiries=[0.25, 0.5, 1.0, 2.0],
        strikes=[80, 100, 120],
        vols=[[vol_level] * 3 for _ in range(4)],
        spot=100, rate=0.05)


def test_concurrent_publish_never_mixes_versions(audit):
    # Hammer the store with many concurrent publishes; every observed
    # snapshot must be a fully self-consistent published version.
    from option_engine.api import store

    surface_id = "concurrent_mix"
    stop = threading.Event()
    errors: list[Exception] = []
    observed: set[tuple[int, float]] = set()

    def publisher(level: float) -> None:
        while not stop.is_set():
            try:
                store.publish(surface_id, _grid(level))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

    def reader() -> None:
        while not stop.is_set():
            try:
                for v in range(1, store.latest_version(surface_id) + 1):
                    snap = store.get(surface_id, v)
                    # A snapshot is internally consistent iff every grid
                    # cell equals that version's single vol level.
                    cells = {x for row in snap.vols for x in row}
                    if len(cells) != 1:
                        errors.append(AssertionError("mixed grid cells"))
                    observed.add((snap.version, next(iter(cells))))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

    threads = [threading.Thread(target=publisher, args=(0.15 + i * 0.05,))
               for i in range(4)]
    readers = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads + readers:
        t.start()
    time.sleep(0.8)
    stop.set()
    for t in threads + readers:
        t.join(timeout=5)

    latest = store.latest_version(surface_id)
    audit("concurrent_publish", {"versions_published": latest,
                                 "snapshots_observed": len(observed)},
          surface_version=f"1..{latest}",
          criterion="no mixed grids; sequential integer versions")
    assert not errors, errors
    assert latest >= 10
    # Every published version is retrievable and immutable.
    for v in range(1, latest + 1):
        snap = store.get(surface_id, v)
        assert snap.version == v


def test_historical_version_replay_and_path_consistency(audit):
    from option_engine.api import service, store

    surface_id = "replay"
    v1 = store.publish(surface_id, _grid(0.20)).version
    v2 = store.publish(surface_id, _grid(0.30)).version
    v3 = store.publish(surface_id, _grid(0.40)).version

    req_kwargs = dict(surface_id=surface_id, option_type="vanilla_call",
                      expiry=1.0, strike=100)

    def paths(version):
        body = client.post("/pricing/price", json={**req_kwargs,
                                                   "version": version}).json()
        job = client.post("/jobs", json={**req_kwargs,
                                         "version": version}).json()
        job_id = job["job_id"]
        for _ in range(100):
            job = client.get(f"/jobs/{job_id}").json()
            if job["status"] in (JobStatus.SUCCEEDED, JobStatus.FAILED,
                                 JobStatus.CANCELLED):
                break
            time.sleep(0.01)
        return body, job

    audit("version_replay", {"versions": [v1, v2, v3]},
          surface_version=[v1, v2, v3],
          criterion="each version reproduces its own price; sync and async"
                    " paths identical; newer publishes do not alter history")
    prices = {}
    for v in (v1, v2, v3):
        sync, job = paths(v)
        assert job["status"] == JobStatus.SUCCEEDED
        assert job["surface_version"] == v
        assert job["result"]["price"] == sync["price"]
        assert job["result"]["vol_used"] == sync["vol_used"] == (
            0.20 if v == v1 else 0.30 if v == v2 else 0.40)
        prices[v] = sync["price"]
    assert prices[v1] < prices[v2] < prices[v3]

    # Replay v1 after v3 is current: identical cached result.
    replay = service.price(PricingRequest(version=v1, **req_kwargs))
    assert replay.price == prices[v1]
    assert replay.surface_version == v1
    assert replay.cached is True


def test_same_inputs_same_version_always_equal_under_load(audit):
    from option_engine.api import service, store

    surface_id = "equal_load"
    store.publish(surface_id, _grid(0.25))
    req = PricingRequest(surface_id=surface_id,
                         option_type="vanilla_call", expiry=1.0, strike=100)
    results: list[float] = []
    lock = threading.Lock()

    def worker() -> None:
        r = service.price(req)
        with lock:
            results.append(r.price)

    threads = [threading.Thread(target=worker) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    audit("equal_under_load", {"threads": 32, "version": 1},
          surface_version=1,
          criterion="all 32 identical prices; only one unique result")
    assert len(set(results)) == 1


def test_long_job_does_not_block_and_failure_is_isolated(audit):
    from option_engine.api import jobs

    # A bad job that fails fast, plus many cheap jobs, all must complete
    # while a deliberately heavy American-put job is still running.
    heavy = PricingRequest(
        surface_id="default", option_type="american_put",
        expiry=1.0, strike=100, steps=2000, max_steps=4000,
        convergence_tol=1e-12)  # cannot converge: guaranteed heavy + failure
    bad = PricingRequest(surface_id="default",
                         option_type="vanilla_call", expiry=99, strike=100)
    cheap = PricingRequest(surface_id="default",
                           option_type="vanilla_call", expiry=1.0, strike=100)

    audit("job_isolation", {"heavy_steps": "2000->4000", "cheap": 16},
          surface_version="latest",
          criterion="cheap jobs finish while heavy runs; heavy ends FAILED;"
                    " service stays healthy")
    heavy_job = jobs.submit(heavy)
    bad_job = jobs.submit(bad)
    cheap_jobs = [jobs.submit(cheap) for _ in range(16)]

    # Cheap jobs finish promptly even though the heavy job occupies a worker.
    deadline = time.time() + 10
    while time.time() < deadline:
        if all(jobs.status(j.job_id).status == JobStatus.SUCCEEDED
               for j in cheap_jobs):
            break
        time.sleep(0.01)
    heavy_status = jobs.status(heavy_job.job_id).status
    assert all(jobs.status(j.job_id).status == JobStatus.SUCCEEDED
               for j in cheap_jobs)
    assert jobs.status(bad_job.job_id).status == JobStatus.FAILED
    assert jobs.status(bad_job.job_id).error_code == "surface_out_of_range"
    assert client.get("/health").json() == {"status": "ok"}

    # Heavy eventually terminates as FAILED with solver code, and the
    # service keeps serving afterwards.
    final = jobs.to_response(jobs.status(heavy_job.job_id))
    deadline = time.time() + 15
    while time.time() < deadline and final.status == JobStatus.RUNNING:
        time.sleep(0.05)
        final = jobs.to_response(jobs.status(heavy_job.job_id))
    assert final.status == JobStatus.FAILED
    assert final.error_code == "solver_non_convergence"
    after = client.post("/pricing/price", json={
        "option_type": "vanilla_call", "expiry": 1.0, "strike": 100})
    assert after.status_code == 200
