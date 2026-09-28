"""HTTP-level contracts: error envelopes/status codes, surface publish API,
version queries, async job lifecycle and cancellation.
"""
import time

from fastapi.testclient import TestClient

from main import app
from option_engine.models import JobStatus

client = TestClient(app)


def test_http_error_envelopes_are_distinguishable(audit):
    cases = [
        ("out_of_range", {"option_type": "vanilla_call", "expiry": 99,
                          "strike": 100}, "surface_out_of_range", 422),
        ("surface_404", {"surface_id": "ghost", "option_type":
                         "vanilla_call", "expiry": 1, "strike": 100},
         "surface_not_found", 404),
        ("bad_param", {"option_type": "vanilla_call", "expiry": -1,
                       "strike": 100}, "invalid_parameter", 422),
        ("solver_fail", {"option_type": "american_put", "surface_id": "tech",
                         "expiry": 0.75, "strike": 200, "spot": 200,
                         "steps": 4, "max_steps": 32,
                         "convergence_tol": 1e-12},
         "solver_non_convergence", 422),
    ]
    for label, body, code, status in cases:
        audit("http_error", {"case": label, **body},
              surface_version=body.get("version", "latest"),
              criterion=f"HTTP {status} with error code '{code}'")
        r = client.post("/pricing/price", json=body)
        assert r.status_code == status, r.text
        err = r.json()
        assert err["error"] == code
        assert err["message"]


def test_publish_and_fetch_versions(audit):
    payload = {"expiries": [0.5, 1.0], "strikes": [95, 100, 105],
               "vols": [[0.21, 0.20, 0.195], [0.22, 0.205, 0.20]],
               "spot": 100, "rate": 0.04}
    r = client.post("/surfaces/api_demo/versions", json=payload)
    assert r.status_code == 200, r.text
    v = r.json()["version"]
    audit("publish_api", {"surface": "api_demo"}, surface_version=v,
          criterion="publish returns version; GET retrieves identical grid")
    got = client.get(f"/surfaces/api_demo/versions/{v}").json()
    assert got["version"] == v and got["rate"] == 0.04
    vol = client.get(f"/surfaces/api_demo/versions/{v}/vol",
                     params={"expiry": 1.0, "strike": 100}).json()
    assert vol["vol"] == 0.205
    assert client.get("/surfaces/api_demo/versions/999").status_code == 404


def test_async_job_lifecycle_and_cancel(audit):
    # Job that cannot converge -> FAILED with a clear code.
    r = client.post("/jobs", json={
        "surface_id": "tech", "option_type": "american_put",
        "expiry": 0.75, "strike": 200, "spot": 200,
        "steps": 4, "max_steps": 32, "convergence_tol": 1e-12})
    assert r.status_code == 202
    job = r.json()
    audit("job_failure", {"job_id": job["job_id"]},
          surface_version="tech/1",
          criterion="terminal FAILED with solver_non_convergence")
    final = client.get(f"/jobs/{job['job_id']}")
    for _ in range(200):
        final = client.get(f"/jobs/{job['job_id']}").json()
        if final["status"] in (JobStatus.SUCCEEDED, JobStatus.FAILED,
                               JobStatus.CANCELLED):
            break
        time.sleep(0.01)
    assert final["status"] == JobStatus.FAILED
    assert final["error_code"] == "solver_non_convergence"
    assert final["surface_version"] is not None  # version pinned pre-compute
    assert client.get("/jobs/does-not-exist").status_code == 404


def test_async_job_carries_fixed_resolution_and_extrapolation(audit):
    body = {
        "surface_id": "tech", "option_type": "american_put",
        "expiry": 2.5, "strike": 250, "allow_extrapolation": True,
        "resolution": 64}
    audit("job_fixed_resolution", body, surface_version="tech/1",
          criterion="async result mirrors sync: fixed/64 + extrapolation audit")
    sync = client.post("/pricing/price", json=body)
    assert sync.status_code == 200, sync.text
    r = client.post("/jobs", json=body)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    final = {}
    for _ in range(300):
        final = client.get(f"/jobs/{job_id}").json()
        if final["status"] in (JobStatus.SUCCEEDED, JobStatus.FAILED,
                               JobStatus.CANCELLED):
            break
        time.sleep(0.01)
    assert final["status"] == JobStatus.SUCCEEDED, final
    res = final["result"]
    assert res["resolution_mode"] == "fixed"
    assert res["resolution_used"] == 64
    assert res["extrapolated"] is True
    assert res["price"] == sync.json()["price"]


def test_cancel_running_job(audit):
    from option_engine.api import jobs
    from option_engine.models import PricingRequest

    # Very heavy request: tiny tol, huge budget -> runs long enough to
    # observe RUNNING and cancel.
    req = PricingRequest(
        surface_id="tech", option_type="american_put",
        expiry=0.75, strike=200, spot=200,
        steps=2000, max_steps=200000, convergence_tol=1e-13)
    job = jobs.submit(req)
    # Wait until it is actually running, then cancel.
    deadline = time.time() + 5
    while time.time() < deadline:
        if jobs.status(job.job_id).status == JobStatus.RUNNING:
            break
        time.sleep(0.005)
    audit("job_cancel", {"job_id": job.job_id}, surface_version="tech/1",
          criterion="event set; terminal CANCELLED; engine still serves")
    jobs.cancel(job.job_id)
    deadline = time.time() + 30
    final = jobs.status(job.job_id)
    while time.time() < deadline and final.status == JobStatus.RUNNING:
        time.sleep(0.02)
        final = jobs.status(job.job_id)
    assert final.status == JobStatus.CANCELLED
    assert final.error_code == "job_cancelled"
