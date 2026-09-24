"""HTTP 层：失败码、历史版本回放、并发曲面更新期间读取一致性。"""

from __future__ import annotations

import logging
import threading
import time

from fastapi.testclient import TestClient

logger = logging.getLogger("test.api")

PRICE_BODY = {"option_type": "european_call", "spot": 100, "strike": 100,
              "maturity": 0.5}


def test_health_and_versions(client: TestClient):
    h = client.get("/api/v1/health").json()
    assert h["status"] == "ok" and h["num_versions"] == 2
    versions = client.get("/api/v1/surfaces").json()
    logger.info("已加载曲面版本=%s，判定依据=启动时按文件名排序发布两个本地文件",
                [(v["version"], v["label"]) for v in versions])
    assert [v["label"] for v in versions] == ["SURF-2026Q2", "SURF-2026Q3"]


def test_http_distinct_failure_reasons(client: TestClient):
    r_range = client.post("/api/v1/pricing/price",
                          json={**PRICE_BODY, "strike": 50, "surface_version": 1})
    r_version = client.post("/api/v1/pricing/price",
                            json={**PRICE_BODY, "surface_version": 42})
    r_params = client.post("/api/v1/pricing/price", json={
        "option_type": "american_put", "spot": 100, "strike": 100,
        "maturity": 1.0, "surface_version": 1})
    r_solver = client.post("/api/v1/pricing/implied-vol", json={
        "option_type": "european_call", "spot": 100, "strike": 100,
        "maturity": 0.5, "target_price": 1e9, "surface_version": 1})

    assert r_range.status_code == 422
    assert r_range.json()["detail"]["error_code"] == "SURFACE_OUT_OF_RANGE"
    assert r_version.status_code == 404
    assert r_version.json()["detail"]["error_code"] == "VERSION_NOT_FOUND"
    assert r_params.json()["detail"]["error_code"] == "INVALID_MODEL_PARAMS"
    assert r_solver.json()["detail"]["error_code"] == "SOLVER_DID_NOT_CONVERGE"
    # 负数值在服务层统一拒绝，同样带业务 error_code（而非通用 schema 错误）
    r_neg = client.post("/api/v1/pricing/price", json={
        **PRICE_BODY, "spot": -10, "surface_version": 1})
    assert r_neg.json()["detail"]["error_code"] == "INVALID_MODEL_PARAMS"
    logger.info("HTTP 四类失败：%s / %s / %s / %s；负数输入=%s",
                r_range.json()["detail"]["error_code"],
                r_version.json()["detail"]["error_code"],
                r_params.json()["detail"]["error_code"],
                r_solver.json()["detail"]["error_code"],
                r_neg.json()["detail"]["error_code"])


def test_version_pinned_replay_identical(client: TestClient):
    v1_first = client.post("/api/v1/pricing/price",
                           json={**PRICE_BODY, "surface_version": 1}).json()
    latest = client.post("/api/v1/pricing/price", json=PRICE_BODY).json()
    time.sleep(0.05)
    v1_replay = client.post("/api/v1/pricing/price",
                            json={**PRICE_BODY, "surface_version": 1}).json()
    logger.info("v1 首次价格=%.12f 当前最新(v%d)价格=%.12f v1 回放=%.12f",
                v1_first["price"], latest["surface_version"],
                latest["price"], v1_replay["price"])
    assert v1_first["price"] == v1_replay["price"]
    assert v1_first["surface_version"] == 1
    assert latest["surface_version"] == 2
    assert latest["price"] != v1_first["price"]


def test_publish_and_concurrent_reads_never_mixed(client: TestClient):
    """发布多个新版本期间并发定价：每次响应只能引用一个完整存在的版本。"""
    errors: list[Exception] = []
    seen_versions: set[int] = set()
    lock = threading.Lock()

    def publisher():
        for i in range(5):
            resp = client.post("/api/v1/surfaces", json={
                "label": f"CONC-{i}", "as_of": f"2026-10-{i+1:02d}",
                "maturities": [0.25, 0.5, 1.0],
                "strikes": [90.0, 95.0, 100.0, 105.0, 110.0],
                "vols": [[0.18 + 0.01 * i + 0.004 * j] * 5
                         for j in range(3)],
                "risk_free_rate": 0.04, "dividend_yield": 0.01})
            assert resp.status_code == 201, resp.text
            time.sleep(0.01)

    def reader():
        try:
            for _ in range(60):
                # 绑定历史版本：即使在更新期间也必须稳定
                r = client.post("/api/v1/pricing/price",
                                json={**PRICE_BODY, "surface_version": 1})
                assert r.status_code == 200
                body = r.json()
                assert body["surface_version"] == 1
                with lock:
                    seen_versions.add(body["surface_version"])
                # 最新版本读取也必须自洽
                r2 = client.post("/api/v1/pricing/price", json=PRICE_BODY)
                b2 = r2.json()
                assert r2.status_code == 200
                assert isinstance(b2["price"], float)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=publisher)]
    threads += [threading.Thread(target=reader) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert seen_versions == {1}
    info = client.get("/api/v1/surfaces").json()
    logger.info("并发更新结束后版本数=%d；历史 v1 读取始终稳定", len(info))
    assert len(info) == 7  # 初始 2 + 新发布 5


def test_async_job_flow_over_http(client: TestClient):
    job = client.post("/api/v1/jobs",
                      json={**PRICE_BODY, "surface_version": 1})
    assert job.status_code == 202
    jid = job.json()["job_id"]
    for _ in range(100):
        rec = client.get(f"/api/v1/jobs/{jid}").json()
        if rec["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.02)
    sync = client.post("/api/v1/pricing/price",
                       json={**PRICE_BODY, "surface_version": 1}).json()
    assert rec["status"] == "succeeded"
    assert rec["result"]["price"] == sync["price"]
    logger.info("异步任务价格=%.12f 与同步路径完全一致，曲面版本=v1",
                rec["result"]["price"])


def test_invalid_surface_publish_rejected(client: TestClient):
    resp = client.post("/api/v1/surfaces", json={
        "label": "bad", "as_of": "x", "maturities": [0.5],
        "strikes": [90.0, 100.0], "vols": [[0.2, 0.2]]})
    assert resp.status_code == 422
    assert resp.json()["detail"]["error_code"] == "SURFACE_DATA_INVALID"
