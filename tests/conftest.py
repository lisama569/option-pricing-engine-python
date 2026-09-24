"""pytest 夹具：独立的仓库/服务/任务管理器与 TestClient。"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from option_engine.app import create_app
from option_engine.config import SURFACE_DATA_DIR
from option_engine.jobs.manager import JobManager
from option_engine.pricing.service import PricingService
from option_engine.surface.loader import publish_directory
from option_engine.surface.store import SurfaceStore

logging.basicConfig(level=logging.INFO)


@pytest.fixture()
def store() -> SurfaceStore:
    s = SurfaceStore()
    publish_directory(s, SURFACE_DATA_DIR)
    return s


@pytest.fixture()
def service(store: SurfaceStore) -> PricingService:
    return PricingService(store)


@pytest.fixture()
def job_manager(service: PricingService) -> JobManager:
    jm = JobManager(service, max_workers=8)
    yield jm
    jm.shutdown(wait=False)


@pytest.fixture()
def client() -> TestClient:
    application = create_app(data_dir=SURFACE_DATA_DIR)
    with TestClient(application) as c:
        yield c


@pytest.fixture()
def flat_surface() -> dict:
    """20% 平坦波动率曲面，用于与解析基准严格对照。"""
    return {
        "label": "FLAT-20PCT",
        "as_of": "2026-09-24",
        "maturities": [0.25, 0.5, 1.0, 2.0],
        "strikes": [80.0, 90.0, 100.0, 110.0, 120.0],
        "vols": [[0.2] * 5 for _ in range(4)],
        "risk_free_rate": 0.05,
        "dividend_yield": 0.0,
    }
