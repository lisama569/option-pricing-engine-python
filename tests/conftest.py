"""Shared fixtures: isolated service stack and structured test logging.

Every test logs its input parameters, the surface version used and the
criterion it is judged against, satisfying the audit/reproducibility
requirement.
"""
from __future__ import annotations

import logging
import sys

import pytest

from option_engine.data import load_bundled_surfaces
from option_engine.jobs import JobManager
from option_engine.service import PricingService
from option_engine.surface import SurfaceStore


@pytest.fixture
def store() -> SurfaceStore:
    return load_bundled_surfaces(SurfaceStore())


@pytest.fixture
def service(store: SurfaceStore) -> PricingService:
    return PricingService(store)


@pytest.fixture
def manager(service: PricingService) -> JobManager:
    mgr = JobManager(service, max_workers=4)
    yield mgr
    mgr.shutdown()


@pytest.fixture
def audit(caplog):
    logger = logging.getLogger("option_engine.tests")
    logger.setLevel(logging.INFO)

    def record(case: str, inputs: dict, surface_version, criterion: str):
        logger.info(
            "CASE=%s | INPUTS=%s | SURFACE_VERSION=%s | CRITERION=%s",
            case, inputs, surface_version, criterion,
        )

    return record


# Ensure log records reach pytest captured log even without -o log_cli.
def pytest_configure(config):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger("option_engine.tests")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
