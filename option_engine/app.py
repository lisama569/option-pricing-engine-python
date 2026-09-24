"""应用装配：仓库 -> 定价服务 -> 任务管理器 -> 路由。

启动时从本地数据目录（默认 data/surfaces，可用 OPTION_ENGINE_DATA_DIR 覆盖）
按文件名排序发布全部曲面。数据目录为空/缺失不阻断启动，服务以空仓库启动，
之后可通过 POST /api/v1/surfaces 发布曲面。
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import MAX_WORKERS, SURFACE_DATA_DIR
from .jobs.manager import JobManager
from .pricing.service import PricingService
from .routes import build_router
from .surface.loader import publish_directory
from .surface.store import SurfaceStore

logging.basicConfig(
    level=os.environ.get("OPTION_ENGINE_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("option_engine.app")


def create_app(
    data_dir=SURFACE_DATA_DIR, max_workers: int = MAX_WORKERS
) -> FastAPI:
    store = SurfaceStore()
    service = PricingService(store)
    jobs = JobManager(service, max_workers=max_workers)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            versions = publish_directory(store, data_dir)
            logger.info(
                "loaded %d surface version(s) from %s: %s",
                len(versions),
                data_dir,
                versions,
            )
        except Exception:
            logger.exception(
                "failed to preload surfaces from %s; starting with empty store",
                data_dir,
            )
        yield
        jobs.shutdown(wait=False)

    app = FastAPI(
        title="option-pricing-engine",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.store = store
    app.state.service = service
    app.state.jobs = jobs
    app.include_router(build_router(store, service, jobs))
    return app


app = create_app()
