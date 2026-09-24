"""全局配置：本地数据目录、线程池规模等。"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SURFACE_DATA_DIR = Path(
    os.environ.get("OPTION_ENGINE_DATA_DIR", BASE_DIR / "data" / "surfaces")
)
MAX_WORKERS = int(os.environ.get("OPTION_ENGINE_MAX_WORKERS", "4"))
