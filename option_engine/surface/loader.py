"""本地曲面数据加载：JSON 文件 -> VolSurfaceData（输入全部来自本地文件）。

- 单个文件经 Pydantic 解析 + 仓库语义校验，非法文件不会污染仓库；
- 目录发布按文件名排序逐个原子发布，发布顺序确定、可复现。
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from ..errors import ErrorCode, PricingEngineError
from ..models import VolSurfaceData
from .store import SurfaceStore


def load_surface_file(path: str | Path) -> VolSurfaceData:
    """读取并解析单个曲面 JSON 文件。"""
    p = Path(path)
    if not p.is_file():
        raise PricingEngineError(
            ErrorCode.SURFACE_DATA_INVALID,
            f"surface file not found: {p}",
        )
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PricingEngineError(
            ErrorCode.SURFACE_DATA_INVALID,
            f"invalid JSON in {p}: {exc}",
        ) from exc
    try:
        data = VolSurfaceData.model_validate(raw)
    except ValidationError as exc:
        raise PricingEngineError(
            ErrorCode.SURFACE_DATA_INVALID,
            f"surface file {p} failed schema validation: {exc}",
        ) from exc
    return data


def publish_directory(store: SurfaceStore, directory: str | Path) -> list[int]:
    """按文件名排序发布目录下全部 .json，返回版本号列表。"""
    d = Path(directory)
    if not d.is_dir():
        raise PricingEngineError(
            ErrorCode.SURFACE_DATA_INVALID,
            f"surface data directory not found: {d}",
        )
    published: list[int] = []
    for path in sorted(d.glob("*.json")):
        data = load_surface_file(path)
        snapshot = store.publish(data)
        published.append(snapshot.version)
    return published
