"""线程安全的版本化曲面仓库。

并发规则：
- 每次发布生成一份全新的 ``PublishedSurface`` 不可变快照（版本号单调递增）；
- 发布时先校验、构造好完整快照，最后在锁内做一次引用替换——
  读者要么看到旧版本、要么看到新版本，绝不会看到部分更新或新旧混杂；
- 快照对象一经发布不再修改，读者取到引用后即可在锁外安全使用。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from ..errors import ErrorCode, PricingEngineError
from ..models import SurfaceVersionInfo, VolSurfaceData
from .grid import validate_surface


@dataclass(frozen=True)
class PublishedSurface:
    """一份不可变的已发布曲面快照。"""

    version: int
    data: VolSurfaceData
    published_at: datetime

    def to_info(self) -> SurfaceVersionInfo:
        return SurfaceVersionInfo(
            version=self.version,
            label=self.data.label,
            as_of=self.data.as_of,
            published_at=self.published_at,
            num_maturities=len(self.data.maturities),
            num_strikes=len(self.data.strikes),
        )


class SurfaceStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._versions: list[PublishedSurface] = []

    def publish(self, data: VolSurfaceData) -> PublishedSurface:
        """校验并原子发布一个新版本，返回该快照。"""
        validate_surface(data)  # 校验在锁外语义上无副作用；放在临界区也很快
        snapshot = PublishedSurface(
            version=0,  # 占位，临界区内赋最终版本号
            data=data.model_copy(deep=True),
            published_at=datetime.now(timezone.utc),
        )
        with self._lock:
            next_version = len(self._versions) + 1
            snapshot = PublishedSurface(
                version=next_version,
                data=snapshot.data,
                published_at=snapshot.published_at,
            )
            self._versions.append(snapshot)
            return snapshot

    def get(self, version: int | None) -> PublishedSurface:
        """按版本取快照；version=None 取最新；不存在则 VERSION_NOT_FOUND。

        返回深拷贝：调用方对返回内容的任何原地修改都不会影响仓库内
        已发布版本，从而保证历史版本可复现。
        """
        with self._lock:
            if not self._versions:
                raise PricingEngineError(
                    ErrorCode.VERSION_NOT_FOUND,
                    "no surface has been published yet",
                )
            if version is None:
                target = self._versions[-1]
            elif version < 1 or version > len(self._versions):
                raise PricingEngineError(
                    ErrorCode.VERSION_NOT_FOUND,
                    f"surface version {version} does not exist; "
                    f"available versions: 1..{len(self._versions)}",
                )
            else:
                target = self._versions[version - 1]
            return PublishedSurface(
                version=target.version,
                data=target.data.model_copy(deep=True),
                published_at=target.published_at,
            )

    def latest_version(self) -> int | None:
        with self._lock:
            return len(self._versions) if self._versions else None

    def list_versions(self) -> list[SurfaceVersionInfo]:
        with self._lock:
            return [s.to_info() for s in self._versions]
