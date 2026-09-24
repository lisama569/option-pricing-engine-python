"""曲面网格、仓库与加载器测试：非法数据、越界、版本化发布。"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import pytest

from option_engine.errors import ErrorCode, PricingEngineError
from option_engine.models import VolSurfaceData
from option_engine.surface.grid import interpolate_vol, validate_surface
from option_engine.surface.loader import load_surface_file, publish_directory
from option_engine.surface.store import SurfaceStore

logger = logging.getLogger("test.surface")


def _surface(**over) -> VolSurfaceData:
    base = dict(
        label="S", as_of="2026-09-24",
        maturities=[0.25, 0.5, 1.0], strikes=[90.0, 100.0, 110.0],
        vols=[[0.2, 0.19, 0.2], [0.22, 0.2, 0.22], [0.24, 0.22, 0.24]],
        risk_free_rate=0.04, dividend_yield=0.01,
    )
    base.update(over)
    return VolSurfaceData(**base)


def test_grid_point_exact_and_bilinear():
    s = _surface()
    validate_surface(s)
    assert interpolate_vol(s, 0.25, 100.0) == 0.19
    mid = interpolate_vol(s, 0.375, 100.0)
    assert mid == pytest.approx(0.195)
    logger.info("双线性插值 (T=0.375,K=100)=%.6f，判定依据=两个相邻到期日切片中点")


@pytest.mark.parametrize("maturity,strike", [
    (0.1, 100.0), (2.0, 100.0), (0.5, 89.99), (0.5, 110.01),
])
def test_extrapolation_rejected_with_distinct_reason(maturity, strike):
    s = _surface()
    with pytest.raises(PricingEngineError) as exc:
        interpolate_vol(s, maturity, strike)
    assert exc.value.code is ErrorCode.SURFACE_OUT_OF_RANGE
    logger.info("输入 T=%s K=%s 被拒绝：%s（判定依据=严格内插，禁止外插）",
                maturity, strike, exc.value.code.value)


def test_boundary_is_interpolation_not_extrapolation():
    s = _surface()
    assert interpolate_vol(s, 0.25, 90.0) == 0.2  # 盒边界点合法


@pytest.mark.parametrize("over", [
    {"maturities": [0.5]},
    {"maturities": [0.5, 0.25]},
    {"strikes": [100.0, 90.0]},
    {"vols": [[0.2, 0.19, 0.2], [0.22, 0.2]]},
    {"vols": [[0.2, 0.0, 0.2], [0.22, 0.2, 0.22], [0.24, 0.22, 0.24]]},
    {"risk_free_rate": -0.01},
])
def test_invalid_surface_rejected(over):
    with pytest.raises(PricingEngineError) as exc:
        validate_surface(_surface(**over))
    assert exc.value.code is ErrorCode.SURFACE_DATA_INVALID


def test_store_publish_versions_and_immutability(store):
    v1 = store.get(1)
    v2 = store.get(2)
    assert v1.data.label != v2.data.label
    assert store.get(None).version == 2
    # 已发布快照不可被后续修改污染
    v1.data.vols[0][0] = 999.0
    assert store.get(1).data.vols[0][0] != 999.0
    with pytest.raises(PricingEngineError) as exc:
        store.get(99)
    assert exc.value.code is ErrorCode.VERSION_NOT_FOUND


def test_concurrent_publish_never_shows_partial_state():
    """并发发布/读取：读者只能看到完整矩形的曲面快照。"""
    s = SurfaceStore()
    failures: list[Exception] = []

    def publish(n):
        for i in range(n):
            try:
                s.publish(_surface(label=f"L{i}"))
            except Exception as exc:  # noqa: BLE001
                failures.append(exc)

    def read(n):
        for _ in range(n):
            v = s.latest_version()
            if not v:
                continue
            d = s.get(v).data
            assert len(d.vols) == len(d.maturities)
            assert all(len(row) == len(d.strikes) for row in d.vols)
            assert all(x > 0 for row in d.vols for x in row)

    threads = [threading.Thread(target=publish, args=(30,)) for _ in range(4)]
    threads += [threading.Thread(target=read, args=(300,)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not failures
    assert s.latest_version() == 120


def test_loader_publishes_bundled_data(store):
    assert [i.label for i in store.list_versions()] == [
        "SURF-2026Q2", "SURF-2026Q3",
    ]


def test_loader_rejects_bad_files(tmp_path):
    (tmp_path / "bad.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(PricingEngineError) as exc:
        load_surface_file(tmp_path / "bad.json")
    assert exc.value.code is ErrorCode.SURFACE_DATA_INVALID
    with pytest.raises(PricingEngineError):
        publish_directory(SurfaceStore(), tmp_path / "missing")
