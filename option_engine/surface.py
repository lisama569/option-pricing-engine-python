"""Volatility surface: immutable snapshots, bilinear interpolation,
versioned store with atomic publish.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .errors import InvalidParameterError, SurfaceNotFoundError
from .models import SurfaceGridPayload


def _interp1(axis: tuple[float, ...], values: tuple[float, ...],
             x: float, allow_extrap: bool) -> float:
    """Linear interpolation with explicit, distinguishable range failure."""
    n = len(axis)
    if x < axis[0] or x > axis[-1]:
        if not allow_extrap:
            raise SurfaceOutOfRangeLike(
                f"coordinate {x} outside support [{axis[0]}, {axis[-1]}]"
            )
    if x <= axis[0]:
        if n == 1:
            return float(values[0])
        w = (x - axis[0]) / (axis[1] - axis[0])
        return float(values[0] + w * (values[1] - values[0]))
    if x >= axis[-1]:
        if n == 1:
            return float(values[-1])
        w = (x - axis[-2]) / (axis[-1] - axis[-2])
        return float(values[-2] + w * (values[-1] - values[-2]))
    for i in range(n - 1):
        if axis[i] <= x <= axis[i + 1]:
            w = (x - axis[i]) / (axis[i + 1] - axis[i])
            return float(values[i] + w * (values[i + 1] - values[i]))
    raise SurfaceOutOfRangeLike(f"coordinate {x} not locatable")  # pragma: no cover


class SurfaceOutOfRangeLike(Exception):
    """Internal marker, converted to SurfaceOutOfRangeError with context."""


@dataclass(frozen=True)
class VolSurfaceSnapshot:
    surface_id: str
    version: int
    created_at: str
    spot: float
    rate: float
    dividend: float
    allow_extrapolation: bool
    expiries: tuple[float, ...]
    strikes: tuple[float, ...]
    vols: tuple[tuple[float, ...], ...] = field(repr=False)

    def vol(self, expiry: float, strike: float) -> float:
        if expiry <= 0:
            raise InvalidParameterError(
                f"expiry must be strictly positive, got {expiry}"
            )
        if strike <= 0:
            raise InvalidParameterError(
                f"strike must be strictly positive, got {strike}"
            )
        if not self.allow_extrapolation:
            if not (self.expiries[0] <= expiry <= self.expiries[-1]):
                from .errors import SurfaceOutOfRangeError
                raise SurfaceOutOfRangeError(
                    f"expiry {expiry} outside surface support "
                    f"[{self.expiries[0]}, {self.expiries[-1]}]"
                )
            if not (self.strikes[0] <= strike <= self.strikes[-1]):
                from .errors import SurfaceOutOfRangeError
                raise SurfaceOutOfRangeError(
                    f"strike {strike} outside surface support "
                    f"[{self.strikes[0]}, {self.strikes[-1]}]"
                )
        try:
            if expiry <= self.expiries[0] or expiry >= self.expiries[-1]:
                row = self._slice_expiry(expiry)
                return _interp1(self.strikes, row, strike,
                                self.allow_extrapolation)
            for i in range(len(self.expiries) - 1):
                e0, e1 = self.expiries[i], self.expiries[i + 1]
                if e0 <= expiry <= e1:
                    w = (expiry - e0) / (e1 - e0)
                    v0 = _interp1(self.strikes, self.vols[i], strike,
                                  self.allow_extrapolation)
                    v1 = _interp1(self.strikes, self.vols[i + 1], strike,
                                  self.allow_extrapolation)
                    return v0 + w * (v1 - v0)
        except SurfaceOutOfRangeLike as exc:
            from .errors import SurfaceOutOfRangeError
            raise SurfaceOutOfRangeError(
                f"({expiry}, {strike}) outside surface "
                f"expiries[{self.expiries[0]},{self.expiries[-1]}] "
                f"strikes[{self.strikes[0]},{self.strikes[-1]}]: {exc}"
            ) from exc
        from .errors import SurfaceOutOfRangeError
        raise SurfaceOutOfRangeError(
            f"expiry {expiry} outside support"
        )  # pragma: no cover

    def _slice_expiry(self, expiry: float) -> tuple[float, ...]:
        """Interpolate/extrapolate every strike column at ``expiry``."""
        try:
            if expiry <= self.expiries[0]:
                if len(self.expiries) == 1:
                    return self.vols[0]
                w = (expiry - self.expiries[0]) / (
                    self.expiries[1] - self.expiries[0]
                )
                return tuple(
                    a + w * (b - a)
                    for a, b in zip(self.vols[0], self.vols[1])
                )
            w = (expiry - self.expiries[-2]) / (
                self.expiries[-1] - self.expiries[-2]
            )
            return tuple(
                a + w * (b - a)
                for a, b in zip(self.vols[-2], self.vols[-1])
            )
        except SurfaceOutOfRangeLike:  # pragma: no cover - expiry handled above
            raise


def _validate_payload(p: SurfaceGridPayload) -> None:
    if not p.expiries or not p.strikes:
        raise InvalidParameterError("expiries and strikes must be non-empty")
    if any(e <= 0 for e in p.expiries):
        raise InvalidParameterError("expiries must be strictly positive")
    if any(s <= 0 for s in p.strikes):
        raise InvalidParameterError("strikes must be strictly positive")
    if list(p.expiries) != sorted(p.expiries) or len(set(p.expiries)) != len(p.expiries):
        raise InvalidParameterError("expiries must be strictly increasing")
    if list(p.strikes) != sorted(p.strikes) or len(set(p.strikes)) != len(p.strikes):
        raise InvalidParameterError("strikes must be strictly increasing")
    if len(p.vols) != len(p.expiries) or any(
        len(row) != len(p.strikes) for row in p.vols
    ):
        raise InvalidParameterError(
            f"vols must have shape {len(p.expiries)}x{len(p.strikes)}"
        )
    for row in p.vols:
        for v in row:
            if v <= 0:
                raise InvalidParameterError(
                    f"volatilities must be strictly positive, got {v}"
                )


class SurfaceStore:
    """Holds immutable snapshots per surface id.

    Every publish creates a brand-new snapshot object and installs it while
    holding the lock; readers always observe a complete snapshot, never a
    partially updated grid.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._versions: dict[str, dict[int, VolSurfaceSnapshot]] = {}
        self._latest: dict[str, int] = {}

    def publish(self, surface_id: str, payload: SurfaceGridPayload) -> VolSurfaceSnapshot:
        _validate_payload(payload)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            versions = self._versions.setdefault(surface_id, {})
            next_version = self._latest.get(surface_id, 0) + 1
            snap = VolSurfaceSnapshot(
                surface_id=surface_id,
                version=next_version,
                created_at=now,
                spot=float(payload.spot),
                rate=float(payload.rate),
                dividend=float(payload.dividend),
                allow_extrapolation=bool(payload.allow_extrapolation),
                expiries=tuple(float(e) for e in payload.expiries),
                strikes=tuple(float(s) for s in payload.strikes),
                vols=tuple(tuple(float(v) for v in row) for row in payload.vols),
            )
            versions[next_version] = snap
            self._latest[surface_id] = next_version
            return snap

    def get(self, surface_id: str, version: int | None = None) -> VolSurfaceSnapshot:
        with self._lock:
            versions = self._versions.get(surface_id)
            if not versions:
                raise SurfaceNotFoundError(
                    f"surface '{surface_id}' does not exist"
                )
            if version is None:
                version = self._latest[surface_id]
            snap = versions.get(version)
            if snap is None:
                raise SurfaceNotFoundError(
                    f"surface '{surface_id}' version {version} does not exist"
                )
            return snap

    def latest_version(self, surface_id: str) -> int:
        with self._lock:
            versions = self._versions.get(surface_id)
            if not versions:
                raise SurfaceNotFoundError(
                    f"surface '{surface_id}' does not exist"
                )
            return self._latest[surface_id]

    def list_versions(self, surface_id: str) -> list[VolSurfaceSnapshot]:
        with self._lock:
            versions = self._versions.get(surface_id)
            if not versions:
                raise SurfaceNotFoundError(
                    f"surface '{surface_id}' does not exist"
                )
            return [versions[v] for v in sorted(versions)]
