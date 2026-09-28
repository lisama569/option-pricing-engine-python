"""Volatility surface: immutable snapshots, bilinear interpolation,
versioned store with atomic publish.

Extrapolation policy: every snapshot either declares an *extrapolation
range* at publish time (``extrapolation_limits``) or forbids extrapolation
outright. Extrapolation is additionally opt-in per request and is rejected
(``SurfaceOutOfRangeError``) when the query point lies beyond the declared
range of the bound version.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .errors import InvalidParameterError, SurfaceNotFoundError, SurfaceOutOfRangeError
from .models import SurfaceGridPayload


def _interp1(axis: tuple[float, ...], values: tuple[float, ...],
             x: float) -> float:
    """Piecewise-linear interpolation, linearly continued beyond the ends.

    Range authorization is handled by the caller (``quote``); this function
    only evaluates the piecewise-linear extension.
    """
    n = len(axis)
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
    raise InvalidParameterError(f"coordinate {x} not locatable")  # pragma: no cover


@dataclass(frozen=True)
class ExtrapolationAudit:
    """How far a vol lookup actually left the grid support."""

    expiry_outside: bool
    strike_outside: bool
    expiry_overhang: float
    strike_overhang: float
    expiry_overhang_fraction: float
    strike_overhang_fraction: float


@dataclass(frozen=True)
class VolSurfaceSnapshot:
    surface_id: str
    version: int
    created_at: str
    spot: float
    rate: float
    dividend: float
    expiries: tuple[float, ...]
    strikes: tuple[float, ...]
    vols: tuple[tuple[float, ...], ...] = field(repr=False)
    # (min_expiry, max_expiry, min_strike, max_strike) or None = forbidden.
    extrapolation_limits: tuple[float, float, float, float] | None = None

    def vol(self, expiry: float, strike: float) -> float:
        """Strict lookup: in-support interpolation, out-of-support rejected."""
        value, _audit = self.quote(expiry, strike, allow_extrap=False)
        return value

    def quote(self, expiry: float, strike: float,
              allow_extrap: bool = False
              ) -> tuple[float, ExtrapolationAudit | None]:
        """Look up a vol with explicit extrapolation authorization.

        Returns ``(vol, audit)`` where ``audit`` is None for an in-support
        interpolation and populated whenever extrapolation was actually
        used. Raises ``SurfaceOutOfRangeError`` for:
          * any out-of-support point when ``allow_extrap`` is False;
          * any out-of-support point when the snapshot declares no range;
          * points beyond the snapshot's declared extrapolation range.
        """
        if expiry <= 0:
            raise InvalidParameterError(
                f"expiry must be strictly positive, got {expiry}"
            )
        if strike <= 0:
            raise InvalidParameterError(
                f"strike must be strictly positive, got {strike}"
            )
        e_lo, e_hi = self.expiries[0], self.expiries[-1]
        s_lo, s_hi = self.strikes[0], self.strikes[-1]
        expiry_outside = expiry < e_lo or expiry > e_hi
        strike_outside = strike < s_lo or strike > s_hi

        if expiry_outside or strike_outside:
            if not allow_extrap:
                raise SurfaceOutOfRangeError(
                    f"({expiry}, {strike}) outside surface support "
                    f"expiries[{e_lo},{e_hi}] strikes[{s_lo},{s_hi}] and "
                    f"extrapolation was not requested"
                )
            if self.extrapolation_limits is None:
                raise SurfaceOutOfRangeError(
                    f"surface version {self.version} declares no "
                    f"extrapolation range; ({expiry}, {strike}) outside "
                    f"support expiries[{e_lo},{e_hi}] strikes[{s_lo},{s_hi}]"
                )
            min_e, max_e, min_s, max_s = self.extrapolation_limits
            if not (min_e <= expiry <= max_e and min_s <= strike <= max_s):
                raise SurfaceOutOfRangeError(
                    f"({expiry}, {strike}) beyond declared extrapolation "
                    f"range expiries[{min_e},{max_e}] "
                    f"strikes[{min_s},{max_s}]"
                )

        vol = self._bilinear(expiry, strike)
        audit = None
        if expiry_outside or strike_outside:
            audit = self._audit(expiry, strike, expiry_outside,
                                strike_outside)
        return vol, audit

    def _bilinear(self, expiry: float, strike: float) -> float:
        if expiry <= self.expiries[0] or expiry >= self.expiries[-1]:
            row = self._slice_expiry(expiry)
            return _interp1(self.strikes, row, strike)
        for i in range(len(self.expiries) - 1):
            e0, e1 = self.expiries[i], self.expiries[i + 1]
            if e0 <= expiry <= e1:
                w = (expiry - e0) / (e1 - e0)
                v0 = _interp1(self.strikes, self.vols[i], strike)
                v1 = _interp1(self.strikes, self.vols[i + 1], strike)
                return v0 + w * (v1 - v0)
        raise InvalidParameterError(
            f"expiry {expiry} not locatable")  # pragma: no cover

    def _audit(self, expiry: float, strike: float,
               expiry_outside: bool, strike_outside: bool
               ) -> ExtrapolationAudit:
        e_lo, e_hi = self.expiries[0], self.expiries[-1]
        s_lo, s_hi = self.strikes[0], self.strikes[-1]
        min_e, max_e, min_s, max_s = self.extrapolation_limits  # type: ignore[misc]
        e_over = max(e_lo - expiry, expiry - e_hi, 0.0)
        s_over = max(s_lo - strike, strike - s_hi, 0.0)
        e_width = max_e - min_e
        s_width = max_s - min_s
        return ExtrapolationAudit(
            expiry_outside=expiry_outside,
            strike_outside=strike_outside,
            expiry_overhang=e_over,
            strike_overhang=s_over,
            expiry_overhang_fraction=e_over / e_width if e_width else 0.0,
            strike_overhang_fraction=s_over / s_width if s_width else 0.0,
        )

    def _slice_expiry(self, expiry: float) -> tuple[float, ...]:
        """Interpolate/extrapolate every strike column at ``expiry``."""
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
    if p.extrapolation_limits is not None:
        lim = p.extrapolation_limits
        if not (lim.min_expiry < lim.max_expiry
                and lim.min_strike < lim.max_strike):
            raise InvalidParameterError(
                "extrapolation_limits must satisfy min < max on both axes"
            )
        e_lo, e_hi = p.expiries[0], p.expiries[-1]
        s_lo, s_hi = p.strikes[0], p.strikes[-1]
        if not (lim.min_expiry <= e_lo and lim.max_expiry >= e_hi
                and lim.min_strike <= s_lo and lim.max_strike >= s_hi):
            raise InvalidParameterError(
                f"extrapolation_limits {lim} must enclose the grid support "
                f"expiries[{e_lo},{e_hi}] strikes[{s_lo},{s_hi}]"
            )


def _limits_tuple(p: SurfaceGridPayload
                  ) -> tuple[float, float, float, float] | None:
    lim = p.extrapolation_limits
    if lim is None:
        return None
    return (float(lim.min_expiry), float(lim.max_expiry),
            float(lim.min_strike), float(lim.max_strike))


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
                expiries=tuple(float(e) for e in payload.expiries),
                strikes=tuple(float(s) for s in payload.strikes),
                vols=tuple(tuple(float(v) for v in row) for row in payload.vols),
                extrapolation_limits=_limits_tuple(payload),
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
