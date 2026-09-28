"""Volatility surface: immutable snapshots, bilinear interpolation,
versioned store with atomic publish, and per-version, caller-authorized,
bounded extrapolation.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .errors import (
    ExtrapolationNotAuthorizedError,
    InvalidParameterError,
    SurfaceNotFoundError,
    SurfaceOutOfRangeError,
)
from .models import SurfaceGridPayload


@dataclass(frozen=True)
class ExtrapolationRange:
    """Immutable copy of the region a published version authorises."""

    min_expiry: float
    max_expiry: float
    min_strike: float
    max_strike: float

    def contains(self, expiry: float, strike: float) -> bool:
        return (self.min_expiry <= expiry <= self.max_expiry
                and self.min_strike <= strike <= self.max_strike)

    def as_dict(self) -> dict:
        return {
            "min_expiry": self.min_expiry,
            "max_expiry": self.max_expiry,
            "min_strike": self.min_strike,
            "max_strike": self.max_strike,
        }


@dataclass(frozen=True)
class VolQuery:
    """Outcome of one vol lookup: the vol and an audit report that is
    non-None exactly when the point was served from outside grid support."""

    vol: float
    audit: dict | None


def _linear(axis: tuple[float, ...], values: tuple[float, ...],
            x: float) -> float:
    """Piecewise linear interpolation/extrapolation along one axis.

    End segments are extended linearly (no clamping), so callers decide via
    range gating which x values may be served.
    """
    n = len(axis)
    if n == 1:
        return float(values[0])
    if x <= axis[0]:
        w = (x - axis[0]) / (axis[1] - axis[0])
        return float(values[0] + w * (values[1] - values[0]))
    if x >= axis[-1]:
        w = (x - axis[-2]) / (axis[-1] - axis[-2])
        return float(values[-2] + w * (values[-1] - values[-2]))
    for i in range(n - 1):
        if axis[i] <= x <= axis[i + 1]:
            w = (x - axis[i]) / (axis[i + 1] - axis[i])
            return float(values[i] + w * (values[i + 1] - values[i]))
    raise InvalidParameterError(f"coordinate {x} not locatable")  # pragma: no cover


def _axis_audit(name: str, x: float, lo: float, hi: float,
                limits: ExtrapolationRange | None) -> dict | None:
    """Report overshoot when ``x`` is outside grid support [lo, hi]."""
    if lo <= x <= hi:
        return None
    if x < lo:
        side, edge, x_limit = "below", lo, limits.min_expiry \
            if name == "expiry" else limits.min_strike
        overshoot = lo - x
        headroom = lo - x_limit
    else:
        side, edge, x_limit = "above", hi, limits.max_expiry \
            if name == "expiry" else limits.max_strike
        overshoot = x - hi
        headroom = x_limit - hi
    fraction = overshoot / headroom if headroom > 0 else 0.0
    return {
        "axis": name, "side": side,
        "grid_edge": edge, "limit_edge": x_limit,
        "overshoot": overshoot,
        "headroom_fraction_used": fraction,
    }


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
    extrapolation_limits: ExtrapolationRange | None = None

    def query_vol(self, expiry: float, strike: float,
                  allow_extrapolation: bool = False) -> VolQuery:
        """Look up a vol with explicit, distinguishable range outcomes.

        * in-grid point: bilinear interpolation, audit is None;
        * out of grid, extrapolation allowed and within declared limits:
          linear extrapolation, audit describes the overshoot on each axis;
        * out of grid, caller did not request extrapolation or the version
          declares no limits: SurfaceOutOfRangeError /
          ExtrapolationNotAuthorizedError;
        * beyond the declared limits even with authorisation:
          SurfaceOutOfRangeError (never silently clamped).
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
        in_grid = (e_lo <= expiry <= e_hi and s_lo <= strike <= s_hi)

        if not in_grid:
            if not allow_extrapolation:
                raise SurfaceOutOfRangeError(
                    f"({expiry}, {strike}) outside grid support "
                    f"expiries[{e_lo},{e_hi}] strikes[{s_lo},{s_hi}] "
                    f"and extrapolation was not requested "
                    f"(set allow_extrapolation=true)"
                )
            limits = self.extrapolation_limits
            if limits is None:
                raise ExtrapolationNotAuthorizedError(
                    f"surface '{self.surface_id}' version {self.version} "
                    f"declares no extrapolation limits; out-of-grid point "
                    f"({expiry}, {strike}) cannot be served"
                )
            if not limits.contains(expiry, strike):
                raise SurfaceOutOfRangeError(
                    f"({expiry}, {strike}) beyond extrapolation limits "
                    f"declared by version {self.version}: "
                    f"expiries[{limits.min_expiry},{limits.max_expiry}] "
                    f"strikes[{limits.min_strike},{limits.max_strike}]"
                )

        # Bilinear interpolation/extrapolation (pure math; range already gated).
        row_lo = tuple(_linear(self.strikes, self.vols[i], strike)
                       for i in range(len(self.expiries)))
        vol = _linear(self.expiries, row_lo, expiry)

        audit = None
        if not in_grid:
            axis_reports = [
                r for r in (
                    _axis_audit("expiry", expiry, e_lo, e_hi,
                                self.extrapolation_limits),
                    _axis_audit("strike", strike, s_lo, s_hi,
                                self.extrapolation_limits),
                ) if r is not None
            ]
            audit = {
                "surface_id": self.surface_id,
                "surface_version": self.version,
                "limits": self.extrapolation_limits.as_dict(),
                "axes": axis_reports,
            }
        return VolQuery(vol=vol, audit=audit)

    def vol(self, expiry: float, strike: float) -> float:
        """Backward-compatible lookup: in-grid interpolation only."""
        return self.query_vol(expiry, strike,
                              allow_extrapolation=False).vol


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
    lim = p.extrapolation_limits
    if lim is not None:
        if lim.min_expiry < 0:
            raise InvalidParameterError(
                "extrapolation_limits.min_expiry must be >= 0")
        if lim.min_expiry > p.expiries[0] or lim.max_expiry < p.expiries[-1]:
            raise InvalidParameterError(
                "extrapolation expiry limits must contain the grid support "
                f"[{p.expiries[0]}, {p.expiries[-1]}]"
            )
        if lim.min_strike > p.strikes[0] or lim.max_strike < p.strikes[-1]:
            raise InvalidParameterError(
                "extrapolation strike limits must contain the grid support "
                f"[{p.strikes[0]}, {p.strikes[-1]}]"
            )
        if lim.max_expiry <= lim.min_expiry or lim.max_strike <= lim.min_strike:
            raise InvalidParameterError(
                "extrapolation limits must be strictly increasing ranges")


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
        limits = None
        if payload.extrapolation_limits is not None:
            lim = payload.extrapolation_limits
            limits = ExtrapolationRange(
                min_expiry=float(lim.min_expiry),
                max_expiry=float(lim.max_expiry),
                min_strike=float(lim.min_strike),
                max_strike=float(lim.max_strike),
            )
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
                extrapolation_limits=limits,
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
