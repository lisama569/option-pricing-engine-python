"""PricingService: binds a request to an immutable surface version, looks up
the vol, runs the chosen model and memoises results.

Determinism guarantee: all pricers are pure functions of (request, surface
snapshot); the snapshot is immutable per version, so identical inputs on the
same version always produce identical results, whether called synchronously
or through an async job. A per-version result cache enforces that literally.
"""
from __future__ import annotations

import hashlib
import json
import threading

from . import pricing
from .errors import InvalidParameterError
from .models import ExtrapolationInfo, PriceResponse, PricingRequest
from .resolution import validate_resolution
from .surface import SurfaceStore


def _request_cache_key(req: PricingRequest) -> str:
    # Canonical, model-dict based serialisation; enums become their value.
    payload = req.model_dump(mode="json", exclude={"surface_id", "version"})
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


class PricingService:
    def __init__(self, store: SurfaceStore) -> None:
        self.store = store
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, int, str], PriceResponse] = {}

    def price(self, req: PricingRequest, cancel_event=None) -> PriceResponse:
        # Resolve the version FIRST and pin it for the whole computation.
        snap = self.store.get(req.surface_id, req.version)
        version = snap.version
        key = (req.surface_id, version, _request_cache_key(req))

        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                return hit.model_copy(update={"cached": True})

        resp = self._compute(req, snap, cancel_event)
        with self._lock:
            existing = self._cache.get(key)
            if existing is not None:
                return existing.model_copy(update={"cached": True})
            self._cache[key] = resp
        return resp

    def _compute(self, req: PricingRequest, snap, cancel_event=None) -> PriceResponse:
        if cancel_event is not None and cancel_event.is_set():
            from .errors import JobCancelledError
            raise JobCancelledError("cancelled before computation")

        # Defence in depth: enforce the exact shared resolution rules at the
        # engine layer too, so a request object that bypassed model
        # validation (e.g. model_construct) can never feed a bool/float/string
        # or out-of-range value into the binomial pricer. No substitution.
        resolution = validate_resolution(req.resolution)

        # Fixed discrete resolution is meaningful only for the binomial
        # American put; reject explicitly instead of silently ignoring it.
        if resolution is not None and req.option_type.value != "american_put":
            raise InvalidParameterError(
                "resolution is supported only for american_put; "
                f"option_type was {req.option_type.value}"
            )

        spot = req.spot if req.spot is not None else snap.spot
        rate = req.rate if req.rate is not None else snap.rate
        dividend = req.dividend if req.dividend is not None else snap.dividend
        opt = req.option_type.value

        if req.target_price is not None:
            # Solve IV from the supplied market price; the surface is not
            # consulted for vol but the version is still bound/reported.
            iv = pricing.implied_vol(
                opt, req.target_price, spot, req.strike, req.expiry,
                rate, dividend,
            )
            if cancel_event is not None and cancel_event.is_set():
                from .errors import JobCancelledError
                raise JobCancelledError("cancelled after IV solve")
            price = pricing.bs_price(
                opt, spot, req.strike, req.expiry, iv, rate, dividend,
                req.payout,
            )
            return PriceResponse(
                price=price, vol_used=iv, surface_version=snap.version,
                method="implied_vol_bisection",
                detail={"target_price": req.target_price},
            )

        vol, audit = snap.quote(req.expiry, req.strike,
                                allow_extrap=req.allow_extrapolation)
        extrap_info = (
            ExtrapolationInfo(**audit.__dict__) if audit is not None else None
        )

        if opt in ("vanilla_call", "vanilla_put", "digital_call", "digital_put"):
            price = pricing.bs_price(
                opt, spot, req.strike, req.expiry, vol, rate, dividend,
                req.payout,
            )
            method = "black_scholes_merton"
            mode = "closed_form"
            steps_used = None
            detail: dict = {"spot": spot, "rate": rate, "dividend": dividend}
        elif opt == "american_put":
            if resolution is not None:
                price, info = pricing.american_put_fixed(
                    spot, req.strike, req.expiry, vol, rate, dividend,
                    resolution, cancel_event=cancel_event,
                )
                mode = "fixed_resolution"
                steps_used = resolution
            else:
                price, info = pricing.american_put_price(
                    spot, req.strike, req.expiry, vol, rate, dividend,
                    start_steps=req.steps, tol=req.convergence_tol,
                    max_steps=req.max_steps, cancel_event=cancel_event,
                )
                mode = "auto_converged"
                steps_used = info.get("steps_used")
            method = "crr_binomial_american_put"
            detail = info
        else:  # pragma: no cover - enum guards this
            raise InvalidParameterError(f"unsupported option type {opt}")

        return PriceResponse(
            price=price, vol_used=vol, surface_version=snap.version,
            method=method, detail=detail,
            pricing_mode=mode, steps_used=steps_used,
            extrapolated=extrap_info is not None,
            extrapolation=extrap_info,
        )

    def cache_size(self) -> int:
        with self._lock:
            return len(self._cache)
