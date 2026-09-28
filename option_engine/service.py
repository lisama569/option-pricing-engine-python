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
from .models import (
    MIN_RESOLUTION,
    MAX_RESOLUTION,
    PriceResponse,
    PricingRequest,
    ResolutionMode,
)
from .surface import SurfaceStore


# The result cache is in-memory and dies with the process, so no on-disk
# schema migration is needed; new PriceResponse fields are populated by
# recomputation on every fresh process.


def _request_cache_key(req: PricingRequest) -> str:
    # Canonical, model-dict based serialisation; enums become their value.
    payload = req.model_dump(
        mode="json",
        exclude={"surface_id", "version", "request_id"})
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

        spot = req.spot if req.spot is not None else snap.spot
        rate = req.rate if req.rate is not None else snap.rate
        dividend = req.dividend if req.dividend is not None else snap.dividend
        opt = req.option_type.value

        # Engine-layer defence for requests constructed via model_construct.
        if req.resolution is not None:
            if not isinstance(req.resolution, int) or isinstance(
                    req.resolution, bool):
                raise InvalidParameterError(
                    f"resolution must be an integer, got {req.resolution!r}")
            if not (MIN_RESOLUTION <= req.resolution <= MAX_RESOLUTION):
                raise InvalidParameterError(
                    f"resolution {req.resolution} outside allowed range "
                    f"[{MIN_RESOLUTION}, {MAX_RESOLUTION}]")
            if opt != "american_put":
                raise InvalidParameterError(
                    "resolution is supported only for option_type "
                    "american_put")

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

        query = snap.query_vol(req.expiry, req.strike,
                               allow_extrapolation=req.allow_extrapolation)
        vol = query.vol

        if req.resolution is not None:
            price = pricing.american_put_fixed(
                spot, req.strike, req.expiry, vol, rate, dividend,
                steps=req.resolution, cancel_event=cancel_event,
            )
            return PriceResponse(
                price=price, vol_used=vol, surface_version=snap.version,
                method="crr_binomial_american_put",
                detail={"steps_used": req.resolution, "converged": False},
                resolution_mode=ResolutionMode.FIXED,
                resolution_used=req.resolution,
                extrapolated=query.audit is not None,
                extrapolation=query.audit,
            )

        if opt in ("vanilla_call", "vanilla_put", "digital_call", "digital_put"):
            price = pricing.bs_price(
                opt, spot, req.strike, req.expiry, vol, rate, dividend,
                req.payout,
            )
            method = "black_scholes_merton"
            detail: dict = {"spot": spot, "rate": rate, "dividend": dividend}
        elif opt == "american_put":
            price, info = pricing.american_put_price(
                spot, req.strike, req.expiry, vol, rate, dividend,
                start_steps=req.steps, tol=req.convergence_tol,
                max_steps=req.max_steps, cancel_event=cancel_event,
            )
            method = "crr_binomial_american_put"
            detail = info
        else:  # pragma: no cover - enum guards this
            raise InvalidParameterError(f"unsupported option type {opt}")

        return PriceResponse(
            price=price, vol_used=vol, surface_version=snap.version,
            method=method, detail=detail,
            resolution_mode=ResolutionMode.CONVERGED,
            resolution_used=detail.get("steps_used")
            if opt == "american_put" else None,
            extrapolated=query.audit is not None,
            extrapolation=query.audit,
        )

    def cache_size(self) -> int:
        with self._lock:
            return len(self._cache)

    def price_batch(self, batch, max_workers: int = 8):
        """Price every item of a batch concurrently, bound to one version.

        The version is resolved exactly once up front (None pins the latest
        at submission), so a publish during the batch can never split items
        across versions. Each item is evaluated through the exact same
        ``price`` path as a standalone request (same cache, same validators,
        same result), and any failure is captured into that item's own error
        slot -- one item's failure or slowness never aborts or blocks the
        others.
        """
        from concurrent.futures import ThreadPoolExecutor

        from .models import BatchItemError, BatchItemResult, BatchPricingResponse
        from .errors import PricingError

        # Pin once for the whole batch BEFORE any item is evaluated.
        snap = self.store.get(batch.surface_id, batch.version)
        version = snap.version

        def run_item(index_item):
            index, item = index_item
            try:
                req = item.to_request(batch.surface_id, version)
                resp = self.price(req)
                return BatchItemResult(index=index, id=item.id, result=resp)
            except PricingError as exc:
                return BatchItemResult(
                    index=index, id=item.id,
                    error=BatchItemError(code=exc.code, message=str(exc)))
            except Exception as exc:  # defensive: isolate unexpected failures
                return BatchItemResult(
                    index=index, id=item.id,
                    error=BatchItemError(code="internal_error",
                                        message=repr(exc)))

        workers = max(1, min(max_workers, len(batch.items)))
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="batch-price") as pool:
            outcomes = list(pool.map(run_item, enumerate(batch.items)))

        return BatchPricingResponse(
            surface_id=batch.surface_id,
            surface_version=version,
            items=outcomes,
        )
