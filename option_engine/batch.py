"""Batch pricing: one surface version, many independent legs.

The surface version is resolved once for the whole batch and every leg is
built into the exact same ``PricingRequest`` shape a standalone caller
would send, so per-leg results are identical to the single-request path.
Legs execute on a dedicated bounded thread pool: a slow or failing leg
neither blocks nor fails any other leg.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from pydantic import ValidationError

from .errors import PricingError, format_validation_errors
from .models import (
    BatchPricingRequest,
    BatchPricingResponse,
    BatchPricingResult,
    PricingRequest,
)
from .service import PricingService


class BatchPricer:
    def __init__(self, service: PricingService, max_workers: int = 8) -> None:
        self._service = service
        self._max_workers = max(1, max_workers)

    def price_batch(self, batch: BatchPricingRequest) -> BatchPricingResponse:
        # Pin the version once for the whole batch. A missing surface/version
        # fails the batch as a whole (nothing leg-specific can be priced).
        snap = self._service.store.get(batch.surface_id, batch.version)
        version = snap.version

        results: list[BatchPricingResult | None] = [None] * len(batch.items)
        workers = min(self._max_workers, len(batch.items))
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="batch") as pool:
            futures = {
                pool.submit(self._price_one, batch.surface_id, version, i, item): i
                for i, item in enumerate(batch.items)
            }
            for fut in futures:
                i = futures[fut]
                results[i] = fut.result()
        return BatchPricingResponse(
            surface_id=batch.surface_id,
            surface_version=version,
            results=[r for r in results if r is not None],
        )

    def _price_one(self, surface_id: str, version: int, index: int,
                   item) -> BatchPricingResult:
        try:
            # Materialise the identical request object a standalone caller
            # would send, bound to the batch-pinned version. Item-level
            # validation failures stay confined to this leg.
            req = PricingRequest(
                surface_id=surface_id,
                version=version,
                option_type=item.option_type,
                expiry=item.expiry,
                strike=item.strike,
                spot=item.spot,
                rate=item.rate,
                dividend=item.dividend,
                payout=item.payout,
                resolution=item.resolution,
                allow_extrapolation=item.allow_extrapolation,
                steps=item.steps,
                convergence_tol=item.convergence_tol,
                max_steps=item.max_steps,
                target_price=item.target_price,
            )
            resp = self._service.price(req)
            return BatchPricingResult(index=index, status="succeeded",
                                      result=resp)
        except ValidationError as exc:
            # Illegal leg parameters (e.g. resolution=true / 5.0 / "5") are
            # rejected HERE, confined to this leg, with the same strict rule
            # and distinguishable code as the single-request path.
            return BatchPricingResult(
                index=index, status="failed",
                error_code="invalid_parameter",
                error_message=format_validation_errors(exc.errors()))
        except PricingError as exc:
            return BatchPricingResult(
                index=index, status="failed",
                error_code=exc.code, error_message=str(exc))
        except Exception as exc:  # defensive: isolate unexpected leg errors
            return BatchPricingResult(
                index=index, status="failed",
                error_code="internal_error", error_message=repr(exc))
