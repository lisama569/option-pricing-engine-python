"""Distinguishable failure reasons for the pricing service."""


def format_validation_errors(errors: list[dict]) -> str:
    """Render pydantic validation errors as ``"<field>: <reason>"`` parts.

    Shared by the HTTP validation handler (sync + async submission) and the
    per-leg batch failure path, so every entry names the offending field and
    explains why the value was rejected using identical wording.
    """
    parts = []
    for err in errors:
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "")
        reason = err.get("msg", "invalid value")
        etype = err.get("type")
        # Carry the machine-readable error-type token (e.g.
        # resolution_bad_type / resolution_out_of_range) so batch legs —
        # which have no structured detail field — stay distinguishable.
        head = f"{loc} [{etype}]" if loc and etype else (loc or etype or "")
        parts.append(f"{head}: {reason}" if head else reason)
    return "request parameter validation failed: " + "; ".join(parts)


class PricingError(Exception):
    """Base class for all service failures.

    ``code`` is a stable, machine-readable failure identifier returned to
    clients so that different failure reasons remain distinguishable.
    """

    code: str = "pricing_error"


class InvalidParameterError(PricingError):
    """Model/request parameters are illegal (non-positive, bad enum, ...)."""

    code = "invalid_parameter"


class SurfaceOutOfRangeError(PricingError):
    """Requested (expiry, strike) lies outside the surface support and
    extrapolation is forbidden."""

    code = "surface_out_of_range"


class SurfaceNotFoundError(PricingError):
    """Requested surface id/version does not exist."""

    code = "surface_not_found"


class SolverNonConvergenceError(PricingError):
    """A numerical solver (IV root find / tree convergence) did not meet its
    tolerance within the allowed work budget."""

    code = "solver_non_convergence"


class JobCancelledError(PricingError):
    """The asynchronous job was cancelled before completion."""

    code = "job_cancelled"
