"""Distinguishable failure reasons for the pricing service."""


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
