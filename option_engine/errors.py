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
    """Requested (expiry, strike) lies outside the region that may be served:

    either outside the grid support with extrapolation not requested / not
    declared by the surface version, or beyond that version's declared
    extrapolation limits."""

    code = "surface_out_of_range"


class ExtrapolationNotAuthorizedError(PricingError):
    """Caller explicitly requested extrapolation against a surface version
    that declares no extrapolation limits at all."""

    code = "extrapolation_not_authorized"


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
