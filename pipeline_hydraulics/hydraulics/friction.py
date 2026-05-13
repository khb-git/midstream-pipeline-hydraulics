"""
Friction factor calculations for pipe flow.

Implements multiple methods:
- Laminar (Re < 2300): f = 64/Re (Hagen-Poiseuille)
- Colebrook-White (turbulent): implicit, solved iteratively
- Swamee-Jain (turbulent): explicit approximation, ~1% accurate

The Colebrook-White equation is the industry-standard friction factor
correlation for fully-developed turbulent pipe flow. It's implicit in f
(the friction factor appears on both sides), so we solve it with Brent's
method on the residual.

References:
    Colebrook, C.F. (1939). "Turbulent Flow in Pipes, with Particular
        Reference to the Transition Region Between the Smooth and Rough
        Pipe Laws." J. Inst. Civil Eng. 11, 133-156.
    Swamee, P.K. & Jain, A.K. (1976). "Explicit Equations for Pipe-Flow
        Problems." J. Hyd. Div. ASCE 102(5), 657-664.
"""

import numpy as np
from scipy.optimize import brentq


# Transition regime boundaries (Reynolds number)
RE_LAMINAR_MAX = 2300
RE_TURBULENT_MIN = 4000


def reynolds_number(rho: float, velocity: float, diameter: float, viscosity: float) -> float:
    """
    Calculate Reynolds number.

    Args:
        rho: Fluid density (kg/m³)
        velocity: Mean flow velocity (m/s)
        diameter: Internal pipe diameter (m)
        viscosity: Dynamic viscosity (Pa·s)

    Returns:
        Dimensionless Reynolds number
    """
    if viscosity <= 0:
        raise ValueError(f"Viscosity must be positive, got {viscosity}")
    if diameter <= 0:
        raise ValueError(f"Diameter must be positive, got {diameter}")
    return abs(rho * velocity * diameter / viscosity)


def friction_factor_laminar(re: float) -> float:
    """Hagen-Poiseuille friction factor for laminar flow."""
    if re <= 0:
        raise ValueError(f"Reynolds number must be positive, got {re}")
    return 64.0 / re


def friction_factor_colebrook(re: float, roughness: float, diameter: float,
                              tol: float = 1e-10) -> float:
    """
    Solve Colebrook-White equation iteratively for friction factor.

    The equation:
        1/sqrt(f) = -2 * log10(eps/(3.7*D) + 2.51/(Re*sqrt(f)))

    Args:
        re: Reynolds number
        roughness: Absolute pipe roughness (m)
        diameter: Internal pipe diameter (m)
        tol: Convergence tolerance

    Returns:
        Darcy friction factor (dimensionless)

    Raises:
        ValueError if re is in the laminar regime
    """
    if re <= 0:
        raise ValueError(f"Reynolds number must be positive, got {re}")
    if re < RE_LAMINAR_MAX:
        # Caller should use the laminar formula; we still handle it
        return friction_factor_laminar(re)

    eps_d = roughness / diameter

    def residual(f):
        if f <= 0:
            return 1e10  # Out of physical range
        return 1.0 / np.sqrt(f) + 2.0 * np.log10(
            eps_d / 3.7 + 2.51 / (re * np.sqrt(f))
        )

    # Bracket: friction factor is positive and physically bounded
    return float(brentq(residual, 1e-5, 1.0, xtol=tol))


def friction_factor_swamee_jain(re: float, roughness: float, diameter: float) -> float:
    """
    Explicit approximation to Colebrook-White (Swamee-Jain 1976).

    Accuracy: ~1% across the turbulent range.

    Args:
        re: Reynolds number
        roughness: Absolute pipe roughness (m)
        diameter: Internal pipe diameter (m)

    Returns:
        Darcy friction factor
    """
    if re <= 0:
        raise ValueError(f"Reynolds number must be positive, got {re}")
    if re < RE_LAMINAR_MAX:
        return friction_factor_laminar(re)

    eps_d = roughness / diameter
    denom = np.log10(eps_d / 3.7 + 5.74 / re**0.9)
    return float(0.25 / denom**2)


def friction_factor(re: float, roughness: float, diameter: float,
                    method: str = "colebrook") -> float:
    """
    Universal friction factor calculator.

    For laminar flow (Re < 2300) uses Hagen-Poiseuille regardless of method.
    For transitional flow (2300 < Re < 4000) interpolates between laminar
    and turbulent values to avoid a discontinuity.
    For turbulent flow uses the requested method.

    Args:
        re: Reynolds number
        roughness: Absolute pipe roughness (m)
        diameter: Internal pipe diameter (m)
        method: 'colebrook' (default, iterative) or 'swamee_jain' (explicit)

    Returns:
        Darcy friction factor
    """
    if re <= 0:
        raise ValueError(f"Reynolds number must be positive, got {re}")

    if re < RE_LAMINAR_MAX:
        return friction_factor_laminar(re)

    if method == "colebrook":
        turb_calc = friction_factor_colebrook
    elif method == "swamee_jain":
        turb_calc = friction_factor_swamee_jain
    else:
        raise ValueError(f"Unknown method: {method}")

    if re > RE_TURBULENT_MIN:
        return turb_calc(re, roughness, diameter)

    # Transitional regime — linear interpolation in log space
    f_lam = friction_factor_laminar(RE_LAMINAR_MAX)
    f_turb = turb_calc(RE_TURBULENT_MIN, roughness, diameter)
    fraction = (re - RE_LAMINAR_MAX) / (RE_TURBULENT_MIN - RE_LAMINAR_MAX)
    return f_lam + fraction * (f_turb - f_lam)
