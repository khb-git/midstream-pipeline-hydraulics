"""
Gas pipeline flow equations.

For gas pipelines, the compressibility of the fluid (density changes
significantly with pressure) means we can't use simple Darcy-Weisbach.
Industry uses empirical correlations derived from the general gas-flow
equation with assumptions about the friction factor:

- Weymouth (1912): assumes fully-turbulent flow, friction factor f = 0.032/D^(1/3).
  Best for short, high-pressure pipelines.
- Panhandle B (1956): assumes partially-turbulent flow with a Reynolds-
  number-dependent friction factor. Best for long transmission lines.

This implementation uses the standard field-unit forms of these equations
(documented in Menon 2005 and GPSA Section 17), converting between SI
and field units internally. All inputs and outputs are in SI; internal
conversion factors are applied so that the verified empirical coefficients
can be used directly.

Standard conditions: 60°F (519.67°R = 288.71 K), 14.7 psia (101.325 kPa).

References:
    Menon, E.S. (2005). Gas Pipeline Hydraulics. CRC Press.
    GPSA Engineering Data Book, Section 17.
"""

from dataclasses import dataclass
import numpy as np


# Standard conditions for gas industry calculations
T_STANDARD_R = 519.67       # 60°F in Rankine
P_STANDARD_PSI = 14.7       # 1 atm in psia
T_STANDARD_K = 288.71       # 60°F in Kelvin

# Unit conversion factors
PSI_PER_PA = 1 / 6894.76
M_PER_MILE = 1609.344
IN_PER_M = 39.3701
K_TO_R = 1.8


@dataclass
class GasFlowResult:
    """Result of a gas pipeline flow calculation."""
    flow_rate_sm3_s: float       # Standard cubic meters per second
    flow_rate_mmscfd: float      # Million standard cubic feet per day
    equation: str                # 'weymouth' or 'panhandle_b'
    p_inlet_pa: float
    p_outlet_pa: float
    length_m: float
    diameter_m: float


def _elevation_correction_psi_squared(
    elevation_change_m: float,
    p_avg_psi: float,
    specific_gravity: float,
    avg_temperature_r: float,
    compressibility: float,
) -> float:
    """
    Elevation correction term (units of psi²) for the general gas-flow eq.

    ΔH = 0.0375 * G * (h2 - h1) * P_avg² / (Z * T_avg)
    where h in feet, P_avg in psia, T_avg in Rankine.
    """
    h_ft = elevation_change_m * 3.28084
    return 0.0375 * specific_gravity * h_ft * p_avg_psi**2 / (compressibility * avg_temperature_r)


def weymouth_flow(
    p_inlet_pa: float,
    p_outlet_pa: float,
    length_m: float,
    diameter_m: float,
    specific_gravity: float,
    avg_temperature_k: float = T_STANDARD_K,
    compressibility: float = 0.95,
    efficiency: float = 0.92,
    elevation_change_m: float = 0.0,
) -> GasFlowResult:
    """
    Weymouth gas flow equation (high-pressure, short pipelines).

    Field-unit form (Menon 2005, eq. 5.5):
        Q_scfd = 433.5 * E * (T_b/P_b) * sqrt((P1² - P2² - ΔH) / (G*T*L*Z)) * D^(8/3)

    Where P in psia, T in Rankine, L in miles, D in inches, Q in SCFD.

    Args:
        p_inlet_pa: Inlet absolute pressure (Pa)
        p_outlet_pa: Outlet absolute pressure (Pa)
        length_m: Pipeline length (m)
        diameter_m: Internal diameter (m)
        specific_gravity: Gas SG relative to air (typical 0.55-0.70)
        avg_temperature_k: Average flowing temperature (K)
        compressibility: Average Z factor (~0.90-0.98)
        efficiency: Pipeline efficiency factor (0.85-0.95)
        elevation_change_m: Outlet minus inlet elevation (m)

    Returns:
        GasFlowResult
    """
    if p_inlet_pa <= p_outlet_pa:
        raise ValueError("Inlet pressure must exceed outlet pressure")

    p1_psi = p_inlet_pa * PSI_PER_PA
    p2_psi = p_outlet_pa * PSI_PER_PA
    p_avg_psi = (p1_psi + p2_psi) / 2
    t_avg_r = avg_temperature_k * K_TO_R
    l_mi = length_m / M_PER_MILE
    d_in = diameter_m * IN_PER_M

    elevation_term = _elevation_correction_psi_squared(
        elevation_change_m, p_avg_psi, specific_gravity, t_avg_r, compressibility,
    )

    pressure_term = p1_psi**2 - p2_psi**2 - elevation_term
    if pressure_term <= 0:
        raise ValueError(
            "Pressure differential insufficient to overcome elevation gain"
        )

    q_scfd = (
        433.5
        * efficiency
        * (T_STANDARD_R / P_STANDARD_PSI)
        * np.sqrt(pressure_term / (specific_gravity * t_avg_r * l_mi * compressibility))
        * d_in ** (8 / 3)
    )

    # SCFD -> Sm³/s (1 Sm³ = 35.3147 SCF, 86400 sec/day)
    q_sm3_s = q_scfd / (35.3147 * 86400)
    q_mmscfd = q_scfd / 1e6

    return GasFlowResult(
        flow_rate_sm3_s=q_sm3_s,
        flow_rate_mmscfd=q_mmscfd,
        equation="weymouth",
        p_inlet_pa=p_inlet_pa,
        p_outlet_pa=p_outlet_pa,
        length_m=length_m,
        diameter_m=diameter_m,
    )


def panhandle_b_flow(
    p_inlet_pa: float,
    p_outlet_pa: float,
    length_m: float,
    diameter_m: float,
    specific_gravity: float,
    avg_temperature_k: float = T_STANDARD_K,
    compressibility: float = 0.95,
    efficiency: float = 0.92,
    elevation_change_m: float = 0.0,
) -> GasFlowResult:
    """
    Panhandle B equation for long-distance gas transmission.

    Field-unit form (Menon 2005, eq. 5.10):
        Q_scfd = 737 * E * (T_b/P_b)^1.02
                 * [(P1² - P2² - ΔH) / (G^0.961 * T * L * Z)]^0.51
                 * D^2.53

    Args:
        Same as weymouth_flow

    Returns:
        GasFlowResult
    """
    if p_inlet_pa <= p_outlet_pa:
        raise ValueError("Inlet pressure must exceed outlet pressure")

    p1_psi = p_inlet_pa * PSI_PER_PA
    p2_psi = p_outlet_pa * PSI_PER_PA
    p_avg_psi = (p1_psi + p2_psi) / 2
    t_avg_r = avg_temperature_k * K_TO_R
    l_mi = length_m / M_PER_MILE
    d_in = diameter_m * IN_PER_M

    elevation_term = _elevation_correction_psi_squared(
        elevation_change_m, p_avg_psi, specific_gravity, t_avg_r, compressibility,
    )

    pressure_term = p1_psi**2 - p2_psi**2 - elevation_term
    if pressure_term <= 0:
        raise ValueError(
            "Pressure differential insufficient to overcome elevation gain"
        )

    q_scfd = (
        737
        * efficiency
        * (T_STANDARD_R / P_STANDARD_PSI) ** 1.02
        * (pressure_term / (specific_gravity ** 0.961 * t_avg_r * l_mi * compressibility)) ** 0.51
        * d_in ** 2.53
    )

    q_sm3_s = q_scfd / (35.3147 * 86400)
    q_mmscfd = q_scfd / 1e6

    return GasFlowResult(
        flow_rate_sm3_s=q_sm3_s,
        flow_rate_mmscfd=q_mmscfd,
        equation="panhandle_b",
        p_inlet_pa=p_inlet_pa,
        p_outlet_pa=p_outlet_pa,
        length_m=length_m,
        diameter_m=diameter_m,
    )
