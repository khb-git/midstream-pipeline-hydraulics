"""
Darcy-Weisbach pressure drop for steady-state incompressible flow.

For liquid (incompressible) flow:
    ΔP_friction = f * (L/D) * (ρ*v²/2)
    ΔP_elevation = ρ * g * Δz
    ΔP_total = ΔP_friction + ΔP_elevation

The elevation term is positive when the outlet is higher than the inlet
(more pressure loss). Negative elevation change (downhill) reduces the
total pressure drop and can even produce pressure recovery.
"""

from dataclasses import dataclass
import numpy as np

from pipeline_hydraulics.hydraulics.friction import friction_factor, reynolds_number


GRAVITY = 9.80665  # Standard gravity (m/s²)


@dataclass
class PressureDropResult:
    """Detailed results of a pressure drop calculation."""
    flow_regime: str        # 'laminar', 'transitional', 'turbulent'
    reynolds: float
    friction_factor: float
    velocity_m_s: float
    dp_friction_pa: float
    dp_elevation_pa: float
    dp_total_pa: float
    hydraulic_grade_line_drop_m: float  # Equivalent head loss in m of fluid

    @property
    def dp_friction_psi(self) -> float:
        return self.dp_friction_pa / 6894.76

    @property
    def dp_elevation_psi(self) -> float:
        return self.dp_elevation_pa / 6894.76

    @property
    def dp_total_psi(self) -> float:
        return self.dp_total_pa / 6894.76


def velocity_from_flow_rate(flow_rate_m3_s: float, diameter_m: float) -> float:
    """Calculate mean velocity from volumetric flow rate and pipe diameter."""
    area = np.pi * (diameter_m / 2) ** 2
    return flow_rate_m3_s / area


def darcy_weisbach(
    length_m: float,
    diameter_m: float,
    density: float,
    viscosity: float,
    roughness_m: float,
    flow_rate_m3_s: float = None,
    velocity_m_s: float = None,
    elevation_change_m: float = 0.0,
    friction_method: str = "colebrook",
) -> PressureDropResult:
    """
    Calculate steady-state pressure drop for liquid flow in a pipe segment.

    Provide either flow_rate_m3_s OR velocity_m_s (not both).

    Args:
        length_m: Pipe length (m)
        diameter_m: Internal diameter (m)
        density: Fluid density (kg/m³)
        viscosity: Dynamic viscosity (Pa·s)
        roughness_m: Absolute pipe roughness (m); commercial steel ~ 4.6e-5
        flow_rate_m3_s: Volumetric flow rate (m³/s)
        velocity_m_s: Mean velocity (m/s); used if flow_rate not provided
        elevation_change_m: Outlet elevation minus inlet elevation (m)
        friction_method: 'colebrook' or 'swamee_jain'

    Returns:
        PressureDropResult with detailed hydraulic information
    """
    if flow_rate_m3_s is None and velocity_m_s is None:
        raise ValueError("Must provide either flow_rate_m3_s or velocity_m_s")
    if flow_rate_m3_s is not None and velocity_m_s is not None:
        raise ValueError("Provide only one of flow_rate_m3_s or velocity_m_s")

    if flow_rate_m3_s is not None:
        v = velocity_from_flow_rate(flow_rate_m3_s, diameter_m)
    else:
        v = velocity_m_s

    re = reynolds_number(density, v, diameter_m, viscosity)
    f = friction_factor(re, roughness_m, diameter_m, method=friction_method)

    # Friction pressure drop
    dp_friction = f * (length_m / diameter_m) * (density * v ** 2 / 2)
    # Elevation pressure change
    dp_elevation = density * GRAVITY * elevation_change_m

    dp_total = dp_friction + dp_elevation
    # Equivalent head loss in m of fluid
    h_loss = dp_total / (density * GRAVITY)

    # Classify flow regime
    from pipeline_hydraulics.hydraulics.friction import RE_LAMINAR_MAX, RE_TURBULENT_MIN
    if re < RE_LAMINAR_MAX:
        regime = "laminar"
    elif re < RE_TURBULENT_MIN:
        regime = "transitional"
    else:
        regime = "turbulent"

    return PressureDropResult(
        flow_regime=regime,
        reynolds=re,
        friction_factor=f,
        velocity_m_s=v,
        dp_friction_pa=dp_friction,
        dp_elevation_pa=dp_elevation,
        dp_total_pa=dp_total,
        hydraulic_grade_line_drop_m=h_loss,
    )
