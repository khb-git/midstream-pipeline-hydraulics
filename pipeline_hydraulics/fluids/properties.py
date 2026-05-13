"""
Fluid properties for pipeline hydraulics.

Defines fluid classes with density, viscosity, and temperature-dependent
property correlations. Includes representative values for common
midstream fluids.

API gravity convention:
    API = (141.5 / SG_60F) - 131.5
    SG_60F = 141.5 / (API + 131.5)
Where SG_60F is specific gravity at 60°F relative to water at 60°F.
"""

from dataclasses import dataclass, field
import numpy as np


# Reference temperature for fluid properties (60°F = 288.71 K)
T_REF_K = 288.71


@dataclass
class Fluid:
    """
    Liquid fluid properties for pipeline hydraulics.

    Attributes:
        name: Descriptive name
        density_kg_m3: Density at reference temperature
        viscosity_pa_s: Dynamic viscosity at reference temperature
        api_gravity: API gravity (computed for liquids; None for water)
        bubble_point_pa: Optional bubble point pressure (Pa)
    """
    name: str
    density_kg_m3: float
    viscosity_pa_s: float
    api_gravity: float = None
    bubble_point_pa: float = None

    @property
    def specific_gravity(self) -> float:
        """Specific gravity at reference temperature (water = 1000 kg/m³)."""
        return self.density_kg_m3 / 1000.0

    def density_at(self, temperature_k: float,
                   thermal_expansion: float = 7e-4) -> float:
        """
        Estimate density at a given temperature using linear thermal expansion.

        Default expansion coefficient (7e-4 /K) is typical for crude oils.
        """
        return self.density_kg_m3 * (1 - thermal_expansion * (temperature_k - T_REF_K))


def api_to_density(api_gravity: float) -> float:
    """Convert API gravity to density (kg/m³) at 60°F."""
    sg = 141.5 / (api_gravity + 131.5)
    return sg * 999.0  # Water density at 60°F


def density_to_api(density_kg_m3: float) -> float:
    """Convert density (kg/m³ at 60°F) to API gravity."""
    sg = density_kg_m3 / 999.0
    return 141.5 / sg - 131.5


# Representative fluid library — values from API/GPSA references
WATER = Fluid(
    name="Water (fresh)",
    density_kg_m3=999.0,
    viscosity_pa_s=1.0e-3,
)

WTI_CRUDE = Fluid(
    name="West Texas Intermediate (39.6° API)",
    density_kg_m3=826.7,
    viscosity_pa_s=4.5e-3,   # ~4.5 cP at 60°F
    api_gravity=39.6,
)

BRENT_CRUDE = Fluid(
    name="Brent (38° API)",
    density_kg_m3=834.5,
    viscosity_pa_s=5.0e-3,
    api_gravity=38.0,
)

WCS_HEAVY = Fluid(
    name="Western Canadian Select (20.5° API)",
    density_kg_m3=929.0,
    viscosity_pa_s=350e-3,   # Very viscous — often diluted for transport
    api_gravity=20.5,
)

DILBIT = Fluid(
    name="Dilbit (diluted bitumen, ~21° API)",
    density_kg_m3=925.0,
    viscosity_pa_s=200e-3,
    api_gravity=21.0,
)

GASOLINE = Fluid(
    name="Gasoline (regular)",
    density_kg_m3=745.0,
    viscosity_pa_s=0.6e-3,
    api_gravity=58.7,
)

DIESEL = Fluid(
    name="Diesel #2",
    density_kg_m3=850.0,
    viscosity_pa_s=2.5e-3,
    api_gravity=34.0,
)


# Pipe roughness (m) for common materials
PIPE_ROUGHNESS = {
    "commercial_steel": 4.6e-5,
    "drawn_tubing": 1.5e-6,
    "cast_iron": 2.6e-4,
    "concrete": 3.0e-4,
    "internally_coated_steel": 2.0e-5,   # FBE coating
    "stainless_steel": 1.5e-5,
}
