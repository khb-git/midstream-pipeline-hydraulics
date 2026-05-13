"""
Pipeline as a sequence of segments with pump stations.

A real pipeline is modeled as a chain of pipe segments separated by
pump stations. Each segment has length, diameter, roughness, and an
elevation profile. Pump stations restore pressure between segments.

This module provides the geometric and equipment model. Hydraulic
analysis (calculating pressure profile along the chain) is in
network.profile.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from src.fluids.properties import Fluid


@dataclass
class PipeSegment:
    """
    A single pipe segment.

    Attributes:
        name: Identifier (e.g., "Cushing to Tulsa")
        length_m: Length (m)
        diameter_m: Internal diameter (m)
        roughness_m: Absolute roughness (m)
        inlet_elevation_m: Elevation at the inlet (m above datum)
        outlet_elevation_m: Elevation at the outlet
        maop_pa: Maximum allowable operating pressure (Pa)
        n_profile_points: How many intermediate points to use when
            walking the elevation profile (for plotting and pinch analysis)
    """
    name: str
    length_m: float
    diameter_m: float
    inlet_elevation_m: float = 0.0
    outlet_elevation_m: float = 0.0
    roughness_m: float = 4.6e-5
    maop_pa: float = 9.93e6     # ~1440 psi, typical Class 1 location
    n_profile_points: int = 50

    @property
    def elevation_change_m(self) -> float:
        return self.outlet_elevation_m - self.inlet_elevation_m

    @property
    def length_mi(self) -> float:
        return self.length_m / 1609.344

    @property
    def diameter_in(self) -> float:
        return self.diameter_m * 39.3701

    def elevation_profile(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Linear elevation profile between endpoints.

        For more realistic terrain you'd override this with actual survey
        data; this default is sufficient for hydraulics calculations
        that only care about cumulative elevation change.

        Returns:
            (distance_m, elevation_m) arrays
        """
        d = np.linspace(0, self.length_m, self.n_profile_points)
        e = np.linspace(
            self.inlet_elevation_m, self.outlet_elevation_m, self.n_profile_points
        )
        return d, e


@dataclass
class PumpStation:
    """
    A pump station between segments.

    A real pump station has multiple pumps in series/parallel and operates
    on a head-vs-flow characteristic curve. For steady-state planning we
    model it as a fixed head boost subject to NPSH and discharge limits.

    Attributes:
        name: Identifier
        max_discharge_head_m: Maximum head the station can deliver (m of fluid)
        efficiency: Combined pump + driver efficiency (0.7-0.85 typical)
        min_suction_pressure_pa: NPSH requirement at suction (Pa absolute)
        max_discharge_pressure_pa: MAOP-driven discharge limit (Pa)
    """
    name: str
    max_discharge_head_m: float = 600.0
    efficiency: float = 0.78
    min_suction_pressure_pa: float = 1.5e5    # 22 psia absolute
    max_discharge_pressure_pa: float = 9.93e6  # ~1440 psi

    def power_required_kw(self, flow_rate_m3_s: float,
                          head_added_m: float, fluid_density: float) -> float:
        """
        Calculate power required at the pump shaft (kW).

        Hydraulic power = ρ * g * Q * H
        Shaft power = hydraulic power / efficiency
        """
        g = 9.80665
        hydraulic_w = fluid_density * g * flow_rate_m3_s * head_added_m
        shaft_w = hydraulic_w / self.efficiency
        return shaft_w / 1000.0


@dataclass
class Pipeline:
    """
    A complete pipeline: ordered sequence of segments, with pump stations
    located at segment boundaries.

    Pump stations are indexed by the segment they precede; station 0 is
    the origin (always present, provides inlet pressure), station k for
    k>0 sits between segment k-1 and segment k.

    Attributes:
        name: Identifier
        fluid: Fluid being transported
        segments: Ordered list of PipeSegment
        stations: Dict mapping segment index -> PumpStation. Station 0
            represents the origin.
        inlet_pressure_pa: Pressure at the origin (head of pipeline)
    """
    name: str
    fluid: Fluid
    segments: list[PipeSegment] = field(default_factory=list)
    stations: dict = field(default_factory=dict)
    inlet_pressure_pa: float = 6.9e6  # ~1000 psi typical origin pressure

    @property
    def total_length_m(self) -> float:
        return sum(s.length_m for s in self.segments)

    @property
    def total_length_mi(self) -> float:
        return self.total_length_m / 1609.344

    @property
    def total_elevation_change_m(self) -> float:
        return sum(s.elevation_change_m for s in self.segments)

    def add_segment(self, segment: PipeSegment) -> "Pipeline":
        self.segments.append(segment)
        return self

    def add_station(self, before_segment_index: int,
                    station: PumpStation) -> "Pipeline":
        self.stations[before_segment_index] = station
        return self

    def summary(self) -> str:
        lines = [
            f"Pipeline: {self.name}",
            f"Fluid: {self.fluid.name} (ρ={self.fluid.density_kg_m3:.0f} kg/m³, "
            f"μ={self.fluid.viscosity_pa_s*1000:.1f} cP)",
            f"Total length: {self.total_length_mi:,.1f} mi ({self.total_length_m/1000:,.0f} km)",
            f"Total elevation change: {self.total_elevation_change_m:+,.0f} m",
            f"Inlet pressure: {self.inlet_pressure_pa/6894.76:,.0f} psi",
            f"Segments: {len(self.segments)}",
            f"Pump stations: {len(self.stations)}",
        ]
        return "\n".join(lines)
