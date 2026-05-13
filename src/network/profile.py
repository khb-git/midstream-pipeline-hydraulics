"""
Hydraulic profile calculation.

Given a pipeline, fluid, and operating conditions, walks the pipeline
from inlet to outlet computing pressure at intermediate points. The
resulting profile is the basis for:
- The hydraulic gradient diagram
- MAOP / minimum pressure constraint checking
- Pump station placement decisions
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

from src.network.pipeline import Pipeline
from src.hydraulics.pressure_drop import darcy_weisbach


@dataclass
class ProfilePoint:
    """A single point in the hydraulic profile."""
    distance_m: float
    elevation_m: float
    pressure_pa: float
    pressure_psi: float
    segment_name: Optional[str] = None
    is_station: bool = False
    station_name: Optional[str] = None


@dataclass
class HydraulicProfile:
    """Complete pressure profile along a pipeline."""
    points: list[ProfilePoint]
    flow_rate_m3_s: float
    maop_violations: list[ProfilePoint] = field(default_factory=list)
    suction_violations: list[ProfilePoint] = field(default_factory=list)
    total_pump_power_kw: float = 0.0
    pipeline_name: str = ""

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "distance_km": p.distance_m / 1000,
                "distance_mi": p.distance_m / 1609.344,
                "elevation_m": p.elevation_m,
                "pressure_pa": p.pressure_pa,
                "pressure_psi": p.pressure_psi,
                "segment": p.segment_name,
                "is_station": p.is_station,
                "station": p.station_name,
            }
            for p in self.points
        ])

    @property
    def is_feasible(self) -> bool:
        """No MAOP violations and no suction-pressure violations."""
        return len(self.maop_violations) == 0 and len(self.suction_violations) == 0

    @property
    def outlet_pressure_pa(self) -> float:
        return self.points[-1].pressure_pa

    @property
    def outlet_pressure_psi(self) -> float:
        return self.points[-1].pressure_psi


def compute_profile(
    pipeline: Pipeline,
    flow_rate_m3_s: float,
    points_per_segment: int = 50,
    friction_method: str = "colebrook",
) -> HydraulicProfile:
    """
    Walk a pipeline calculating pressure at intermediate points.

    Args:
        pipeline: Pipeline definition
        flow_rate_m3_s: Volumetric flow rate (m³/s) — assumed uniform
        points_per_segment: Discretization resolution within each segment
        friction_method: Friction factor method

    Returns:
        HydraulicProfile with pressure at each point, plus violation lists
    """
    points: list[ProfilePoint] = []
    maop_violations: list[ProfilePoint] = []
    suction_violations: list[ProfilePoint] = []
    total_power_kw = 0.0

    current_distance = 0.0
    current_pressure = pipeline.inlet_pressure_pa

    # Origin point
    origin = ProfilePoint(
        distance_m=0.0,
        elevation_m=pipeline.segments[0].inlet_elevation_m if pipeline.segments else 0.0,
        pressure_pa=current_pressure,
        pressure_psi=current_pressure / 6894.76,
        segment_name=None,
        is_station=True,
        station_name="Origin",
    )
    points.append(origin)

    fluid = pipeline.fluid

    for seg_idx, segment in enumerate(pipeline.segments):
        # Apply pump station if one precedes this segment (and isn't the origin)
        if seg_idx > 0 and seg_idx in pipeline.stations:
            station = pipeline.stations[seg_idx]
            # Verify NPSH
            if current_pressure < station.min_suction_pressure_pa:
                suction_violations.append(points[-1])

            # Add head — limited by station max head and MAOP
            head_added_m = station.max_discharge_head_m
            head_added_pa = head_added_m * fluid.density_kg_m3 * 9.80665
            new_pressure = current_pressure + head_added_pa
            new_pressure = min(new_pressure, station.max_discharge_pressure_pa)

            actual_head_added = (new_pressure - current_pressure) / (fluid.density_kg_m3 * 9.80665)
            power = station.power_required_kw(flow_rate_m3_s, actual_head_added, fluid.density_kg_m3)
            total_power_kw += power

            current_pressure = new_pressure
            # Insert a station point at the current location
            station_point = ProfilePoint(
                distance_m=current_distance,
                elevation_m=segment.inlet_elevation_m,
                pressure_pa=current_pressure,
                pressure_psi=current_pressure / 6894.76,
                segment_name=segment.name,
                is_station=True,
                station_name=station.name,
            )
            points.append(station_point)

        # Walk the segment in small steps
        n = points_per_segment
        sub_lengths = np.diff(np.linspace(0, segment.length_m, n + 1))
        elevations = np.linspace(segment.inlet_elevation_m, segment.outlet_elevation_m, n + 1)

        for i, dl in enumerate(sub_lengths):
            dz = elevations[i + 1] - elevations[i]
            dp_result = darcy_weisbach(
                length_m=dl,
                diameter_m=segment.diameter_m,
                density=fluid.density_kg_m3,
                viscosity=fluid.viscosity_pa_s,
                roughness_m=segment.roughness_m,
                flow_rate_m3_s=flow_rate_m3_s,
                elevation_change_m=dz,
                friction_method=friction_method,
            )
            current_pressure -= dp_result.dp_total_pa
            current_distance += dl

            point = ProfilePoint(
                distance_m=current_distance,
                elevation_m=elevations[i + 1],
                pressure_pa=current_pressure,
                pressure_psi=current_pressure / 6894.76,
                segment_name=segment.name,
                is_station=False,
            )
            points.append(point)

            # MAOP check
            if current_pressure > segment.maop_pa:
                maop_violations.append(point)

    return HydraulicProfile(
        points=points,
        flow_rate_m3_s=flow_rate_m3_s,
        maop_violations=maop_violations,
        suction_violations=suction_violations,
        total_pump_power_kw=total_power_kw,
        pipeline_name=pipeline.name,
    )
