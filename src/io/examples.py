"""
Pre-configured example pipelines for testing and demonstration.

These approximate real-world pipelines with publicly known dimensions
but use synthetic elevation profiles for illustration. They're not meant
to be design-accurate replicas of specific operating pipelines.
"""

import numpy as np

from src.fluids.properties import WTI_CRUDE, WCS_HEAVY, GASOLINE
from src.network.pipeline import Pipeline, PipeSegment, PumpStation


def cushing_to_houston_example() -> Pipeline:
    """
    A representative Cushing OK to Houston TX crude pipeline.

    ~500 miles, 30" diameter, mostly downhill (Cushing ~270 m elevation,
    Houston ~10 m). Approximates the topology of pipelines like the
    Seaway twin lines without claiming to be either of them specifically.
    """
    fluid = WTI_CRUDE
    inlet_pressure = 8.5e6  # ~1230 psi origin
    pipeline = Pipeline(
        name="Cushing-to-Houston Example (Light Crude)",
        fluid=fluid,
        inlet_pressure_pa=inlet_pressure,
    )

    # 5 segments of roughly equal length, with synthesized terrain
    segment_definitions = [
        # (name, length_mi, inlet_el_m, outlet_el_m)
        ("Cushing to Paris TX", 150, 270, 170),
        ("Paris to Longview", 110, 170, 110),
        ("Longview to Lufkin", 90, 110, 80),
        ("Lufkin to Conroe", 90, 80, 40),
        ("Conroe to Houston Terminal", 60, 40, 10),
    ]

    diameter_m = 30 * 0.0254   # 30 inch nominal -> approximate ID
    maop_pa = 9.93e6           # ~1440 psi MAOP

    for name, length_mi, e_in, e_out in segment_definitions:
        seg = PipeSegment(
            name=name,
            length_m=length_mi * 1609.344,
            diameter_m=diameter_m,
            inlet_elevation_m=e_in,
            outlet_elevation_m=e_out,
            roughness_m=4.6e-5,
            maop_pa=maop_pa,
        )
        pipeline.add_segment(seg)

    # Pump stations between segments — typical spacing ~100 miles
    # Station 0 = origin (defined by inlet_pressure)
    # Station k (k > 0) sits between segments k-1 and k
    pipeline.add_station(1, PumpStation(
        name="Paris Station", max_discharge_head_m=900,
        efficiency=0.78, max_discharge_pressure_pa=maop_pa,
    ))
    pipeline.add_station(3, PumpStation(
        name="Lufkin Station", max_discharge_head_m=900,
        efficiency=0.78, max_discharge_pressure_pa=maop_pa,
    ))

    return pipeline


def heavy_crude_pipeline() -> Pipeline:
    """
    Heavy crude pipeline — Western Canadian Select.

    Same dimensions as the example above but transporting a much more
    viscous fluid. Useful for demonstrating the dramatic impact of
    viscosity on pipeline hydraulics.
    """
    pipeline = cushing_to_houston_example()
    pipeline.name = "WCS Heavy Crude (Same Route)"
    pipeline.fluid = WCS_HEAVY
    pipeline.inlet_pressure_pa = 9.5e6  # Higher origin pressure
    return pipeline


def short_products_line() -> Pipeline:
    """
    A short refined-products pipeline.

    Lower density (gasoline) and lower viscosity than crude — quite
    different hydraulic behavior. About 100 miles, 12" diameter.
    """
    fluid = GASOLINE
    pipeline = Pipeline(
        name="Refined Products Line — Gasoline",
        fluid=fluid,
        inlet_pressure_pa=7.0e6,
    )

    diameter_m = 12 * 0.0254

    pipeline.add_segment(PipeSegment(
        name="Refinery to Distribution Terminal",
        length_m=100 * 1609.344,
        diameter_m=diameter_m,
        inlet_elevation_m=50,
        outlet_elevation_m=120,    # Net climb of 70 m
        roughness_m=4.6e-5,
        maop_pa=9.93e6,
    ))

    return pipeline
