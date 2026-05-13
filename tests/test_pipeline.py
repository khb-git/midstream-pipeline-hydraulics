"""
Tests for the pipeline network model and hydraulic profile calculation.

Strategy: build small test pipelines with known geometry and verify that
profile calculations agree with manual single-segment Darcy-Weisbach
calculations. Also verify that pump stations correctly restore pressure.
"""

import numpy as np
import pytest

from pipeline_hydraulics.fluids.properties import WTI_CRUDE, WATER
from pipeline_hydraulics.network.pipeline import Pipeline, PipeSegment, PumpStation
from pipeline_hydraulics.network.profile import compute_profile
from pipeline_hydraulics.hydraulics.pressure_drop import darcy_weisbach


class TestPipelineConstruction:
    def test_total_length_aggregates(self):
        pipe = Pipeline(name="Test", fluid=WATER)
        pipe.add_segment(PipeSegment("s1", length_m=100, diameter_m=0.1))
        pipe.add_segment(PipeSegment("s2", length_m=200, diameter_m=0.1))
        pipe.add_segment(PipeSegment("s3", length_m=300, diameter_m=0.1))
        assert pipe.total_length_m == 600

    def test_elevation_change_aggregates(self):
        pipe = Pipeline(name="Test", fluid=WATER)
        pipe.add_segment(PipeSegment("s1", length_m=100, diameter_m=0.1,
                                     inlet_elevation_m=0, outlet_elevation_m=50))
        pipe.add_segment(PipeSegment("s2", length_m=100, diameter_m=0.1,
                                     inlet_elevation_m=50, outlet_elevation_m=20))
        assert pipe.total_elevation_change_m == 20  # 0 to 20


class TestProfileSingleSegment:
    """Verify multi-step profile calc matches a single Darcy-Weisbach calc."""

    def test_pressure_drop_matches_single_dw(self):
        # Build a single-segment pipeline
        pipe = Pipeline(name="Single", fluid=WATER, inlet_pressure_pa=1e7)
        pipe.add_segment(PipeSegment(
            name="only_segment",
            length_m=10_000,
            diameter_m=0.3,
            inlet_elevation_m=100,
            outlet_elevation_m=50,
            roughness_m=4.6e-5,
        ))

        flow_rate = 0.05  # m³/s
        profile = compute_profile(pipe, flow_rate_m3_s=flow_rate, points_per_segment=100)

        # Single-shot Darcy-Weisbach for the whole segment
        single = darcy_weisbach(
            length_m=10_000,
            diameter_m=0.3,
            density=WATER.density_kg_m3,
            viscosity=WATER.viscosity_pa_s,
            roughness_m=4.6e-5,
            flow_rate_m3_s=flow_rate,
            elevation_change_m=-50,
        )

        actual_drop = pipe.inlet_pressure_pa - profile.outlet_pressure_pa
        # Should match within 0.1% (discretization error)
        assert np.isclose(actual_drop, single.dp_total_pa, rtol=0.001)


class TestPumpStations:
    def test_station_boost_increases_pressure(self):
        # Pipeline with two segments and a station between them
        pipe = Pipeline(name="WithStation", fluid=WATER, inlet_pressure_pa=5e6)
        pipe.add_segment(PipeSegment("s1", length_m=50_000, diameter_m=0.3))
        pipe.add_segment(PipeSegment("s2", length_m=50_000, diameter_m=0.3))
        pipe.add_station(1, PumpStation(
            name="mid_station",
            max_discharge_head_m=500,
            max_discharge_pressure_pa=1e7,
        ))

        with_station = compute_profile(pipe, flow_rate_m3_s=0.05)

        # Same pipeline without the station
        pipe_no_station = Pipeline(name="NoStation", fluid=WATER, inlet_pressure_pa=5e6)
        pipe_no_station.add_segment(PipeSegment("s1", length_m=50_000, diameter_m=0.3))
        pipe_no_station.add_segment(PipeSegment("s2", length_m=50_000, diameter_m=0.3))
        without_station = compute_profile(pipe_no_station, flow_rate_m3_s=0.05)

        # Outlet pressure should be higher with the station
        assert with_station.outlet_pressure_pa > without_station.outlet_pressure_pa

    def test_station_power_calculation(self):
        station = PumpStation(
            name="test", max_discharge_head_m=500, efficiency=0.75,
        )
        # Hydraulic: ρgQH = 1000 * 9.80665 * 0.1 * 500 = 490 kW
        # Shaft: 490 / 0.75 = 654 kW
        power = station.power_required_kw(0.1, 500, 1000)
        assert np.isclose(power, 1000 * 9.80665 * 0.1 * 500 / 0.75 / 1000)


class TestProfileViolations:
    def test_excessive_inlet_pressure_flags_maop(self):
        # Set inlet pressure above MAOP — the origin point itself violates
        pipe = Pipeline(name="overpressure", fluid=WATER, inlet_pressure_pa=2e7)
        pipe.add_segment(PipeSegment(
            "s1", length_m=10_000, diameter_m=0.3, maop_pa=1.5e7,
        ))
        # Some early points in the segment will exceed MAOP
        profile = compute_profile(pipe, flow_rate_m3_s=0.05)
        assert len(profile.maop_violations) > 0
        assert not profile.is_feasible


class TestExamples:
    def test_cushing_houston_example_runs(self):
        from pipeline_hydraulics.io.examples import cushing_to_houston_example
        pipe = cushing_to_houston_example()
        profile = compute_profile(pipe, flow_rate_m3_s=0.5)  # ~270k bpd
        # Should produce a valid profile
        assert len(profile.points) > 100
        assert profile.outlet_pressure_pa > 0

    def test_short_products_line_runs(self):
        from pipeline_hydraulics.io.examples import short_products_line
        pipe = short_products_line()
        profile = compute_profile(pipe, flow_rate_m3_s=0.05)
        assert profile.outlet_pressure_pa > 0
