"""
Tests for gas pipeline flow equations.

Weymouth and Panhandle B are empirical correlations; we validate by
checking:
- Physical limits: flow rate decreases as outlet pressure approaches inlet
- Scaling: doubling length should reduce flow rate by ~1/sqrt(2)
- Equation comparison: Weymouth and Panhandle B should be the same order
  of magnitude on the same problem
"""

import numpy as np
import pytest

from src.hydraulics.gas_flow import weymouth_flow, panhandle_b_flow


class TestWeymouth:
    def test_inlet_greater_than_outlet_required(self):
        with pytest.raises(ValueError):
            weymouth_flow(
                p_inlet_pa=5e6, p_outlet_pa=6e6,
                length_m=100_000, diameter_m=0.6,
                specific_gravity=0.65,
            )

    def test_flow_increases_with_pressure_differential(self):
        kwargs = dict(
            length_m=100_000, diameter_m=0.6, specific_gravity=0.65,
        )
        r1 = weymouth_flow(p_inlet_pa=8e6, p_outlet_pa=7e6, **kwargs)
        r2 = weymouth_flow(p_inlet_pa=8e6, p_outlet_pa=5e6, **kwargs)
        assert r2.flow_rate_sm3_s > r1.flow_rate_sm3_s

    def test_flow_decreases_with_length(self):
        kwargs = dict(
            p_inlet_pa=8e6, p_outlet_pa=5e6, diameter_m=0.6,
            specific_gravity=0.65,
        )
        short = weymouth_flow(length_m=50_000, **kwargs)
        long_pipe = weymouth_flow(length_m=200_000, **kwargs)
        # Length appears under sqrt, so 4x length ~ 1/2 flow
        ratio = long_pipe.flow_rate_sm3_s / short.flow_rate_sm3_s
        assert np.isclose(ratio, 0.5, rtol=0.05)

    def test_flow_scales_with_diameter_to_8_3(self):
        kwargs = dict(
            p_inlet_pa=8e6, p_outlet_pa=5e6, length_m=100_000,
            specific_gravity=0.65,
        )
        r_small = weymouth_flow(diameter_m=0.3, **kwargs)
        r_large = weymouth_flow(diameter_m=0.6, **kwargs)
        # Q scales as D^(8/3); doubling diameter -> 2^(8/3) = 6.35x flow
        ratio = r_large.flow_rate_sm3_s / r_small.flow_rate_sm3_s
        assert np.isclose(ratio, 2 ** (8/3), rtol=0.001)


class TestPanhandleB:
    def test_flow_increases_with_pressure_differential(self):
        kwargs = dict(
            length_m=100_000, diameter_m=0.9, specific_gravity=0.65,
        )
        r1 = panhandle_b_flow(p_inlet_pa=8e6, p_outlet_pa=7e6, **kwargs)
        r2 = panhandle_b_flow(p_inlet_pa=8e6, p_outlet_pa=5e6, **kwargs)
        assert r2.flow_rate_sm3_s > r1.flow_rate_sm3_s

    def test_flow_scales_with_diameter_to_2_53(self):
        kwargs = dict(
            p_inlet_pa=8e6, p_outlet_pa=5e6, length_m=100_000,
            specific_gravity=0.65,
        )
        r_small = panhandle_b_flow(diameter_m=0.6, **kwargs)
        r_large = panhandle_b_flow(diameter_m=1.2, **kwargs)
        ratio = r_large.flow_rate_sm3_s / r_small.flow_rate_sm3_s
        assert np.isclose(ratio, 2 ** 2.53, rtol=0.001)


class TestEquationComparison:
    def test_same_order_of_magnitude(self):
        """For a typical transmission case, both equations should give
        results within a factor of ~2 of each other."""
        kwargs = dict(
            p_inlet_pa=7e6, p_outlet_pa=5e6,
            length_m=100_000, diameter_m=0.9,
            specific_gravity=0.65,
        )
        wm = weymouth_flow(**kwargs)
        pb = panhandle_b_flow(**kwargs)
        # Both should be in the same ballpark
        assert 0.4 < wm.flow_rate_sm3_s / pb.flow_rate_sm3_s < 2.5


class TestElevation:
    def test_uphill_reduces_flow(self):
        kwargs = dict(
            p_inlet_pa=8e6, p_outlet_pa=6e6,
            length_m=100_000, diameter_m=0.6,
            specific_gravity=0.65,
        )
        flat = weymouth_flow(elevation_change_m=0, **kwargs)
        uphill = weymouth_flow(elevation_change_m=500, **kwargs)
        assert uphill.flow_rate_sm3_s < flat.flow_rate_sm3_s
