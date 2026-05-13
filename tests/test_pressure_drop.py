"""
Tests for pressure drop calculations.

Validation strategy: verify the components of the calculation
independently (friction term, elevation term) and check that the
result matches first-principles derivations.

I'm deliberately *not* citing specific numerical values from textbook
examples I haven't independently verified — instead the tests focus
on internal consistency and physical limits.
"""

import numpy as np
import pytest

from pipeline_hydraulics.hydraulics.pressure_drop import (
    darcy_weisbach,
    velocity_from_flow_rate,
    GRAVITY,
)


class TestVelocity:
    def test_velocity_from_flow(self):
        # 0.1 m³/s through 0.1 m diameter pipe
        # Area = π * 0.05² = 0.00785 m²
        # v = 0.1 / 0.00785 = 12.73 m/s
        v = velocity_from_flow_rate(0.1, 0.1)
        assert np.isclose(v, 0.1 / (np.pi * 0.05 ** 2))


class TestElevationTerm:
    def test_pure_elevation_loss(self):
        """No flow, pure hydrostatic elevation effect."""
        # 100 m elevation gain with water density should produce
        # ΔP = ρgh = 1000 * 9.80665 * 100 = 980,665 Pa
        # Use very low flow rate so friction is negligible
        result = darcy_weisbach(
            length_m=1.0,             # Tiny segment
            diameter_m=1.0,            # Very large pipe so v is small
            density=1000,
            viscosity=1e-3,
            roughness_m=4.6e-5,
            flow_rate_m3_s=1e-6,
            elevation_change_m=100,
        )
        expected = 1000 * GRAVITY * 100
        assert np.isclose(result.dp_elevation_pa, expected)
        # Friction should be negligible
        assert result.dp_friction_pa < expected * 0.001


class TestFrictionTerm:
    def test_zero_elevation_friction_only(self):
        """Horizontal segment: total ΔP equals friction ΔP."""
        result = darcy_weisbach(
            length_m=1000,
            diameter_m=0.1,
            density=1000,
            viscosity=1e-3,
            roughness_m=4.6e-5,
            flow_rate_m3_s=0.01,
            elevation_change_m=0,
        )
        assert result.dp_elevation_pa == 0
        assert result.dp_total_pa == result.dp_friction_pa
        assert result.dp_friction_pa > 0

    def test_friction_scales_with_length(self):
        kw = dict(diameter_m=0.1, density=1000, viscosity=1e-3,
                  roughness_m=4.6e-5, flow_rate_m3_s=0.01,
                  elevation_change_m=0)
        r1 = darcy_weisbach(length_m=1000, **kw)
        r2 = darcy_weisbach(length_m=2000, **kw)
        assert np.isclose(r2.dp_friction_pa / r1.dp_friction_pa, 2.0)

    def test_friction_scales_with_velocity_squared(self):
        kw = dict(length_m=1000, diameter_m=0.1, density=1000, viscosity=1e-3,
                  roughness_m=4.6e-5, elevation_change_m=0)
        r1 = darcy_weisbach(velocity_m_s=1.0, **kw)
        r2 = darcy_weisbach(velocity_m_s=2.0, **kw)
        # f changes slightly between Re=1e5 and Re=2e5, so won't be exactly 4x
        ratio = r2.dp_friction_pa / r1.dp_friction_pa
        assert 3.6 < ratio < 4.0  # Roughly v² with some friction-factor variation


class TestFlowRegimeClassification:
    def test_laminar_classified(self):
        # Heavy oil at slow speed -> laminar
        result = darcy_weisbach(
            length_m=100, diameter_m=0.1, density=900, viscosity=0.5,
            roughness_m=4.6e-5, velocity_m_s=0.01,
        )
        assert result.flow_regime == "laminar"

    def test_turbulent_classified(self):
        # Water at typical speed -> turbulent
        result = darcy_weisbach(
            length_m=100, diameter_m=0.1, density=1000, viscosity=1e-3,
            roughness_m=4.6e-5, velocity_m_s=2.0,
        )
        assert result.flow_regime == "turbulent"


class TestInputValidation:
    def test_must_provide_flow_or_velocity(self):
        with pytest.raises(ValueError):
            darcy_weisbach(
                length_m=100, diameter_m=0.1, density=1000,
                viscosity=1e-3, roughness_m=4.6e-5,
            )

    def test_cannot_provide_both(self):
        with pytest.raises(ValueError):
            darcy_weisbach(
                length_m=100, diameter_m=0.1, density=1000,
                viscosity=1e-3, roughness_m=4.6e-5,
                flow_rate_m3_s=0.01, velocity_m_s=1.0,
            )
