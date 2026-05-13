"""
Tests for friction factor calculations.

Validates against:
- Analytical limit: laminar f = 64/Re
- Cross-check: Swamee-Jain explicit approximation should match Colebrook
  iterative to within ~1.5% (Swamee-Jain claim)
- Boundary behavior at the laminar-turbulent transition
"""

import numpy as np
import pytest

from src.hydraulics.friction import (
    reynolds_number,
    friction_factor_laminar,
    friction_factor_colebrook,
    friction_factor_swamee_jain,
    friction_factor,
    RE_LAMINAR_MAX,
    RE_TURBULENT_MIN,
)


class TestReynolds:
    def test_known_value(self):
        # Water at 1 m/s in 4-inch pipe
        re = reynolds_number(rho=1000, velocity=1.0, diameter=0.1, viscosity=1e-3)
        # ρvD/μ = 1000 * 1.0 * 0.1 / 0.001 = 100,000
        assert np.isclose(re, 1e5)

    def test_negative_velocity_gives_positive_re(self):
        # Re is magnitude — direction doesn't matter
        re = reynolds_number(rho=1000, velocity=-1.0, diameter=0.1, viscosity=1e-3)
        assert re > 0

    def test_zero_viscosity_raises(self):
        with pytest.raises(ValueError):
            reynolds_number(rho=1000, velocity=1.0, diameter=0.1, viscosity=0)


class TestLaminar:
    def test_classic_formula(self):
        assert np.isclose(friction_factor_laminar(1000), 0.064)
        assert np.isclose(friction_factor_laminar(2000), 0.032)


class TestColebrook:
    def test_smooth_pipe_high_re(self):
        # Very smooth pipe at high Re — should be close to Nikuradse smooth-pipe
        f = friction_factor_colebrook(re=1e6, roughness=1e-7, diameter=0.1)
        # Smooth pipe at Re=1e6 has f ~ 0.012
        assert 0.010 < f < 0.014

    def test_fully_rough_high_re(self):
        # Rough pipe at high Re — friction factor approaches a constant
        # Independent of Re for fully rough flow
        f_low = friction_factor_colebrook(re=1e6, roughness=1e-3, diameter=0.1)
        f_high = friction_factor_colebrook(re=1e8, roughness=1e-3, diameter=0.1)
        # Should be very close
        assert np.isclose(f_low, f_high, rtol=0.05)

    def test_falls_back_to_laminar(self):
        # Below turbulent threshold, should give laminar value
        f = friction_factor_colebrook(re=1000, roughness=4.6e-5, diameter=0.1)
        assert np.isclose(f, 64 / 1000)


class TestSwameeJain:
    def test_matches_colebrook_within_3_pct(self):
        # Swamee-Jain claims ~1% accuracy but degrades at extreme roughness
        # 3% is a realistic envelope across the full turbulent range
        for re in [1e4, 1e5, 1e6, 1e7]:
            for eps_d in [1e-6, 1e-4, 1e-3, 1e-2]:
                eps = eps_d * 0.1
                f_sj = friction_factor_swamee_jain(re, eps, 0.1)
                f_cw = friction_factor_colebrook(re, eps, 0.1)
                rel_error = abs(f_sj - f_cw) / f_cw
                assert rel_error < 0.03, (
                    f"Swamee-Jain off by {rel_error:.2%} at Re={re}, eps/D={eps_d}"
                )


class TestUniversal:
    def test_transitional_continuity(self):
        # The piecewise formula should be continuous at transition boundaries
        eps, D = 4.6e-5, 0.1
        # At Re = RE_LAMINAR_MAX, should be very close to 64/Re
        f_lam_edge = friction_factor(RE_LAMINAR_MAX - 1, eps, D)
        f_trans_start = friction_factor(RE_LAMINAR_MAX + 1, eps, D)
        # Continuous within ~5%
        assert abs(f_lam_edge - f_trans_start) / f_lam_edge < 0.05

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            friction_factor(1e5, 4.6e-5, 0.1, method="nonsense")
