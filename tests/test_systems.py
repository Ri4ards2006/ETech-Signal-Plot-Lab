import numpy as np
import pytest

from signal_lab import systems

T = np.linspace(0, 10, 5001)


def test_first_order_step_reaches_63_percent_at_tau():
    K, tau = 2.0, 0.5
    t, y = systems.step(systems.first_order(K, tau), np.linspace(0, 5 * tau, 501))
    assert y[0] == pytest.approx(0.0, abs=1e-12)
    assert t[100] == pytest.approx(tau)
    assert y[100] == pytest.approx(K * (1 - np.exp(-1)), rel=1e-9)  # 63.2 %
    assert y[-1] == pytest.approx(K * (1 - np.exp(-5)), rel=1e-9)  # 99.3 %


def test_first_order_numeric_matches_analytic():
    K, tau = 1.5, 0.8
    sys = systems.first_order(K, tau)
    np.testing.assert_allclose(systems.step(sys, T)[1], systems.first_order_step(T, K, tau), atol=1e-9)
    np.testing.assert_allclose(systems.impulse(sys, T)[1], systems.first_order_impulse(T, K, tau), atol=1e-9)


@pytest.mark.parametrize("zeta", [0.0, 0.1, 0.5, 0.707, 1.0, 1.5, 3.0])
def test_second_order_numeric_matches_closed_form(zeta):
    wn, K = 2.0, 1.3
    sys = systems.second_order(zeta, wn, K)
    np.testing.assert_allclose(systems.step(sys, T)[1], systems.second_order_step(T, zeta, wn, K), atol=1e-8)
    np.testing.assert_allclose(systems.impulse(sys, T)[1], systems.second_order_impulse(T, zeta, wn, K), atol=1e-8)


def test_closed_form_is_continuous_across_critical_damping():
    a = systems.second_order_step(T, 1.0 - 1e-7, 2.0)
    b = systems.second_order_step(T, 1.0, 2.0)
    c = systems.second_order_step(T, 1.0 + 1e-7, 2.0)
    np.testing.assert_allclose(a, b, atol=1e-5)
    np.testing.assert_allclose(c, b, atol=1e-5)


@pytest.mark.parametrize("zeta", [0.1, 0.3, 0.5, 0.707])
def test_measured_overshoot_and_peak_time_match_formulas(zeta):
    wn = 5.0
    t = np.linspace(0, 10, 200001)
    m = systems.step_metrics(t, systems.second_order_step(t, zeta, wn), final=1.0)
    assert m["overshoot_pct"] == pytest.approx(systems.overshoot_percent(zeta), abs=1e-3)
    assert m["peak_time"] == pytest.approx(systems.peak_time(zeta, wn), abs=1e-4)


def test_overshoot_formula_reference_values():
    # exp(-pi*zeta/sqrt(1-zeta^2)): textbook values
    assert systems.overshoot_percent(0.5) == pytest.approx(16.303, abs=1e-3)
    assert systems.overshoot_percent(1 / np.sqrt(2)) == pytest.approx(4.321, abs=1e-3)
    assert systems.overshoot_percent(1.0) == 0.0
    assert systems.overshoot_percent(2.0) == 0.0


def test_no_overshoot_for_zeta_ge_1():
    for zeta in (1.0, 2.0):
        m = systems.step_metrics(T, systems.second_order_step(T, zeta, 3.0))
        assert m["overshoot_pct"] == pytest.approx(0.0, abs=1e-9)


def test_settling_time_first_order_is_tau_ln50():
    tau = 0.4
    t = np.linspace(0, 5, 500001)
    m = systems.step_metrics(t, systems.first_order_step(t, 1.0, tau), final=1.0)
    assert m["settling_time"] == pytest.approx(tau * np.log(50), abs=1e-4)  # 2 % band
    # 10-90 % rise time of a first-order system: tau * ln 9
    assert m["rise_time"] == pytest.approx(tau * np.log(9), abs=1e-4)


def test_invalid_parameters():
    with pytest.raises(ValueError):
        systems.first_order(1.0, 0.0)
    with pytest.raises(ValueError):
        systems.second_order(-0.1, 1.0)
