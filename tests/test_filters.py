import numpy as np
import pytest

from signal_lab import filters

FS = 8000.0


def gain_db(sos, f):
    f_axis, h = filters.frequency_response(sos, FS, f=np.atleast_1d(f))
    return 20 * np.log10(np.abs(h))


@pytest.mark.parametrize("order", [2, 4, 7])
def test_butterworth_lowpass_matches_bilinear_analytic_magnitude(order):
    fc = 500.0
    sos = filters.design("butter", "low", order, fc, FS)
    f = np.array([50.0, 250.0, 500.0, 1000.0, 2000.0])
    # bilinear-transformed Butterworth: |H|² = 1 / (1 + (tan(πf/fs)/tan(πfc/fs))^(2N))
    ratio = np.tan(np.pi * f / FS) / np.tan(np.pi * fc / FS)
    analytic = -10 * np.log10(1 + ratio ** (2 * order))
    np.testing.assert_allclose(gain_db(sos, f), analytic, atol=1e-6)
    assert gain_db(sos, fc)[0] == pytest.approx(-3.0103, abs=1e-4)


def test_butterworth_highpass_corner_and_stopband():
    sos = filters.design("butter", "high", 4, 300.0, FS)
    assert gain_db(sos, 300.0)[0] == pytest.approx(-3.0103, abs=1e-4)
    assert gain_db(sos, 30.0)[0] < -75  # 4th order: ~-80 dB one decade down


def test_bandpass_edges_and_centre():
    sos = filters.design("butter", "band", 3, (400.0, 1600.0), FS)
    lo, hi = gain_db(sos, [400.0, 1600.0])
    assert lo == pytest.approx(-3.0103, abs=1e-4)
    assert hi == pytest.approx(-3.0103, abs=1e-4)
    assert gain_db(sos, 800.0)[0] == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("btype,edge", [("low", 1000.0), ("high", 1000.0)])
def test_chebyshev_ripple_and_edge(btype, edge):
    rp = 1.0
    sos = filters.design("cheby1", btype, 5, edge, FS, ripple_db=rp)
    # Chebyshev I: gain at the band edge equals -ripple
    assert gain_db(sos, edge)[0] == pytest.approx(-rp, abs=1e-4)
    passband = np.linspace(10, 990, 2000) if btype == "low" else np.linspace(1010, 3990, 2000)
    g = gain_db(sos, passband)
    assert g.max() == pytest.approx(0.0, abs=1e-3)
    assert g.min() == pytest.approx(-rp, abs=1e-3)  # equiripple between 0 and -rp


def test_chebyshev_is_steeper_than_butterworth_same_order():
    b = filters.design("butter", "low", 5, 1000.0, FS)
    c = filters.design("cheby1", "low", 5, 1000.0, FS, ripple_db=1.0)
    assert gain_db(c, 2000.0)[0] < gain_db(b, 2000.0)[0] - 10


def test_apply_lowpass_recovers_clean_tone():
    t = np.arange(int(FS)) / FS
    clean = np.sin(2 * np.pi * 50 * t)
    noisy = clean + 0.5 * np.sin(2 * np.pi * 1500 * t)
    sos = filters.design("butter", "low", 6, 200.0, FS)
    y = filters.apply(sos, noisy)  # zero-phase -> no delay to compensate
    core = slice(400, -400)  # ignore edge transients
    err = np.sqrt(np.mean((y[core] - clean[core]) ** 2))
    assert err < 1e-3


def test_apply_causal_has_group_delay_zero_phase_does_not():
    t = np.arange(int(FS)) / FS
    x = np.sin(2 * np.pi * 50 * t)
    sos = filters.design("butter", "low", 4, 200.0, FS)
    lag = lambda y: np.argmax(np.correlate(y[2000:6000], x[2000:6000], "full")) - 3999
    assert lag(filters.apply(sos, x, zero_phase=True)) == 0
    assert lag(filters.apply(sos, x, zero_phase=False)) > 0


def test_invalid_arguments():
    with pytest.raises(ValueError):
        filters.design("elliptic", "low", 4, 100.0, FS)
    with pytest.raises(ValueError):
        filters.design("butter", "low", 4, 5000.0, FS)  # above Nyquist
    with pytest.raises(ValueError):
        filters.design("butter", "band", 4, 100.0, FS)  # band needs two edges
