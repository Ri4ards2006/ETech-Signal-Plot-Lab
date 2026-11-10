import numpy as np
import pytest
from scipy.special import sici

from signal_lab import fourier


FS = 1000.0
N = 1000


def tone(f, amp=1.0, n=N, fs=FS):
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * f * t)


def test_frequency_axis_resolution_and_nyquist():
    freqs, _ = fourier.spectrum(tone(50), FS, window="rect", db=False)
    assert len(freqs) == N // 2 + 1
    assert freqs[1] - freqs[0] == pytest.approx(FS / N)
    assert freqs[-1] == pytest.approx(FS / 2)


@pytest.mark.parametrize("window", ["rect", "hann", "blackman"])
def test_peak_bin_and_amplitude_for_bin_centred_tone(window):
    freqs, mag = fourier.spectrum(tone(50, amp=2.0), FS, window=window, db=False)
    k = np.argmax(mag)
    assert k == 50  # 50 Hz / (fs/N = 1 Hz)
    assert freqs[k] == pytest.approx(50.0)
    # coherent-gain correction -> amplitude is recovered for every window
    assert mag[k] == pytest.approx(2.0, rel=1e-9)


def test_dc_is_not_doubled():
    _, mag = fourier.spectrum(np.full(N, 0.7), FS, window="rect", db=False)
    assert mag[0] == pytest.approx(0.7)


def test_db_scale():
    _, db1 = fourier.spectrum(tone(50, amp=1.0), FS, window="rect", db=True)
    _, db01 = fourier.spectrum(tone(50, amp=0.1), FS, window="rect", db=True)
    assert db1[50] == pytest.approx(0.0, abs=1e-9)
    assert db01[50] == pytest.approx(-20.0, abs=1e-9)
    assert np.all(np.isfinite(db1))  # no -inf for empty bins


def test_window_choice_controls_leakage():
    x = tone(50.5)  # off-bin tone -> leakage
    _, rect = fourier.spectrum(x, FS, window="rect")
    _, hann = fourier.spectrum(x, FS, window="hann")
    _, black = fourier.spectrum(x, FS, window="blackman")
    near = 60  # ~10 bins away from the tone
    # rect leaks ~ -30 dB here, Hann ~ -68 dB, Blackman ~ -76 dB
    assert rect[near] > -35
    assert rect[near] - hann[near] > 30
    assert hann[near] - black[near] > 5


def test_unknown_window_rejected():
    with pytest.raises(ValueError):
        fourier.spectrum(tone(50), FS, window="kaiser-ish")


def test_square_series_single_harmonic_is_4_over_pi_sine():
    t = np.linspace(0, 1, 1001)
    y = fourier.square_wave_series(t, f0=1.0, n_harmonics=1)
    np.testing.assert_allclose(y, 4 / np.pi * np.sin(2 * np.pi * t), atol=1e-12)


def test_square_series_converges_away_from_jumps():
    t = np.array([0.25, 0.75])  # middle of the high / low half-period
    y = fourier.square_wave_series(t, f0=1.0, n_harmonics=2000)
    np.testing.assert_allclose(y, [1.0, -1.0], atol=1e-3)


def test_gibbs_overshoot_approaches_analytic_constant():
    # limit: overshoot / jump = Si(pi)/pi - 1/2  ~= 0.0895 (~9 %)
    analytic = sici(np.pi)[0] / np.pi - 0.5
    assert analytic == pytest.approx(0.0895, abs=1e-4)
    assert fourier.gibbs_overshoot(200) == pytest.approx(analytic, abs=5e-4)
    # it does NOT vanish with more harmonics
    assert fourier.gibbs_overshoot(10) > 0.085
