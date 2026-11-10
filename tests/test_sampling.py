import numpy as np
import pytest

from signal_lab import fourier, sampling


@pytest.mark.parametrize(
    "f,fs,expected",
    [(100, 1000, 100), (400, 1000, 400), (500, 1000, 500), (600, 1000, 400),
     (900, 1000, 100), (1000, 1000, 0), (1100, 1000, 100), (2350, 1000, 350)],
)
def test_alias_frequency_folds_into_0_to_nyquist(f, fs, expected):
    assert sampling.alias_frequency(f, fs) == pytest.approx(expected)


def test_alias_frequency_vectorised():
    np.testing.assert_allclose(sampling.alias_frequency(np.array([900, 1900]), 1000), [100, 100])


def test_nyquist_rate_and_check():
    assert sampling.nyquist_rate(440.0) == 880.0
    assert sampling.is_aliased(440.0, 800.0)
    assert not sampling.is_aliased(440.0, 1000.0)


def test_undersampled_tone_samples_equal_alias_tone():
    # sin(2π·900·n/1000) = −sin(2π·100·n/1000): the samples are identical
    _, x900 = sampling.sample(900.0, 1000.0, duration=0.1)
    _, x100 = sampling.sample(100.0, 1000.0, duration=0.1)
    np.testing.assert_allclose(x900, -x100, atol=1e-9)


def test_fft_of_undersampled_tone_peaks_at_alias():
    fs = 1000.0
    _, x = sampling.sample(900.0, fs, duration=1.0)
    f, mag = fourier.spectrum(x, fs, window="hann", db=False)
    assert f[np.argmax(mag)] == pytest.approx(100.0)


def test_sinc_reconstruction_recovers_band_limited_tone():
    fs, f = 1000.0, 120.0
    tn, xn = sampling.sample(f, fs, duration=1.0)
    t = np.linspace(0.4, 0.6, 501)  # far from the record edges
    y = sampling.sinc_reconstruct(t, tn, xn, fs)
    np.testing.assert_allclose(y, np.sin(2 * np.pi * f * t), atol=5e-3)


def test_sinc_reconstruction_of_aliased_tone_gives_alias():
    fs = 1000.0
    tn, xn = sampling.sample(900.0, fs, duration=1.0)
    t = np.linspace(0.4, 0.6, 501)
    y = sampling.sinc_reconstruct(t, tn, xn, fs)
    np.testing.assert_allclose(y, -np.sin(2 * np.pi * 100.0 * t), atol=5e-3)
