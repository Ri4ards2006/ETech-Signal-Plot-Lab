import numpy as np
import pytest

from signal_lab import bode

R, C, L = 1e3, 100e-9, 10e-3  # 1 kΩ, 100 nF, 10 mH


def test_rc_lowpass_minus_3db_is_1_over_2pi_rc():
    tf = bode.rc_lowpass(R, C)
    (fc,) = bode.minus_3db_frequencies(tf)
    assert fc == pytest.approx(1 / (2 * np.pi * R * C), rel=1e-6)
    mag, phase = bode.response(tf, [fc])
    assert mag[0] == pytest.approx(-10 * np.log10(2), abs=1e-9)  # -3.01 dB
    assert phase[0] == pytest.approx(-45.0, abs=1e-6)


def test_rc_lowpass_rolloff_20db_per_decade():
    tf = bode.rc_lowpass(R, C)
    fc = 1 / (2 * np.pi * R * C)
    mag, phase = bode.response(tf, [100 * fc, 1000 * fc])
    assert mag[0] - mag[1] == pytest.approx(20.0, abs=0.01)
    assert phase[1] == pytest.approx(-90.0, abs=0.1)


def test_rc_highpass():
    tf = bode.rc_highpass(R, C)
    (fc,) = bode.minus_3db_frequencies(tf)
    assert fc == pytest.approx(1 / (2 * np.pi * R * C), rel=1e-6)
    _, phase = bode.response(tf, [fc])
    assert phase[0] == pytest.approx(45.0, abs=1e-6)


@pytest.mark.parametrize("make", [bode.rl_lowpass, bode.rl_highpass])
def test_rl_corner_is_r_over_2pi_l(make):
    (fc,) = bode.minus_3db_frequencies(make(R, L))
    assert fc == pytest.approx(R / (2 * np.pi * L), rel=1e-6)


def test_rlc_bandpass_centre_and_bandwidth():
    tf = bode.rlc_bandpass(R, L, C)
    f0 = 1 / (2 * np.pi * np.sqrt(L * C))
    mag, phase = bode.response(tf, [f0])
    assert mag[0] == pytest.approx(0.0, abs=1e-9)  # unity gain at resonance
    assert phase[0] == pytest.approx(0.0, abs=1e-6)
    f_lo, f_hi = bode.minus_3db_frequencies(tf)
    assert f_hi - f_lo == pytest.approx(R / (2 * np.pi * L), rel=1e-6)  # B = R/(2πL)
    assert np.sqrt(f_lo * f_hi) == pytest.approx(f0, rel=1e-6)  # geometric centre


def test_rlc_lowpass_resonance_and_q():
    r = 50.0
    tf = bode.rlc_lowpass(r, L, C)
    f0 = 1 / (2 * np.pi * np.sqrt(L * C))
    q = np.sqrt(L / C) / r
    mag, phase = bode.response(tf, [f0])
    assert mag[0] == pytest.approx(20 * np.log10(q), abs=1e-9)  # |H(f0)| = Q
    assert phase[0] == pytest.approx(-90.0, abs=1e-6)


def test_invalid_components_rejected():
    with pytest.raises(ValueError):
        bode.rc_lowpass(-1.0, C)


def test_rlc_lowpass_cutoff_is_referenced_to_dc_gain():
    f0 = 1 / (2 * np.pi * np.sqrt(L * C))
    # zeta = 1/sqrt(2) (Butterworth): -3 dB exactly at f0
    r_bw = np.sqrt(2) * np.sqrt(L / C)
    assert bode.minus_3db_frequencies(bode.rlc_lowpass(r_bw, L, C)) == [pytest.approx(f0, rel=1e-6)]
    # high Q: one cutoff above the resonance peak, not two points around it
    (fc,) = bode.minus_3db_frequencies(bode.rlc_lowpass(50.0, L, C))
    assert fc > f0
    mag, _ = bode.response(bode.rlc_lowpass(50.0, L, C), [fc])
    assert mag[0] == pytest.approx(-10 * np.log10(2), abs=1e-9)
