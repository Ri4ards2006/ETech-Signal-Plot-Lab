"""Bode plots of RC, RL and RLC networks from component values.

    uv run python -m examples.bode_demo
    uv run python -m examples.bode_demo --save docs/img
"""

from examples._cli import run
from signal_lab import bode

R, C, L = 1e3, 100e-9, 10e-3  # 1 kΩ, 100 nF, 10 mH


def make_figures():
    return {
        "bode_rc_lowpass.png": bode.plot_bode(bode.rc_lowpass(R, C)),
        "bode_rl_highpass.png": bode.plot_bode(bode.rl_highpass(R, L)),
        "bode_rlc_bandpass.png": bode.plot_bode(bode.rlc_bandpass(R, L, C)),
        "bode_rlc_lowpass.png": bode.plot_bode(bode.rlc_lowpass(50.0, L, C)),
    }


if __name__ == "__main__":
    run(make_figures, __doc__)
