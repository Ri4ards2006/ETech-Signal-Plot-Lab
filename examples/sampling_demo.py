"""Sampling & aliasing: drag the fs slider and watch the tone fold.

    uv run python -m examples.sampling_demo              # interactive fs slider
    uv run python -m examples.sampling_demo --save docs/img
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from examples._cli import run
from signal_lab import sampling

F_SIGNAL = 100.0  # Hz
T_VIEW = 0.05  # s shown


def _draw(ax, fs):
    # sample 3 view-lengths and show the middle one, so the finite-record
    # error of the sinc interpolation stays out of view
    tn, xn = sampling.sample(F_SIGNAL, fs, 3 * T_VIEW)
    t = np.linspace(T_VIEW, 2 * T_VIEW, 4000)
    view = (tn >= T_VIEW) & (tn <= 2 * T_VIEW)
    fa = sampling.alias_frequency(F_SIGNAL, fs)
    ax.clear()
    ax.plot(t * 1e3, np.sin(2 * np.pi * F_SIGNAL * t), color="0.6", lw=1, label=f"x(t), {F_SIGNAL:g} Hz")
    ax.stem(tn[view] * 1e3, xn[view], linefmt="C0-", markerfmt="C0o", basefmt=" ", label=f"samples, fs = {fs:g} Hz")
    ax.plot(t * 1e3, sampling.sinc_reconstruct(t, tn, xn, fs), "C3", lw=1.4,
            label=f"sinc reconstruction → {fa:g} Hz")
    status = "aliased!" if sampling.is_aliased(F_SIGNAL, fs) else "OK (fs > 2f)"
    ax.set_title(f"fs = {fs:g} Hz, Nyquist = {fs / 2:g} Hz — {status}")
    ax.set_xlim(T_VIEW * 1e3, 2 * T_VIEW * 1e3)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("t [ms]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def _fold_plot(ax):
    f = np.linspace(0, 1000, 2001)
    fs = 400.0
    ax.plot(f, sampling.alias_frequency(f, fs), lw=1.5)
    for k in range(1, 6):
        ax.axvline(k * fs / 2, color="0.8", lw=0.8, ls="--")
    ax.set_title(f"Frequency folding at fs = {fs:g} Hz")
    ax.set_xlabel("true frequency [Hz]")
    ax.set_ylabel("apparent frequency [Hz]")
    ax.grid(alpha=0.3)


def make_figures():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, fs in zip(axes.flat[:3], (1000.0, 250.0, 110.0)):
        _draw(ax, fs)
    _fold_plot(axes.flat[3])
    fig.tight_layout()
    return {"sampling_aliasing.png": fig}


def interactive():
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(bottom=0.2)
    slider = Slider(fig.add_axes([0.15, 0.06, 0.7, 0.04]), "fs [Hz]", 50, 1500, valinit=1000, valstep=5)
    slider.on_changed(lambda fs: (_draw(ax, fs), fig.canvas.draw_idle()))
    _draw(ax, slider.val)
    return slider


if __name__ == "__main__":
    run(make_figures, __doc__, interactive=interactive)
