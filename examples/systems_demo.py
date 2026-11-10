"""Step / impulse responses of 1st- and 2nd-order systems vs. damping ratio ζ.

    uv run python -m examples.systems_demo
    uv run python -m examples.systems_demo --save docs/img
"""

import matplotlib.pyplot as plt
import numpy as np

from examples._cli import run
from signal_lab import systems

ZETAS = (0.1, 0.3, 0.5, 0.707, 1.0, 2.0)


def make_figures():
    wn = 1.0
    t = np.linspace(0, 15, 3000)
    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(12, 8))

    for z in ZETAS:
        sys = systems.second_order(z, wn)
        a1.plot(t, systems.step(sys, t)[1], lw=1.3, label=f"ζ = {z:g}")
        a2.plot(t, systems.impulse(sys, t)[1], lw=1.3, label=f"ζ = {z:g}")
    a1.set_title("2nd order: step response (ωn = 1 rad/s)")
    a2.set_title("2nd order: impulse response")

    for tau in (0.5, 1.0, 2.0):
        sys = systems.first_order(1.0, tau)
        (line,) = a3.plot(t, systems.step(sys, t)[1], lw=1.3, label=f"τ = {tau:g} s")
        a3.plot(tau, 1 - np.exp(-1), "o", color=line.get_color())
    a3.axhline(1 - np.exp(-1), color="0.6", ls="--", lw=0.8)
    a3.text(6, 0.57, "63.2 % at t = τ", color="0.4")
    a3.set_title("1st order: step response K/(τs+1)")
    a3.set_xlim(0, 12)

    z = np.linspace(0.01, 0.99, 300)
    a4.plot(z, [systems.overshoot_percent(v) for v in z], lw=1.5, label="100·exp(−πζ/√(1−ζ²))")
    measured = [systems.step_metrics(t, systems.second_order_step(t, v, wn), final=1.0)["overshoot_pct"]
                for v in ZETAS]
    a4.plot(ZETAS, measured, "o", label="measured from simulated step")
    a4.set_title("Peak overshoot vs. ζ")
    a4.set_xlabel("ζ")
    a4.set_ylabel("overshoot [%]")

    for ax in (a1, a2, a3):
        ax.set_xlabel("t [s]")
    for ax in (a1, a2, a3, a4):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return {"systems_responses.png": fig}


if __name__ == "__main__":
    run(make_figures, __doc__)
