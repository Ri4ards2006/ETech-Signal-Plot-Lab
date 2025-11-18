import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
from scipy.integrate import odeint

# Funktion zur Simulation eines RLC-Kreis
def rlc_circuit(y, t, R, L, C):
    # y[0] = Spannung q(t), y[1] = Strom i(t) = q'(t)
    q, i = y
    dqdt = i
    didt = -(R/L)*i - (1/(L*C))*q
    return [dqdt, didt]

# Interaktive Plot-Funktion
def plot_rlc(R=1.0, L=1.0, C=1.0, q0=1.0, i0=0.0, t_max=10.0):
    t = np.linspace(0, t_max, 1000)  # Zeitachse
    y0 = [q0, i0]                     # Anfangsbedingungen: Spannung q0, Strom i0
    sol = odeint(rlc_circuit, y0, t, args=(R, L, C))
    
    plt.figure(figsize=(10,5))
    plt.plot(t, sol[:,0], label="Spannung q(t)", color="#2c7fb8", linewidth=2)
    plt.plot(t, sol[:,1], label="Strom i(t)", color="#f15854", linewidth=1.5)
    plt.title(f"RLC-Kreis Simulation (R={R}Ω, L={L}H, C={C}F)", fontsize=14)
    plt.xlabel("Zeit (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Interaktive Slider
interact(plot_rlc,
         R=FloatSlider(min=0, max=10, step=0.1, value=1, description="R [Ω]"),
         L=FloatSlider(min=0.1, max=5, step=0.1, value=1, description="L [H]"),
         C=FloatSlider(min=0.01, max=2, step=0.01, value=1, description="C [F]"),
         q0=FloatSlider(min=-5, max=5, step=0.1, value=1, description="q0 [V]"),
         i0=FloatSlider(min=-5, max=5, step=0.1, value=0, description="i0 [A]"),
         t_max=FloatSlider(min=1, max=20, step=0.5, value=10, description="t_max [s]"));
