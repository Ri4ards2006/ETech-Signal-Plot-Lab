import numpy as np
import matplotlib.pyplot as plt

def plot_noise(
    amplitude: float = 1.0,    # Rausch-Amplitude (Größe der Schwankungen)
    sample_rate: int = 1000,   # Abtastrate (Hz) → Zeitachse berechnet
    duration: float = 2.0,    # Dauer des Signals (Sekunden)
    style: str = "seaborn",    # Plot-Stil (siehe Matplotlib Styles)
    title: str = "Gaußsches Rauschen",  # Plot-Titel
    xlabel: str = "Zeit (s)",  # x-Achsen-Beschriftung
    ylabel: str = "Amplitude", # y-Achsen-Beschriftung
    figsize: tuple = (10, 5)   # Größe des Plots (Breite, Höhe)
):
    # Generiere Zeitachse
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generiere Gaußsches Rauschen (Mittelwert 0, Standardabweichung = amplitude)
    noise = np.random.normal(0, amplitude, len(t))
    
    # Plot-Einstellungen
    plt.style.use(style)
    plt.figure(figsize=figsize)
    
    # Plotte das Signal
    plt.plot(t, noise, color="#2c7fb8", linewidth=1.2, label="Rauschsignal")  # Blauer Farbton
    
    # Hinzufügen von Grid, Legende, Titel und Achsen-Beschriftungen
    plt.grid(True, linestyle="--", alpha=0.7)  # Weak dashed grid
    plt.legend()
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()  # Optimiert Layout
    
    # Anzeige oder Speichern
    plt.show()
    # plt.savefig("noise_plot.png", dpi=300)  # Optional: Speichere als Bild

# Beispielaufruf mit "ultra baba"-ähnlichen Parametern (pass an deine Bedürfnisse an!)
plot_noise(
    amplitude=0.5,        # Leicht schwache Schwankungen
    sample_rate=5000,     # Höhere Abtastrate für detaillierter Plot
    duration=1.5,         # Kurzzeit-Plot
    style="ggplot",       # Alternativer Stil (ggplot2-ähnlich)
    title="Ultra Baba Noise Signal",  # Referenz zu deinem Projekt
    xlabel="Zeit (s)",
    ylabel="Amplitude (V)"
)
