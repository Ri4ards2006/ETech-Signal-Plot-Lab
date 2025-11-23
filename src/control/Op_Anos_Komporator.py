# Importiere erforderliche Bibliotheken
# NumPy: Für numerische Berechnungen und Array-Operationen
import numpy as np  
# Matplotlib: Für die Erstellung von Diagrammen und Plots
import matplotlib.pyplot as plt  
# ipywidgets: Für interaktive Eingabewidgets (z. B. Slider, Dropdown) in Jupyter-Notebooks
from ipywidgets import interact, FloatSlider, Dropdown  


def compute_signals(t, Vin_amp, Vin_freq, waveform):
    """Berechnet das Eingangssignal (Vin) basierend auf Zeit, Amplitude, Frequenz und Wellentyp.
    
    Args:
        t (np.ndarray): Zeitachse (Array mit Zeitwerten in Sekunden)
        Vin_amp (float): Amplitude des Eingangssignals in Volt (V)
        Vin_freq (float): Frequenz des Eingangssignals in Hertz (Hz)
        waveform (str): Typ der Wellenform ('Sine', 'Square', 'Triangle', 'Zero')
    
    Returns:
        np.ndarray: Eingangssignal-Array Vin in Volt (V)
    """
    # Berechne den Winkel φ(t) = ω*t, wobei ω = 2π*Frequenz (Angular Frequency)
    # Dieser Winkel wird für alle trigonometrischen Berechnungen benötigt
    x = 2 * np.pi * Vin_freq * t  
    
    # Initialisiere Vin als Array mit Nullen (Fallback für unerwartete Wellenformen)
    Vin = np.zeros_like(t)  
    
    # Prüfe die gewählte Wellenform und berechne Vin entsprechend
    if waveform == 'Sine':  # Sinuswellenform
        # Vin = Amplitude * sin(φ(t))
        # np.sin(x) berechnet den Sinus für jeden Winkel im Array x
        Vin = Vin_amp * np.sin(x)  
    elif waveform == 'Square':  # Rechteckwellenform
        # Schalte zwischen 1 und -1: Wenn sin(φ(t)) ≥ 0 → 1, sonst -1
        # np.where(condition, x, y): Gibt für Elemente, die condition erfüllen, x zurück, sonst y
        # Multipliziere mit Amplitude, um die Spannungshöhe zu erreichen
        Vin = Vin_amp * np.where(np.sin(x) >= 0, 1, -1)  
    elif waveform == 'Triangle':  # Dreieckwellenform
        # Grundprinzip: sin(φ(t)) generiert eine Sinuswelle. Durch arcsin(sin(φ(t))) wird sie in eine lineare Rampe umgewandelt.
        # Schritt-für-Schritt:
        # 1. np.sin(x) → Standard-Sinuswelle mit Werten [-1, 1]
        # 2. np.arcsin(np.sin(x)): Arkussinus invertiert den Sinus, aber nur im Intervall [-π/2, π/2].
        #    Dadurch wird die Sinuswelle 'gestrafft' zu einer linearen Funktion zwischen -π/2 und π/2.
        # 3. (2/π) scaling: Arkussinus liefert Werte [-π/2, π/2]. Multiplikation mit (2/π) konvertiert dies zu [-1, 1]
        # 4. Multiplikation mit Amplitude → Vin in der gewünschten Amplitude
        Vin = Vin_amp * (2 / np.pi) * np.arcsin(np.sin(x))  
    elif waveform == 'Zero':  # Explizite Nullwellenform (hinzugefügt)
        # Setze Vin auf 0 V für alle Zeitpunkte
        Vin = np.zeros_like(t)  
    # else: (optional, falls unbekannte Wellenform übergeben wird)
    #     Vin bleibt als Null-Array (wurde bereits initialisiert)
    
    return Vin


def plot_comparator(Vin_amp=5.0, Vin_freq=1.0, t_max=10.0,
                    waveform='Sine', Vcc=15.0, Vref=0.0):
    """Simuliert einen idealen Komparator und visualisiert Eingang-, Referenz- und Ausgangssignal.
    
    Args:
        Vin_amp (float): Amplitude des Eingangssignals (Standard: 5.0 V)
        Vin_freq (float): Frequenz des Eingangssignals (Standard: 1.0 Hz)
        t_max (float): Maximale Simulationszeit (Standard: 10.0 s)
        waveform (str): Wellenform des Eingangssignals (Standard: 'Sine')
        Vcc (float): Versorgungsspannung des Komparators (Standard: 15.0 V)
        Vref (float): Referenzspannung (Threshold) des Komparators (Standard: 0.0 V)
    
    Returns:
        None (zeigt das Diagramm an)
    """
    # Erstelle Zeitachse: 2000 gleichmäßig verteilte Punkte zwischen 0 und t_max (s)
    # Hoher Punktenzahl gewährleistet eine glatte Darstellung der Signale
    t = np.linspace(0, t_max, 2000)  
    
    # Generiere Eingangssignal Vin mithilfe der compute_signals-Funktion
    Vin = compute_signals(t, Vin_amp, Vin_freq, waveform)  
    
    # -------------------- Komparator-Logik --------------------
    # Idealisiertes Verhalten eines Komparators:
    # - Wenn Eingangssignal Vin > Referenzspannung Vref → Ausgang Vout = +Vcc
    # - Wenn Eingangssignal Vin < Referenzspannung Vref → Ausgang Vout = -Vcc
    # (realen Komparatorcharakteristika wie Übertragungszeit oder Hysteresis werden hier nicht berücksichtigt)
    # np.where() führt eine elementweise Bedingungsprüfung durch und ersetzt Werte basierend auf der Bedingung
    Vout = np.where(Vin > Vref, Vcc, -Vcc)  # Syntax: np.where(condition, wert_if_true, wert_if_false)
    # ---------------------------------------------------------
    
    # Initialisiere eine Plot-Figur mit fixer Größe (12 Zentimeter Breite, 6 Zentimeter Höhe)
    plt.figure(figsize=(12, 6))  
    
    # Plotte Eingangssignal Vin:
    # - Linienfarbe: #2c7fb8 (blau-grau)
    # - Linienstärke: 2 (für bessere Sichtbarkeit)
    # - Label: 'Eingangssignal Vin' für die Legende
    plt.plot(t, Vin, label='Eingangssignal Vin', color='#2c7fb8', linewidth=2)  
    
    # Plotte Referenzlinie Vref:
    # - Typ: waagerechte Linie (axhline = axis horizontal line)
    # - Farbe: orange
    # - Linienstil: gestrichelt (--)
    # - Label: Dynamisch generiert mit aktueller Vref-Wert (2 Nachkommastellen)
    plt.axhline(Vref, color='orange', linestyle='--', label=f'Referenzspannung Vref = {Vref:.2f} V')  
    
    # Plotte Ausgangssignal Vout:
    # - Linienfarbe: #e53434 (rot)
    # - Linienstärke: 2
    # - Label: 'Komparator-Ausgang'
    plt.plot(t, Vout, label='Komparator-Ausgang', color='#e53434', linewidth=2)  
    
    # Titel des Diagramms
    plt.title('Ideal-Komparator Simulation')  
    # X-Achse: Zeit in Sekunden (s)
    plt.xlabel('Zeit (s)')  
    # Y-Achse: Spannung in Volt (V)
    plt.ylabel('Spannung (V)')  
    
    # Zeige Gitternetz im Plot an:
    # - grid(True): Aktiviert Gitter
    # - linestyle='--': Gestrichelter Linienstil
    # - alpha=0.5: Transparenz (Hintergrundlinien sind nicht zu auffällig)
    plt.grid(True, linestyle='--', alpha=0.5)  
    
    # Füge Legende hinzu, um Linien zu beschriften
    plt.legend()  
    
    # Skalierung der Y-Achse: Von -1.2*Vcc bis +1.2*Vcc, um Vcc/-Vcc gut sichtbar zu machen
    plt.ylim(-Vcc * 1.2, Vcc * 1.2)  
    
    # Passe Layout an, um Label, Titel und Legende optimal zu fitten
    plt.tight_layout()  
    
    # Zeige das fertige Diagramm an
    plt.show()  


# Interaktive UI-Eingabe für Parameterübergabe
# Die interact()-Funktion von ipywidgets erstellt automatisch Eingabefelder (Widgets) für die Funktion plot_comparator
# Jeder Parameter wird mit einem Widget verbunden, das den Benutzer bei der Einstellung unterstützt
interact(plot_comparator,
         # Slider für Eingangs-Amplitude:
         # - min=0.1: Minimalwert (V)
         # - max=10: Maximalwert (V)
         # - step=0.1: Schrittweite (V)
         # - value=5: Standardwert (V)
         # - description: Beschriftung für den Benutzer
         Vin_amp=FloatSlider(min=0.1, max=10, step=0.1, value=5,
                             description='Eingangs-Amplitude [V]'),
         # Slider für Eingangs-Frequenz:
         # - min=0.1: Minimalwert (Hz)
         # - max=10: Maximalwert (Hz)
         # - step=0.1: Schrittweite (Hz)
         # - value=1: Standardwert (Hz)
         Vin_freq=FloatSlider(min=0.1, max=10, step=0.1, value=1,
                              description='Eingangs-Frequenz [Hz]'),
         # Slider für Simulationszeit:
         # - min=1: Minimalwert (s)
         # - max=20: Maximalwert (s)
         # - step=0.5: Schrittweite (s)
         # - value=10: Standardwert (s)
         t_max=FloatSlider(min=1, max=20, step=0.5, value=10,
                           description='Simulationszeit [s]'),
         # Dropdown-Menü für Wellenform:
         # - options: Liste der verfügbaren Wellenformen
         # - value='Sine': Standardwert
         # - description: Beschriftung
         waveform=Dropdown(options=['Sine', 'Square', 'Triangle', 'Zero'], value='Sine',  # 'Zero' hinzugefügt
                           description='Wellenform'),
         # Slider für Versorgungsspannung Vcc:
         # - min=5: Minimalwert (V)
         # - max=30: Maximalwert (V)
         # - step=1: Schrittweite (V)
         # - value=15: Standardwert (V)
         Vcc=FloatSlider(min=5, max=30, step=1, value=15,
                         description='Vcc Versorgung [V]'),
         # Slider für Referenzspannung Vref:
         # - min=-10: Minimalwert (V)
         # - max=10: Maximalwert (V)
         # - step=0.1: Schrittweite (V)
         # - value=0: Standardwert (V)
         Vref=FloatSlider(min=-10, max=10, step=0.1, value=0,
                          description='Referenzspannung Vref [V]'));