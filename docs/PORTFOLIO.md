# ETech-Signal-Plot-Lab — Portfolio

[English](#english) · [Deutsch](#deutsch)

---

## English

**What it is.** A Python lab for electrical-engineering signal processing. The
`signal_lab` package turns textbook DSP and circuit theory into small,
importable modules, and each one is tested against a closed-form result.

**Modules**

- **Fourier.** Windowed FFT spectrum (rect / Hann / Blackman) with correct
  frequency axis, coherent-gain-corrected amplitudes and dB scale. A
  Fourier-series demo builds a square wave from odd harmonics and measures the
  Gibbs overshoot, which converges to Si(π)/π − ½ ≈ 8.95 % of the jump and does
  not shrink as harmonics are added.
- **Bode.** RC, RL and series-RLC networks built from component values. The
  plots show magnitude and phase, with −3 dB corners found by root-finding
  rather than read off a grid.
- **Filter design.** Butterworth and Chebyshev I low-, high- and band-pass
  filters via `scipy.signal`, stored as second-order sections. They can be
  applied causally or with zero phase.
- **Sampling & aliasing.** Nyquist check, frequency folding and Whittaker–Shannon
  reconstruction, with an interactive sampling-rate slider.
- **Step / impulse response.** 1st- and 2nd-order systems across the damping
  ratio ζ. Measured overshoot, peak, rise and settling times are compared with
  the formulas.

**Engineering choices**

- Computation is kept separate from plotting, so the numbers can be tested
  without a display.
- There are 69 pytest tests, for example: FFT peak bin, f_c = 1/(2πRC),
  B = R/(2πL), Butterworth −3.01 dB at f_c, Chebyshev ripple, alias of 900 Hz at
  1 kHz = 100 Hz, and overshoot = e^(−πζ/√(1−ζ²)).
- GitHub Actions runs the tests on Python 3.10 and 3.13, treats warnings as
  errors, and publishes the example plots as an artifact.
- Dependencies are limited to NumPy, SciPy and Matplotlib, with a reproducible
  `uv` lock file.

**Pitfalls caught by the tests**

- A resonant RLC low-pass must be referenced to its DC gain, not its resonance
  peak. Otherwise the "−3 dB points" end up on both sides of the peak.
- Window choice decides whether a −60 dB tone next to a strong tone is visible
  at all. With a rectangular window, leakage at 70 Hz from the tone is already
  around −50 dB, so the weak tone is buried.

**Stack:** Python · NumPy · SciPy · Matplotlib · pytest · uv · GitHub Actions

---

## Deutsch

**Worum es geht.** Ein Python-Labor für Signalverarbeitung in der
Elektrotechnik. Das Paket `signal_lab` setzt DSP- und Schaltungstheorie aus dem
Lehrbuch in kleine, importierbare Module um, und jedes davon wird gegen ein
analytisches Ergebnis getestet.

**Module**

- **Fourier.** FFT-Spektrum mit Fensterwahl (Rechteck / Hann / Blackman),
  korrekter Frequenzachse, Amplitudenkorrektur über den Coherent Gain und
  dB-Skala. Eine Fourier-Reihen-Demo baut ein Rechteck aus ungeraden
  Harmonischen und misst das Gibbs-Überschwingen. Es konvergiert gegen
  Si(π)/π − ½ ≈ 8,95 % des Sprungs und wird mit mehr Harmonischen nicht kleiner.
- **Bode.** RC-, RL- und Serien-RLC-Netzwerke aus Bauteilwerten. Die Plots
  zeigen Betrag und Phase; die −3-dB-Eckfrequenzen werden per Nullstellensuche
  bestimmt, nicht aus einem Raster abgelesen.
- **Filterentwurf.** Butterworth- und Tschebyscheff-I-Tief-, Hoch- und
  Bandpässe über `scipy.signal`, gespeichert als SOS. Sie lassen sich kausal
  oder nullphasig anwenden.
- **Abtastung & Aliasing.** Nyquist-Prüfung, Frequenzfaltung und
  Whittaker-Shannon-Rekonstruktion, mit interaktivem Schieberegler für die
  Abtastrate.
- **Sprung-/Impulsantwort.** Systeme 1. und 2. Ordnung über den Dämpfungsgrad ζ.
  Gemessenes Überschwingen sowie Spitzen-, Anstiegs- und Einschwingzeit werden
  mit den Formeln verglichen.

**Technische Entscheidungen**

- Berechnung und Darstellung sind getrennt, damit sich die Zahlen ohne
  Bildschirm testen lassen.
- Es gibt 69 pytest-Tests, z. B.: FFT-Peak-Bin, f_c = 1/(2πRC), B = R/(2πL),
  Butterworth −3,01 dB bei f_c, Tschebyscheff-Welligkeit, Alias von 900 Hz bei
  1 kHz = 100 Hz und Überschwingen = e^(−πζ/√(1−ζ²)).
- GitHub Actions führt die Tests unter Python 3.10 und 3.13 aus, behandelt
  Warnungen als Fehler und stellt die Beispielplots als Artefakt bereit.
- Die Abhängigkeiten beschränken sich auf NumPy, SciPy und Matplotlib, mit
  reproduzierbarem `uv`-Lockfile.

**Fallstricke, die die Tests aufgedeckt haben**

- Ein resonanter RLC-Tiefpass muss auf seine DC-Verstärkung bezogen werden,
  nicht auf den Resonanzpeak. Sonst landen die „−3-dB-Punkte“ links und rechts
  vom Peak.
- Die Fensterwahl entscheidet, ob ein −60-dB-Ton neben einem starken Ton
  überhaupt sichtbar ist. Mit Rechteckfenster liegt das Leakage 70 Hz neben dem
  Ton bereits bei rund −50 dB, der schwache Ton geht darin unter.

**Stack:** Python · NumPy · SciPy · Matplotlib · pytest · uv · GitHub Actions
