# Architektur- und Entwicklungsplan: Vollständig virtuelle 3D-Raketensimulation

## Statusbewertung (Ist)
- **Translation:** Euler–Cromer, Punktmasse (`physics/translation.py`, `simulation/loop.py`). Kräfte = Gravitation + optionales `a_cmd` (ohne g). 
- **Zustand:** `State` enthält pos/vel/acc, `q` (scalar-first), `omega`, `alpha`, `force_total`, `moment_total` – Felder vorhanden, Rotation aber **nicht integriert**.
- **Sensorik:** Modelle (`sensors/*`, `sensors/observed.py`) vorhanden; Trennung TRUE/MEASUREMENT/ESTIMATE realisiert; `delay_s` teilweise deklariert, nicht aktiv.
- **Schätzung:** `LowPassEstimator` + `SimpleEstimator`; kein Kalman. RMS (Schätzung vs Wahrheit) nur für Bewertung.
- **Controller:** `LandingController` gibt `a_cmd` (Beschleunigung ohne g) – heuristisch, keine Momente/Gimbal.
- **Visualisierung:** 3D-Trajektorie vorhanden, Studio mit Zeitplots – zeigt Trajektorie/Status, keine echte Rumpforientierung mit schwenkbarer Düse.
- **Tests:** 124 passed (umfangreich). Experimente reproduzierbar.

Ziel: 6-DOF-ähnliches, nachvollziehbares virtuelles Modell (Lernsimulation) mit echter Orientierung, Drehbewegung, Düsenschwenkung, konsistenter Sensorik→Schätzung→Steuerung→Physik.

## Prinzipien
- **Trennung:** true state (Physik), sensor readings (virtuell verrauscht/Dropout), estimate (Schätzer), controller output (Entscheidung).
- **Keine Doppelgravitation:** `a_cmd` ohne Gravitation (bestehender Vertrag beibehalten).
- **Opt-in:** neue 6-DOF-Pfade ohne Breaking der bestehenden Tests.
- **Tests zuerst bei kritischen Schritten:** Unit-Tests + Konsistenztests.
- **Nur virtuelle Simulation:** keine Hardware/Flugsteuerung.

## Phasen

### Phase 1: Rotationsintegration (minimal, konsistent)
**Ziel:** `q, omega, alpha, moment_total` tatsächlich integrieren.
**Betroffene Dateien:**
- `physics/rotation.py` (erweitern: Integration, Ableitungen)
- `physics/rigidbody.py` (neu, optional) oder direkt Integration in Loop
- `simulation/loop.py` (Rotationsschritt ergänzen)
- `state/core.py` (klar halten)

**Voraussetzungen:** quaternion utils vorhanden.
**Tests:** Quaternion norm bleibt nahe 1; Winkelgeschwindigkeit ändert q konsistent (kleiner Zeitschritt), `moment_total` wirkt.
**Ergebnis:** Rakete rotiert physikalisch korrekt bei Momenten.

### Phase 2: Thrust-Modell + Düsenorientierung
**Ziel:** virtuelles Triebwerk, schwenkbare Düse (TVC) → Kraft+Moment.
- `physics/thrust.py` (neu): Thrust vector in Bodyframe, Düsenschwenkwinkel (2 Achsen), Begrenzung, Schub (N) oder Beschleunigung.
- `physics/controller_tvc.py` (neu) oder `controller.py` erweitern (rückwärtskompatibel).
- `simulation/loop.py`: Body→Inertial für Schubvektor.

**Tests:** TVC erzeugt erwartete Momente; keine Doppelgravitation.
**Ergebnis:** sichtbare Düsenorientierung beeinflusst Flugbahn.

### Phase 3: Sensorik→Schätzung (konsistent)
- `sensors/imu.py` nutzen, `observed.py` um Orientation erweitern.
- `estimator/` erweitern (LPF reicht vorerst), RMS Tracking.
**Tests:** Trennung bleibt gewahrt.

### Phase 4: Visualisierung synchronisieren
- `visualization/viz3d_mpl.py`/`viz3d_studio.py`: Rumpf + Düse darstellen (quaternion), Anzeige Mess/Schätzung/Steuerung.
**Tests:** keine Fake-Daten.

### Phase 5: Integrationstests
- Szenarien reproduzierbar, 124+ Tests weiterhin grün.

## Risiken
- Quaternion-Norm drift (renorm)
- dt zu groß
- Breaking alter Tests (deshalb opt-in)

## Empfohlene erste Aufgabe
**Phase 1, erster Schritt:** Erweitere `physics/rotation.py` um `integrate_rotation(state, dt)` (Euler für omega→alpha? oder semi-implicit), normalisiere q. Füge optionalen Rotationsschritt in `Simulator` hinter Translation (noch nicht aktiviert standardmäßig) + einfacher Test. 
Betroffene Dateien: `physics/rotation.py`, `simulation/loop.py` (opt-in Flag), `tests/test_rotation.py` (neu). 
Ziele: q ändert sich nur bei Momenten, ||q||~1.
