# Autoland – Lernsimulation (Design)

> Reine Lernsimulation (vereinfachtes Punktmassen-Modell). Keine Hardware,
> keine reale Raketen-/Triebwerksregelung. Die Ergebnisse sind NICHT auf echte
> Systeme übertragbar.

## Ist-Zustand (berechnet)
- pos[3], vel[3], acc[3], q[4], omega[3], alpha[3], t
- mass_kg, fuel_kg, dry_mass_kg, height_m
- force_total[3], moment_total[3]
- Kräfte: Gravitation `F_g = m * [0,0,-g0]` (optional abschaltbar) plus
  optionaler Guidance-/Thrust-Anteil `F_cmd = m * a_cmd`.
- Translation: Euler-Cromer. Rotation: vorhanden, aber NICHT integriert.

## Guidance-/Controller-Schnittstelle (S1/S2)
Vertrag, klar dokumentiert in `physics/controller.py`:

- `controller.compute(state, target_pos, mass_kg) -> a_cmd` (Form `(3,)`, m/s²)
- `a_cmd` ist eine **Schub-Beschleunigung ohne Gravitation** im Inertialframe.
- Der Simulator addiert die Gravitation separat, also:
  `a_total = g + a_cmd` mit `g = [0,0,-g0]`.
- Folge: Zum Schweben muss `a_cmd_z = +g0` kommandiert werden. Gravitation
  wird nie doppelt oder falsch gerichtet angewendet.
- `Simulator(..., guidance=None)` (Default) => Verhalten exakt wie vorher
  (nur Gravitation). Ohne `target` wird die Guidance nicht aufgerufen.

`LandingController` (vereinfacht):
- Horizontal: PD-Zug zum Zielpunkt (X,Y).
- Vertikal: P-Regelung auf eine Soll-Sinkrate (`descent_speed_m_s`, Default 0.5).
- Sinkraten-Gate: volle Sinkrate erst, wenn horizontal zentriert UND langsam;
  so wird die horizontale Geschwindigkeit vor dem Aufsetzen gedämpft und die
  Gesamtaufsetzgeschwindigkeit bleibt bei ~0.5 m/s.

## Landebewertung (S3)
`evaluation/landing.py: evaluate_landing(history, target_pos, ...)`:

- Gültiger Bodenkontakt: `|height - ground_z| <= h_tol` (Default `h_tol=0.05 m`).
  Der Endzustand muss also IN der Bodenzone liegen. Ein Zustand weit UNTER der
  Bodenebene (durchgetunnelt bzw. über den Boden hinaus integriert) zählt NICHT
  als Kontakt.
- `LANDED`   : gültiger Kontakt UND Gesamtgeschwindigkeit `<= 0.5 m/s` UND
  horizontale Distanz zum Ziel `<= 1.0 m`.
- `TOO_FAST` : gültiger Kontakt, aber Gesamtgeschwindigkeit `> 0.5 m/s`.
- `FAILED`   : kein gültiger Kontakt bis zum Ende (z. B. Schweben oder
  Zeitüberschreitung in der Luft, `height > h_tol`) ODER Kontakt außerhalb der
  Ziel-Toleranz ODER unter den Boden getunnelt.
- `NO_DATA`  : leere History.
Kleine Fließkomma-Toleranz (1e-9) verhindert Fehlklassifikation bei exakt 0.5.
Wichtig: Ein Abbruch/Zeitlimit (`n_steps` erreicht) in der Luft ist `FAILED`,
niemals `LANDED`.

## Automatischer Szenariolauf
`evaluation/landing_matrix.py` fährt mehrere Startzustände/Ziele durch dieselbe
Simulation + Controller + Bewertung und gibt eine Tabelle aus:

```bash
cd selflanding_rocket
PYTHONPATH=. python3 evaluation/landing_matrix.py
```

Enthaltene Fälle u. a.: nominal (zentriert/versetzt), große Höhe, weites Ziel,
niedriger Start, anderes `dt`, zu kurzes `tf` (Zeitüberschreitung), freier Fall,
Schweben ohne Aufsetzen, Durchtunneln ohne `stop_at_ground`. Die Invarianten
dazu werden in `tests/test_landing_matrix.py` geprüft.

## Status
Der Controller ist jetzt **gekoppelt** und beeinflusst die simulierte Bewegung
nachweisbar. Für die Standard-Szenarien (Start 20–40 m, Ziele bis ~5 m
versetzt) erreicht die Simulation tatsächlich `LANDED` mit `v ≈ 0.5 m/s`.

## Studio & Szenario (interaktive Oberfläche)
- `simulation/scenario.py`: `Scenario` (Start-Hoehe, horizontaler Startversatz,
  Ziel X/Y, v0 vertikal/horizontal, `dt`) mit `validate()` und
  `run_scenario()`. Reine Orchestrierung vorhandener Bausteine, keine
  parallele Physik/Regelung.
- `visualization/viz3d_studio.py`: `LandingStudio` – reine Darstellung/Wiring.
  Textboxen für Einstellungen, Buttons Start / Pause / Reset / Neu berechnen,
  3D-Flugbahn + Ziel, Zeitdiagramme (Hoehe, Geschwindigkeit, Zielabstand),
  Live-Werte (`AKTUELL`) und Gesamtergebnis (`GESAMTERGEBNIS`).
- `landing_studio_app.py`: Startpunkt.
- Jeder Lauf wird neu berechnet; `Reset` spult nur den aktuellen Lauf zurueck.

## Experiment-/Auswertungsmodus
- `evaluation/experiment.py`: `Trial`, `run_trial()`, `run_experiment()`,
  `summarize()`, `write_results_csv()`, `plot_results()`, `main()`.
  Nutzt ausschliesslich `run_scenario()` + `evaluate_landing()`.
- Sweeps (begrenzt): `start_height_m`, `target_dist_m`, `v0_vertical_m_s`,
  `dt`, `max_time_s` (Timeout-Check).
- Gemessen pro Versuch: Szenario/Anfangsbedingungen, Status, Dauer, Endhoehe,
  vertikale/horizontale/gesamte Aufsetzgeschwindigkeit, Zielabstand,
  Schrittzahl, Kontakt-Flag.
- Zeitueberschreitung (Ende in der Luft) ist `FAILED` und zaehlt nie als
  erfolgreiche Landung (`success = status == LANDED`).
- Ausgaben: Konsolentabelle, CSV (`output/experiment/experiment_results.csv`),
  Plot (`output/experiment/experiment_results.png`).
- Start: `landing_experiment_app.py` bzw. `evaluation/experiment.py`.

## Sensorunsicherheit & Zustandsschaetzung
Erweiterung als **messbare, rein virtuelle Untersuchung** – ohne den
Physik-/Regel-Kontrakt zu aendern.

Drei strikt getrennte Dinge:
- **Wahrer Zustand**: das von `simulation/loop.py` integrierte `State`.
- **Messung**: simulierter Sensorwert (Rauschen, Bias, evtl. fehlend).
- **Schaetzung**: das, was die Regelung tatsaechlich benutzt.

Regeln (im Code erzwungen):
- Eine Messung veraendert den wahren Zustand NIE; es wird nur eine Kopie
  gelesen (`sensors/observed.py`).
- Die Regelung bekommt ausschliesslich die **Schaetzung**, nie die Wahrheit.
- Bei `ideal` ist die Schaetzung = Wahrheit; sonst nicht.

Bausteine:
- `sensors/observed.py`:
  - `SensorConfig` (dokumentiert, virtuell): `pos_std_m=1.0`,
    `vel_std_m_s=0.3`, Bias 0, `dropout_prob=0.1`, `seed=1234`; `validate()`.
  - `MeasurementModel.sample(true_pos, true_vel, t) -> Measurement`
    (`pos`/`vel` je `None` bei Dropout, `*_valid`-Flags). Nutzt die
    vorhandenen `PositionSensor`/`VelocitySensor` (`dt_sample=0`), Rauschen aus
    EINEM geseedeten `default_rng` -> reproduzierbar.
  - `ObservedController(inner, cfg, estimator=None)`: wrappt eine innere
    Regelung (z. B. `LandingController`). Sample-and-Hold bei fehlenden Werten,
    optionaler `estimator`. Loggt den RMS-Fehler (Schaetzung vs. Wahrheit);
    das dient nur der Bewertung und steuert nie.
- `estimator/base.py`: zusaetzlich `LowPassEstimator(alpha)` – bewusst
  einfach, `x <- x + alpha*(z - x)`, fehlende Messung (`None`) wird
  uebersprungen (letzte Schaetzung bleibt). KEIN Kalman-Filter.
- `simulation/scenario.py`: `run_scenario_with(scenario, guidance)` als
  gemeinsamer Runner; `run_scenario()` ruft ihn mit `LandingController` auf
  (identisches Ergebnis wie zuvor).

Vergleichsexperiment:
- `evaluation/sensor_experiment.py`: gleiche Szenarien in vier Gruppen
  `ideal` / `noise` / `dropout` / `estimation` bei **identischen
  Anfangsbedingungen**. Nur das Guidance-Objekt unterscheidet sich. Pro Versuch
  werden Status, Endgeschwindigkeit, Zielabstand, RMS-Schaetzfehler, fehlende
  Samples, Schritte und Dauer gemessen (`GroupTrial`, enthaelt den `seed`).
- `run_repeated(..., seeds)`: laeuft den kompletten Vergleich einmal pro Seed.
  Szenarien, Anfangsbedingungen und Controller-Parameter sind in jedem Lauf
  identisch; nur der Sensor-`seed` aendert sich. Ein Lauf ist deterministisch.
- `verify_fair_comparison(scenarios)`: stellt sicher, dass alle Modi dieselben
  Szenarien/Anfangsbedingungen nutzen (`scenario_signature`), damit die
  `noise`/`dropout`-Trennung und gleiche Erfolgsquoten nicht durch versteckte
  Unterschiede verfaelscht werden.
- Statistik (`summarize_statistics`, `stats_for`): je (Modus, Szenario) und je
  Modus gesamt - Versuchszahl `trials`, `landed`, `failed`, Erfolgsquote sowie
  Statuszaehler `status_LANDED/TOO_FAST/FAILED/NO_DATA` (Summe = `trials`;
  fehlgeschlagene/abgebrochene Versuche verschwinden NICHT), mittlere und
  maximale Endgeschwindigkeit, mittlere Zielentfernung bei Bodenkontakt
  (`n_contact`), mittlere RMS-Schaetzfehler und die verwendeten Seeds.
- Ausgaben: Trials-CSV (je Versuch inkl. Seed), Summary-CSV (je Modus+Szenario
  + `ALL`), Statistik-Plot (`output/sensor_experiment/`).
- Start: `landing_sensor_experiment_app.py` mit `--seeds` (z. B. `--seeds 1,2,3`).

Beispielergebnis (5 Szenarien, 5 Seeds, 100 Versuche):
`ideal 25/25`, `noise 25/25`, `dropout 25/25`, `estimation 17/25` (8x
`TOO_FAST`, 0x `FAILED`). Ueber mehrere Seeds stabil: Allein die `estimation`
-Gruppe schlaegt fehl - der Low-Pass-Filter halbiert in etwa den Schaetzfehler
(pos 1.74 -> 0.80 m), verzoegert aber die Messung; die geschaetzte Sinkrate ist
kurzzeitig zu niedrig, die Regelung bremst spaeter -> Touchend auf bis zu
~0.59 m/s (knapp ueber dem 0.5 m/s-Limit). Es wurde noch KEIN Filter optimiert;
die Auswertung zeigt nur, wie stabil dieser Befund ist.

## Diagnose der estimation-Fehlschlaege (TOO_FAST)
Ziel: WANN und um WIEVIEL weicht die Schaetzung von der Wahrheit ab - als
Messung + Plot, ohne Parameter zu optimieren und ohne Controller-Aenderung.

- `ObservedController` kann mit `record_series=True` pro Zeitschritt den
  WAHREN Zustand, die rohe MESSUNG (inkl. Dropout-Luecken) und die SCHAETZUNG
  live aufzeichnen (`self.record`). Das Aufzeichnen aendert weder den
  RNG-Strom noch die Simulation - ein aufgezeichneter Lauf ist numerisch
  identisch mit einem ohne Aufzeichnung (per Test abgesichert).
- `evaluation/sensor_diagnostics.py`:
  - `find_failures(...)`: selektiert Fehlschlaege (`TOO_FAST`/`FAILED`)
    der estimation-Modi - keine wird verschwiegen.
  - `lag_metrics(trial, series)`: aus der echten Zeitreihe - Kontaktzeit,
    wahrer/geschaetzter vz beim Aufsetzen sowie deren Differenz (Lag-Mass),
    max. Abweichung inkl. Zeitpunkt, mittlere |Abweichung| ueber den Lauf und
    ueber die letzten 2 s, erster Zeitpunkt > 0.05 m/s Abweichung.
  - `plot_speed_diagnostics(...)`: 3-Panel-Figur aus den echten gespeicherten
    Simulationsdaten (Hoehe, vz mit wahr/Messung/Schaetzung, |v| mit v_tol),
    Bodenkontakt markiert, `TOO_FAST` beschriftet. KEINE kuenstlichen Kurven
    (Test vergleicht Plotdaten mit der Zeitreihe).
- Verwendet dieselben Szenarien + Seeds wie das Experiment; jede Fehlschlags-
  (Szenario, Seed)-Kombination wird reproduzierbar nachgefahren.
- Ausgaben: `output/sensor_diagnostics/failures.csv` + eine Figur pro Fehlschlag
  (`failures/estimation__<szenario>__seed<s>.png`).
- Start: `landing_sensor_diagnostics_app.py` mit `--seeds`.

Befund (Default 5 Seeds, 8 Fehlschlaege): Beim Aufsetzen ist die Schaetzung
meist noch nicht auf dem wahren Wert (Delta vz 0.04..0.32 m/s) - der Filter
haengt der Endphase nach, die Regelung bremst einen Moment zu spaet. Die groesste
Abweichung tritt bei t=0 auf (Filter startet bei Null -> erster Schaetzwert
springt) oder beim Sinkraten-Anstieg bei grossen Hoehen (t~25-65 s). In den
letzten 2 s wandert die Schaetzung im Mittel um ~0.08-0.14 m/s um die Wahrheit
(Rauschen + Lag), was einzelne Landungen knapp ueber 0.5 m/s bringt.

## Annahmen/Grenzen
- Punktmasse; keine Schubgrenzen, keine Lage/Rotation im Schub, kein
  Triebwerksmodell, unbegrenzte Schub-Beschleunigung.
- Der horizontale PD und das Sinkraten-Gate sind Lern-Heuristiken, keine
  flugmechanisch hergeleitete Regelung.
- Massenmodell vorhanden, aber die Guidance skaliert nicht mit der Masse.
- Sensor-/Schaetzmodell bewusst einfach: skalares weisses Rauschen + konstanter
  Bias, Dropouts unabhaengig pro Kanal, keine Verzoegerung, keine IMU-
  Integration, kein Kalman-Filter, keine Sensor-Validierung/Plausibilitaet.
- Nur Lernsimulation.
