# Vorschlag: Konsistente virtuelle Starrkörper-Rotationsdynamik

## Ist-Zustand (kurz)
- `q` scalar-first `[w,x,y,z]` (body->inertial), `omega`, `alpha`, `moment_total` im `State` vorhanden.
- `integrate_rotation` (opt-in, default False) integriert aktuell `q` aus `omega` und aktualisiert `omega += alpha*dt` – vereinfacht.
- Keine Trägheitsmatrix, keine klare Unterscheidung Body/Inertial für `omega/alpha/moments` im bestehenden Pfad.

## 1) Koordinatensysteme
- **Inertial (I):** ENU wie bisher (X East, Y North, Z Up) – für pos/vel/acc, Kräfte, Gravitation.
- **Körperfest (B):** fest am virtuellen Rumpf. Orientierung `q_{B->I}` (scalar-first) bildet Vektoren B→I.
- **Empfehlung:** `omega`, `alpha`, `moment_total` **konsistent** einer Frame-Definition zuordnen. Für echten Starrkörper gilt: `I*alpha_b + omega_b × (I*omega_b)=M_b` (bodyframe) oder äquivalent inertial. Der aktuelle einfache Schritt ist ein Kompromiss – für Lernsimulation sollte die gewählte Zuordnung dokumentiert und testbar sein.
- **Annahme (offen):** Bevorzugt **bodyframe** für `omega_b, alpha_b, M_b` und Quaternion-Update über `omega_b` (Standard). Alternative: inertial – beides möglich, aber inkonsistent, wenn Kräfte/Momente im Bodyframe entstehen.

## 2) Zusammenhang Quaternion/Winkelgeschwindigkeit/Beschleunigung
- `q_dot = 0.5 * q ⊗ [0, omega_b]` (bodyframe `omega_b`) – gängige Form. Falls `omega` als inertial behandelt wird, ist die Form leicht anders; wichtig, nur eine Variante zu wählen.
- `alpha_b = d/dt omega_b` (bodyframe) folgt aus Euler-Gleichungen: `alpha_b = I_b^{-1}(M_b - omega_b × (I_b*omega_b))`.
- Integration: Semi-implizit oder explizit Euler (kleines dt). q nach jedem Schritt normalisieren.

## 3) Zusätzliche Zustandsgrößen & Parameter
- **Trägheitstensor:** `I_b` – symmetrisch 3x3 (oder Hauptträgheitsmomente + Orientierung, minimal: diagonale Hauptachsen). Für einfache Rakete Start mit Diagonale `I_xx,I_yy,I_zz`.
- **Massenverteilung konstant** (rigid body) oder vereinfacht.
- **Optionale Felder im State:** `omega_b`, `I_b` (oder Referenz) – aber bestehender `State` soll minimal erweitert werden (Rückwärtskompatibel). Alternativ Parameter in Physikmodul (`RigidBody`).
- **Empfehlung:** `physics/rigidbody.py` mit `RigidBody(I_diag=[...], mass_kg=...)`. Simulator kann RB-Instanz optional erhalten.

## 4) Unabhängige Testbarkeit der Rotationsdynamik
- **Isolation:** Nur Rotationsintegration testen (ohne Translation). Unit-Tests für Euler-Gleichungen (z. B. freies Kreiselverhalten mit bekannten I, konstantes Moment).
- **Konsistenz:** `||q||==1` nach jedem Schritt, Energie/Drall sinnvoll für Lernfälle (nicht zwingend exakt, aber nicht offensichtlich falsch).
- **Opt-in:** Neuer Pfad parallel, alter Pfad unverändert (119+ Tests bleiben grün).
- **Deterministisch:** mit Seed/gleichem dt reproduzierbar.

## 5) Kopplung an Visualisierung
- `q_{B->I}` steuert Rumpforientierung. Düsenorientierung (TVC) ergibt `moment_total_b` (später), Kraft im Bodyframe → inertial via `q`.
- Visualisierung liest `q` aus History und zeichnet Rumpf+Ausrichtung + Düsenwinkel. Keine Fake-Daten.

## Empfohlene erste Implementierungsaufgabe
**Phase 1b: Frame-klarer Rotationspfad vorbereiten**
- Datei: `physics/rigidbody.py` (neu) – Klasse `RigidBody` mit `I_diag` (3x3 oder numpy), Methode `angular_accel(omega_b, M_b)` → `alpha_b`.
- Datei: `physics/rotation.py` – Funktion `integrate_rotation_body(state, dt, rb=None)` (bodyframe-Variante), dokumentiere Frames. Bestehende `integrate_rotation` bleibt unverändert.
- Datei: `simulation/loop.py` – `SimConfig.integrate_rotation_body` (opt-in), nutzt bodyframe-Pfad falls gesetzt.
- Test: `tests/test_rigidbody.py` (neu) – Einheitskreisrotation bei konstantem Moment, `||q||==1`.
- Annahmen dokumentieren (bodyframe für omega/Momente).

**Warum:** trennt sauber von bestehendem Pfad, macht Trägheit explizit, testbar ohne TVC/Thrust.
