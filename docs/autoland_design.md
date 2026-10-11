# Autoland – Lernsimulation (Design)

## Ist-Zustand (berechnet)
- pos[3], vel[3], acc[3], q[4], omega[3], alpha[3], t
- mass_kg, fuel_kg, dry_mass_kg, height_m
- force_total[3], moment_total[3]
Kräfte: aktuell nur Gravitation (g=[0,0,-g0]) in SimLoop. Translation: Euler-Cromer. Rotation: vorhanden, nicht integriert.

## Fehlende Voraussetzungen
- Thrust-Modell (Schubrichtung/Größe, burn) gekoppelt an Kräfte
- 6DOF-Kopplung (Rotation aktiv) für richtungsabhängigen Schub
- Inertia/CG genutzt bei Rotation/Momenten
- Steuerbefehle getrennt von Physik (Schnittstelle)

## Schnittstellen
- Target(pos): Ziel (Landepunkt, z nahe 0). Distanz-Helper.
- LandingController: placeholder, gibt gewünschte Beschleunigung a_des (m/s^2) zurück (virtuell). Kein Realhardware-Zugriff.
- SimLoop: bleibt unabhängig (Controller optional).

## Status
Controller nicht an Simulation gekoppelt (bewusst). Visu zeigt Status 'NO_CTRL' bzw. heuristisch. Ziel v_final <= 0.5 m/s noch nicht erreicht mit vorhandenem Physikmodell ohne aktiven Schub.

## Annahmen/Grenzen
- Nur Lernsimulation, keine reale Regelung. Vereinfachungen dokumentiert.
