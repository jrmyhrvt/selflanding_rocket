# Sensor-Diagnose – Analyse der TOO_FAST-Fälle

**Stand:** Analyse der Diagnose-Ausgaben (reine Lese-/Re-Analyse, **keine Codeänderung**).
**Datei(en):** `output/sensor_diagnostics/failures.csv` (8 fehlgeschlagene Versuche) und die daraus erzeugten Plots in `output/sensor_diagnostics/failures/`.
**Zusätzlich:** zur Kontrastierung wurden alle 25 `estimation`-Versuche (5 Szenarien × 5 Seeds) schreibfrei über die vorhandenen Runner erneut ausgewertet (gleiche Seeds/Parameter, deterministisch).

## 1. Datenbasis und Methodik

- `failures.csv` enthält je Fehlschlag: `v_end`, Kontakt-/Zielabstand, Schätzfehler (RMS), `vz_true`/`vz_est` beim Aufsetzen, deren Differenz `vz_end_delta`, `max_abs_dev` (inkl. Zeitpunkt `t_max_dev`), mittlere |Abweichung| im Lauf und in den letzten 2 s (`mean_abs_dev_last_2s`), erster Zeitpunkt mit Abweichung > 0.05 m/s (`first_dev_gt_0p05`).
- Die PNG-Diagramme könnten vom Berichterstatter nicht direkt angezeigt werden (Modell ohne Bild-Eingabe). Die Plots bestehen jedoch aus genau denselben registrierten Zeitreihen (`ObservedController.record`, wahr / Messung / Schätzung); eine frühere Testprüfung bestätigte, dass die geplotteten Linienkoordinaten mit den gespeicherten Reihen identisch sind. **Diese Analyse basiert damit auf denselben numerischen Daten, die auch in den Bildern stehen.**

**Begriffe:**
- `Δ vz = vz_est – vz_true` **beim Bodenkontakt** → Maß für den Schätz-Lag in der kritischen Endphase (positiv = Schätzung meldet geringere Sinkrate als tatsächlich vorhanden).
- `max|dev|`/`t_max` → größte Abweichung im gesamten Flug und deren Zeitpunkt.
- `|dev| letzte 2 s` → mittlere Abweichung kurz vor dem Aufsetzen (die Phase, die die Aufsetzgeschwindigkeit bestimmt).

## 2. Ergebnisse aller 25 estimation-Versuche

| Szenario | Seed | Status    | v_end | vz_true | vz_est | Δ vz  | max|dev| | t_max | dev 2s |
|---|---|---|---|---|---|---|---|---|---|
| h20_d0  | 1234 | LANDED    | 0.394 | -0.383 | -0.280 | +0.102 | 0.874 | 0.00 | 0.141 |
| h20_d0  | 101  | TOO_FAST  | **0.588** | -0.588 | -0.552 | +0.037 | 0.476 | 25.50 | 0.132 |
| h20_d0  | 202  | TOO_FAST  | 0.508 | -0.498 | -0.587 | **-0.090** | 0.452 | 0.00 | 0.140 |
| h20_d0  | 303  | TOO_FAST  | 0.504 | -0.506 | -0.542 | -0.036 | 0.487 | 0.00 | 0.093 |
| h20_d0  | 404  | LANDED    | 0.378 | -0.375 | -0.500 | -0.125 | 0.362 | 37.75 | 0.081 |
| h25_d3  | 1234 | LANDED    | 0.492 | -0.488 | -0.482 | +0.007 | 0.874 | 0.00 | 0.130 |
| h25_d3  | 101  | TOO_FAST  | 0.509 | -0.480 | -0.211 | **+0.269** | 0.475 | 25.50 | 0.103 |
| h25_d3  | 202  | LANDED    | 0.421 | -0.411 | -0.309 | +0.102 | 0.452 | 0.00 | 0.065 |
| h25_d3  | 303  | LANDED    | 0.378 | -0.358 | -0.267 | +0.091 | 0.487 | 0.00 | 0.093 |
| h25_d3  | 404  | LANDED    | 0.431 | -0.432 | -0.449 | -0.017 | 0.378 | 52.50 | 0.067 |
| h40_d6  | 1234 | LANDED    | 0.439 | -0.432 | -0.441 | -0.009 | 0.874 | 0.00 | 0.107 |
| h40_d6  | 101  | LANDED    | 0.346 | -0.331 | -0.328 | +0.004 | 0.475 | 25.50 | 0.090 |
| h40_d6  | 202  | LANDED    | 0.497 | -0.490 | -0.550 | -0.061 | 0.452 | 0.00 | 0.128 |
| h40_d6  | 303  | TOO_FAST  | 0.543 | -0.525 | -0.303 | **+0.222** | 0.487 | 0.00 | 0.122 |
| h40_d6  | 404  | TOO_FAST  | 0.522 | -0.507 | -0.326 | **+0.181** | 0.433 | 65.35 | 0.101 |
| h30_d2y | 1234 | TOO_FAST  | 0.522 | -0.496 | -0.173 | **+0.323** | 0.874 | 0.00 | 0.075 |
| h30_d2y | 101  | LANDED    | 0.391 | -0.375 | -0.370 | +0.005 | 0.708 | 0.55 | 0.086 |
| h30_d2y | 202  | LANDED    | 0.479 | -0.480 | -0.497 | -0.017 | 0.552 | 0.65 | 0.081 |
| h30_d2y | 303  | LANDED    | 0.499 | -0.447 | -0.364 | +0.083 | 0.487 | 0.00 | 0.117 |
| h30_d2y | 404  | LANDED    | 0.486 | -0.483 | -0.427 | +0.057 | 0.598 | 0.20 | 0.086 |
| h60_d10 | 1234 | TOO_FAST  | 0.564 | -0.566 | -0.518 | +0.048 | 0.874 | 0.00 | 0.087 |
| h60_d10 | 101  | LANDED    | 0.480 | -0.457 | -0.553 | -0.096 | 0.474 | 25.50 | 0.090 |
| h60_d10 | 202  | LANDED    | 0.459 | -0.445 | -0.424 | +0.022 | 0.452 | 0.00 | 0.094 |
| h60_d10 | 303  | LANDED    | 0.473 | -0.465 | -0.458 | +0.007 | 0.487 | 0.00 | 0.100 |
| h60_d10 | 404  | LANDED    | 0.483 | -0.485 | -0.520 | -0.035 | 0.433 | 65.35 | 0.069 |

**Zusammenfassung:**
| Gruppe | n | v_end Mittel (Spanne) | Δ v_z Mittel (Spanne) | dev 2 s Mittel | max|dev| Mittel |
|---|---|---|---|---|---|
| LANDED | 17 | 0.443 (0.346–0.499) | +0.007 (−0.125 … +0.102) | 0.096 | 0.554 |
| TOO_FAST | 8 | 0.532 (0.504–0.588) | +0.119 (−0.090 … +0.323) | 0.107 | 0.570 |

## 3. Wann und wie stark weicht die Schätzung ab? (gemessen)

- **Abweichung von > 0.05 m/s existiert praktisch vom ersten Zeitschritt an** (`first_dev_gt_0p05 = 0.0 s` in 7 von 8 Fehlschlägen). Das ist kein Lag, sondern der **Kaltstart**: Der Filter startet bei 0 und übernimmt erst die erste (verrauschte) Messung – die größte Absolutabweichung liegt daher in 5 von 8 Fehlschlägen bei `t ≈ 0` und ist eine einzelne, rauschdeterminierte Stichprobe (max|dev| bis 0.87 m/s; ein ~3σ-Ausreißer des Vel-Rauschens, unabhängig vom Flug).
- **Zweitgrößte Abweichungen bei der Sinkraten-Rampe** großer Höhenflüge: `t_max_dev ≈ 25.5 s` (Seed 101) bzw. `65.35 s` (h40_d6/h60_d10, Seed 404). Auch das liegt weit vor dem Bodenkontakt (t_contact 46–147 s).
- **In den letzten 2 s vor dem Aufsetzen** pendelt die Schätzung im Mittel um `0.07–0.14 m/s` um die Wahrheit – kleiner als die Spitzen, aber dies ist die entscheidende Phase.
- **Beim Bodenkontakt selbst** liegt der systematische Offset `Δ vz` zwischen `−0.125 und +0.323 m/s` über alle Versuche; bei Fehlschlägen im Mittel `+0.119 m/s` (max. `+0.323`), bei Erfolgen im Mittel `+0.007 m/s`.

**Korrelationsanalyse (gemessen):** `corr(v_end, Δ vz) = 0.28`, `corr(v_end, dev 2 s) = 0.30`, `corr(v_end, max|dev|) = 0.07`. Die Reaktion auf den Bodenkontakt (dev 2 s, Δ vz) ist mit der Aufsetzgeschwindigkeit schwach positiv verknüpft; die globale Spitzenabweichung (Kaltstart/Rampe) hat praktisch keinen Zusammenhang mit dem Ausgang.

## 4. Gegenüberstellung nach Szenario und Seed

**Nach Szenario (Fehlschlagsquote):**
| Szenario | Fehlschläge / 5 Seeds | Charakter |
|---|---|---|
| h20_d0 (kurzer Sinkflug, ~46 s) | **3/5** (101, 202, 303)| kleiner, gemischter Δ (neg. und pos.) |
| h40_d6 | 2/5 (303, 404) | positives Δ (+0.18 … +0.22) |
| h25_d3 | 1/5 (101) | deutlich positives Δ (+0.269) |
| h30_d2y | 1/5 (1234) | größtes positives Δ (+0.323) |
| h60_d10 | 1/5 (1234) | kleines positives Δ (+0.048) |

Kurz zusammengefasst: **Der kürzeste Sinkflug (h20_d0) ist am fehleranfälligsten – mit gemischtem Vorzeichen der Schätzung.** Die Fehlschläge der höheren/steileren Szenarien tragen dagegen fast durchweg ein **positives** Δ (Schätzung meldet weniger Sinkrate als vorhanden → zu spät gebremst). Die Zielabstände aller Fehlschläge sind klein (0.03–0.31 m) – es ist **kein** Zielproblem, sondern ausschließlich ein Geschwindigkeitsüberschreiten (Definition von `TOO_FAST`).

**Nach Seed:** Seed 101 → 2 Fehlschläge (inkl. dem schnellsten Aufsetzen 0.588 m/s), Seed 1234 → 2, Seed 303 → 2, Seed 202/404 → je 1. Kein einzelner Seed ist „kaputt“; die Fehlschläge verteilen sich über alle Seeds (stochastisch bedingt, deterministisch reproduzierbar).

## 5. Messwerte – plausible Erklärungen – unbewiesene Vermutungen

**Gemessen (Fakten aus CSV/Zeitreihen):**
1. 8/25 `estimation`-Versuche = `TOO_FAST`, keiner = `FAILED`; `ideal`/`noise`/`dropout` 25/25 erfolgreich.
2. Alle Fehlschläge liegen knapp über dem Limit: `v_end ∈ [0.504, 0.588] m/s` (LANDED-Maximum 0.499). Es sind marginale Schwellwertüberschreitungen, keine Abstürze; alle mit gültigem Bodenkontakt am Ziel.
3. `Δ vz` beim Kontakt: Fehlschläge im Mittel +0.119 (max +0.323), Erfolge im Mittel +0.007. 5/8 Fehlschläge haben Δ > +0.05; 2 der 8 haben negatives Δ (−0.090, −0.036).
4. `max|dev|` bis 0.87 m/s tritt fast nie beim Kontakt auf, sondern beim Kaltstart (t=0) oder der Sinkraten-Rampe und korreliert nicht mit v_end (r=0.07).
5. `dev 2 s` (kurz vor Kontakt): Fehlschläge 0.107 vs. Erfolge 0.096 (kleiner Unterschied).
6. Szenario-Pattern: h20_d0 (kürzester Flug) 3/5 Fehlschläge; höhere Szenarien 1–2/5.
7. Gruppenvergleich aus dem Statistiklauf: `noise` (kein Filter) landet 25/25 mit v_end-Mittel 0.351; `estimation` (mit Filter) hat v_end-Mittel 0.471 (inkl. Fehlschläge bis 0.588). Das Filtern verschiebt die touchdown-Verteilung spürbar Richtung Limit.

**Plausible Erklärungen (durch Messung gestützt, aber nicht strikt bewiesen):**
- **Lag der Endphase:** Bei positiven Δ (5/8 Fehlschläge) meldet die Schätzung kurz vor dem Aufsetzen eine geringere Sinkrate, als wirklich vorliegt → Bremsbefehl kommt etwas zu spät → v_end über 0.5. Gestützt durch: deutlich höheres mittleres Δ der Fehlschläge, die Fallbeispiele h30_d2y (+0.32) und h25_d3 (+0.27) mit den größten Δ, und die schwache Korrelation Δ↔v_end.
- **Schwellwerteffekt + Rauschen:** Da alle Fehlschläge dicht am 0.5-Limit liegen, genügen kleine Zufallsschwankungen im Regelkreis, um einzelne Läufe über die Grenze zu kippen – gerade bei den beiden Fehlschlägen mit negativem Δ, wo die wahre Sinkrate selbst (−0.50 … −0.59) schon nahe/über dem Soll liegt.
- **Kurze Sinkflüge sind kritischer (h20_d0):** wenig Zeit zum Einschwingen des Filters + kurze Abfangphase → höhere Streuung. Gestützt durch Quote 3/5 und gemischte Δ.

**Noch unbewiesen (Spekulation):**
- Ob der Lag **kausal** und nicht nur assoziiert ist (Korrelationen sind mit 0.28–0.30 schwach; eine kausale Zuordnung ist aus diesen Daten nicht ableitbar).
- Ob eine Änderung des Filterparameters (α), bzw. überhaupt ein anderer Schätzer, die Fehlschläge tatsächlich reduziert – wurde explizit nicht getestet.
- Ob Messrauschen allein (ohne Filter-Lag) unter anderen Rauschleveln dieselben Marginal-Fehler erzeugen würde. Im vorliegenden Setup: `noise` (identisches Rauschen, kein Filter) = 25/25, d. h. rauschinduziert, reicht hier nicht für einen Fehlschlag.

## 6. Drei wichtigste Erkenntnisse

1. **Die Fehlschläge sind marginale Schwellwert-Peaks, kein Versagen der Landung.** Alle 8 `TOO_FAST` liegen bei v_end 0.504–0.588 m/s (alle anderen Modi 0/25), mit gültigem Kontakt am Ziel; das Symptom ist eine um wenige cm/s zu hohe Aufsetzgeschwindigkeit.
2. **Der Schätz-Lag in den letzten zwei Sekunden ist messbar und richtungskonform, aber nicht Ursache-nr. 1 über alle Fälle.** Fehlschläge haben im Mittel Δ vz = +0.119 m/s (Erfolge +0.007) und leicht höheres dev-2-s (0.107 vs. 0.096); die größten Absolutabweichungen (bis 0.87 m/s) treten dagegen beim Kaltstart/der Sink-Rampe auf und korrelieren nicht mit dem Ergebnis (r=0.07).
3. **Das Filtern selbst verschiebt die Verteilung Richtung Limit, und kurze Sinkflüge sind am fehleranfälligsten.** Mit ungefiltertem Rauschen (`noise`) 25/25 bei mittlerem v_end 0.351; mit dem Filter steigt das mittlere v_end auf 0.471 (Fehlschläge bis 0.588), und der kürzeste Flug (h20_d0) hat die höchste Fehlschlagsquote (3/5) mit beiden Δ-Vorzeichen – d. h. dort dominiert Streuung, nicht Lag.

## 7. Empfehlung: genau ein nächster Schritt

**Hypothesen-Test durch kontrollierten, rein virtuellen Zeitkonstanten-Sweep (kein Controller-/Physik-Eingriff):** Führe mit denselben Szenarien und Seeds (COMPARISON_SCENARIOS × DEFAULT_SEEDS) und derselben Statistik einen `alpha`-Sweep des vorhandenen `LowPassEstimator` durch (z. B. `α ∈ {0.1, 0.25, 0.5, 1.0}`; α=1 = keine Filterung). Prüfe zwei klare Vorhersagen der Lag-Hypothese: (a) die `TOO_FAST`-Quote sinkt monoton mit steigendem α, und (b) das mittlere `Δ vz` beim Kontakt schrumpft entsprechend. Erst dieses Ergebnis (Messung, keine Optimierung) entscheidet, ob ein Filter-Tuning überhaupt sinnvoll ist – und würde gleichzeitig die Frage „Lag vs. Rauschen/Schwellwert“ für kurze Sinkflüge (h20_d0) auflösen.

---
*Rein virtuelle Lernsimulation (Punktmasse). Keine Hardware, keine reale Regelung. Keine Codeänderung – nur Analyse.*