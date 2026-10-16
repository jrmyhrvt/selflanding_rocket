# Estimator Robustness Test

## Method
- Compared `LowPassEstimator(alpha=0.25, 0.25)` (default, baseline, unchanged) vs `alpha=0.7, 0.7` (less smoothing, less lag) under 8 sensor configurations:
  - base (pos_std=1.0, vel_std=0.3, dropout=0.1)
  - low_noise (0.5,0.15,0.1)
  - med_noise (1.5,0.45,0.1)
  - high_noise (2.0,0.6,0.1)
  - low_dropout (1.0,0.3,0.05)
  - med_dropout (1.0,0.3,0.15)
  - high_dropout (1.0,0.3,0.25)
  - noisy_highdrop (1.5,0.45,0.2)
- Same scenarios and seeds as sensor experiment: 5 scenarios × 5 seeds = 25 runs per (group, config). Total 400 trials (2×8×25).
- Metrics: success rate (LANDED), TOO_FAST count, est pos/vel RMS (truth vs estimate), missing samples (mean), v_end mean/max.
- No controller/physics change. Purely virtual simulation.

## Results
Robustness summary (see output/estimator_robustness/robust_summary.csv):

| config | alpha=0.25 TOO_FAST/success | alpha=0.7 TOO_FAST/success | est vel RMS (0.25/0.7) |
|---|---|---|---|
| base | 8/0.68 | 0/1.0 | 0.212/0.386 |
| low_noise | 11/0.56 | 2/0.92 | 0.109/0.193 |
| med_noise | 6/0.76 | 0/1.0 | 0.316/0.579 |
| high_noise | 12/0.52 | 1/0.96 | 0.421/0.771 |
| low_dropout | 8/0.68 | 0/1.0 | 0.211/0.385 |
| med_dropout | 9/0.64 | 0/1.0 | 0.213/0.387 |
| high_dropout | 10/0.60 | 1/0.96 | 0.217/0.391 |
| noisy_highdrop | 6/0.76 | 3/0.88 | 0.320/0.582 |

- alpha=0.7 consistently reduces TOO_FAST (often to 0) and raises success rate across configs.
- Cost: est vel RMS increases (less smoothing) — ~1.7–1.8× in most cases (matches expectation). est pos RMS similar trend.
- Under stronger noise (med/high/noisy_highdrop), alpha=0.7 still prevents most TOO_FAST but est RMS grows; one case (noisy_highdrop) left 3/25 TOO_FAST.
- Missing samples tracked correctly (higher dropout -> more missing; alpha doesn’t create/destroy measurements).

## Conclusion
Higher alpha reduces lag-induced TOO_FAST but trades noise suppression for responsiveness. Default alpha=0.25 remains a conservative baseline (lower RMS). No automatic switch recommended.

## Limits
Only first-order LPF, fixed scenarios/seeds, virtual phenomenological noise/dropouts (no delay). Results specific to this setup.
