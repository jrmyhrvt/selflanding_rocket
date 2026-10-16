# Estimator Comparison: Reducing Lag Without Unnecessary Noise Loss

**Scope:** Virtual learning simulation only. No controller change, no physics change. Same scenarios and seeds as in sensor experiment (5 scenarios × 5 seeds = 25 per variant). Evaluated against estimation mode (noisy+dropout + estimator) with existing behavior.

## Current Implementation
- `LowPassEstimator(alpha_pos, alpha_vel)`, default `(0.25, 0.25)`: `x ← x + alpha*(z-x)`, `None` (missing) → keep last estimate, `alpha in (0,1]`, `alpha=1` = passthrough.
- With `alpha=0.25`: 17/25 LANDED, 8/25 TOO_FAST (diagnosed: end-phase lag Δ vz positive in 5/8 failures). Est RMS pos/vel 0.802/0.212.
- Noise-only (`noise`, no estimator): 25/25 LANDED, but larger RMS vs filtered.

## Experimental Comparison of Variants

Tested higher alpha (less smoothing, less lag):
- `alpha=0.35`: 19/25 LANDED (+2), 6/25 TOO_FAST. estRMS pos 0.924 (+15%), vel 0.250 (+18%). v_end mean/max 0.455/0.570 (vs 0.471/0.588).
- `alpha=0.5`:  21/25 LANDED (+4), 4/25 TOO_FAST. estRMS pos 1.106 (+38%), vel 0.308 (+45%). v_end 0.436/0.569.
- `alpha=0.7`:  25/25 LANDED (+8), 0/25 TOO_FAST. estRMS pos 1.357 (+69%), vel 0.386 (+82%). v_end 0.391/0.484. Δ vz at contact still present but not enough to exceed v_tol in these runs.
- `alpha=1.0`:  25/25 LANDED, 0/25 TOO_FAST (passthrough). estRMS pos 1.794 (+124%), vel 0.523 (+147%). v_end 0.336/0.444 (closest to noise mode behavior).

Failure details for remaining cases: at higher alpha, end-phase lag (Δ vz) becomes smaller in magnitude on average but the estimate tracks measurements more closely (higher RMS). Missing-sample handling remains correct (samples_missing tracked; filter skips `None`).

## Observations
- Increasing alpha reduces lag → reduces TOO_FAST count (8→6→4→0). 
- Cost: RMS error increases (less smoothing). Trade-off is expected.
- alpha=0.7 eliminates TOO_FAST on this test set while keeping some smoothing relative to passthrough.
- No controller/physics change. Same scenarios/seeds reproduce previous baseline.

## Recommendation
Keep existing default `alpha=0.25` (documented baseline). Do **not** auto-replace it. If reducing lag is desired later, `alpha=0.7` is the smallest tested increase that removes TOO_FAST on these cases but increases est RMS substantially; this is a tuning choice (noise vs lag), not obviously “better” in general.

## Limitations
- Only 5 scenarios × 5 seeds (virtual). Results may differ under other noise/dropout params.
- Only first-order LPF tested (simple). No delay modeled; no IMU fusion.
- alpha-sweep is experimental; no optimization performed.

