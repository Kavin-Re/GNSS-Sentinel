# GNSS-Sentinel

A **defensive, passive, simulation-validated** toolkit that flags GNSS interference and spoofing *indicators* from
receiver-level measurements (C/N0, AGC, noise floor, satellite count, clock bias, position/velocity, residuals, multipath).
It reads logs; it never transmits, and it contains no signal-generation or RF code. The test-data generator emits abstract
feature arrays only (numbers per epoch), not waveforms or real-frequency signals.

## Limits (read first)
* **All quantitative results are on SYNTHETIC data** from this repo's own generator. Thresholds and the classifier were fitted
  on the same generator family (different seeds), so the numbers are optimistic. A "shifted" noisier profile is included to show
  the sensitivity: false alarms rise from ~1.1/h to ~7/h (fused) and classifier accuracy falls from 0.95 to 0.83.
* Not validated on any real receiver or recorded interference. Real AGC/C/N0 reporting, antennas, multipath and clock steering differ.
* Scenario severity ranges are the author's choices; weak cases (small clock steps, slow drag, weak CW) are undetectable by design.
  Slow position drag is the weakest case (fused detection 0.74 nominal, 0.82 shifted, mostly via CUSUM that assumes Doppler velocity is not dragged too).
* A spoofer that keeps per-satellite power structure, geometry and clock consistent is not modelled and would evade these detectors.
* Alerts are indicators, not attribution. The softmax "hint" is a convenience, not evidence.
* NavIC/IRNSS-specific behaviour is **not modelled**. NMEA parsing accepts any 2-letter talker (incl. GI) by sentence type only.
* First 60 epochs are treated as clean baseline.

## Layout
`gnss_sentinel/` synth.py (scenarios), detectors.py, pipeline.py (streaming, hysteresis), classifier.py, logio.py (CSV/NMEA),
cli.py, dashboard.py, evaluate.py. `tests/` unittest/pytest. `results/` results.md + results.json. `examples/` generated logs + dashboard.

## Scenarios (labelled)
clean, jam_ramp (wideband jamming ramp), cw (narrowband CW), meacon (meaconing-style common delay), spoof_uniform
(all-satellites-same-power), drag (slow position drag). Generator also has a sat_fault option used for RAIM testing.

## Detectors
1. cn0_agc: C/N0 drop (median, worst-3) plus AGC and noise-floor rise vs baseline.
2. uniformity: C/N0 spread collapse / elevation-correlation loss (all satellites at same power).
3. clock: clock-bias increment jump vs recent drift.
4. raim: residual chi-square consistency (weighted), with largest-residual satellite named.
5. innovation: Kalman-filter innovation monitor on position/velocity.
6. cusum: Page test on position motion not explained by Doppler velocity.
Each uses a hysteresis latch. Thresholds = 99th percentile of per-run max score on 200 clean synthetic runs (`default_thresholds.json`).

## Usage
```
python -m gnss_sentinel gen meacon -o m.csv [--nmea m.nmea] --seed 11 [--profile shifted]
python -m gnss_sentinel stream m.csv [--dashboard out.html] [--quiet] [--no-classifier]   # '-' = stdin; --format csv|nmea|auto
python -m gnss_sentinel dashboard m.csv -o out.html     # offline single-file HTML, alert timeline, no network
python -m gnss_sentinel evaluate [--quick] --out results  # full run ~15 min
python -m pytest -q
```
Requires Python 3 and numpy only.

## CSV format
Header-driven, '#' comments, empty cell = missing. Columns: `t,n_sat,agc_db,noise_db,clock_m,e_m,n_m,u_m,ve_mps,vn_mps,vu_mps,label`
then per slot k=0..15: `prn{k},el{k},az{k},cn0{k},res{k},mp{k}`. `label` is optional ground truth, never read by detectors.
Detectors whose input fields are absent report "n/a" (e.g. NMEA has no AGC, clock, residuals or multipath).

## NMEA
GGA, RMC, GSV only (checksums verified; optional NMEA 4.10 signal-ID field accepted). Round-trip tested on the synthetic generator only;
untested against real receiver logs. Position/velocity resolution is limited by NMEA decimals (~0.19 m, ~0.05 m/s).

## Evaluation
`results/results.md`: ROC AUC, detection probability and median delay per scenario x detector (ablation + leave-one-out), false-alarm
rate per hour, Wilson / exact-Poisson / bootstrap 95% CIs, classifier confusion matrices, nominal and shifted profiles.
