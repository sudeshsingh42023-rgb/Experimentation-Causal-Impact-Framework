# Project 2: Experimentation & Causal-Impact Framework for Product Growth
Target role: Data Scientist Associate, JPMorganChase (Chase UK, Business Analytics), Job ID 210788748

All data is **synthetic** (seeded). True effects are known by construction so estimators can be validated
against ground truth. Economic parameters (NIM, cost per nudge) are assumptions. Say "simulated" in interviews.

**Stack:** Python, SciPy, NumPy, pandas, scikit-learn (propensity scores), matplotlib

## Resume bullets
- Designed and analysed a randomised A/B test (40K customers) of an in-app savings-activation nudge: pre-registered sample size (12,208/arm for an 8% to 9% MDE at 80% power; empirical power 79.5%), SRM check (p=0.87) and an A/A simulation (2,000 runs, 5.4% false positives) to validate the pipeline.
- Measured a +0.76pp activation lift (8.05% to 8.81%, 95% CI 0.21 to 1.30pp, p=0.006); applied CUPED to cut variance of 90-day balance estimates by 38%, and Holm correction across 8 segment cuts (only the under-25 effect, +1.5pp, survived).
- Ran a guardrail non-inferiority test on notification opt-outs (+0.35pp; upper bound 0.63pp vs 0.5pp margin), so non-inferiority was not established and the recommendation was a staged rollout. Unit economics (assumed 2% NIM, £0.05 cost) gave £0.44 incremental annual margin per treated customer (CI 0.13 to 0.77).
- Built a quasi-experimental module for self-selected feature opt-ins: naive comparison overstated impact at £71.8 vs a known £22 true effect; propensity matching (SMD on pre-period spend 0.75 to 0.00) plus DiD recovered £22.9 (CI 22.4 to 23.4). Also showed when not to use it: with diverging pre-trends (£3.2/month, p<0.001) DiD overstated impact by 88%, which the event-study diagnostic flagged.

## Run
```
pip install pandas numpy scipy scikit-learn matplotlib
mkdir -p outputs && python3 experimentation_framework.py
```

## Files
- `experimentation_framework.py` - full code (Part A: A/B test; Part B: matching + DiD)
- `outputs/results_project2.json` - all metrics
- `outputs/segment_effects_holm.csv` - segment lifts with Holm-adjusted p-values
- `outputs/experimentation_diagnostics.png` - CUPED CI, segment lifts, event study

## Caveats (know these for interviews)
- The +0.76pp lift is significant but below the 1pp MDE the test was sized for.
- The opt-out guardrail result is reported as inconclusive rather than hidden.
