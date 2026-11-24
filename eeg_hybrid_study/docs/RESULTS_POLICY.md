# Results Policy

One complete preliminary seed-11 run is committed in `reported_runs/seed11_preliminary` for transparency. A single seed, a partial run, or a mixture of outputs from the older audit repository must not be reported as the final result of this protocol.

To create a reportable result set:

1. Complete all predeclared models for seeds 11, 23, and 37.
2. Confirm every run has `run_info.json` showing `scaling_scope: fit_on_inner_training_fold`.
3. Run `scripts/analyze_study_results.py` and retain its per-seed and aggregate tables.
4. For graph claims, complete the topology ablation with the same seeds and optimizer settings.
5. State that the primary split is trial-random and not participant-independent.

The historical `cnnlstmeeg` repository remains the preserved audited record for its earlier protocol and results.
