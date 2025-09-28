# Bayesian Avatar Realism (Study 2 → Study 3)

This repository contains the Study 2 / Study 3 replication pipeline used to analyse avatar realism and quality ratings. The current codebase is fully R-based (tidyverse + brms) and replaces the earlier Pyro/PyTorch prototype that lived in this repository.

## Data

Place the long-format CSVs under `data/` (headers are normalised by the scripts):

- `data/study2_long.csv`
- `data/study3_long.csv`

Required columns (case-insensitive, truncated headers tolerated):

- `participant_code`
- `avatar_type` (Study 2) / `avatar` (Study 3)
- `disclosure_sentiment`
- `real_person_rating`, `enjoyment_rating`

## Environment setup

```bash
# Install R dependencies (CmdStan suggested; rstan fallback)
Rscript R/00_setup.R
```

## Running the pipeline

Each numbered script can be executed independently or orchestrated through `R/run_pipeline.R`.

1. `R/01_priors_from_study2.R`
   - Fits Study 2 Gaussian mixed models to derive priors and saves them in `models/prior_config.json`.
2. `R/02_anova_study3.R`
   - Classical ANOVA baselines for Study 3.
3. `R/03_bayes_linear_study3.R`
   - Study 3 Bayesian mixed models (quality + realism composites).
4. `R/04_information_gain_study3.R`
   - Computes per-participant KL-based information gain (bits) for random effects.
5. `R/05_effect_sizes.R`
   - Summarises posterior contrasts and effect sizes.
6. `R/01_effect_convergence.R`
   - Builds the convergence table (ηp², d statistics).

Helper:

```bash
# Runs selected steps (auto-runs prior derivation if needed)
Rscript R/run_pipeline.R 3 2 4 1 5
```

## Outputs

Key outputs are written to `models/` (prior + fitted model objects) and `reports/` (CSV/TXT summaries):

- `reports/anova_study3_{quality,realism}.{csv,txt}`
- `reports/bayes_linear_study3_{quality,realism}.csv`
- `reports/effect_convergence_study3.csv`
- `reports/effect_information_gain_bits_summary.csv`
- `reports/info_gain_study3_{quality,realism}.csv`
- `reports/info_gain_summary.csv`
- `reports/effect_size_summary.csv`

## Scripts

`AGENT.md` documents research context and agent prompts. The `scripts/run_pipeline.sh` helper wraps the numbered R scripts for quick runs.

## License

See `AGENT.md` for usage constraints. All Study 2/3 data files are expected to be anonymised prior to committing.
