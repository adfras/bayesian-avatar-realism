# Bayesian Avatar Realism

This repository contains the Bayesian analysis pipeline that we used to study how disclosure style and avatar fidelity affect perceived realism and quality ratings in our Study 2 → Study 3 replication programme.  The current code base centres on Pyro/PyTorch models with Study‑2 informed hyperpriors and a battery of post-processing utilities for the Study 3 replication.  Legacy R scripts that powered the original prototype are still shipped for reference, but the supported workflow is fully Python based.

## Repository layout

```
├── data_utils.py                 # Shared helpers to clean/load Study 2 & 3 CSV files
├── phase1_study2_hyperpriors.py  # Phase 1: derive Study 2 hyperpriors for Study 3 models
├── phase2_study3_ordinal.py      # Phase 2 (main): ordinal regression for Study 3 Likert outcomes
├── phase2_study3_continuous.py   # Phase 2 (optional): Student-t LMM on composite 1..5 outcomes
├── make_*.py                     # Figure/report generators fed by Phase 2 results
├── reports/                      # CSV/TXT summaries from the legacy R pipeline (read-only)
├── R/                            # Archived tidyverse/brms implementation of the pipeline
└── scripts/run_pipeline.sh       # Wrapper that targets the legacy R pipeline (not maintained)
```

## Data requirements

The Pyro workflow expects the long-format CSV exports from both studies in the repository root:

- `study2_long.csv`
- `study3_long.csv`

Headers are normalised automatically (case-insensitive, punctuation stripped).  Each file must contain:

- `participant_code`
- `avatar_type`
- `disclosure_sentiment`
- `real_person_rating`
- `enjoyment_rating`

Study 3 files may additionally include item-level realism/quality questions if you plan to run the continuous composite models (`phase2_study3_continuous.py`).

## Environment setup

Use Python 3.9+ (PyTorch double precision is required).  Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you need to revisit the archived brms implementation you will also need a working R toolchain, but that is optional for the supported workflow.

## Running the Pyro pipeline

1. **Derive Study 2 hyperpriors** (Phase 1):

   ```bash
   python phase1_study2_hyperpriors.py
   ```

   This fits Gaussian mixed models to Study 2 realism/quality ratings and stores the resulting hyperpriors in `study2_hyperpriors.json`.

2. **Fit Study 3 ordinal models** (Phase 2 main analysis):

   ```bash
   python phase2_study3_ordinal.py            # defaults to Study-2 informed priors
   python phase2_study3_ordinal.py --prior weak  # optional sensitivity run
   ```

   These runs produce posterior draws (`results/study3_*_idata.nc`), fixed-effect summaries, condition-level probabilities for ratings ≥ 4, posterior predictive checks, and random-effect information gain tables under `results/`.

3. **(Optional) Continuous composite models** for realism/quality Likert composites:

   ```bash
   python phase2_study3_continuous.py
   ```

   Outputs are written to `results_cont/` and mirror the ordinal reports (effects, condition means, information gain, etc.).

4. **Generate publication figures/tables** once the Phase 2 results exist:

   ```bash
   python make_plots.py                             # publication-quality forest/condition/PPC plots
   python make_prior_vs_posterior_predictive.py     # prior vs posterior predictive comparison
   python make_ppc_obs_vs_model.py                  # overlays observed vs simulated category shares
   python make_hyperprior_transfer_plots.py         # diagnostics for Study-2 → Study-3 transfer
   python make_info_gain_vs_anova.py                # compare Bayesian info gain with ANOVA ηp²
   python make_contrast_plots.py                    # summarise posterior contrasts/effect sizes
   ```

   Each script writes PNG/PDF figures to `figs/` (created on demand) and, where relevant, additional CSV summaries under `results/`.

## Outputs

- `study2_hyperpriors.json` — Phase 1 hyperprior definitions consumed by Study 3 models.
- `results/` — Ordinal-model outputs: posterior draws (`*.nc`), effect tables, condition probabilities, posterior predictive checks, and participant-level information gain summaries.
- `results_cont/` — Continuous-model counterparts when the optional phase is executed.
- `figs/` — Figures used in manuscripts and supplementary materials.
- `reports/` — Historical brms/ANOVA outputs kept for provenance.

## Legacy R pipeline

The `R/` folder and `scripts/run_pipeline.sh` contain the earlier tidyverse + brms pipeline.  They are preserved for transparency but are not exercised by the current automation; Pyro is the reference implementation.

## Data handling & privacy

All Study 2/3 datasets must be anonymised prior to committing.  The repository does not include participant-identifying information, and any derivative outputs inherit that expectation.
