# Bayesian Avatar Realism — Study 2 → Study 3 (Pyro/PyTorch)

This repository reproduces the analyses for two within‑subject VR experiments (Study 2 and Study 3) on avatar **realism** and **enjoyment/quality** ratings. We fit a Bayesian hierarchical **ordinal** regression to Study 3 using **Pyro** (NUTS/HMC on **PyTorch**), with **Study 2‑informed hyperpriors** to test whether individual differences (random intercepts and slopes) generalize across studies. We compare these Bayesian results against standard **ANOVAs** and quantify **information gain** at the participant‑level random effects. A complementary Student‑t model computes **Cohen’s d**‑style standardized contrasts.

---

## Data

- **study2_long.csv** — long‑format Study 2 data: must include at least `participant_code`, `avatar_type`, `disclosure_sentiment`, `real_person_rating`, `enjoyment_rating`.
- **study3_long.csv** — long‑format Study 3 data: must include at least `participant_code`, `avatar` (Sync/Unreal or equivalent), `disclosure_sentiment` (Positive/Negative), `real_person_rating`, `enjoyment_rating`.
- Column names are auto‑cleaned to lowercase underscore format and coding columns are constructed (e.g., `avatar_c = −0.5 / +0.5`) in **`data_utils.py`**.
- Ratings are expected on a **1–5** scale; invalid/missing values are dropped.

## Environment

```bash
# Python 3.10+ recommended
python -m venv .venv
# Windows (cmd): .venv\Scripts\activate.bat
# PowerShell:    .venv\Scripts\Activate.ps1
# macOS/Linux:   source .venv/bin/activate
pip install -r requirements.txt
```

## Repro pipeline (commands)

From the repo root:

```bash
# 1) Frequentist baselines (Study 2 & 3); outputs -> results_anova/
python run_anovas_study2_study3.py

# 2) Phase‑1: Fit Study‑2 LMM to derive hyperpriors; writes study2_hyperpriors.json
python phase1_study2_hyperpriors.py

# 3) Phase‑2 (ORDINAL): Hierarchical cumulative‑logit for Study‑3 (uses hyperpriors)
#    Outputs -> results/: idata (.nc), summaries, random‑effects info‑gain, PPCs, contrasts
python phase2_study3_ordinal.py

# 4) Phase‑2 (CONTINUOUS): Student‑t LMM for effect sizes (Cohen's d) and checks
#    Outputs -> results_cont/: idata (.nc), effect_size_d_*.csv, info‑gain tables
python phase2_study3_continuous.py

# 5) Posterior predictive checks: observed vs model (reads results/ppc_*.csv)
python make_ppc_obs_vs_model.py

# 6) Participant random‑effects figures (reads results/*.nc)
python make_re_plots_random_effects.py

# 7) Contrast plots (Unreal−Sync; Positive−Negative) with 89% CrIs
python make_contrast_plots.py

# 8) Compare ANOVA vs Bayesian information‑gain (“bits”)
python make_info_gain_vs_anova.py

# 9) NEW: Hyperprior transfer plots — overlay Study‑2‑informed priors vs Study‑3 posteriors (σ)
python make_hyperprior_transfer_plots.py

# 10) NEW: Prior vs posterior predictive for a NEW participant (P(rating ≥ 4) by condition)
python make_prior_vs_posterior_predictive.py
```

## What each script produces

- **`phase1_study2_hyperpriors.py`** → `study2_hyperpriors.json` (prior centers for β and LogNormal params for σ/SDs), derived from a hierarchical Gaussian LMM on Study 2.
- **`phase2_study3_ordinal.py`** → under `results/` per outcome (`realism`, `quality`):
  - `study3_{tag}_idata.nc` (ArviZ InferenceData), `summary_{tag}.csv` (posterior summaries),
  - `effects_{tag}.csv` (main/interaction contrasts), `cond_probs_{tag}_*.csv` (category probabilities),
  - `ppc_{tag}.csv` (posterior predictive), `trace_{tag}.png` (diagnostics),
  - `info_gain_random_effects_{tag}_by_pid.csv` and `_by_param.csv` (participant‑level “bits”).  
- **`phase2_study3_continuous.py`** → under `results_cont/`:
  - `study3_cont_{tag}_idata.nc` (InferenceData),
  - `effect_size_d_{tag}.csv` (Cohen’s d for avatar, valence, interaction),
  - `info_gain_cont_random_{tag}_by_pid.csv` and `_by_param.csv`.
- **`run_anovas_study2_study3.py`** → under `results_anova/`:
  - `study{2|3}_anova_{realism3|quality3}.csv`, `study{2|3}_cellmeans_{...}.csv`, and `study3_anova_all.csv`.
- **`make_ppc_obs_vs_model.py`** → uses `results/ppc_*.csv` to create observed vs simulated PPC plots.
- **`make_re_plots_random_effects.py`** → plots participant random intercepts/slopes from `results/*.nc`.
- **`make_contrast_plots.py`** → clean contrast figures (Unreal−Sync; Positive−Negative) with 89% CrIs.
- **`make_info_gain_vs_anova.py`** → side‑by‑side figures/tables comparing ANOVA vs Bayesian information‑gain.
- **NEW `make_hyperprior_transfer_plots.py`** → saves
  - `figs/hyperprior_vs_posterior_sigma_realism.png` and `..._quality.png` by overlaying the Study‑2–informed **LogNormal** priors for σ with Study‑3 posteriors. Inputs: `study2_hyperpriors.json` and `results/study3_{tag}_idata.nc`.
- **NEW `make_prior_vs_posterior_predictive.py`** → saves
  - `figs/prior_vs_posterior_new_participant_realism.png` and `..._quality.png`, comparing **prior‑predictive** vs **posterior‑predictive** P(rating ≥ 4) for a **new participant** across the four condition cells (First/Third × Negative/Positive). Inputs: `results/study3_{tag}_idata.nc`.

## Repository layout (key files)

- `data_utils.py` – shared loaders/cleaners and contrast coding (centered −0.5 / +0.5).
- `phase1_study2_hyperpriors.py` – derive Study‑2 hyperpriors (writes `study2_hyperpriors.json`).
- `phase2_study3_ordinal.py` – hierarchical cumulative‑logit model for Study 3 with Study‑2‑informed priors.
- `phase2_study3_continuous.py` – Student‑t LMM for effect sizes and info‑gain cross‑checks.
- Plotters: `make_ppc_obs_vs_model.py`, `make_re_plots_random_effects.py`, `make_contrast_plots.py`,
  `make_info_gain_vs_anova.py`, **`make_hyperprior_transfer_plots.py`**, **`make_prior_vs_posterior_predictive.py`**.

## Outputs & folders

- `results/` — ordinal model artifacts (NetCDF, CSVs, PNGs)
- `results_cont/` — continuous model artifacts (NetCDF, CSVs)
- `results_anova/` — frequentist baselines
- `figs/` — publication‑quality figures

> Ensure these folders are **git‑ignored** (see `.gitignore`) so large artifacts don’t get committed.

---

If you want the README to include a short excerpt from the report, I can paste the abstract/overview section verbatim on request.
