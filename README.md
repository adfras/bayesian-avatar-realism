# Bayesian Avatar Realism — Study 2 → Study 3 (Pyro/PyTorch)

This repository reproduces the analyses for two within‑subject VR experiments (Study 2 and Study 3) on avatar **realism** and **enjoyment/quality** ratings. We fit a Bayesian hierarchical **ordinal** regression to Study 3 using **Pyro** (NUTS/HMC on **PyTorch**), with **Study 2‑informed hyperpriors** to test whether individual differences (random intercepts and slopes) generalize across studies. We compare these Bayesian results against standard **ANOVAs** and quantify **information gain** at the participant‑level random effects. A complementary continuous (Student‑t) model computes **Cohen’s d** style standardized contrasts.

---

## Data

- **study2_long.csv** — long‑format Study 2 data: required columns include at least
  `participant_code`, `avatar_type`, `disclosure_sentiment`, `real_person_rating`, `enjoyment_rating`.
- **study3_long.csv** — long‑format Study 3 data: required columns include at least
  `participant_code`, `avatar` (Sync/Unreal or equivalent), `disclosure_sentiment` (Positive/Negative),
  `real_person_rating`, `enjoyment_rating`.
- Column names are cleaned to lowercase underscore format automatically (see `data_utils.py`).

> Tip: The ordinal model expects ratings on a 1–5 scale; invalid/missing values are dropped by the loader.

## Environment

```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate   # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` includes: numpy, pandas, matplotlib, arviz, xarray, pyro-ppl, torch, statsmodels.

## Reproducing the pipeline

From the repo root:

```bash
# 1) (Optional) Frequentist baselines for both studies (outputs -> results_anova/)
python run_anovas_study2_study3.py

# 2) Hyperpriors from Study 2 (writes study2_hyperpriors.json)
python phase1_study2_hyperpriors.py

# 3) Bayesian ORDINAL model on Study 3 with Study-2-informed hyperpriors
#    Outputs -> results/: idata, summaries, random-effects info_gain, PPCs, etc.
python phase2_study3_ordinal.py

# 4) Bayesian CONTINUOUS (Student‑t) model on Study 3 to compute Cohen's d
#    Outputs -> results_cont/: idata and info_gain tables
python phase2_study3_continuous.py

# 5) Posterior predictive checks (reads results/ppc_*.csv)
python make_ppc_obs_vs_model.py

# 6) Random‑effects visualizations (reads results/*.nc)
python make_re_plots_random_effects.py

# 7) Contrast plots (Unreal−Sync; Positive−Negative) and publication figures
python make_contrast_plots.py

# 8) Compare ANOVA vs Bayesian information‑gain with figures
python make_info_gain_vs_anova.py
```

Outputs are written under `results/`, `results_cont/`, `results_anova/`, and `figs/`. The ordinal
script creates its folders as needed; if any folder is missing, create it manually.

## Repository layout

- `data_utils.py` – shared loaders/cleaners and coding (e.g., centered contrasts, 1–5 checks)
- `phase1_study2_hyperpriors.py` – estimates hyperpriors from Study 2 and writes `study2_hyperpriors.json`
- `phase2_study3_ordinal.py` – hierarchical cumulative‑logit model (random intercepts and slopes)
- `phase2_study3_continuous.py` – Student‑t LMM for continuous approximation + Cohen’s d
- `run_anovas_study2_study3.py` – within‑subjects ANOVAs and cell means for Studies 2/3
- `make_ppc_obs_vs_model.py` – posterior predictive checks (observed vs simulated)
- `make_re_plots_random_effects.py` – participant random‑effect plots (intercepts, slopes)
- `make_contrast_plots.py` – Unreal−Sync and Positive−Negative contrasts with 89% CrIs
- `make_info_gain_vs_anova.py` – side‑by‑side “bits” from ANOVA vs Bayesian RE information
- `make_plots.py` – additional publication‑quality figures

## Notes

- The models run on CPU by default; **PyTorch** is used as the compute backend and **Pyro** for NUTS.
- Study 3 rating outcomes are treated as **ordinal (1–5)**; the Student‑t model is for effect‑size reporting.
- Information‑gain tables are produced both **by‑parameter** and **by‑participant** for realism and quality.

---

### Short excerpt from the report

```
Bayesian Hierarchical Ordinal Regression for Avatar Realism and Enjoyment (Study 3)
Overview and Data Sources
In this session, we built a Bayesian hierarchical ordinal regression model to analyze Study 3, focusing on two key outcome measures: perceived avatar realism and conversation enjoyment (conversation quality). The model is hierarchical because it accounts for participant-level variability (each participant has their own baseline and sensitivity to conditions), and it’s ordinal because the ratings are on an ordered Likert scale (e.g. 1–5 stars). Crucially, we incorporated prior information from Study 2 to inform our model. Study 2 was a baseline study where participants rated realism and quality in simpler scenarios; we extracted prior means for the fixed effects from Study 2; random-effect scales were not set from Study 2 (they are learned in Study 3 with Half-Normal(0.5) priors).[1]. By anchoring the model with realistic prior expectations of rating behavior (e.g. reasonable fixedeffect means informed by Study 2), we add stability and interpretability to the analysis. The overall goal of this modeling was to estimate how experimental conditions in Study 3 affected realism and enjoyment ratings while accounting for individual differences.
Prior Analyses and the Need for Information Gain: Prior work analyzing these data with traditional ANOVA confirmed the presence of significant main effects of avatar realism and disclosure valence on both outcome measures, but such me
```

If you prefer not to include any report excerpt in the README, delete the section above.
