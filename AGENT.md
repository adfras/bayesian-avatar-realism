# Agent: Study 2 → Priors → Study 3 (Composites + Information Gain; No Ordinal)

## Objective
Create an R pipeline that:
1) Builds **composite outcomes** for Study 2 and Study 3.
2) **Learns priors from Study 2** (Gaussian mixed models).
3) Fits **Study 3** using those priors (Gaussian mixed models).
4) Runs **ANOVA** on Study 3.
5) Computes **information gain in bits** (KL) for participant random effects in Study 3.
6) Aggregates ANOVA + Bayesian effect sizes.

## Project structure to create
```
R/
  00_setup.R
  01_priors_from_study2.R
  02_anova_study3.R
  03_bayes_linear_study3.R
  04_information_gain_study3.R
  01_effect_convergence.R
  run_pipeline.R
data/
models/
reports/
```

## Actions

### A) Create directories
Ensure `R/`, `data/`, `models/`, `reports/` exist.

### B) Create the following R scripts with **exact** contents

#### File: `R/00_setup.R`
```r
#!/usr/bin/env Rscript
options(warn = 1)
seed <- as.integer(Sys.getenv("SEED", "2025"))
set.seed(seed)

need <- c("tidyverse","brms","afex","emmeans","posterior","jsonlite","cmdstanr","rstan")
ensure <- function(pkgs) {
  to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
  if (length(to_install)) install.packages(to_install, repos = "https://cloud.r-project.org")
}
ensure(need)

# Prefer cmdstanr if available
use_cmdstan <- FALSE
suppressWarnings({
  if (requireNamespace("cmdstanr", quietly = TRUE)) {
    library(cmdstanr)
    ver <- tryCatch(cmdstanr::cmdstan_version(), error = function(e) NULL)
    if (is.null(ver)) {
      message("CmdStan not found; attempting installation.")
      try(cmdstanr::install_cmdstan(), silent = TRUE)
      ver <- tryCatch(cmdstanr::cmdstan_version(), error = function(e) NULL)
    }
    use_cmdstan <- !is.null(ver)
  }
})

if (use_cmdstan) {
  message("Using BRMS backend = cmdstanr")
  brms_backend <- "cmdstanr"
} else {
  message("Using BRMS backend = rstan (fallback)")
  brms_backend <- "rstan"
  rstan::rstan_options(auto_write = TRUE)
  options(mc.cores = parallel::detectCores())
}

dir.create("models", showWarnings = FALSE, recursive = TRUE)
writeLines(brms_backend, con = "models/brms_backend.txt")
message("Setup complete.")
```

#### File: `R/01_priors_from_study2.R`
```r
#!/usr/bin/env Rscript
library(tidyverse)
library(brms)
library(posterior)
library(jsonlite)

seed <- as.integer(Sys.getenv("SEED", "2025")); set.seed(seed)
data_dir <- Sys.getenv("DATA_DIR", "data")
backend <- if (file.exists("models/brms_backend.txt")) readLines("models/brms_backend.txt", warn = FALSE) else "rstan"

path <- file.path(data_dir, "study2_long.csv")
if (!file.exists(path)) stop("Missing data file: ", path)

df2 <- readr::read_csv(path, show_col_types = FALSE)

# --- Robust column detection (supports truncated headers) ---
pick <- function(df, candidates, label) {
  present <- intersect(candidates, names(df))
  if (length(present) == 0) stop("Missing expected column for ", label, ". Tried: ", paste(candidates, collapse = ", "))
  present[1]
}

enjoy_nm   <- pick(df2, c("enjoyment_rating","enjoyment","enjoymen"), "enjoyment")
comfort_nm <- pick(df2, c("comfort_rating","comfort","comfort_r"), "comfort")
pleas_nm   <- pick(df2, c("pleasantness_rating","pleasantness","pleasantn"), "pleasantness")

realp_nm   <- pick(df2, c("real_person_rating","real_person","real_pers"), "real-person")
facial_nm  <- pick(df2, c("facial_realism_rating","facial_realism","facial_rea"), "facial realism")
body_nm    <- pick(df2, c("body_realism_rating","body_realism","body_real"), "body realism")

# Build composites (row means)
df2$quality_rating <- rowMeans(df2[, c(enjoy_nm, comfort_nm, pleas_nm)], na.rm = TRUE)
df2$realism_rating <- rowMeans(df2[, c(realp_nm, facial_nm, body_nm)], na.rm = TRUE)

# Participant and factors
if (!"participant_code" %in% names(df2)) stop("Missing participant_code in Study 2")
if (!"avatar_type" %in% names(df2)) stop("Missing avatar_type in Study 2")
if (!"disclosure_sentiment" %in% names(df2)) stop("Missing disclosure_sentiment in Study 2")

df2 <- df2 |>
  mutate(
    participant_code = factor(participant_code),
    avatar_lowhigh = case_when(
      str_detect(tolower(avatar_type), "high|unreal|iclone") ~ "high",
      str_detect(tolower(avatar_type), "low|sync|vive")      ~ "low",
      TRUE ~ tolower(as.character(avatar_type))
    ),
    # Drop any middle level if present (e.g., "medium")
    avatar_lowhigh = ifelse(avatar_lowhigh %in% c("medium","med","mid"), NA, avatar_lowhigh)
  ) |>
  filter(avatar_lowhigh %in% c("low","high")) |>
  mutate(
    avatar_c   = if_else(avatar_lowhigh == "high",  0.5, -0.5),
    sentiment_c = if_else(tolower(disclosure_sentiment) == "positive", 0.5, -0.5)
  )

form <- bf(y ~ 1 + avatar_c + sentiment_c + avatar_c:sentiment_c +
             (1 + avatar_c + sentiment_c | participant_code))

pri_weak <- c(
  prior(normal(0, 2), class = "b"),
  prior(normal(0, 5), class = "Intercept"),
  prior(exponential(1), class = "sigma"),
  prior(exponential(1), class = "sd")
)

fit_study2 <- function(yvar) {
  dat <- df2 |> drop_na(avatar_c, sentiment_c) |> rename(y = all_of(yvar))
  brm(
    formula = form, family = gaussian(),
    data = dat, prior = pri_weak,
    chains = 2, iter = 2000, warmup = 1000, seed = seed,
    backend = backend, refresh = 0
  )
}

message("Fitting Study 2 (Gaussian) for QUALITY composite...")
m2_q <- fit_study2("quality_rating"); saveRDS(m2_q, "models/study2_quality_linear.rds")

message("Fitting Study 2 (Gaussian) for REALISM composite...")
m2_r <- fit_study2("realism_rating"); saveRDS(m2_r, "models/study2_realism_linear.rds")

summarise_priors <- function(fit) {
  dr <- as_draws_df(fit)
  list(
    b_means = list(
      avatar      = mean(dr$b_avatar_c),
      sentiment   = mean(dr$b_sentiment_c),
      interaction = mean(dr$`b_avatar_c:sentiment_c`)
    ),
    b_scales = list(  # adjustable: scales for Normal priors on fixed effects
      avatar = 1.0, sentiment = 1.0, interaction = 1.0
    ),
    sd_lognormal = list(
      intercept = list(mu = mean(log(dr$`sd_participant_code__Intercept`)), sigma = sd(log(dr$`sd_participant_code__Intercept`))),
      avatar    = list(mu = mean(log(dr$`sd_participant_code__avatar_c`)),    sigma = sd(log(dr$`sd_participant_code__avatar_c`))),
      sentiment = list(mu = mean(log(dr$`sd_participant_code__sentiment_c`)), sigma = sd(log(dr$`sd_participant_code__sentiment_c`)))
    ),
    sigma_lognormal = list(
      mu = mean(log(dr$sigma)), sigma = sd(log(dr$sigma))
    )
  )
}

priors <- list(
  quality = summarise_priors(m2_q),
  realism = summarise_priors(m2_r)
)

dir.create("models", showWarnings = FALSE, recursive = TRUE)
write_json(priors, "models/prior_config.json", auto_unbox = TRUE, pretty = TRUE)
message("Wrote Study-2-informed priors to models/prior_config.json")
```

#### File: `R/03_bayes_linear_study3.R`  (step **3**)
```r
#!/usr/bin/env Rscript
library(tidyverse)
library(brms)
library(posterior)
library(jsonlite)

seed <- as.integer(Sys.getenv("SEED", "2025")); set.seed(seed)
data_dir <- Sys.getenv("DATA_DIR", "data")
backend <- if (file.exists("models/brms_backend.txt")) readLines("models/brms_backend.txt", warn = FALSE) else "rstan"

# Auto-run Study 2 priors if missing and data is available
if (!file.exists("models/prior_config.json") && file.exists(file.path(data_dir, "study2_long.csv"))) {
  message("prior_config.json not found. Running R/01_priors_from_study2.R to generate priors...")
  status <- system("Rscript R/01_priors_from_study2.R"); if (status != 0) stop("Failed to generate priors")
}

pri_cfg <- if (file.exists("models/prior_config.json")) jsonlite::read_json("models/prior_config.json", simplifyVector = TRUE) else NULL

df3 <- readr::read_csv(file.path(data_dir, "study3_long.csv"), show_col_types = FALSE)

# --- Robust column detection (supports truncated headers) ---
pick <- function(df, candidates, label) {
  present <- intersect(candidates, names(df))
  if (length(present) == 0) stop("Missing expected column for ", label, ". Tried: ", paste(candidates, collapse = ", "))
  present[1]
}

enjoy_nm   <- pick(df3, c("enjoyment_rating","enjoyment","enjoymen"), "enjoyment")
comfort_nm <- pick(df3, c("comfort_rating","comfort","comfort_r"), "comfort")
pleas_nm   <- pick(df3, c("pleasantness_rating","pleasantness","pleasantn"), "pleasantness")

realp_nm   <- pick(df3, c("real_person_rating","real_person","real_pers"), "real-person")
facial_nm  <- pick(df3, c("facial_realism_rating","facial_realism","facial_rea"), "facial realism")
body_nm    <- pick(df3, c("body_realism_rating","body_realism","body_real"), "body realism")

# Build composites (row means, keep 1–5 scale)
df3$quality_rating <- rowMeans(df3[, c(enjoy_nm, comfort_nm, pleas_nm)], na.rm = TRUE)
df3$realism_rating <- rowMeans(df3[, c(realp_nm, facial_nm, body_nm)], na.rm = TRUE)

df3 <- df3 |>
  mutate(
    participant_code = factor(participant_code),
    avatar_c = if_else(avatar_type == "unreal",  0.5, -0.5),
    sentiment_c = if_else(disclosure_sentiment == "positive", 0.5, -0.5)
  ) |>
  select(participant_code, avatar_c, sentiment_c, quality_rating, realism_rating)

mk_priors <- function(section) {
  if (is.null(pri_cfg)) {
    message("No Study-2 priors found; falling back to weakly-informative priors.")
    return(c(
      prior(normal(0, 2), class = "b"),
      prior(normal(0, 5), class = "Intercept"),
      prior(exponential(1), class = "sigma"),
      prior(exponential(1), class = "sd")
    ))
  }
  bm <- pri_cfg[[section]]$b_means
  bs <- pri_cfg[[section]]$b_scales
  lg <- pri_cfg[[section]]$sd_lognormal
  sg <- pri_cfg[[section]]$sigma_lognormal
  c(
    prior(normal(bm$avatar,      bs$avatar),      class = "b", coef = "avatar_c"),
    prior(normal(bm$sentiment,   bs$sentiment),   class = "b", coef = "sentiment_c"),
    prior(normal(bm$interaction, bs$interaction), class = "b", coef = "avatar_c:sentiment_c"),
    prior(normal(0, 5), class = "Intercept"),
    prior(lognormal(lg$intercept$mu, lg$intercept$sigma), class = "sd", group = "participant_code", coef = "Intercept"),
    prior(lognormal(lg$avatar$mu,    lg$avatar$sigma),    class = "sd", group = "participant_code", coef = "avatar_c"),
    prior(lognormal(lg$sentiment$mu, lg$sentiment$sigma), class = "sd", group = "participant_code", coef = "sentiment_c"),
    prior(lognormal(sg$mu, sg$sigma), class = "sigma")
  )
}

form <- bf(y ~ 1 + avatar_c + sentiment_c + avatar_c:sentiment_c +
             (1 + avatar_c + sentiment_c | participant_code))

fit_outcome <- function(yvar, section) {
  dat <- df3 |> rename(y = all_of(yvar))
  brm(
    formula = form, family = gaussian(),
    data = dat, prior = mk_priors(section),
    chains = 2, iter = 2000, warmup = 1000, seed = seed,
    backend = backend, refresh = 0
  )
}

dir.create("models", showWarnings = FALSE, recursive = TRUE)
dir.create("reports", showWarnings = FALSE, recursive = TRUE)

message("Fitting Study 3 (Gaussian) for QUALITY composite with Study-2 priors...")
m_qual <- fit_outcome("quality_rating", "quality"); saveRDS(m_qual, "models/study3_quality_linear.rds")
message("Fitting Study 3 (Gaussian) for REALISM composite with Study-2 priors...")
m_real <- fit_outcome("realism_rating", "realism"); saveRDS(m_real, "models/study3_realism_linear.rds")

summarise_d <- function(fit) {
  dr <- as_draws_df(fit)
  tibble(
    term = c("avatar_c","sentiment_c","avatar_c:sentiment_c"),
    d_median = c(
      median(dr$b_avatar_c / dr$sigma),
      median(dr$b_sentiment_c / dr$sigma),
      median(dr$`b_avatar_c:sentiment_c` / dr$sigma)
    ),
    d_ci_low = c(
      quantile(dr$b_avatar_c / dr$sigma, probs = 0.025),
      quantile(dr$b_sentiment_c / dr$sigma, probs = 0.025),
      quantile(dr$`b_avatar_c:sentiment_c` / dr$sigma, probs = 0.025)
    ),
    d_ci_high = c(
      quantile(dr$b_avatar_c / dr$sigma, probs = 0.975),
      quantile(dr$b_sentiment_c / dr$sigma, probs = 0.975),
      quantile(dr$`b_avatar_c:sentiment_c` / dr$sigma, probs = 0.975)
    )
  )
}

summarise_d(m_qual) |> write_csv("reports/bayes_linear_study3_quality.csv")
summarise_d(m_real) |> write_csv("reports/bayes_linear_study3_realism.csv")
message("Wrote Bayesian continuous summaries to reports/")
```

#### File: `R/02_anova_study3.R`  (step **2**)
```r
#!/usr/bin/env Rscript
library(tidyverse)
library(afex)
library(emmeans)

data_dir <- Sys.getenv("DATA_DIR", "data")

df3 <- readr::read_csv(file.path(data_dir, "study3_long.csv"), show_col_types = FALSE)

# Robust column detection
pick <- function(df, candidates, label) {
  present <- intersect(candidates, names(df))
  if (length(present) == 0) stop("Missing expected column for ", label, ". Tried: ", paste(candidates, collapse = ", "))
  present[1]
}

enjoy_nm   <- pick(df3, c("enjoyment_rating","enjoyment","enjoymen"), "enjoyment")
comfort_nm <- pick(df3, c("comfort_rating","comfort","comfort_r"), "comfort")
pleas_nm   <- pick(df3, c("pleasantness_rating","pleasantness","pleasantn"), "pleasantness")

realp_nm   <- pick(df3, c("real_person_rating","real_person","real_pers"), "real-person")
facial_nm  <- pick(df3, c("facial_realism_rating","facial_realism","facial_rea"), "facial realism")
body_nm    <- pick(df3, c("body_realism_rating","body_realism","body_real"), "body realism")

df3$quality_rating <- rowMeans(df3[, c(enjoy_nm, comfort_nm, pleas_nm)], na.rm = TRUE)
df3$realism_rating <- rowMeans(df3[, c(realp_nm, facial_nm, body_nm)], na.rm = TRUE)

df3 <- df3 |>
  mutate(
    participant_code = factor(participant_code),
    avatar_type = factor(avatar_type, levels = c("sync","unreal")),
    disclosure_sentiment = factor(disclosure_sentiment, levels = c("negative","positive"))
  )

dir.create("reports", showWarnings = FALSE, recursive = TRUE)

run_anova <- function(yvar, outfile_txt, outfile_csv) {
  dat <- df3 |> select(participant_code, avatar_type, disclosure_sentiment, all_of(yvar))
  colnames(dat)[4] <- "y"
  a <- suppressMessages(
    afex::aov_ez(
      id = "participant_code",
      dv = "y",
      within = c("avatar_type","disclosure_sentiment"),
      data = dat,
      anova_table = list(es = "pes")
    )
  )

  # Write human-readable report
  sink(outfile_txt); on.exit(sink(), add = TRUE)
  cat("Repeated-measures ANOVA for", yvar, "\n\n"); print(a)
  cat("\nEstimated marginal means:\n")
  print(emmeans::emmeans(a, ~ avatar_type * disclosure_sentiment))

  # Write machine-friendly ηp²
  tab <- as.data.frame(a$anova_table)
  tab <- tibble::rownames_to_column(tab, var = "Effect")
  tab <- tab |>
    filter(Effect %in% c("avatar_type","disclosure_sentiment","avatar_type:disclosure_sentiment")) |>
    select(Effect, pes)
  readr::write_csv(tab, outfile_csv)
}

run_anova("quality_rating",
          "reports/anova_study3_quality.txt",
          "reports/anova_study3_quality.csv")
run_anova("realism_rating",
          "reports/anova_study3_realism.txt",
          "reports/anova_study3_realism.csv")
message("ANOVA reports and CSVs written to reports/")
```

#### File: `R/04_information_gain_study3.R`  (step **4**)
```r
#!/usr/bin/env Rscript
library(tidyverse)
library(brms)
library(posterior)
library(jsonlite)

# KL(post || prior) between Normals, in bits
kl_norm_bits <- function(mu_post, sd_post, sd_prior, mu_prior = 0) {
  if (sd_post <= 0 || sd_prior <= 0) return(NA_real_)
  kl_nats <- log(sd_prior / sd_post) + (sd_post^2 + (mu_post - mu_prior)^2) / (2 * sd_prior^2) - 0.5
  kl_nats / log(2)  # convert to bits
}

compute_info <- function(fit, outcome_label, pri_cfg_section) {
  dr <- as_draws_df(fit)
  cols <- names(dr)

  # Prior SDs from Study 2 hyperpriors (take LogNormal median exp(mu))
  sd_prior <- list(
    Intercept   = exp(pri_cfg_section$sd_lognormal$intercept$mu),
    avatar_c    = exp(pri_cfg_section$sd_lognormal$avatar$mu),
    sentiment_c = exp(pri_cfg_section$sd_lognormal$sentiment$mu)
  )

  # Helper to extract per-participant stats for a given effect
  extract_eff <- function(effect, pretty_name) {
    pat <- paste0("^r_participant_code\[(.*?),", effect, "\]$")
    eff_cols <- grep(pat, cols, value = TRUE)
    if (!length(eff_cols)) return(tibble())
    map_dfr(eff_cols, function(cn) {
      id <- sub(pat, "\\1", cn)
      v <- dr[[cn]]
      tibble(
        outcome   = outcome_label,
        participant_code = id,
        effect    = pretty_name,
        post_mean = mean(v),
        post_sd   = sd(v),
        prior_sd  = sd_prior[[effect]],
        bits      = kl_norm_bits(mean(v), sd(v), sd_prior[[effect]])
      )
    })
  }

  bind_rows(
    extract_eff("Intercept",   "Intercept (baseline bias)"),
    extract_eff("avatar_c",    "Avatar slope (fidelity sensitivity)"),
    extract_eff("sentiment_c", "Valence slope (disclosure sensitivity)")
  )
}

dir.create("reports", showWarnings = FALSE, recursive = TRUE)

pri_cfg <- jsonlite::read_json("models/prior_config.json", simplifyVector = TRUE)
if (is.null(pri_cfg)) stop("Missing models/prior_config.json. Run R/01_priors_from_study2.R first.")

paths <- c(
  quality = "models/study3_quality_linear.rds",
  realism = "models/study3_realism_linear.rds"
)
missing <- paths[!file.exists(paths)]
if (length(missing)) {
  stop("Missing fitted models: ", paste(missing, collapse = ", "),
       "\nRun step 3 (R/03_bayes_linear_study3.R) first.")
}

m_qual <- readRDS(paths["quality"])
m_real <- readRDS(paths["realism"])

info_q <- compute_info(m_qual, "Quality", pri_cfg$quality)
info_r <- compute_info(m_real, "Realism", pri_cfg$realism)

readr::write_csv(info_q, "reports/info_gain_study3_quality.csv")
readr::write_csv(info_r, "reports/info_gain_study3_realism.csv")

summary_tbl <- bind_rows(info_q, info_r) |>
  group_by(outcome, effect) |>
  summarise(
    mean_bits = mean(bits, na.rm = TRUE),
    median_bits = median(bits, na.rm = TRUE),
    .groups = "drop"
  ) |>
  arrange(outcome, effect)

readr::write_csv(summary_tbl, "reports/info_gain_summary.csv")
message("Wrote info gain tables to reports/")
```

#### File: `R/01_effect_convergence.R`  (step **1**)
```r
#!/usr/bin/env Rscript
library(tidyverse)

paths <- list(
  anova_qual = "reports/anova_study3_quality.csv",
  anova_real = "reports/anova_study3_realism.csv",
  bayes_qual = "reports/bayes_linear_study3_quality.csv",
  bayes_real = "reports/bayes_linear_study3_realism.csv",
  info_sum   = "reports/info_gain_summary.csv"
)

needed <- unlist(paths[1:4])
miss <- needed[!file.exists(needed)]
if (length(miss)) {
  stop("Missing prerequisite outputs:\n", paste(miss, collapse = "\n"),
       "\nRun steps 3 and 2 before step 1 (and 4 if you want bits included).")
}

an_q <- readr::read_csv(paths$anova_qual, show_col_types = FALSE)
an_r <- readr::read_csv(paths$anova_real, show_col_types = FALSE)
bl_q <- readr::read_csv(paths$bayes_qual, show_col_types = FALSE)
bl_r <- readr::read_csv(paths$bayes_real, show_col_types = FALSE)

# Map term names for a unified table
map_lin <- tibble(
  term = c("avatar_c","sentiment_c","avatar_c:sentiment_c"),
  Effect = c("Avatar fidelity","Disclosure valence","Avatar × Valence")
)

recode_effect <- function(x) dplyr::recode(
  x,
  "avatar_type"                      = "Avatar fidelity",
  "disclosure_sentiment"             = "Disclosure valence",
  "avatar_type:disclosure_sentiment" = "Avatar × Valence",
  .default = x
)

an_q2 <- an_q |> mutate(Effect = recode_effect(Effect)) |> select(Effect, Quality_eta_p2 = pes)
an_r2 <- an_r |> mutate(Effect = recode_effect(Effect)) |> select(Effect, Realism_eta_p2 = pes)

bl_q2 <- bl_q |> inner_join(map_lin, by = "term") |>
  select(Effect, Quality_d = d_median, Quality_d_low = d_ci_low, Quality_d_high = d_ci_high)
bl_r2 <- bl_r |> inner_join(map_lin, by = "term") |>
  select(Effect, Realism_d = d_median, Realism_d_low = d_ci_low, Realism_d_high = d_ci_high)

tbl <- map_lin |> select(Effect) |>
  left_join(bl_r2, by = "Effect") |>
  left_join(bl_q2, by = "Effect") |>
  left_join(an_r2, by = "Effect") |>
  left_join(an_q2, by = "Effect") |>
  mutate(across(where(is.numeric), ~round(., 3)))

dir.create("reports", showWarnings = FALSE, recursive = TRUE)
readr::write_csv(tbl, "reports/effect_convergence_study3.csv")

# Sidecar: info-gain summary if present
if (file.exists(paths$info_sum)) {
  info <- readr::read_csv(paths$info_sum, show_col_types = FALSE)
  readr::write_csv(info, "reports/effect_information_gain_bits_summary.csv")
}

message("Wrote reports/effect_convergence_study3.csv (and bits summary if available).")
```

#### File: `R/run_pipeline.R`
```r
#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  message("Usage: Rscript R/run_pipeline.R [3] [2] [4] [1]")
  quit(save = "no", status = 1)
}
steps <- sort(unique(as.integer(args)), decreasing = TRUE)

run <- function(cmd) {
  message("\n>>> ", cmd)
  status <- system(cmd)
  if (status != 0) stop("Command failed: ", cmd, call. = FALSE)
}

# Always run setup
run("Rscript R/00_setup.R")

# If step 3 or 4 is requested and priors are missing, learn from Study 2
need_priors <- (!file.exists("models/prior_config.json")) && ( (3 %in% steps) || (4 %in% steps) )
if (need_priors) {
  run("Rscript R/01_priors_from_study2.R")
}

if (3 %in% steps) run("Rscript R/03_bayes_linear_study3.R")
if (2 %in% steps) run("Rscript R/02_anova_study3.R")
if (4 %in% steps) run("Rscript R/04_information_gain_study3.R")
if (1 %in% steps) run("Rscript R/01_effect_convergence.R")

message("\nRequested steps complete. See ./reports and ./models.")
```

### C) How to run

```
bash
# Learn Study 2 priors → Fit Study 3 (Bayes) → ANOVA → Info gain → Aggregate
Rscript R/run_pipeline.R 3 2 4 1
```

You can run any subset; if priors are absent and steps **3** or **4** are requested, the agent auto‑runs the Study 2 prior step.

## Acceptance Criteria
- Study 2 models fitted and **priors written** to `models/prior_config.json`.
- Study 3 Bayesian models **use those priors**.
- Composite outcomes are used throughout.
- ANOVA + Bayesian outputs created for Study 3.
- **Information gain in bits** computed for participant random effects using **Study 2 priors**.
- Convergence table written to `reports/effect_convergence_study3.csv`.
