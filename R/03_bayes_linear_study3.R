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
  fmt <- function(x) formatC(x, digits = 9, format = "g")
  bm <- pri_cfg[[section]]$b_means
  bs <- pri_cfg[[section]]$b_scales
  lg <- pri_cfg[[section]]$sd_lognormal
  sg <- pri_cfg[[section]]$sigma_lognormal
  c(
    prior_string(sprintf("normal(%s, %s)", fmt(bm$avatar), fmt(bs$avatar)),
                 class = "b", coef = "avatar_c"),
    prior_string(sprintf("normal(%s, %s)", fmt(bm$sentiment), fmt(bs$sentiment)),
                 class = "b", coef = "sentiment_c"),
    prior_string(sprintf("normal(%s, %s)", fmt(bm$interaction), fmt(bs$interaction)),
                 class = "b", coef = "avatar_c:sentiment_c"),
    prior(normal(0, 5), class = "Intercept"),
    prior_string(sprintf("lognormal(%s, %s)", fmt(lg$intercept$mu), fmt(lg$intercept$sigma)),
                 class = "sd", group = "participant_code", coef = "Intercept"),
    prior_string(sprintf("lognormal(%s, %s)", fmt(lg$avatar$mu), fmt(lg$avatar$sigma)),
                 class = "sd", group = "participant_code", coef = "avatar_c"),
    prior_string(sprintf("lognormal(%s, %s)", fmt(lg$sentiment$mu), fmt(lg$sentiment$sigma)),
                 class = "sd", group = "participant_code", coef = "sentiment_c"),
    prior_string(sprintf("lognormal(%s, %s)", fmt(sg$mu), fmt(sg$sigma)),
                 class = "sigma")
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
