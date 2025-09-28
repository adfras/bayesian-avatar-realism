#!/usr/bin/env Rscript
library(tidyverse)
library(brms)
library(posterior)
library(jsonlite)

# KL(post || prior) between Normals, in bits
kl_norm_bits <- function(mu_post, sd_post, sd_prior, mu_prior = 0) {
  if (is.null(sd_post) || is.null(sd_prior) ||
      is.na(sd_post) || is.na(sd_prior) ||
      sd_post <= 0 || sd_prior <= 0) {
    return(NA_real_)
  }
  kl_nats <- log(sd_prior / sd_post) + (sd_post^2 + (mu_post - mu_prior)^2) / (2 * sd_prior^2) - 0.5
  kl_nats / log(2)  # convert to bits
}

compute_info <- function(fit, outcome_label, pri_cfg_section) {
  dr <- as_draws_df(fit)
  cols <- names(dr)

  # Prior SDs from Study 2 hyperpriors (take LogNormal median exp(mu))
  sd_prior <- c(
    Intercept   = exp(pri_cfg_section$sd_lognormal$intercept$mu),
    avatar_c    = exp(pri_cfg_section$sd_lognormal$avatar$mu),
    sentiment_c = exp(pri_cfg_section$sd_lognormal$sentiment$mu)
  )

  # Helper to extract per-participant stats for a given effect
  extract_eff <- function(effect, pretty_name) {
    pat <- paste0("^r_participant_code\\[(.*?),", effect, "\\]$")
    eff_cols <- grep(pat, cols, value = TRUE)
    if (!length(eff_cols)) return(tibble())
    prior_sd_val <- sd_prior[[effect]]
    map_dfr(eff_cols, function(cn) {
      id <- sub(pat, "\\1", cn)
      v <- dr[[cn]]
      tibble(
        outcome   = outcome_label,
        participant_code = id,
        effect    = pretty_name,
        post_mean = mean(v),
        post_sd   = sd(v),
        prior_sd  = prior_sd_val,
        bits      = kl_norm_bits(mean(v), sd(v), prior_sd_val)
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
