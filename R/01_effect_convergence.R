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
