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
if (5 %in% steps) run("Rscript R/05_effect_sizes.R")

message("\nRequested steps complete. See ./reports and ./models.")
