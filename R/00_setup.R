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
