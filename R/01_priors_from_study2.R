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
