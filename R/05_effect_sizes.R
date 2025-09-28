#!/usr/bin/env Rscript
library(tidyverse)
library(brms)
library(posterior)

seed <- as.integer(Sys.getenv("SEED", "2025"))
set.seed(seed)

build_study3_data <- function(data_dir = Sys.getenv("DATA_DIR", "data")) {
  path <- file.path(data_dir, "study3_long.csv")
  if (!file.exists(path)) stop("Missing data file: ", path)
  df <- readr::read_csv(path, show_col_types = FALSE)

  pick <- function(df, candidates, label) {
    present <- intersect(candidates, names(df))
    if (length(present) == 0) stop(
      "Missing expected column for ", label,
      ". Tried: ", paste(candidates, collapse = ", ")
    )
    present[1]
  }

  enjoy_nm   <- pick(df, c("enjoyment_rating","enjoyment","enjoymen"), "enjoyment")
  comfort_nm <- pick(df, c("comfort_rating","comfort","comfort_r"), "comfort")
  pleas_nm   <- pick(df, c("pleasantness_rating","pleasantness","pleasantn"), "pleasantness")
  realp_nm   <- pick(df, c("real_person_rating","real_person","real_pers"), "real-person")
  facial_nm  <- pick(df, c("facial_realism_rating","facial_realism","facial_rea"), "facial realism")
  body_nm    <- pick(df, c("body_realism_rating","body_realism","body_real"), "body realism")

  df$quality_rating <- rowMeans(df[, c(enjoy_nm, comfort_nm, pleas_nm)], na.rm = TRUE)
  df$realism_rating <- rowMeans(df[, c(realp_nm, facial_nm, body_nm)], na.rm = TRUE)

  df |> mutate(
    participant_code = factor(participant_code),
    avatar_c = if_else(avatar_type == "unreal",  0.5, -0.5),
    sentiment_c = if_else(disclosure_sentiment == "positive", 0.5, -0.5)
  ) |> select(participant_code, avatar_c, sentiment_c, quality_rating, realism_rating)
}

draw_limit <- function(fit, max_draws = 1000) {
  nd <- posterior::ndraws(as_draws(fit))
  min(nd, max_draws)
}

select_draw_ids <- function(fit, draws) {
  all_draws <- posterior::ndraws(as_draws(fit))
  if (draws >= all_draws) {
    seq_len(all_draws)
  } else {
    sort(sample(seq_len(all_draws), size = draws, replace = FALSE))
  }
}

posterior_long <- function(fit, newdata, draw_ids) {
  if (length(draw_ids) == posterior::ndraws(as_draws(fit))) {
    preds <- posterior_predict(fit, newdata = newdata)
  } else {
    preds <- posterior_predict(fit, newdata = newdata, draw_ids = draw_ids)
  }
  tibble(
    draw = rep(draw_ids, each = nrow(newdata)),
    row_id = rep(seq_len(nrow(newdata)), times = length(draw_ids)),
    y_tilde = as.vector(t(preds))
  ) |>
    left_join(newdata |> mutate(row_id = row_number()), by = "row_id")
}

summarise_draws_ci <- function(x) {
  if (all(is.na(x))) return(c(NA_real_, NA_real_, NA_real_))
  c(
    median(x, na.rm = TRUE),
    quantile(x, 0.025, na.rm = TRUE),
    quantile(x, 0.975, na.rm = TRUE)
  )
}

make_newdata <- function(study3, effect) {
  ids <- levels(study3$participant_code)
  if (effect == "avatar") {
    tibble(
      participant_code = factor(rep(ids, each = 2), levels = ids),
      avatar_c = rep(c(-0.5, 0.5), times = length(ids)),
      sentiment_c = 0,
      y = NA_real_
    )
  } else {
    tibble(
      participant_code = factor(rep(ids, each = 2), levels = ids),
      avatar_c = 0,
      sentiment_c = rep(c(-0.5, 0.5), times = length(ids)),
      y = NA_real_
    )
  }
}

paired_effect_metrics <- function(long_df, effect) {
  if (!effect %in% c("avatar", "sentiment")) {
    return(list(dz = c(NA_real_, NA_real_, NA_real_),
                dav = c(NA_real_, NA_real_, NA_real_)))
  }

  if (effect == "avatar") {
    long_df <- long_df |>
      mutate(level = factor(if_else(avatar_c > 0, "unreal", "sync"),
                             levels = c("sync", "unreal")))
    hi_col <- "unreal"; lo_col <- "sync"
  } else {
    long_df <- long_df |>
      mutate(level = factor(if_else(sentiment_c > 0, "positive", "negative"),
                             levels = c("negative", "positive")))
    hi_col <- "positive"; lo_col <- "negative"
  }

  paired <- long_df |>
    group_by(draw, participant_code, level) |>
    summarise(mu = mean(y_tilde), .groups = "drop") |>
    pivot_wider(names_from = level, values_from = mu) |>
    drop_na(any_of(c(hi_col, lo_col))) |>
    mutate(diff = .data[[hi_col]] - .data[[lo_col]])

  dz_draws <- paired |>
    group_by(draw) |>
    summarise(
      diff_mean = mean(diff),
      diff_sd   = sd(diff),
      .groups = "drop"
    ) |>
    mutate(dz = if_else(diff_sd < 1e-6, NA_real_, diff_mean / diff_sd)) |>
    pull(dz)

  unpaired <- long_df |>
    group_by(draw, level) |>
    summarise(mu = mean(y_tilde), sd = sd(y_tilde), .groups = "drop") |>
    pivot_wider(names_from = level, values_from = c(mu, sd))

  dav_draws <- unpaired |>
    mutate(
      mu_hi = .data[[paste0("mu_", hi_col)]],
      mu_lo = .data[[paste0("mu_", lo_col)]],
      sd_hi = .data[[paste0("sd_", hi_col)]],
      sd_lo = .data[[paste0("sd_", lo_col)]],
      sd_pool = sqrt((sd_hi^2 + sd_lo^2) / 2),
      dav = if_else(sd_pool < 1e-6, NA_real_, (mu_hi - mu_lo) / sd_pool)
    ) |>
    pull(dav)

  list(
    dz = summarise_draws_ci(dz_draws),
    dav = summarise_draws_ci(dav_draws)
  )
}

effect_summary <- function(fit, study3, outcome_label, draws) {
  draws_df <- as_draws_df(fit)
  draw_ids <- select_draw_ids(fit, draws)

  newdata_avatar <- make_newdata(study3, "avatar")
  newdata_sent   <- make_newdata(study3, "sentiment")

  preds_avatar <- posterior_long(fit, newdata_avatar, draw_ids)
  preds_sent   <- posterior_long(fit, newdata_sent, draw_ids)

  effects <- list(
    list(id = "avatar", label = "Avatar (unreal - sync)", beta_name = "b_avatar_c", preds = preds_avatar),
    list(id = "sentiment", label = "Sentiment (positive - negative)", beta_name = "b_sentiment_c", preds = preds_sent),
    list(id = "interaction", label = "Interaction (sentiment x avatar)", beta_name = "b_avatar_c:sentiment_c", preds = NULL)
  )

  bind_rows(lapply(effects, function(eff) {
    beta <- draws_df[[eff$beta_name]]
    beta_ci <- quantile(beta, probs = c(0.025, 0.5, 0.975))
    pd <- mean(beta > 0) * 100

    if (!is.null(eff$preds)) {
      stats <- paired_effect_metrics(eff$preds, eff$id)
      dz_vals <- stats$dz
      dav_vals <- stats$dav
      dz_method <- "posterior_predict"
      dav_method <- "posterior_predict"
    } else {
      dz_vals <- c(NA_real_, NA_real_, NA_real_)
      dav_vals <- c(NA_real_, NA_real_, NA_real_)
      dz_method <- NA_character_
      dav_method <- NA_character_
    }

    tibble(
      outcome = outcome_label,
      effect = eff$label,
      beta_median = beta_ci[[2]],
      beta_low = beta_ci[[1]],
      beta_high = beta_ci[[3]],
      pd = pd,
      dz_method = dz_method,
      dz_median = dz_vals[1],
      dz_low = dz_vals[2],
      dz_high = dz_vals[3],
      dav_method = dav_method,
      dav_median = dav_vals[1],
      dav_low = dav_vals[2],
      dav_high = dav_vals[3]
    )
  }))
}

study3 <- build_study3_data()
quality_fit <- readRDS("models/study3_quality_linear.rds")
realism_fit <- readRDS("models/study3_realism_linear.rds")

quality_draws <- draw_limit(quality_fit)
realism_draws <- draw_limit(realism_fit)

quality_summary <- effect_summary(quality_fit, study3, "Quality", quality_draws)
realism_summary <- effect_summary(realism_fit, study3, "Realism", realism_draws)

summary_tbl <- bind_rows(quality_summary, realism_summary)

dir.create("reports", showWarnings = FALSE, recursive = TRUE)
readr::write_csv(summary_tbl, "reports/effect_size_summary.csv")
message("Wrote posterior effect size summary to reports/effect_size_summary.csv")
