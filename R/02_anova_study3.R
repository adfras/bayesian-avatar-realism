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
