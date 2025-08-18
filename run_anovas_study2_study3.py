#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run within-subjects ANOVAs on Study 2 (3×2) and Study 3 (2×2) using long-format CSVs.

Composites:
  - quality3 = mean(enjoyment_rating, comfort_rating, pleasantness_rating)
  - realism3 = mean(real_person_rating, facial_realism_rating, body_realism_rating)
    (Study 3 EXCLUDES room_realism_rating and space_realism_rating for comparability.)

Auto-detects factor levels:
  - avatar_type: Study 2 -> Low/Medium/High (3); Study 3 -> Sync/Unreal (2)
  - disclosure_sentiment: Negative/Positive

Outputs -> ./results_anova/ :
  - study{2|3}_anova_{realism3|quality3}.csv      # F, df1, df2, p, partial_eta2
  - study{2|3}_cellmeans_{realism3|quality3}.csv  # condition means, sd, se, n
  - study{2|3}_anova_all.csv                      # both DVs stacked
  - study{2|3}_cellmeans_all.csv                  # both DVs stacked
  - study{2|3}_n_subjects.txt                     # N participants with complete cells per DV
"""

import os
import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM

RESULTS_DIR = "results_anova"
os.makedirs(RESULTS_DIR, exist_ok=True)

QUALITY_ITEMS = ["enjoyment_rating", "comfort_rating", "pleasantness_rating"]
REALISM_ITEMS = ["real_person_rating", "facial_realism_rating", "body_realism_rating"]

# ---------- helpers ----------
def _require_cols(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing {label} column(s): {missing}")

def _canonicalize_avatar(s: str) -> str:
    s = str(s).strip().lower()
    # Study 3 labels
    if s in {"sync", "synchronous", "vive"}: return "Sync"
    if s in {"unreal", "iclone"}: return "Unreal"
    # Study 2 labels
    if s in {"low", "medium", "high"}: return s.capitalize()
    return s.capitalize() or s

def _canonicalize_valence(s: str) -> str:
    s = str(s).strip().lower()
    if s.startswith("pos"): return "Positive"
    if s.startswith("neg"): return "Negative"
    return s.capitalize() or s

def _id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "participant_id" in df.columns:
        return df.rename(columns={"participant_id":"participant_id"})
    if "participant_code" in df.columns:
        df = df.rename(columns={"participant_code":"participant_id"})
        return df
    raise KeyError("Need participant identifier: 'participant_id' or 'participant_code'.")

def _build_composites(df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(df, QUALITY_ITEMS, "quality items")
    _require_cols(df, REALISM_ITEMS, "realism items")
    out = df.copy()
    out["quality3"] = out[QUALITY_ITEMS].mean(axis=1)
    out["realism3"] = out[REALISM_ITEMS].mean(axis=1)
    # Drop rows with invalid/missing item values (expect 1..5)
    def ok(s): return s.notna() & (s >= 1) & (s <= 5)
    keep = pd.Series(True, index=out.index)
    for c in QUALITY_ITEMS + REALISM_ITEMS:
        keep &= ok(out[c])
    dropped = int((~keep).sum())
    if dropped:
        print(f"[clean] Dropping {dropped} rows with invalid/missing rating(s) (must be 1..5).")
    return out.loc[keep].copy()

def _prepare_factors(df: pd.DataFrame) -> pd.DataFrame:
    if "avatar_type" not in df or "disclosure_sentiment" not in df:
        raise KeyError("Need factor columns 'avatar_type' and 'disclosure_sentiment'.")
    df = df.copy()
    df["avatar_type"] = df["avatar_type"].map(_canonicalize_avatar)
    df["disclosure_sentiment"] = df["disclosure_sentiment"].map(_canonicalize_valence)

    # Determine levels present & set ordered Categoricals
    a_levels = pd.Index(sorted(df["avatar_type"].dropna().unique().tolist(), key=lambda x: ["Low","Medium","High","Sync","Unreal"].index(x) if x in ["Low","Medium","High","Sync","Unreal"] else x))
    v_levels = pd.Index(sorted(df["disclosure_sentiment"].dropna().unique().tolist(), key=lambda x: ["Negative","Positive"].index(x) if x in ["Negative","Positive"] else x))

    df["avatar_type"] = pd.Categorical(df["avatar_type"], categories=a_levels, ordered=True)
    df["disclosure_sentiment"] = pd.Categorical(df["disclosure_sentiment"], categories=v_levels, ordered=True)
    return df

def _aggregate_subject_cell_means(df: pd.DataFrame, dv: str):
    """One row per participant × avatar × valence (mean over repeats)."""
    a_levels = df["avatar_type"].cat.categories
    v_levels = df["disclosure_sentiment"].cat.categories
    required_cells = len(a_levels) * len(v_levels)

    grp = (df.groupby(["participant_id","avatar_type","disclosure_sentiment"], observed=True)[dv]
             .mean().reset_index())

    counts = grp.groupby("participant_id")[dv].count()
    keep_ids = counts[counts == required_cells].index
    dropped_rows = int((~grp["participant_id"].isin(keep_ids)).sum())
    if dropped_rows:
        print(f"[aggregate:{dv}] Dropping {dropped_rows} rows from participants with incomplete {len(a_levels)}×{len(v_levels)} cells.")
    grp = grp[grp["participant_id"].isin(keep_ids)].copy()
    n_subj = grp["participant_id"].nunique()
    return grp, n_subj, list(a_levels), list(v_levels), required_cells

def _partial_eta_squared(F, df1, df2):
    # For within-subjects effects in RM-ANOVA:
    # ηp² = (F * df1) / (F * df1 + df2)
    return (F * df1) / (F * df1 + df2) if np.isfinite(F) else np.nan

def _anova_any(grp: pd.DataFrame, dv: str, within_factors: list, study_label: str) -> pd.DataFrame:
    aov = AnovaRM(data=grp, depvar=dv, subject="participant_id", within=within_factors).fit()
    tab = aov.anova_table.reset_index().rename(columns={"index":"effect"})
    # Normalize column names across statsmodels versions
    tab = tab.rename(columns={"F Value":"F","Num DF":"df1","Den DF":"df2","Pr > F":"p"})
    tab["partial_eta2"] = [_partial_eta_squared(F, df1, df2) for F,df1,df2 in zip(tab["F"], tab["df1"], tab["df2"])]
    tab.insert(0,"study", study_label)
    tab.insert(1,"dv", dv)
    return tab

def _save_cell_means(grp: pd.DataFrame, dv: str, study_label: str) -> pd.DataFrame:
    cell = (grp.groupby(["avatar_type","disclosure_sentiment"], observed=True)[dv]
              .agg(mean="mean", sd="std", n="count"))
    cell["se"] = cell["sd"] / np.sqrt(cell["n"])
    cell = cell.reset_index()
    cell.insert(0,"study", study_label)
    cell.insert(1,"dv", dv)
    return cell

def run_for_study(study_csv_path: str, study_label: str):
    print(f"\n=== {study_label}: reading {study_csv_path} ===")
    df = pd.read_csv(study_csv_path)
    df = _id_column(df)
    df = _build_composites(df)
    df = _prepare_factors(df)

    results = []
    cells_all = []
    for dv in ["realism3","quality3"]:
        grp, n_subj, a_levels, v_levels, n_cells = _aggregate_subject_cell_means(df, dv)
        print(f"[{study_label} - {dv}] Levels: avatar={a_levels}, valence={v_levels} -> required cells per participant={n_cells}")
        print(f"[{study_label} - {dv}] N participants with complete cells: {n_subj}")

        # If no participants remain, skip gracefully
        if n_subj == 0:
            print(f"[{study_label} - {dv}] Skipping ANOVA (no complete within-subject cells).")
            continue

        # Save N
        with open(os.path.join(RESULTS_DIR, f"{study_label.lower()}_n_subjects.txt"), "a") as f:
            f.write(f"{dv}: {n_subj}\n")

        # Save cell means
        cells = _save_cell_means(grp, dv, study_label)
        cells_all.append(cells)
        cells.to_csv(os.path.join(RESULTS_DIR, f"{study_label.lower()}_cellmeans_{dv}.csv"), index=False)

        # ANOVA with the detected within factors
        within = ["avatar_type","disclosure_sentiment"]
        tab = _anova_any(grp, dv, within, study_label)
        results.append(tab)
        tab.to_csv(os.path.join(RESULTS_DIR, f"{study_label.lower()}_anova_{dv}.csv"), index=False)

        # console summary of main effects + interaction
        def row(effect_name):
            m = tab.loc[tab["effect"].str.lower()==effect_name.lower()]
            return m.iloc[0] if len(m) else None

        for eff in ["avatar_type", "disclosure_sentiment", "avatar_type:disclosure_sentiment"]:
            r = row(eff)
            if r is not None:
                print(f"[{study_label} - {dv}] {eff}: F({int(r['df1'])},{int(r['df2'])})={r['F']:.2f}, "
                      f"p={r['p']:.4f}, ηp²={r['partial_eta2']:.3f}")

    # Save combined tables for convenience
    if results:
        all_tab = pd.concat(results, ignore_index=True)
        all_tab.to_csv(os.path.join(RESULTS_DIR, f"{study_label.lower()}_anova_all.csv"), index=False)
    if cells_all:
        all_cells = pd.concat(cells_all, ignore_index=True)
        all_cells.to_csv(os.path.join(RESULTS_DIR, f"{study_label.lower()}_cellmeans_all.csv"), index=False)

    print(f"✓ Saved outputs for {study_label} in '{RESULTS_DIR}/'.")

# ---------- main ----------
if __name__ == "__main__":
    STUDY2_CSV = "study2_long.csv"
    STUDY3_CSV = "study3_long.csv"

    run_for_study(STUDY2_CSV, "Study2")  # 3×2 (Low/Medium/High × Negative/Positive)
    run_for_study(STUDY3_CSV, "Study3")  # 2×2 (Sync/Unreal × Negative/Positive)

    print("\nAll done. See the 'results_anova/' folder for CSVs and summaries.")
