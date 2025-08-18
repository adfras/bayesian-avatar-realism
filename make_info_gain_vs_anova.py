#!/usr/bin/env python3
# Compare ANOVA bits vs Bayesian info-gain bits with publication-quality figures.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- paths (unchanged) -------------------------------------------------------
ANOVA_PATH   = "results_anova/study3_anova_all.csv"
ORD_REALISM  = "results/info_gain_random_effects_realism_by_param.csv"
ORD_QUALITY  = "results/info_gain_random_effects_quality_by_param.csv"
CONT_REALISM = "results_cont/info_gain_cont_random_realism_by_param.csv"
CONT_QUALITY = "results_cont/info_gain_cont_random_quality_by_param.csv"

OUT_DIR = "results_figs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- style -------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.18,
    "grid.linestyle": "-",
})

def eta2_to_bits(eta2):
    eta2 = np.clip(np.asarray(eta2, dtype=float), 0.0, 0.999999999999)
    return -0.5 * np.log2(1.0 - eta2)

def load_anova_bits(path):
    df = pd.read_csv(path)
    keep = df["effect"].str.lower().isin(["avatar_type", "disclosure_sentiment"])
    df = df.loc[keep].copy()
    df["Effect"] = df["effect"].map({"avatar_type": "Avatar", "disclosure_sentiment": "Valence"})
    df["Outcome"] = df["dv"].map({"realism3": "Realism", "quality3": "Quality"})
    df["ANOVA_bits"] = eta2_to_bits(df["partial_eta2"].astype(float))
    return df[["Outcome", "Effect", "ANOVA_bits", "partial_eta2"]].reset_index(drop=True)

def load_bayes_bits(path_realism, path_quality):
    if not (os.path.exists(path_realism) and os.path.exists(path_quality)):
        raise FileNotFoundError("Bayesian info-gain files not found.")
    def prep(path, outcome):
        d = pd.read_csv(path)
        re_col   = "re_param" if "re_param" in d.columns else "param"
        bits_col = "KL_bits_mean" if "KL_bits_mean" in d.columns else "KL_bits"
        map_effect = {
            "Intercept":"Intercept", "avatar_c":"Avatar", "avatar":"Avatar",
            "Avatar":"Avatar", "type_c":"Valence", "valence":"Valence", "Valence":"Valence"
        }
        d["Effect"] = d[re_col].map(map_effect)
        d = d.dropna(subset=["Effect"])
        d = (d.groupby("Effect", observed=True)[bits_col]
               .mean().reset_index()
               .rename(columns={bits_col: "Bayes_bits"}))
        d["Outcome"] = outcome
        return d[["Outcome","Effect","Bayes_bits"]]
    return pd.concat([prep(path_realism,"Realism"), prep(path_quality,"Quality")], ignore_index=True)

# Try ordinal first, fall back to continuous
try:
    bayes_bits = load_bayes_bits(ORD_REALISM, ORD_QUALITY)
except Exception:
    bayes_bits = load_bayes_bits(CONT_REALISM, CONT_QUALITY)

anova_bits = load_anova_bits(ANOVA_PATH)

pairable = (anova_bits.merge(bayes_bits, on=["Outcome","Effect"], how="inner")
                        .sort_values(["Outcome","Effect"]))
labels = [f"{o}\n{e}" for o, e in zip(pairable["Outcome"], pairable["Effect"])]
x = np.arange(len(labels))
width = 0.38

# ---- Figure 1: ANOVA vs Bayesian (Avatar & Valence) -------------------------
fig1, ax1 = plt.subplots(figsize=(12, 6))
b1 = ax1.bar(x - width/2, pairable["ANOVA_bits"].values, width, label="ANOVA (ηp² → bits)")
b2 = ax1.bar(x + width/2, pairable["Bayes_bits"].values, width, label="Bayesian (random slopes)")

# annotate bars clearly (always above the bar top)
for bars in (b1, b2):
    for rect in bars:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2.0, h + 0.03, f"{h:.2f}",
                 ha="center", va="bottom", fontsize=11)

ax1.set_xticks(x, labels=labels)
ax1.set_ylabel("Bits of information")
ax1.set_title("Information Gain (bits): ANOVA vs Bayesian (Avatar & Valence)")
ax1.margins(x=0.08)
y_top = max(1.1, float(np.nanmax(np.r_[pairable["ANOVA_bits"], pairable["Bayes_bits"]])) + 0.25)
ax1.set_ylim(0, y_top)

# 1-bit guide
ax1.axhline(1.0, color="0.5", linestyle="--", linewidth=1)
ax1.text(ax1.get_xlim()[1] - 0.02*(ax1.get_xlim()[1]-ax1.get_xlim()[0]),
         1.0 + 0.02*(y_top-0), "≈1 bit (halve uncertainty)",
         ha="right", va="bottom", fontsize=10, color="0.35")

ax1.legend(frameon=False, loc="upper left")
fig1.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, "info_gain_bits_anova_vs_bayes.png"), bbox_inches="tight")

# ---- Figure 2: Intercepts (Bayes-only) --------------------------------------
intercepts = bayes_bits[bayes_bits["Effect"] == "Intercept"].copy()
if not intercepts.empty:
    fig2, ax2 = plt.subplots(figsize=(8.5, 5.6))
    x2 = np.arange(len(intercepts))
    bars = ax2.bar(x2, intercepts["Bayes_bits"].values, width=0.55, label="Bayesian (intercepts)")

    for rect in bars:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2.0, h + 0.04, f"{h:.2f}",
                 ha="center", va="bottom", fontsize=11)

    ax2.set_xticks(x2, labels=intercepts["Outcome"].values)
    ax2.set_ylabel("Bits of information")
    ax2.set_title("Information Gain (bits): Bayesian Intercepts (no ANOVA counterpart)")
    ax2.margins(x=0.12)
    y2_top = max(1.2, float(np.nanmax(intercepts["Bayes_bits"])) + 0.35)
    ax2.set_ylim(0, y2_top)

    ax2.axhline(1.0, color="0.5", linestyle="--", linewidth=1)
    ax2.text(ax2.get_xlim()[1] - 0.02*(ax2.get_xlim()[1]-ax2.get_xlim()[0]),
             1.0 + 0.02*(y2_top-0), "≈1 bit (halve uncertainty)",
             ha="right", va="bottom", fontsize=10, color="0.35")

    ax2.legend(frameon=False, loc="upper left")
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "info_gain_bits_bayes_intercepts.png"), bbox_inches="tight")

print("Saved:")
print(" -", os.path.join(OUT_DIR, "info_gain_bits_anova_vs_bayes.png"))
if not intercepts.empty:
    print(" -", os.path.join(OUT_DIR, "info_gain_bits_bayes_intercepts.png"))
