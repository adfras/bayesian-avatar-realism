#!/usr/bin/env python3
# make_plots.py — Beautiful, readable plots for Study-3 results
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import transforms as mtrans

# ---------- setup ----------
plt.style.use("seaborn-v0_8")
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

RESULTS = Path("results")
FIGS    = Path("figs")
FIGS.mkdir(exist_ok=True)

# nice labels for effects and conditions
EFFECT_LABELS = {
    "avatar_c": "Avatar: Unreal > Sync",
    "type_c": "Disclosure: Positive > Negative",
    "avatar_c:type_c": "Interaction (Avatar × Disclosure)",
    "Intercept": "Intercept",
}
COND_ORDER = ["first_neg", "first_pos", "third_neg", "third_pos"]
COND_LABELS = {
    "first_neg":  "First • Negative",
    "first_pos":  "First • Positive",
    "third_neg":  "Third • Negative",
    "third_pos":  "Third • Positive",
}

# small helper
def pct(x): return 100 * np.asarray(x)

# ---------- 1) Forest plots (odds ratios) ----------
def forest_plot(kind):
    """kind in {'realism','quality'}"""
    f = RESULTS / f"effects_{kind}.csv"
    df = pd.read_csv(f)

    # We plot only slopes (OR meaningful). Intercept omitted from OR forest.
    df = df[df["param"] != "Intercept"].copy()
    # enforce a stable, human order
    order = ["type_c", "avatar_c", "avatar_c:type_c"]
    df["__order__"] = df["param"].map({k:i for i,k in enumerate(order)})
    df = df.sort_values("__order__")

    y = np.arange(len(df))[::-1]  # top-to-bottom
    x_med = df["OR_50%"].values
    x_lo  = df["OR_2.5%"].values
    x_hi  = df["OR_97.5%"].values
    xerr  = np.vstack([x_med - x_lo, x_hi - x_med])

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.errorbar(x_med, y, xerr=xerr, fmt="o", lw=1.8, capsize=3)
    ax.axvline(1.0, color="k", lw=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio (log scale)")
    ax.set_yticks(y)
    ax.set_yticklabels([EFFECT_LABELS.get(p, p) for p in df["param"]])
    ax.set_title(f"Fixed Effects (Odds Ratios) — {kind.capitalize()}")

    # annotate Pr(β>0)
    trans = mtrans.blended_transform_factory(ax.transAxes, ax.transData)
    for yi, p in zip(y, df["Pr(beta>0)"].values):
        ax.text(0.99, yi, f"Pr(β>0) = {p:.2f}", transform=trans, ha="right", va="center")

    fig.tight_layout()
    out = FIGS / f"forest_{kind}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")

# ---------- 2) Condition plots: P(rating ≥ 4) ----------
def cond_plot():
    # Prefer population-average; optionally include new-participant as a second panel row
    rows_pop, rows_new = [], []
    for kind in ["realism","quality"]:
        pop = RESULTS / f"cond_probs_{kind}_population.csv"
        new = RESULTS / f"cond_probs_{kind}_new_participant.csv"
        if pop.exists():
            dfp = pd.read_csv(pop); dfp["kind"] = kind.capitalize(); rows_pop.append(dfp)
        if new.exists():
            dfn = pd.read_csv(new); dfn["kind"] = kind.capitalize(); rows_new.append(dfn)

    assert rows_pop, "No condition files found. Run phase2 script first."

    def prep(df):
        df["condition"] = pd.Categorical(df["condition"], categories=COND_ORDER, ordered=True)
        df = df.sort_values(["kind","condition"]).copy()
        df["label"] = df["condition"].map(COND_LABELS)
        return df

    pop_all = prep(pd.concat(rows_pop, ignore_index=True))
    have_new = len(rows_new) > 0
    if have_new:
        new_all = prep(pd.concat(rows_new, ignore_index=True))

    nrows = 2 if have_new else 1
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 3.6*nrows), sharey=True)
    axes = np.atleast_2d(axes)

    def draw(ax, sub, title_suffix):
        x   = np.arange(len(sub))
        y   = 100*sub["P(Y>=4)_50%"].values
        lo  = 100*sub["P(Y>=4)_2.5%"].values
        hi  = 100*sub["P(Y>=4)_97.5%"].values
        err = np.vstack([y-lo, hi-y])
        ax.bar(x, y, yerr=err, capsize=3)
        ax.set_xticks(x, sub["label"], rotation=20, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("P(rating ≥ 4) [%]")
        ax.set_title(title_suffix)
        for xi, yi in zip(x, y):
            ax.text(xi, yi + 2, f"{yi:.0f}%", ha="center", va="bottom")

    # Row 1: population-average
    for ax, (kind, sub) in zip(axes[0], pop_all.groupby("kind")):
        draw(ax, sub, f"{kind} — Population average")

    # Row 2: new-participant predictive (if present)
    if have_new:
        for ax, (kind, sub) in zip(axes[1], new_all.groupby("kind")):
            draw(ax, sub, f"{kind} — New participant")

    fig.suptitle("Condition-level probability of high ratings (≥4)")
    fig.tight_layout()
    out = FIGS / "conditions_probs.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"✓ {out}")


# ---------- 3) Posterior Predictive Checks ----------
def ppc_plot(kind):
    """kind in {'realism','quality'}"""
    f = RESULTS / f"ppc_{kind}.csv"
    df = pd.read_csv(f).sort_values("cat")

    # categories are 0..4 -> show as 1..5
    cats   = (df["cat"].values + 1).astype(int)
    sim_lo = pct(df["sim_2.5%"].values)
    sim_md = pct(df["sim_50%"].values)
    sim_hi = pct(df["sim_97.5%"].values)
    obs    = pct(df["obs_prop"].values)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.fill_between(cats, sim_lo, sim_hi, alpha=0.25, label="95% posterior predictive")
    ax.plot(cats, sim_md, lw=2, label="posterior predictive median")
    ax.plot(cats, obs, "o", ms=6, label="observed")

    ax.set_xticks(cats)
    ax.set_xlabel("Likert category")
    ax.set_ylabel("Proportion [%]")

    # Robust ymax: at least 35%, or 1.2× the largest of sim_hi / obs, capped at 100%
    ymax = float(max(35, np.nanmax(np.concatenate([sim_hi, obs])) * 1.2))
    ymax = min(100, ymax)
    ax.set_ylim(0, ymax)

    ax.set_title(f"Posterior predictive check — {kind.capitalize()}")
    ax.legend(frameon=True)
    fig.tight_layout()
    out = FIGS / f"ppc_{kind}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {out}")


# ---------- main ----------
if __name__ == "__main__":
    # 1) Forest plots
    forest_plot("realism")
    forest_plot("quality")

    # 2) Condition plots
    cond_plot()

    # 3) PPC plots
    ppc_plot("realism")
    ppc_plot("quality")

    print("\nAll figures saved in:", FIGS.resolve())
