# make_ppc_obs_vs_model.py  — FIXED: align categories (0-based vs 1-based) + use cleaned data
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_utils import load_study3  # same cleaner the model used

# ---------- read PPC (robust) ----------
def read_ppc(csv_path: str) -> pd.DataFrame:
    ppc = pd.read_csv(csv_path)

    # find category column
    cat_col = None
    for c in ppc.columns:
        lc = c.lower()
        if lc in ("k", "category", "cat", "rating"):
            cat_col = c
            break
    if cat_col is None:
        # fall back to 1..N if nothing found
        ppc = ppc.copy()
        ppc["k"] = np.arange(1, len(ppc) + 1)
        cat_col = "k"

    # locate predictive columns (common names from our pipeline)
    def pick(*names, contains=None):
        # exact name first
        for n in names:
            for c in ppc.columns:
                if c.lower() == n.lower():
                    return c
        # then substring fallback
        if contains is not None:
            for c in ppc.columns:
                if contains.lower() in c.lower():
                    return c
        return None

    med = pick("sim_50%", "median", "pred_med", contains="50")
    lo  = pick("sim_2.5%", "lower", "pred_lo", contains="2.5")
    hi  = pick("sim_97.5%", "upper", "pred_hi", contains="97.5")

    missing = [name for name, col in [("median", med), ("lower", lo), ("upper", hi)] if col is None]
    if missing:
        raise ValueError(f"Missing PPC columns {missing} in {csv_path}. Columns: {list(ppc.columns)}")

    out = ppc[[cat_col, med, lo, hi]].copy()
    out.columns = ["k", "pred_med", "pred_lo", "pred_hi"]

    # ---- CRUCIAL FIX: if PPC categories are 0..4, convert to 1..5
    # (this is what causes the “dot for 5 plotted at 4” symptom)
    kmin, kmax = pd.to_numeric(out["k"], errors="coerce").min(), pd.to_numeric(out["k"], errors="coerce").max()
    if (kmin == 0) and (kmax == 4):
        out["k"] = out["k"] + 1

    out = out.sort_values("k")
    for c in ["k", "pred_med", "pred_lo", "pred_hi"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

# ---------- observed from same cleaned analysis dataset ----------
def observed_from_model_dataset(study3_csv_path: str, outcome: str) -> pd.DataFrame:
    df = load_study3(study3_csv_path)  # applies the exact modeling filters
    col = "realism" if outcome.lower().startswith("real") else "quality"
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    x = x[(x >= 1) & (x <= 5)]

    counts = x.value_counts().sort_index()
    for k in range(1, 6):
        if k not in counts.index:
            counts.loc[k] = 0
    counts = counts.sort_index()
    prop = (counts / counts.sum()).reset_index()
    prop.columns = ["k", "obs_prop"]
    return prop

# ---------- plotting ----------
def make_panel(ppc_file: str, study3_file: str, which: str, title: str, outfile: str):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    ppc = read_ppc(ppc_file)                               # k, pred_med, pred_lo, pred_hi
    obs = observed_from_model_dataset(study3_file, which)  # k, obs_prop

    # tiny diagnostic so you can verify alignment in the terminal
    print(f"\n[{which}] PPC k values: {ppc['k'].tolist()}")
    print(f"[{which}] OBS k values: {obs['k'].tolist()}")

    df = pd.merge(ppc, obs, on="k", how="outer").sort_values("k")
    df["obs_prop"] = df["obs_prop"].fillna(0.0)

    plt.figure(figsize=(9.8, 5.6))
    ax = plt.gca()

    ax.bar(df["k"], df["obs_prop"], width=0.8, color="0.82", edgecolor="0.55", label="Observed")
    ax.errorbar(
        df["k"], df["pred_med"],
        yerr=[df["pred_med"] - df["pred_lo"], df["pred_hi"] - df["pred_med"]],
        fmt="o", color="k", ecolor="k", elinewidth=2, capsize=4,
        label="Model Prediction 95% CI"
    )

    label = "Realism" if which.lower().startswith("real") else "Enjoyment"
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel(f"Rating Category ({label})")
    ax.set_ylabel("Proportion of responses")
    ax.set_ylim(0, max(0.5, float(df[["obs_prop", "pred_hi"]].max().max()) * 1.15))
    ax.set_title(title)
    ax.legend(frameon=True, loc="upper right")

    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"✓ Saved {outfile}")

def main():
    make_panel(
        ppc_file="results/ppc_realism.csv",
        study3_file="study3_long.csv",
        which="realism",
        title="Realism Ratings Distribution: Observed vs Model Prediction",
        outfile="figs/ppc_obs_vs_model_realism.png",
    )
    make_panel(
        ppc_file="results/ppc_quality.csv",
        study3_file="study3_long.csv",
        which="quality",
        title="Enjoyment Ratings Distribution: Observed vs Model Prediction",
        outfile="figs/ppc_obs_vs_model_quality.png",
    )

if __name__ == "__main__":
    main()
