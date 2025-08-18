#!/usr/bin/env python3
# make_re_plots_random_effects.py
# Visualize participant-level random effects from your Pyro ordinal models.

from pathlib import Path
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import xarray as xr

# ---------- locations ----------
RES  = Path("results")
FIGS = Path("figs"); FIGS.mkdir(exist_ok=True)

# ---------- pretty defaults ----------
plt.style.use("seaborn-v0_8")
plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 300,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.25,
})

EFFECT_ORDER = ["Intercept", "avatar_c", "type_c"]
EFFECT_LABEL = {
    "Intercept": "Random intercept (baseline leniency)",
    "avatar_c":  "Random slope: Avatar (Unreal > Sync)",
    "type_c":    "Random slope: Disclosure (Positive > Negative)",
}
EFFECT_SHORT = {"Intercept":"intercept", "avatar_c":"avatar", "type_c":"type"}

def _load_pid_codes(csv_path="study3_long.csv"):
    """Recover participant codes in the same order np.unique would have used."""
    df = pd.read_csv(csv_path)
    # make column name robust
    cols = {c.lower().strip(): c for c in df.columns}
    key = None
    for cand in ["participant_code", "participantid", "participant_id"]:
        if cand in cols:
            key = cols[cand]
            break
    if key is None:
        # fallback: numbered labels
        n = len(pd.unique(df[df.columns[0]]))
        return [f"P{j+1:02d}" for j in range(n)]
    return sorted(df[key].astype(str).unique().tolist())

def _extract_b_samples(idata):
    """
    Return numpy array with shape [samples, subjects, 3] for random effects b.
    Works even if ArviZ stores dims as 'b_dim_0','b_dim_1', etc.
    """
    b = idata.posterior["b"]

    # Stack chains & draws into a single 'sample' dim
    if "chain" in b.dims and "draw" in b.dims:
        b = b.stack(sample=("chain", "draw"))

    # Remaining dims should be ['sample', <subjects>, <effects>]
    cand_dims = [d for d in b.dims if d != "sample"]
    if len(cand_dims) != 2:
        raise RuntimeError(f"Unexpected dims for 'b': {b.dims} (expected 2 non-sample dims)")

    # Identify the effects dim as the one with size == 3 (Intercept, avatar_c, type_c)
    sizes = {d: int(b.sizes[d]) for d in cand_dims}
    eff_dim = next((d for d, sz in sizes.items() if sz == 3), None)

    if eff_dim is None:
        # Fallback: if neither dim is 3, try last dim as effects
        eff_dim = cand_dims[-1]

    sub_dim = [d for d in cand_dims if d != eff_dim][0]

    # Reorder to [sample, subject, effect]
    b = b.transpose("sample", sub_dim, eff_dim)
    b_np = np.asarray(b)

    # If we guessed wrong, swap and check again
    if b_np.shape[2] != 3 and b_np.shape[1] == 3:
        b_np = np.transpose(b_np, (0, 2, 1))
        sub_dim, eff_dim = eff_dim, sub_dim

    if b_np.shape[2] != 3:
        raise RuntimeError(
            f"Couldn't infer effect axis. Shape={b_np.shape}, dims={b.dims}, sizes={sizes}"
        )

    return b_np, sub_dim


def _summarize_re(b_np, pid_codes, kind):
    """Return long DataFrame with per-participant summaries for each random effect."""
    S = b_np.shape[1]  # subjects
    if len(pid_codes) != S:
        # if mismatch, create generic labels
        pid_codes = [f"P{j+1:02d}" for j in range(S)]

    rows = []
    for e_idx, eff in enumerate(EFFECT_ORDER):
        draws = b_np[:, :, e_idx]  # [samples, subjects]
        q = np.quantile(draws, [0.025, 0.5, 0.975], axis=0)
        mean = draws.mean(axis=0)
        p_gt0 = (draws > 0).mean(axis=0)
        for j in range(S):
            rows.append(dict(
                outcome=kind.capitalize(),
                participant=pid_codes[j],
                effect=eff,
                mean=mean[j],
                q2_5=q[0, j], q50=q[1, j], q97_5=q[2, j],
                p_gt0=p_gt0[j],
            ))
    df = pd.DataFrame(rows)
    out = RES / f"re_summary_{kind}.csv"
    df.to_csv(out, index=False)
    print(f"✓ wrote {out}")
    return df

def _caterpillar(df_sub, title, out_png):
    """Horizontal 'caterpillar' plot: 95% CrI + median per participant for a given effect."""
    df = df_sub.sort_values("q50")
    y = np.arange(len(df))
    x_lo, x_md, x_hi = df["q2_5"].values, df["q50"].values, df["q97_5"].values
    colors = np.where(df["p_gt0"] >= 0.9, "#177245",    # strong positive (green)
              np.where(df["p_gt0"] <= 0.1, "#B22222",   # strong negative (red)
                       "#6C757D"))                      # uncertain (grey)

    h = max(6, 0.22*len(df) + 1.8)
    fig, ax = plt.subplots(figsize=(8.5, h))
    ax.hlines(y, x_lo, x_hi, color="#38598A", lw=1.6, alpha=0.85)
    ax.plot(x_md, y, "o", ms=4, color="#1f77b4")
    for yi, col in zip(y, colors):
        ax.plot([x_lo[yi], x_hi[yi]], [yi, yi], color=col, lw=3, alpha=0.25)
    ax.axvline(0.0, color="k", lw=1, alpha=0.6)
    ax.set_yticks(y, df["participant"].values)
    ax.set_xlabel("Effect (latent logit units; 0 = population average)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)
    print(f"✓ {out_png}")

def _hist_and_scatter(df, kind, out_prefix):
    """(a) Histograms of participant means for each effect; (b) scatter of intercept vs slopes."""
    # (a) histograms
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, eff in zip(axes, EFFECT_ORDER):
        s = df.query("effect == @eff")["mean"]
        ax.hist(s, bins=20, alpha=0.85)
        ax.axvline(0, color="k", lw=1)
        ax.set_title(EFFECT_LABEL[eff])
        ax.set_xlabel("Participant mean effect")
        ax.set_ylabel("Count")
    fig.suptitle(f"Participant heterogeneity — {kind.capitalize()}")
    fig.tight_layout()
    p_hist = FIGS / f"{out_prefix}_hist.png"
    fig.savefig(p_hist, bbox_inches="tight"); plt.close(fig)
    print(f"✓ {p_hist}")

    # (b) scatter: intercept vs slopes (posterior means)
    def _scatter(ax, xeff, yeff, label):
        x = df.query("effect == @xeff")[["participant","mean"]].set_index("participant")
        y = df.query("effect == @yeff")[["participant","mean"]].set_index("participant")
        m = x.join(y, lsuffix="_x", rsuffix="_y").dropna()
        ax.scatter(m["mean_x"], m["mean_y"], s=25, alpha=0.85)
        ax.axvline(0, color="k", lw=1); ax.axhline(0, color="k", lw=1)
        r = np.corrcoef(m["mean_x"], m["mean_y"])[0,1]
        ax.set_xlabel("Intercept (baseline)"); ax.set_ylabel(label)
        ax.set_title(f"r = {r:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    _scatter(axes[0], "Intercept", "avatar_c",
             "Avatar slope (Unreal > Sync)")
    _scatter(axes[1], "Intercept", "type_c",
             "Disclosure slope (Positive > Negative)")
    fig.suptitle(f"Intercept–slope relationships — {kind.capitalize()}")
    fig.tight_layout()
    p_scat = FIGS / f"{out_prefix}_scatter.png"
    fig.savefig(p_scat, bbox_inches="tight"); plt.close(fig)
    print(f"✓ {p_scat}")

def make_plots_for(kind):
    """kind ∈ {'realism','quality'}; reads results/study3_{kind}_idata.nc."""
    nc = RES / f"study3_{kind}_idata.nc"
    if not nc.exists():
        raise FileNotFoundError(f"Cannot find {nc}. Run your phase-2 script first.")

    idata = az.from_netcdf(nc)
    b_np, sub_dim = _extract_b_samples(idata)
    pid_codes = _load_pid_codes("study3_long.csv")

    # Summarize & save
    df = _summarize_re(b_np, pid_codes, kind)

    # Caterpillars
    for eff in EFFECT_ORDER:
        _caterpillar(
            df_sub=df.query("effect == @eff"),
            title=f"{EFFECT_LABEL[eff]} — {kind.capitalize()}",
            out_png=FIGS / f"re_caterpillar_{kind}_{EFFECT_SHORT[eff]}.png"
        )

    # Histograms + intercept–slope scatter
    _hist_and_scatter(df, kind, out_prefix=f"re_{kind}")

if __name__ == "__main__":
    for k in ["realism", "quality"]:
        make_plots_for(k)
    print("\nDone. See the 'figs/' folder and 'results/re_summary_*.csv'.")
