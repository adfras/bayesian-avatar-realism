# make_hyperprior_transfer_plots.py
# Figure S1: Overlay Study-2–informed LogNormal prior for σ with Study-3 posterior of σ.
from pathlib import Path
import json, numpy as np, pandas as pd
import arviz as az
import matplotlib.pyplot as plt

FIGS = Path("figs"); FIGS.mkdir(exist_ok=True)

def _sd_names(resp_tag):
    return [
        f"sd_participant_id__{resp_tag}_Intercept",
        f"sd_participant_id__{resp_tag}_avatar_c",
        f"sd_participant_id__{resp_tag}_type_c",
    ]

def _lognormal_pdf(x, mu, s):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    m = (x > 0)
    out[m] = (1.0 / (x[m] * s * np.sqrt(2*np.pi))) * np.exp( - (np.log(x[m]) - mu)**2 / (2*s**2) )
    return out

def _load_prior_params(hyper, resp_tag):
    names = _sd_names(resp_tag)
    lmu_map = hyper.get("sd_log_mu", {})
    lsd_map = hyper.get("sd_log_sd", {})
    mus = [float(lmu_map.get(n, np.log(float(hyper.get("sd_means", {}).get(n, 0.5))))) for n in names]
    sds = [float(max(lsd_map.get(n, 0.4), 1e-4)) for n in names]
    return names, np.array(mus), np.array(sds)

def _load_posterior_draws(nc_path):
    idata = az.from_netcdf(nc_path)
    sd = idata.posterior["sd_re"].values  # shape [chain, draw, R]
    sd = sd.reshape(-1, sd.shape[-1])     # [Samps, R]
    return sd

def _panel_ax(ax, xgrid, prior_mu, prior_sd, post_draws_1d, title):
    # posterior density via simple histogram (no scipy)
    density, edges = np.histogram(post_draws_1d, bins=60, range=(xgrid.min(), xgrid.max()), density=True)
    mids = 0.5*(edges[:-1] + edges[1:])
    ax.plot(mids, density, lw=2, label="Posterior (Study 3)")
    ax.plot(xgrid, _lognormal_pdf(xgrid, prior_mu, prior_sd), lw=2, linestyle="--", label="Prior (Study-2-informed)")
    ax.set_title(title); ax.set_xlabel("σ (random-effect SD)"); ax.set_ylabel("Density")
    ax.legend(frameon=False)

def make_figure_for(resp_tag, nc_path, out_png):
    with open("study2_hyperpriors.json","r") as f: hyper = json.load(f)
    names, mus, sds = _load_prior_params(hyper, resp_tag)
    sd = _load_posterior_draws(nc_path)  # [Samps, 3]
    # grid per dimension
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for j, (nm, mu, s) in enumerate(zip(names, mus, sds)):
        draws = sd[:, j]
        x_max = max(np.percentile(draws, 99.5), np.exp(mu + 4*s))
        xgrid = np.linspace(1e-3, float(x_max), 600)
        _panel_ax(axes[j], xgrid, mu, s, draws, title=nm.split("__")[-1])
    fig.suptitle(f"Hyperprior transfer check (σ) — {resp_tag.capitalize()}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("✓", out_png)

if __name__ == "__main__":
    make_figure_for("realism", "results/study3_realism_idata.nc", "figs/hyperprior_vs_posterior_sigma_realism.png")
    make_figure_for("quality", "results/study3_quality_idata.nc", "figs/hyperprior_vs_posterior_sigma_quality.png")
