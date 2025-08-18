# make_prior_vs_posterior_predictive.py
# Figure S2: Compare prior-predictive vs posterior-predictive P(Y>=4) for a NEW participant.
import json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import torch

plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 300})

CONDS = [
    ("first_neg", -0.5, -0.5),
    ("first_pos", -0.5,  0.5),
    ("third_neg",  0.5, -0.5),
    ("third_pos",  0.5,  0.5),
]

def _sd_names(tag): return [
    f"sd_participant_id__{tag}_Intercept",
    f"sd_participant_id__{tag}_avatar_c",
    f"sd_participant_id__{tag}_type_c",
]

def _kappa_from_kraw(k_raw):
    inc = np.log1p(np.exp(k_raw)) + 1e-3
    kap = np.cumsum(inc, axis=1)
    return kap - kap.mean(axis=1, keepdims=True)

def _p_ge4(beta, kappa, a, t):
    eta = beta[:,0] + beta[:,1]*a + beta[:,2]*t + (beta[:,3] if beta.shape[1] > 3 else 0.0)*(a*t)
    # K=5 -> index 2 is threshold between 3 and 4
    return 1.0 - (1.0 / (1.0 + np.exp(-(kappa[:,2] - eta))))

def _qtile(x, level=0.89):
    lo = (1.0 - level)/2.0; hi = 1.0 - lo
    return np.quantile(x, [lo, 0.5, hi])

def prior_predictive(tag, hyper, S=4000):
    # Priors: beta ~ N(b_means, 0.3); sd_re ~ LogNormal(mu, sd); k_raw ~ N(0,1)
    names = _sd_names(tag)
    b_names = [f"b_{tag}_Intercept", f"b_{tag}_avatar_c", f"b_{tag}_type_c", f"b_{tag}_avatar_c:type_c"]
    loc = np.array([float(hyper.get("b_means", {}).get(n, 0.0)) for n in b_names])
    beta = np.random.normal(loc=loc, scale=0.3, size=(S, len(b_names)))
    mu = np.array([float(hyper.get("sd_log_mu", {}).get(n, np.log(0.5))) for n in names])
    sd = np.array([float(hyper.get("sd_log_sd", {}).get(n, 0.4)) for n in names])
    sd_re = np.exp(np.random.normal(loc=mu, scale=sd, size=(S, 3)))
    b_new = np.random.normal(loc=0.0, scale=sd_re)  # S x 3
    # draw thresholds
    k_raw = np.random.normal(0.0, 1.0, size=(S, 4))
    kappa = _kappa_from_kraw(k_raw)
    # compute p>=4 by condition for NEW participant
    out = {}
    for name, a, t in CONDS:
        X = np.array([1., a, t, a*t])
        Z = np.array([1., a, t])
        eta_beta = beta @ X
        eta_b    = (b_new @ Z)
        p = 1.0 - (1.0 / (1.0 + np.exp(-(kappa[:,2] - (eta_beta + eta_b)))))
        out[name] = p
    return out  # dict name -> S-vector

def posterior_predictive(tag, nc_path):
    idata = az.from_netcdf(nc_path)
    beta  = idata.posterior["beta"].values.reshape(-1, idata.posterior["beta"].shape[-1])
    sd_re = idata.posterior["sd_re"].values.reshape(-1, idata.posterior["sd_re"].shape[-1])
    k_raw = idata.posterior["k_raw"].values.reshape(-1, idata.posterior["k_raw"].shape[-1])
    kappa = _kappa_from_kraw(k_raw)
    S = min(len(beta), 8000)
    idx = np.random.choice(len(beta), size=S, replace=False)
    beta, sd_re, kappa = beta[idx], sd_re[idx], kappa[idx]
    b_new = np.random.normal(loc=0.0, scale=sd_re)  # S x 3
    out = {}
    for name, a, t in CONDS:
        X = np.array([1., a, t, a*t])
        Z = np.array([1., a, t])
        eta_beta = beta @ X
        eta_b    = (b_new @ Z)
        p = 1.0 - (1.0 / (1.0 + np.exp(-(kappa[:,2] - (eta_beta + eta_b)))))
        out[name] = p
    return out

def make_panel(tag, nc_path, fig_path):
    with open("study2_hyperpriors.json","r") as f:
        hyper = json.load(f)

    prior = prior_predictive(tag, hyper)
    post  = posterior_predictive(tag, nc_path)

    names = [c[0] for c in CONDS]
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    x = np.arange(len(names))
    w = 0.38

    def summary(arr): lo, med, hi = _qtile(arr, 0.89); return lo, med, hi

    # bars with error bars
    prior_stats = np.array([summary(prior[n]) for n in names])
    post_stats  = np.array([summary(post[n])  for n in names])

    ax.bar(x - w/2, post_stats[:,1], w, yerr=[post_stats[:,1]-post_stats[:,0], post_stats[:,2]-post_stats[:,1]],
           capsize=3, label="Posterior predictive (Study 3)")
    ax.bar(x + w/2, prior_stats[:,1], w, yerr=[prior_stats[:,1]-prior_stats[:,0], prior_stats[:,2]-prior_stats[:,1]],
           capsize=3, label="Prior predictive (Study‑2‑informed)")

    ax.set_xticks(x); ax.set_xticklabels(["First•Neg","First•Pos","Third•Neg","Third•Pos"], rotation=15)
    ax.set_ylabel("P(rating ≥ 4)")
    ax.set_ylim(0, 1)
    ax.set_title(f"New‑participant predictions — {tag.capitalize()}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("✓", fig_path)

if __name__ == "__main__":
    make_panel("realism", "results/study3_realism_idata.nc", "figs/prior_vs_posterior_new_participant_realism.png")
    make_panel("quality", "results/study3_quality_idata.nc", "figs/prior_vs_posterior_new_participant_quality.png")
