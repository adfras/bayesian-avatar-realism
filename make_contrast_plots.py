# make_contrast_plots.py
# Produce clean contrast plots: Unreal−Sync and Positive−Negative, with 89% CrIs.
import os
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def _kappa_from_kraw(k_raw):
    """
    Transform unconstrained k_raw (K-1 thresholds) into ordered, centered kappas.
    Same transform we used in the Pyro model: softplus + cumsum, then center.
    """
    increments = np.log1p(np.exp(k_raw)) + 1e-3       # softplus + jitter
    kappa = np.cumsum(increments, axis=1)            # enforce ordering
    kappa = kappa - kappa.mean(axis=1, keepdims=True)  # center per draw
    return kappa

def _load_draws(nc_path):
    idata = az.from_netcdf(nc_path)
    post = idata.posterior
    # (chain, draw, P) -> (S, P)
    beta = post["beta"].values
    beta = beta.reshape(-1, beta.shape[-1])
    k_raw = post["k_raw"].values
    k_raw = k_raw.reshape(-1, k_raw.shape[-1])
    kappa = _kappa_from_kraw(k_raw)
    return beta, kappa

def _eta(beta, a, t):
    # beta columns: [Intercept, avatar, type, (interaction?)]
    eta = beta[:, 0] + beta[:, 1]*a + beta[:, 2]*t
    if beta.shape[1] > 3:
        eta = eta + beta[:, 3]*(a*t)
    return eta

def p_ge4(beta, kappa, a, t):
    """
    For K=5 categories: P(Y >= 4) = 1 - logistic(kappa3 - eta)
    kappa[:, 2] is the 3rd threshold (between 3 and 4), 0-based index.
    """
    eta_val = _eta(beta, a, t)
    return 1.0 - _sigmoid(kappa[:, 2] - eta_val)

def qtile(x, level=0.89):
    lo = (1.0 - level) / 2.0
    hi = 1.0 - lo
    return np.quantile(x, [lo, 0.5, hi])

def make_contrast_figure(beta, kappa, title, savepath):
    # Main effects at the coded "grand mean" of the other factor (0.0)
    # (This matches effect coding with -0.5/+0.5.)
    avatar_main = p_ge4(beta, kappa, +0.5, 0.0) - p_ge4(beta, kappa, -0.5, 0.0)
    type_main   = p_ge4(beta, kappa, 0.0, +0.5) - p_ge4(beta, kappa, 0.0, -0.5)

    # Simple effects (optional, often useful)
    avatar_pos  = p_ge4(beta, kappa, +0.5, +0.5) - p_ge4(beta, kappa, -0.5, +0.5)
    avatar_neg  = p_ge4(beta, kappa, +0.5, -0.5) - p_ge4(beta, kappa, -0.5, -0.5)
    type_sync   = p_ge4(beta, kappa, -0.5, +0.5) - p_ge4(beta, kappa, -0.5, -0.5)
    type_unreal = p_ge4(beta, kappa, +0.5, +0.5) - p_ge4(beta, kappa, +0.5, -0.5)

    items = [
        ("Avatar (main)", avatar_main),
        ("Disclosure (main)", type_main),
        ("Avatar @ Positive", avatar_pos),
        ("Avatar @ Negative", avatar_neg),
        ("Disclosure @ Sync", type_sync),
        ("Disclosure @ Unreal", type_unreal),
    ]

    meds, los, his, pr_gt0 = [], [], [], []
    for _, arr in items:
        lo, med, hi = qtile(arr, 0.89)
        meds.append(med); los.append(lo); his.append(hi)
        pr_gt0.append((arr > 0).mean())

    y = np.arange(len(items))
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.hlines(y, los, his, lw=6, alpha=0.35)
    ax.plot(meds, y, "o", ms=7)
    ax.vlines(0, -1, len(items), color="k", lw=1, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{name}  (Pr>0 = {p:.2f})" for (name, _), p in zip(items, pr_gt0)], fontsize=10)
    ax.set_xlabel("Difference in probability of rating ≥ 4")
    ax.set_title(title)
    ax.set_xlim(-0.5, 0.5)  # adjust if your effects are larger/smaller
    fig.tight_layout()
    os.makedirs("figs", exist_ok=True)
    fig.savefig(savepath, dpi=220)
    plt.close(fig)

if __name__ == "__main__":
    # Realism
    beta, kappa = _load_draws("results/study3_realism_idata.nc")
    make_contrast_figure(beta, kappa, "Contrasts — Realism", "figs/contrasts_realism.png")

    # Quality
    beta, kappa = _load_draws("results/study3_quality_idata.nc")
    make_contrast_figure(beta, kappa, "Contrasts — Quality", "figs/contrasts_quality.png")
