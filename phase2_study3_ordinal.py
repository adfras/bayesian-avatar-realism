# phase2_study3_ordinal.py
import os, json, numpy as np, torch, pyro
import pandas as pd
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive
import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import load_study3

torch.set_default_dtype(torch.double)
pyro.set_rng_seed(314159)

NUM_SAMPLES = 800
WARMUP      = 800
K_CATS      = 5  # Likert 1..5 -> model uses 0..4
EPS         = 1e-9

# ---------- design & link ----------
def build_design_ordinal(df, outcome_col, K=K_CATS):
    X = np.column_stack([
        np.ones(len(df)),
        df["avatar_c"].values,
        df["type_c"].values,
        (df["avatar_c"]*df["type_c"]).values
    ])
    Z = np.column_stack([
        np.ones(len(df)),
        df["avatar_c"].values,
        df["type_c"].values
    ])
    pid_codes, pid_idx = np.unique(df["participant_id"].values, return_inverse=True)
    y = df[outcome_col].values.astype(int) - 1  # 0..K-1
    assert y.min() >= 0 and y.max() < K, "Outcome must be 1..K"
    return (torch.tensor(X),
            torch.tensor(Z),
            torch.tensor(pid_idx, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
            pid_codes)

def ordered_probs(eta, cutpoints):
    """
    eta: (N,)
    cutpoints: (K-1,)
    returns probs (N,K) using cumulative logit: P(Y<=k)=sigmoid(kappa_k - eta)
    """
    cdf   = torch.sigmoid(cutpoints - eta.unsqueeze(-1))  # (N, K-1)
    first = cdf[:, :1]
    middle= cdf[:, 1:] - cdf[:, :-1] if cutpoints.shape[0] > 1 else torch.empty(eta.shape[0],0, dtype=eta.dtype, device=eta.device)
    last  = 1 - cdf[:, -1:]
    return torch.cat([first, middle, last], dim=1)

def kappa_from_raw(k_raw):
    # Enforce ordered cutpoints via positive increments, then center
    inc = torch.nn.functional.softplus(k_raw) + 1e-3
    kap = torch.cumsum(inc, dim=-1)
    return kap - kap.mean(dim=-1, keepdim=True)

# ---------- model ----------
def ordinal_model(X, Z, pid_idx, y, hyper=None, resp_tag="realism"):
    N, P = X.shape
    _, R = Z.shape
    S = int(pid_idx.max().item()) + 1
    K = int(y.max().item()) + 1 if y is not None else K_CATS

    # Prior means for betas from Study-2 when available
    default_loc = torch.zeros(P)
    if hyper is not None:
        names = [f"b_{resp_tag}_Intercept", f"b_{resp_tag}_avatar_c",
                 f"b_{resp_tag}_type_c", f"b_{resp_tag}_avatar_c:type_c"]
        loc = [float(hyper.get("b_means", {}).get(n, 0.0)) for n in names]
        default_loc = torch.tensor(loc, dtype=torch.double)

    beta  = pyro.sample("beta",  dist.Normal(default_loc, 0.3*torch.ones(P)))
    sd_re = pyro.sample("sd_re", dist.HalfNormal(0.5*torch.ones(R)))
    with pyro.plate("subjects", S):
        b = pyro.sample("b", dist.Normal(torch.zeros(R), sd_re).to_event(1))  # (S,R)

    k_raw = pyro.sample("k_raw", dist.Normal(torch.zeros(K-1), torch.ones(K-1)))
    kappa = kappa_from_raw(k_raw)  # ordered cutpoints

    eta   = (X @ beta) + (Z * b[pid_idx]).sum(-1)  # (N,)
    probs = ordered_probs(eta, kappa)              # (N,K)
    pyro.sample("y", dist.Categorical(probs=probs), obs=y)

def fit_ordinal(X, Z, pid_idx, y, hyper, resp_tag, num_samples=NUM_SAMPLES, warmup=WARMUP):
    nuts = NUTS(lambda X,Z,pid_idx,y: ordinal_model(X,Z,pid_idx,y,hyper,resp_tag),
                target_accept_prob=0.9)
    mcmc = MCMC(nuts, num_samples=num_samples, warmup_steps=warmup)
    mcmc.run(X, Z, pid_idx, y)
    return mcmc

# ---------- reporting helpers ----------
def save_arviz_and_traces(mcmc, tag):
    os.makedirs("results", exist_ok=True)
    idata = az.from_pyro(mcmc)
    idata.to_netcdf(f"results/study3_{tag}_idata.nc")
    summ  = az.summary(idata, var_names=["beta","sd_re"], kind="stats")
    summ.to_csv(f"results/summary_{tag}.csv")
    az.plot_trace(idata, var_names=["beta","sd_re"])
    plt.savefig(f"results/trace_{tag}.png", bbox_inches="tight"); plt.close()
    print(f"✓ saved results/summary_{tag}.csv and results/trace_{tag}.png")
    return idata

def effect_table(mcmc, tag):
    coef_names = ["Intercept","avatar_c","type_c","avatar_c:type_c"]
    post = mcmc.get_samples()
    beta = post["beta"].numpy()           # [S,4]
    OR   = np.exp(beta)                   # odds ratios
    q    = lambda x: np.quantile(x, [0.025, 0.5, 0.975], axis=0)
    q_beta, q_or = q(beta), q(OR)
    p_gt0 = (beta > 0).mean(axis=0)

    df = pd.DataFrame({
        "param": coef_names,
        "beta_2.5%":  q_beta[0], "beta_50%":  q_beta[1], "beta_97.5%": q_beta[2],
        "OR_2.5%":    q_or[0],   "OR_50%":    q_or[1],   "OR_97.5%":   q_or[2],
        "Pr(beta>0)": p_gt0
    })
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/effects_{tag}.csv", index=False)
    print(f"✓ saved results/effects_{tag}.csv")

def summarize_conditions(mcmc, tag, K=K_CATS, which="population"):
    """
    Summarize P(Y>=4) for 4 conditions under either:
      which='population'     -> population-average participant (b = 0)
      which='new_participant'-> new participant drawn from RE SDs
    """
    post  = mcmc.get_samples()
    beta  = post["beta"]       # [S,4]
    sd_re = post["sd_re"]      # [S,3]
    k_raw = post["k_raw"]      # [S,K-1]
    kappa = kappa_from_raw(k_raw)

    conds = [
        ("first_neg", -0.5, -0.5),
        ("first_pos", -0.5,  0.5),
        ("third_neg",  0.5, -0.5),
        ("third_pos",  0.5,  0.5),
    ]
    Xc = torch.tensor([[1., a, t, a*t] for _,a,t in conds], dtype=torch.double)   # [4,4]
    Zc = torch.tensor([[1., a, t]       for _,a,t in conds], dtype=torch.double)  # [4,3]

    if which == "population":
        b_draw = torch.zeros_like(sd_re)                 # mean participant
    elif which == "new_participant":
        b_draw = torch.randn_like(sd_re) * sd_re         # random new participant
    else:
        raise ValueError("which must be 'population' or 'new_participant'")

    eta   = (beta @ Xc.T) + (b_draw @ Zc.T)              # [S,4]
    cdf   = torch.sigmoid(kappa[:,None,:] - eta[:,:,None])
    first = cdf[:,:,:1]
    middle= cdf[:,:,1:] - cdf[:,:,:-1] if K>2 else torch.empty(eta.shape[0], eta.shape[1], 0, dtype=eta.dtype)
    last  = 1 - cdf[:,:,-1:]
    probs = torch.cat([first, middle, last], dim=-1)     # [S,4,K]
    p_top2= probs.index_select(-1, torch.tensor([3,4])).sum(-1)  # [S,4]
    qs    = torch.quantile(p_top2, torch.tensor([0.025,0.5,0.975], dtype=torch.double), dim=0)

    out = pd.DataFrame({
        "condition": [c for c,_,_ in conds],
        "P(Y>=4)_2.5%": qs[0].numpy(),
        "P(Y>=4)_50%":  qs[1].numpy(),
        "P(Y>=4)_97.5%":qs[2].numpy(),
        "which": which
    })
    os.makedirs("results", exist_ok=True)
    out.to_csv(f"results/cond_probs_{tag}_{which}.csv", index=False)
    print(f"✓ saved results/cond_probs_{tag}_{which}.csv")

def posterior_predictive_check(mcmc, model_fn, X, Z, pid_idx, y_obs, tag):
    pred = Predictive(model_fn, posterior_samples=mcmc.get_samples(), return_sites=["y"])
    sim  = pred(X, Z, pid_idx, None)["y"].numpy()           # [S, N]
    K    = int(sim.max()) + 1
    obs_prop = np.bincount(y_obs, minlength=K) / len(y_obs)
    sim_prop = np.stack([(sim==k).mean(axis=1) for k in range(K)], axis=1)  # [S,K]
    lo, med, hi = np.quantile(sim_prop, [0.025,0.5,0.975], axis=0)

    ppc_df = pd.DataFrame({"cat": np.arange(K),
                           "obs_prop": obs_prop,
                           "sim_2.5%": lo, "sim_50%": med, "sim_97.5%": hi})
    os.makedirs("results", exist_ok=True)
    ppc_df.to_csv(f"results/ppc_{tag}.csv", index=False)
    print(f"✓ saved results/ppc_{tag}.csv")

# ---------- NEW: information gain for random effects ----------
def _kl_normal_1d(mu_q, sd_q, mu_p, sd_p):
    # KL(N(mu_q, sd_q) || N(mu_p, sd_p)) in nats (1D)
    sd_q = max(sd_q, EPS)
    sd_p = max(sd_p, EPS)
    return np.log(sd_p/sd_q) + (sd_q**2 + (mu_q - mu_p)**2)/(2*sd_p**2) - 0.5

def information_gain_random_effects(mcmc, pid_codes, tag_out):
    """
    Compute per-participant information gain for each random-effect dimension:
      - KL divergence (nats, bits) from prior N(0, sd_re[r]) to posterior N(mu_ir, sd_ir)
      - shrinkage = 1 - sd_ir / sd_re_prior
    Writes:
      results/info_gain_random_effects_{tag}_by_pid.csv
      results/info_gain_random_effects_{tag}_by_param.csv
    """
    os.makedirs("results", exist_ok=True)
    post   = mcmc.get_samples()
    if "b" not in post or "sd_re" not in post:
        raise RuntimeError("Posterior samples missing 'b' or 'sd_re'.")

    b      = post["b"].numpy()          # [Samps, Subjects, R]
    sd_re  = post["sd_re"].numpy()      # [Samps, R]
    Samps, Subjects, R = b.shape
    re_names = ["Intercept","avatar_c","type_c"]

    # Use posterior-mean sd_re as the prior SD for each RE dimension
    prior_sd = sd_re.mean(axis=0)  # [R]

    rows = []
    for i in range(Subjects):
        pid = pid_codes[i]
        for r in range(R):
            draws = b[:, i, r]
            mu    = float(np.mean(draws))
            sd    = float(np.std(draws, ddof=1))
            psd   = float(prior_sd[r])

            kl_nats = _kl_normal_1d(mu_q=mu, sd_q=sd, mu_p=0.0, sd_p=psd)
            kl_bits = kl_nats / np.log(2.0)
            shrink  = 1.0 - (sd / (psd + EPS))
            q025, q50, q975 = np.quantile(draws, [0.025, 0.5, 0.975])

            rows.append({
                "pid": pid,
                "re_param": re_names[r],
                "post_mean": mu,
                "post_sd": sd,
                "prior_sd": psd,
                "KL_nats": kl_nats,
                "KL_bits": kl_bits,
                "shrinkage": shrink,
                "Pr>0": float((draws > 0).mean()),
                "q2.5%": q025, "q50%": q50, "q97.5%": q975
            })

    df = pd.DataFrame(rows)
    df.to_csv(f"results/info_gain_random_effects_{tag_out}_by_pid.csv", index=False)

    # Aggregated (this is where your original error came from: need explicit aggs)
    summ = df.groupby("re_param").agg(
        KL_bits_mean   = ("KL_bits", "mean"),
        KL_bits_median = ("KL_bits", "median"),
        KL_bits_sum    = ("KL_bits", "sum"),
        KL_bits_sd     = ("KL_bits", "std"),
        KL_nats_mean   = ("KL_nats", "mean"),
        shrinkage_mean = ("shrinkage","mean"),
        shrinkage_median=("shrinkage","median"),
        post_sd_mean   = ("post_sd","mean"),
        prior_sd_mean  = ("prior_sd","mean"),
        n_participants = ("pid","nunique")
    ).reset_index()
    summ.to_csv(f"results/info_gain_random_effects_{tag_out}_by_param.csv", index=False)

    print(f"✓ saved results/info_gain_random_effects_{tag_out}_by_pid.csv "
          f"and results/info_gain_random_effects_{tag_out}_by_param.csv")

# ---------- main ----------
def main():
    # 1) Load hyperpriors from Phase-1 (Study-2 fit)
    with open("study2_hyperpriors.json","r") as f:
        hyper = json.load(f)

    # 2) Load Study-3 data
    df3 = load_study3("study3_long.csv")  # same study design as described in Study-3 paper

    # 3) Build designs (y in 0..4)
    Xr, Zr, pidr, yr, pid_codes_r = build_design_ordinal(df3, "realism", K=K_CATS)
    Xq, Zq, pidq, yq, pid_codes_q = build_design_ordinal(df3, "quality", K=K_CATS)

    # 4) Fit realism & quality models
    mcmc_r = fit_ordinal(Xr, Zr, pidr, yr, hyper, "realism")
    print("✓ Study-3 REALISM ordinal fit done")
    mcmc_q = fit_ordinal(Xq, Zq, pidq, yq, hyper, "quality")
    print("✓ Study-3 QUALITY ordinal fit done")

    # 5) Save diagnostics & traceplots
    save_arviz_and_traces(mcmc_r, "realism")
    save_arviz_and_traces(mcmc_q, "quality")

    # 6) Effects & condition probabilities
    effect_table(mcmc_r, "realism")
    effect_table(mcmc_q, "quality")
    # population-average (tight, presentation-friendly)
    summarize_conditions(mcmc_r, "realism", which="population")
    summarize_conditions(mcmc_q, "quality", which="population")
    # new-participant predictive (wide, conservative)
    summarize_conditions(mcmc_r, "realism", which="new_participant")
    summarize_conditions(mcmc_q, "quality", which="new_participant")

    # 7) Posterior predictive checks
    model_r = lambda X,Z,pid_idx,y=None: ordinal_model(X,Z,pid_idx,y,hyper,"realism")
    model_q = lambda X,Z,pid_idx,y=None: ordinal_model(X,Z,pid_idx,y,hyper,"quality")
    posterior_predictive_check(mcmc_r, model_r, Xr, Zr, pidr, yr.numpy(), "realism")
    posterior_predictive_check(mcmc_q, model_q, Xq, Zq, pidq, yq.numpy(), "quality")

    # 8) Information gain (random effects) — per participant and per RE dimension
    information_gain_random_effects(mcmc_r, pid_codes_r, tag_out="realism")
    information_gain_random_effects(mcmc_q, pid_codes_q, tag_out="quality")

if __name__ == "__main__":
    main()
