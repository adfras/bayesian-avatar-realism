# phase2_study3_ordinal.py
import os, json, argparse, numpy as np, torch, pyro
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
K_CATS      = 5
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
    cdf   = torch.sigmoid(cutpoints - eta.unsqueeze(-1))  # (N, K-1)
    first = cdf[:, :1]
    middle= cdf[:, 1:] - cdf[:, :-1] if cutpoints.shape[0] > 1 else torch.empty(eta.shape[0],0, dtype=eta.dtype, device=eta.device)
    last  = 1 - cdf[:, -1:]
    return torch.cat([first, middle, last], dim=1)

def kappa_from_raw(k_raw):
    inc = torch.nn.functional.softplus(k_raw) + 1e-3
    kap = torch.cumsum(inc, dim=-1)
    return kap - kap.mean(dim=-1, keepdim=True)

# ---------- helpers for Study-2-informed hyperpriors ----------
def _sd_names(resp_tag):
    return [
        f"sd_participant_id__{resp_tag}_Intercept",
        f"sd_participant_id__{resp_tag}_avatar_c",
        f"sd_participant_id__{resp_tag}_type_c",
    ]

def _get_sd_log_params(hyper, resp_tag, R, prior_kind="study2"):
    names = _sd_names(resp_tag)

    # Fallback defaults from legacy 'sd_means'
    fallback_means = [float(hyper.get("sd_means", {}).get(n, 0.5)) for n in names]
    fallback_means = [max(v, 1e-3) for v in fallback_means]
    fallback_log_mu = torch.log(torch.tensor(fallback_means, dtype=torch.double))
    fallback_log_sd = 0.40 * torch.ones(R, dtype=torch.double)

    if prior_kind == "weak":
        # Use weak priors by emulating HalfNormal(0.5) with a broad LogNormal approx
        return torch.log(torch.full((R,), 0.5, dtype=torch.double)), torch.full((R,), 0.6, dtype=torch.double)

    sd_log_mu_map = hyper.get("sd_log_mu", {})
    sd_log_sd_map = hyper.get("sd_log_sd", {})

    if not sd_log_mu_map or not sd_log_sd_map:
        return fallback_log_mu, fallback_log_sd

    log_mu = []
    log_sd = []
    for nm in names:
        mu = sd_log_mu_map.get(nm, None)
        sd = sd_log_sd_map.get(nm, None)
        if mu is None or sd is None:
            log_mu.append(float(fallback_log_mu[len(log_mu)].item()))
            log_sd.append(float(fallback_log_sd[len(log_sd)].item()))
        else:
            log_mu.append(float(mu))
            log_sd.append(float(max(sd, 1e-4)))
    return torch.tensor(log_mu, dtype=torch.double), torch.tensor(log_sd, dtype=torch.double)

# ---------- model ----------
def ordinal_model(X, Z, pid_idx, y, hyper=None, resp_tag="realism", prior_kind="study2"):
    N, P = X.shape
    _, R = Z.shape
    S = int(pid_idx.max().item()) + 1
    K = int(y.max().item()) + 1 if y is not None else K_CATS

    # Fixed-effect priors centered on Study 2 (when available)
    default_loc = torch.zeros(P)
    if hyper is not None:
        names = [f"b_{resp_tag}_Intercept", f"b_{resp_tag}_avatar_c",
                 f"b_{resp_tag}_type_c", f"b_{resp_tag}_avatar_c:type_c"]
        loc = [float(hyper.get("b_means", {}).get(n, 0.0)) for n in names]
        default_loc = torch.tensor(loc, dtype=torch.double)

    beta = pyro.sample("beta", dist.Normal(default_loc, 0.3*torch.ones(P)))

    # Study-2–informed hyperpriors for participant SDs (or weak)
    if prior_kind == "study2":
        log_mu, log_sd = _get_sd_log_params(hyper or {}, resp_tag, R, prior_kind="study2")
        sd_re = pyro.sample("sd_re", dist.LogNormal(log_mu, log_sd))
    else:
        sd_re = pyro.sample("sd_re", dist.HalfNormal(0.5*torch.ones(R)))

    with pyro.plate("subjects", S):
        b = pyro.sample("b", dist.Normal(torch.zeros(R), sd_re).to_event(1))  # (S,R)

    k_raw = pyro.sample("k_raw", dist.Normal(torch.zeros(K-1), torch.ones(K-1)))
    kappa = kappa_from_raw(k_raw)

    eta   = (X @ beta) + (Z * b[pid_idx]).sum(-1)
    probs = ordered_probs(eta, kappa)
    pyro.sample("y", dist.Categorical(probs=probs), obs=y)

def fit_ordinal(X, Z, pid_idx, y, hyper, resp_tag, num_samples=NUM_SAMPLES, warmup=WARMUP, prior_kind="study2"):
    nuts = NUTS(lambda X,Z,pid_idx,y: ordinal_model(X,Z,pid_idx,y,hyper,resp_tag,prior_kind),
                target_accept_prob=0.9)
    mcmc = MCMC(nuts, num_samples=num_samples, warmup_steps=warmup)
    mcmc.run(X, Z, pid_idx, y)
    return mcmc

# ---------- reporting helpers ----------
def save_arviz_and_traces(mcmc, tag, suffix=""):
    os.makedirs("results", exist_ok=True)
    idata = az.from_pyro(mcmc)
    idata.to_netcdf(f"results/study3_{tag}{suffix}_idata.nc")
    summ  = az.summary(idata, var_names=["beta","sd_re"], kind="stats")
    summ.to_csv(f"results/summary_{tag}{suffix}.csv")
    az.plot_trace(idata, var_names=["beta","sd_re"])
    plt.savefig(f"results/trace_{tag}{suffix}.png", bbox_inches="tight"); plt.close()
    print(f"✓ saved results/summary_{tag}{suffix}.csv and results/trace_{tag}{suffix}.png")
    return idata

def effect_table(mcmc, tag, suffix=""):
    coef_names = ["Intercept","avatar_c","type_c","avatar_c:type_c"]
    post = mcmc.get_samples()
    beta = post["beta"].numpy()
    OR   = np.exp(beta)
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
    df.to_csv(f"results/effects_{tag}{suffix}.csv", index=False)
    print(f"✓ saved results/effects_{tag}{suffix}.csv")

def summarize_conditions(mcmc, tag, K=K_CATS, which="population", suffix=""):
    post  = mcmc.get_samples()
    beta  = post["beta"]
    sd_re = post["sd_re"]
    k_raw = post["k_raw"]
    inc   = torch.log1p(torch.exp(k_raw)) + 1e-3
    kappa = torch.cumsum(inc, dim=-1) - torch.cumsum(inc, dim=-1).mean(dim=-1, keepdim=True)

    conds = [
        ("first_neg", -0.5, -0.5),
        ("first_pos", -0.5,  0.5),
        ("third_neg",  0.5, -0.5),
        ("third_pos",  0.5,  0.5),
    ]
    Xc = torch.tensor([[1., a, t, a*t] for _,a,t in conds], dtype=torch.double)
    Zc = torch.tensor([[1., a, t]       for _,a,t in conds], dtype=torch.double)

    if which == "population":
        b_draw = torch.zeros_like(sd_re)
    elif which == "new_participant":
        b_draw = torch.randn_like(sd_re) * sd_re
    else:
        raise ValueError("which must be 'population' or 'new_participant'")

    eta   = (beta @ Xc.T) + (b_draw @ Zc.T)
    cdf   = torch.sigmoid(kappa[:,None,:] - eta[:,:,None])
    first = cdf[:,:,:1]
    middle= cdf[:,:,1:] - cdf[:,:,:-1] if K>2 else torch.empty(eta.shape[0], eta.shape[1], 0, dtype=eta.dtype)
    last  = 1 - cdf[:,:,-1:]
    probs = torch.cat([first, middle, last], dim=-1)
    p_top2= probs.index_select(-1, torch.tensor([3,4])).sum(-1)
    qs    = torch.quantile(p_top2, torch.tensor([0.025,0.5,0.975], dtype=torch.double), dim=0)

    out = pd.DataFrame({
        "condition": [c for c,_,_ in conds],
        "P(Y>=4)_2.5%": qs[0].numpy(),
        "P(Y>=4)_50%":  qs[1].numpy(),
        "P(Y>=4)_97.5%":qs[2].numpy(),
        "which": which
    })
    os.makedirs("results", exist_ok=True)
    out.to_csv(f"results/cond_probs_{tag}{suffix}_{which}.csv", index=False)
    print(f"✓ saved results/cond_probs_{tag}{suffix}_{which}.csv")

def posterior_predictive_check(mcmc, model_fn, X, Z, pid_idx, y_obs, tag, suffix=""):
    pred = Predictive(model_fn, posterior_samples=mcmc.get_samples(), return_sites=["y"])
    sim  = pred(X, Z, pid_idx, None)["y"].numpy()
    K    = int(sim.max()) + 1
    obs_prop = np.bincount(y_obs, minlength=K) / len(y_obs)
    sim_prop = np.stack([(sim==k).mean(axis=1) for k in range(K)], axis=1)
    lo, med, hi = np.quantile(sim_prop, [0.025,0.5,0.975], axis=0)
    ppc_df = pd.DataFrame({"cat": np.arange(K),
                           "obs_prop": obs_prop,
                           "sim_2.5%": lo, "sim_50%": med, "sim_97.5%": hi})
    os.makedirs("results", exist_ok=True)
    ppc_df.to_csv(f"results/ppc_{tag}{suffix}.csv", index=False)
    print(f"✓ saved results/ppc_{tag}{suffix}.csv")

# ---------- info gain ----------
def _kl_normal_1d(mu_q, sd_q, mu_p, sd_p):
    sd_q = max(sd_q, EPS); sd_p = max(sd_p, EPS)
    return np.log(sd_p/sd_q) + (sd_q**2 + (mu_q - mu_p)**2)/(2*sd_p**2) - 0.5

def information_gain_random_effects(mcmc, pid_codes, tag_out, suffix=""):
    os.makedirs("results", exist_ok=True)
    post   = mcmc.get_samples()
    b      = post["b"].numpy()
    sd_re  = post["sd_re"].numpy()
    Samps, Subjects, R = b.shape
    re_names = ["Intercept","avatar_c","type_c"]
    prior_sd = sd_re.mean(axis=0)

    rows = []
    for i in range(Subjects):
        pid = pid_codes[i]
        for r in range(R):
            draws = b[:, i, r]
            mu    = float(np.mean(draws))
            sd    = float(np.std(draws, ddof=1))
            psd   = float(prior_sd[r])
            kl_nats = _kl_normal_1d(mu_q=mu, sd_q=sd, mu_p=0.0, sd_p=psd)
            rows.append({
                "pid": pid, "re_param": re_names[r],
                "post_mean": mu, "post_sd": sd, "prior_sd": psd,
                "KL_nats": kl_nats, "KL_bits": kl_nats/np.log(2.0),
                "shrinkage": 1.0 - (sd / (psd + EPS))
            })
    df = pd.DataFrame(rows)
    df.to_csv(f"results/info_gain_random_effects_{tag_out}{suffix}_by_pid.csv", index=False)
    summ = (df.groupby("re_param")
              .agg(KL_bits_mean=("KL_bits","mean"),
                   KL_bits_median=("KL_bits","median"),
                   KL_bits_sum=("KL_bits","sum"),
                   KL_bits_sd=("KL_bits","std"),
                   shrinkage_mean=("shrinkage","mean"),
                   n_participants=("pid","nunique"))
              .reset_index())
    summ.to_csv(f"results/info_gain_random_effects_{tag_out}{suffix}_by_param.csv", index=False)
    print(f"✓ saved info-gain CSVs for {tag_out}{suffix}")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", choices=["study2","weak"], default="study2",
                    help="Use Study-2-informed hyperpriors (default) or weak HalfNormal(0.5).")
    args = ap.parse_args()

    with open("study2_hyperpriors.json","r") as f:
        hyper = json.load(f)

    df3 = load_study3("study3_long.csv")

    Xr, Zr, pidr, yr, pid_codes_r = build_design_ordinal(df3, "realism", K=K_CATS)
    Xq, Zq, pidq, yq, pid_codes_q = build_design_ordinal(df3, "quality", K=K_CATS)

    # Fit realism & quality
    mcmc_r = fit_ordinal(Xr, Zr, pidr, yr, hyper, "realism", prior_kind=args.prior)
    print("✓ Study-3 REALISM ordinal fit done")
    mcmc_q = fit_ordinal(Xq, Zq, pidq, yq, hyper, "quality", prior_kind=args.prior)
    print("✓ Study-3 QUALITY ordinal fit done")

    suffix = "" if args.prior == "study2" else "_weak"

    save_arviz_and_traces(mcmc_r, "realism", suffix=suffix)
    save_arviz_and_traces(mcmc_q, "quality", suffix=suffix)

    effect_table(mcmc_r, "realism", suffix=suffix)
    effect_table(mcmc_q, "quality", suffix=suffix)

    summarize_conditions(mcmc_r, "realism", which="population", suffix=suffix)
    summarize_conditions(mcmc_q, "quality", which="population", suffix=suffix)
    summarize_conditions(mcmc_r, "realism", which="new_participant", suffix=suffix)
    summarize_conditions(mcmc_q, "quality", which="new_participant", suffix=suffix)

    model_r = lambda X,Z,pid_idx,y=None: ordinal_model(X,Z,pid_idx,y,hyper,"realism",args.prior)
    model_q = lambda X,Z,pid_idx,y=None: ordinal_model(X,Z,pid_idx,y,hyper,"quality",args.prior)
    posterior_predictive_check(mcmc_r, model_r, Xr, Zr, pidr, yr.numpy(), "realism", suffix=suffix)
    posterior_predictive_check(mcmc_q, model_q, Xq, Zq, pidq, yq.numpy(), "quality", suffix=suffix)

    information_gain_random_effects(mcmc_r, pid_codes_r, tag_out="realism", suffix=suffix)
    information_gain_random_effects(mcmc_q, pid_codes_q, tag_out="quality", suffix=suffix)

if __name__ == "__main__":
    main()
