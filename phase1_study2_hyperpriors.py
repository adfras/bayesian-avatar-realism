# phase1_study2_hyperpriors.py
import json, numpy as np, torch, pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from data_utils import load_study2

torch.set_default_dtype(torch.double)
pyro.set_rng_seed(42)

NUM_SAMPLES = 800
WARMUP      = 800

def build_design(df, outcome_col):
    # Fixed: [Intercept, avatar_c, type_c, avatar_c:type_c]
    X = np.column_stack([
        np.ones(len(df)),
        df["avatar_c"].values,
        df["type_c"].values,
        (df["avatar_c"]*df["type_c"]).values
    ])
    # Random by participant: intercept, avatar_c, type_c
    Z = np.column_stack([
        np.ones(len(df)),
        df["avatar_c"].values,
        df["type_c"].values
    ])
    pid_codes, pid_idx = np.unique(df["participant_id"].values, return_inverse=True)
    y = df[outcome_col].values
    return (torch.tensor(X),
            torch.tensor(Z),
            torch.tensor(pid_idx, dtype=torch.long),
            torch.tensor(y),
            pid_codes)

def lmm_model(X, Z, pid_idx, y):
    """
    Hierarchical Gaussian LMM with independent REs:
      y ~ Normal(X beta + (Z * b[pid])·1, sigma_obs)
      b_jp ~ Normal(0, sd_re[p])
    """
    N, P = X.shape
    _, R = Z.shape
    S = int(pid_idx.max().item()) + 1

    beta     = pyro.sample("beta",   dist.Normal(torch.zeros(P), torch.ones(P)))
    sd_re    = pyro.sample("sd_re",  dist.HalfNormal(0.5*torch.ones(R)))
    sigma    = pyro.sample("sigma",  dist.HalfNormal(1.0))

    with pyro.plate("subjects", S):
        b = pyro.sample("b", dist.Normal(torch.zeros(R), sd_re).to_event(1))  # (S,R)

    mu = (X @ beta) + (Z * b[pid_idx]).sum(-1)
    pyro.sample("y", dist.Normal(mu, sigma), obs=y)

def fit_lmm(X, Z, pid_idx, y, num_samples=NUM_SAMPLES, warmup=WARMUP):
    nuts = NUTS(lmm_model, target_accept_prob=0.9)
    mcmc = MCMC(nuts, num_samples=num_samples, warmup_steps=warmup)
    mcmc.run(X, Z, pid_idx, y)
    return mcmc

def summarize_hyperpriors(mcmc, resp_tag):
    post = mcmc.get_samples()
    beta_mean  = post["beta"].mean(0).numpy()
    sd_re_mean = post["sd_re"].mean(0).numpy()
    b_names  = [f"b_{resp_tag}_Intercept", f"b_{resp_tag}_avatar_c",
                f"b_{resp_tag}_type_c", f"b_{resp_tag}_avatar_c:type_c"]
    sd_names = [f"sd_participant_id__{resp_tag}_Intercept",
                f"sd_participant_id__{resp_tag}_avatar_c",
                f"sd_participant_id__{resp_tag}_type_c"]
    return ({n: float(v) for n, v in zip(b_names,  beta_mean)},
            {n: float(v) for n, v in zip(sd_names, sd_re_mean)})

def main():
    df = load_study2("study2_long.csv")

    # realism
    Xr, Zr, pidr, yr, _ = build_design(df, "realism")
    mcmc_r = fit_lmm(Xr, Zr, pidr, yr)
    b_r, sd_r = summarize_hyperpriors(mcmc_r, "realism")

    # quality
    Xq, Zq, pidq, yq, _ = build_design(df, "quality")
    mcmc_q = fit_lmm(Xq, Zq, pidq, yq)
    b_q, sd_q = summarize_hyperpriors(mcmc_q, "quality")

    hyper = {
        "b_means": {**b_r, **b_q},
        "sd_means": {**sd_r, **sd_q},
        "cor_means": {}   # (independent REs in this pass)
    }
    with open("study2_hyperpriors.json","w") as f:
        json.dump(hyper, f, indent=2)
    print("✓ wrote study2_hyperpriors.json")

if __name__ == "__main__":
    main()
