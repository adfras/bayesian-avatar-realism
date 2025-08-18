# phase2_study3_continuous.py
import os, json, numpy as np, torch, pyro, pandas as pd
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive
import arviz as az
from data_utils import load_study3

torch.set_default_dtype(torch.double)
pyro.set_rng_seed(314159)

NUM_SAMPLES = 800
WARMUP      = 800
EPS         = 1e-8

def load_study3_continuous(csv_path, realism_items, quality_items):
    import pandas as pd
    df = pd.read_csv(csv_path)

    # 1) Make 3-item composites (the same items you used in the ANOVAs)
    miss_r = [c for c in realism_items if c not in df.columns]
    miss_q = [c for c in quality_items if c not in df.columns]
    if miss_r: raise KeyError(f"Realism items missing: {miss_r}")
    if miss_q: raise KeyError(f"Quality items missing: {miss_q}")

    df["realism3"] = df[realism_items].mean(axis=1)
    df["quality3"] = df[quality_items].mean(axis=1)

    # 2) Drop rows with any invalid/missing ratings (1..5)
    def valid_1to5(s): 
        return s.notna() & (s >= 1) & (s <= 5)
    keep = valid_1to5(df["realism3"]) & valid_1to5(df["quality3"])
    dropped = (~keep).sum()
    if dropped:
        print(f"[load_study3_continuous] Dropping {dropped} rows with invalid/missing composite ratings (must be 1..5).")
    df = df.loc[keep].copy()

    # 3) Add centered contrasts and participant id (to mirror your ordinal build_design)
    #    sync/negative -> -0.5 ; unreal/positive -> +0.5
    df["avatar_c"] = np.where(df["avatar_type"].str.lower().eq("unreal"), +0.5, -0.5)
    df["type_c"]   = np.where(df["disclosure_sentiment"].str.lower().eq("positive"), +0.5, -0.5)

    # 4) Participant id
    if "participant_code" in df.columns:
        df["participant_id"] = df["participant_code"]
    elif "participant_id" not in df.columns:
        raise KeyError("Need a participant identifier column (participant_code or participant_id).")

    return df


# ---------- utilities ----------
def build_design(df, outcome_col):
    X = np.column_stack([
        np.ones(len(df)),
        df["avatar_c"].values,
        df["type_c"].values,
        (df["avatar_c"]*df["type_c"]).values
    ])
    Z = np.column_stack([
        np.ones(len(df)),          # random intercept
        df["avatar_c"].values,     # random avatar slope
        df["type_c"].values        # random valence slope
    ])
    pid_codes, pid_idx = np.unique(df["participant_id"].values, return_inverse=True)
    y = df[outcome_col].values.astype(float)
    return (torch.tensor(X), torch.tensor(Z),
            torch.tensor(pid_idx, dtype=torch.long),
            torch.tensor(y, dtype=torch.double),
            pid_codes)

def add_composites(df, realism_items, quality_items):
    out = df.copy()
    if realism_items:
        missing = [c for c in realism_items if c not in out.columns]
        if missing:
            raise KeyError(f"Realism items missing from dataframe: {missing}")
        out["realism3"] = out[realism_items].mean(axis=1)
    if quality_items:
        missing = [c for c in quality_items if c not in out.columns]
        if missing:
            raise KeyError(f"Quality items missing from dataframe: {missing}")
        out["quality3"] = out[quality_items].mean(axis=1)
    # optional: drop any rows where the composite is NaN
    out = out.dropna(subset=[c for c in ["realism3","quality3"] if c in out.columns])
    return out


# ---------- model: Student-t LMM ----------
def t_lmm_model(X, Z, pid_idx, y, hyper=None, resp_tag="realism"):
    N, P = X.shape
    _, R = Z.shape
    S = int(pid_idx.max().item()) + 1

    # Prior means for betas from Study-2 hyperpriors (keeps alignment with Phase-1)
    default_loc = torch.zeros(P)
    if hyper is not None:
        names = [f"b_{resp_tag}_Intercept", f"b_{resp_tag}_avatar_c",
                 f"b_{resp_tag}_type_c", f"b_{resp_tag}_avatar_c:type_c"]
        loc = [float(hyper.get("b_means", {}).get(n, 0.0)) for n in names]
        default_loc = torch.tensor(loc, dtype=torch.double)

    beta  = pyro.sample("beta",  dist.Normal(default_loc, 0.3*torch.ones(P)))
    sd_re = pyro.sample("sd_re", dist.HalfNormal(0.5*torch.ones(R)))
    sigma = pyro.sample("sigma", dist.HalfNormal(0.7))  # observation scale on 1..5

    with pyro.plate("subjects", S):
        b = pyro.sample("b", dist.Normal(torch.zeros(R), sd_re).to_event(1))  # (S,R)

    nu_raw = pyro.sample("nu_raw", dist.Exponential(1.0))  # >0
    nu = nu_raw + 3.0  # df > 3 for finite variance

    mu = (X @ beta) + (Z * b[pid_idx]).sum(-1)
    pyro.sample("y", dist.StudentT(df=nu, loc=mu, scale=sigma), obs=y)

def fit_t_lmm(X, Z, pid_idx, y, hyper, resp_tag):
    nuts = NUTS(lambda X,Z,pid_idx,y: t_lmm_model(X,Z,pid_idx,y,hyper,resp_tag),
                target_accept_prob=0.9)
    mcmc = MCMC(nuts, num_samples=NUM_SAMPLES, warmup_steps=WARMUP)
    mcmc.run(X, Z, pid_idx, y)
    return mcmc

# ---------- reporting ----------
def effects_table(mcmc, tag):
    post = mcmc.get_samples()
    beta = post["beta"].numpy()
    q = lambda x: np.quantile(x, [0.025, 0.5, 0.975], axis=0)
    q_beta = q(beta)
    p_gt0  = (beta > 0).mean(axis=0)
    df = pd.DataFrame({
        "param": ["Intercept","avatar_c","type_c","avatar_c:type_c"],
        "beta_2.5%": q_beta[0], "beta_50%": q_beta[1], "beta_97.5%": q_beta[2],
        "Pr(beta>0)": p_gt0
    })
    os.makedirs("results_cont", exist_ok=True)
    df.to_csv(f"results_cont/effects_cont_{tag}.csv", index=False)

def cond_means(mcmc, tag):
    post  = mcmc.get_samples()
    beta  = post["beta"]       # [S,4]
    sd_re = post["sd_re"]      # [S,3]

    conds = [
        ("first_neg", -0.5, -0.5),
        ("first_pos", -0.5,  0.5),
        ("third_neg",  0.5, -0.5),
        ("third_pos",  0.5,  0.5),
    ]
    Xc = torch.tensor([[1., a, t, a*t] for _,a,t in conds], dtype=torch.double)
    Zc = torch.tensor([[1., a, t]       for _,a,t in conds], dtype=torch.double)

    # population average (b=0)
    b0 = torch.zeros_like(sd_re)
    mu = (beta @ Xc.T) + (b0 @ Zc.T)   # [S,4]
    qs = torch.quantile(mu, torch.tensor([0.025,0.5,0.975], dtype=torch.double), dim=0).numpy()

    df = pd.DataFrame({
        "condition": [c for c,_,_ in conds],
        "E[composite]_2.5%": qs[0], "E[composite]_50%": qs[1], "E[composite]_97.5%": qs[2],
        "which":"population"
    })
    os.makedirs("results_cont", exist_ok=True)
    df.to_csv(f"results_cont/cond_means_cont_{tag}.csv", index=False)

def cohens_d_from_posterior(mcmc, tag):
    """
    Compute Cohen's d-style standardized effects from the Student-t LMM:
      d_avatar      = beta[:,1] / sigma
      d_valence     = beta[:,2] / sigma
      d_interaction = beta[:,3] / sigma

    Interpretation: standardized mean difference on the residual (within-trial)
    scale. With centered coding (-0.5/+0.5), the main-effect contrast is 1.0.
    """
    import numpy as np
    import pandas as pd
    import os

    post  = mcmc.get_samples()
    beta  = post["beta"].numpy()        # [S, 4] -> [Intercept, avatar, valence, interaction]
    sigma = post["sigma"].numpy().reshape(-1)  # [S]

    if beta.shape[1] < 4:
        raise RuntimeError("Expected 4 fixed-effect columns [Intercept, avatar, valence, interaction].")

    def summarize(arr):
        lo, med, hi = np.quantile(arr, [0.025, 0.5, 0.975])
        return float(lo), float(med), float(hi), float((arr > 0).mean())

    d_avatar      = beta[:, 1] / sigma
    d_valence     = beta[:, 2] / sigma
    d_interaction = beta[:, 3] / sigma

    rows = []
    for name, arr in [
        ("avatar_c",      d_avatar),
        ("type_c",        d_valence),
        ("avatar_c:type_c", d_interaction),
    ]:
        lo, med, hi, pgt0 = summarize(arr)
        rows.append({
            "param": name,
            "d_2.5%": lo,
            "d_50%":  med,
            "d_97.5%": hi,
            "Pr(d>0)": pgt0
        })

    df = pd.DataFrame(rows)
    os.makedirs("results_cont", exist_ok=True)
    df.to_csv(f"results_cont/effect_size_d_{tag}.csv", index=False)
    print(f"✓ saved results_cont/effect_size_d_{tag}.csv")
    

def info_gain_random_effects(mcmc, pid_codes, tag_out):
    post = mcmc.get_samples()
    b    = post["b"].numpy()        # [S, n_subj, 3]
    sd_re= post["sd_re"].numpy()    # [S, 3]
    prior_sd = sd_re.mean(axis=0)   # [3]
    re_names = ["Intercept","avatar_c","type_c"]

    rows = []
    for j, pid in enumerate(pid_codes):
        for r, name in enumerate(re_names):
            draws = b[:, j, r]
            mu_q  = float(draws.mean())
            sd_q  = float(draws.std(ddof=1))
            sd_p  = float(max(prior_sd[r], EPS))
            # KL between N(mu_q, sd_q) || N(0, sd_p)
            kl = np.log(sd_p/sd_q) + (sd_q**2 + mu_q**2)/(2*sd_p**2) - 0.5
            rows.append({
                "pid": pid, "re_param": name,
                "post_mean": mu_q, "post_sd": sd_q, "prior_sd": sd_p,
                "KL_nats": kl, "KL_bits": kl/np.log(2.0),
                "shrinkage": 1.0 - sd_q/sd_p
            })
    df = pd.DataFrame(rows)
    os.makedirs("results_cont", exist_ok=True)
    df.to_csv(f"results_cont/info_gain_cont_random_{tag_out}_by_pid.csv", index=False)
    (df.groupby("re_param")
       .agg(KL_bits_mean=("KL_bits","mean"), KL_bits_median=("KL_bits","median"),
            KL_bits_sum=("KL_bits","sum"), shrinkage_mean=("shrinkage","mean"),
            n=("pid","nunique"))
       ).reset_index().to_csv(f"results_cont/info_gain_cont_random_{tag_out}_by_param.csv", index=False)

def run(tag, outcome_col, item_cols):
    with open("study2_hyperpriors.json","r") as f:
        hyper = json.load(f)

    # NEW: build composites + contrasts directly from raw long CSV
    df3 = load_study3_continuous(
        "study3_long.csv",
        realism_items=item_cols["realism"],
        quality_items=item_cols["quality"]
    )

    X, Z, pid_idx, y, pid_codes = build_design(df3, outcome_col)
    mcmc = fit_t_lmm(X, Z, pid_idx, y, hyper, resp_tag=tag)

    idata = az.from_pyro(mcmc)
    os.makedirs("results_cont", exist_ok=True)
    idata.to_netcdf(f"results_cont/study3_cont_{tag}_idata.nc")

    effects_table(mcmc, tag)
    cond_means(mcmc, tag)
    cohens_d_from_posterior(mcmc, tag)    
    info_gain_random_effects(mcmc, pid_codes, tag_out=tag)

if __name__ == "__main__":
    # The exact 3-item composites you used in the ANOVAs:
    item_cols = {
        "realism": ["real_person_rating", "facial_realism_rating", "body_realism_rating"],
        "quality": ["enjoyment_rating", "comfort_rating", "pleasantness_rating"],
    }
    # Fit on the composites we just created:
    run(tag="realism", outcome_col="realism3", item_cols=item_cols)
    run(tag="quality", outcome_col="quality3", item_cols=item_cols)
