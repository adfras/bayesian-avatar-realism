# phase2_study3_continuous.py
import os, json, argparse, numpy as np, torch, pyro, pandas as pd
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
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
    miss_r = [c for c in realism_items if c not in df.columns]
    miss_q = [c for c in quality_items if c not in df.columns]
    if miss_r: raise KeyError(f"Realism items missing: {miss_r}")
    if miss_q: raise KeyError(f"Quality items missing: {miss_q}")
    df["realism3"] = df[realism_items].mean(axis=1)
    df["quality3"] = df[quality_items].mean(axis=1)
    def valid_1to5(s): return s.notna() & (s >= 1) & (s <= 5)
    keep = valid_1to5(df["realism3"]) & valid_1to5(df["quality3"])
    dropped = (~keep).sum()
    if dropped:
        print(f"[load_study3_continuous] Dropping {dropped} rows with invalid/missing 1..5 composites.")
    df = df.loc[keep].copy()
    df["avatar_c"] = np.where(df["avatar_type"].str.lower().eq("unreal"), +0.5, -0.5)
    df["type_c"]   = np.where(df["disclosure_sentiment"].str.lower().eq("positive"), +0.5, -0.5)
    if "participant_code" in df.columns:
        df["participant_id"] = df["participant_code"]
    elif "participant_id" not in df.columns:
        raise KeyError("Need participant identifier (participant_code or participant_id).")
    return df

def build_design(df, outcome_col):
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
    y = df[outcome_col].values.astype(float)
    return (torch.tensor(X), torch.tensor(Z),
            torch.tensor(pid_idx, dtype=torch.long),
            torch.tensor(y, dtype=torch.double),
            pid_codes)

# ---- Study-2-informed hyperprior helpers ----
def _sd_names(resp_tag):
    return [
        f"sd_participant_id__{resp_tag}_Intercept",
        f"sd_participant_id__{resp_tag}_avatar_c",
        f"sd_participant_id__{resp_tag}_type_c",
    ]

def _get_sd_log_params(hyper, resp_tag, R, prior_kind="study2"):
    names = _sd_names(resp_tag)
    fallback_means = [float(hyper.get("sd_means", {}).get(n, 0.5)) for n in names]
    fallback_means = [max(v, 1e-3) for v in fallback_means]
    fallback_log_mu = torch.log(torch.tensor(fallback_means, dtype=torch.double))
    fallback_log_sd = 0.40 * torch.ones(R, dtype=torch.double)
    if prior_kind == "weak":
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

# ---------- model: Student-t LMM ----------
def t_lmm_model(X, Z, pid_idx, y, hyper=None, resp_tag="realism", prior_kind="study2"):
    N, P = X.shape
    _, R = Z.shape
    S = int(pid_idx.max().item()) + 1

    default_loc = torch.zeros(P)
    if hyper is not None:
        names = [f"b_{resp_tag}_Intercept", f"b_{resp_tag}_avatar_c",
                 f"b_{resp_tag}_type_c", f"b_{resp_tag}_avatar_c:type_c"]
        loc = [float(hyper.get("b_means", {}).get(n, 0.0)) for n in names]
        default_loc = torch.tensor(loc, dtype=torch.double)

    beta  = pyro.sample("beta",  dist.Normal(default_loc, 0.3*torch.ones(P)))

    if prior_kind == "study2":
        log_mu, log_sd = _get_sd_log_params(hyper or {}, resp_tag, R, prior_kind)
        sd_re = pyro.sample("sd_re", dist.LogNormal(log_mu, log_sd))
    else:
        sd_re = pyro.sample("sd_re", dist.HalfNormal(0.5*torch.ones(R)))

    sigma = pyro.sample("sigma", dist.HalfNormal(0.7))

    with pyro.plate("subjects", S):
        b = pyro.sample("b", dist.Normal(torch.zeros(R), sd_re).to_event(1))

    nu_raw = pyro.sample("nu_raw", dist.Exponential(1.0))
    nu = nu_raw + 3.0

    mu = (X @ beta) + (Z * b[pid_idx]).sum(-1)
    pyro.sample("y", dist.StudentT(df=nu, loc=mu, scale=sigma), obs=y)

def fit_t_lmm(X, Z, pid_idx, y, hyper, resp_tag, prior_kind="study2"):
    nuts = NUTS(lambda X,Z,pid_idx,y: t_lmm_model(X,Z,pid_idx,y,hyper,resp_tag,prior_kind),
                target_accept_prob=0.9)
    mcmc = MCMC(nuts, num_samples=NUM_SAMPLES, warmup_steps=WARMUP)
    mcmc.run(X, Z, pid_idx, y)
    return mcmc

# ---------- reporting ----------
def effects_table(mcmc, tag, outdir="results_cont", suffix=""):
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
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, f"effects_cont_{tag}{suffix}.csv"), index=False)

def cond_means(mcmc, tag, outdir="results_cont", suffix=""):
    post  = mcmc.get_samples()
    beta  = post["beta"]
    sd_re = post["sd_re"]
    conds = [
        ("first_neg", -0.5, -0.5),
        ("first_pos", -0.5,  0.5),
        ("third_neg",  0.5, -0.5),
        ("third_pos",  0.5,  0.5),
    ]
    Xc = torch.tensor([[1., a, t, a*t] for _,a,t in conds], dtype=torch.double)
    Zc = torch.tensor([[1., a, t]       for _,a,t in conds], dtype=torch.double)
    b0 = torch.zeros_like(sd_re)
    mu = (beta @ Xc.T) + (b0 @ Zc.T)
    qs = torch.quantile(mu, torch.tensor([0.025,0.5,0.975], dtype=torch.double), dim=0).numpy()
    df = pd.DataFrame({
        "condition": [c for c,_,_ in conds],
        "E[composite]_2.5%": qs[0], "E[composite]_50%": qs[1], "E[composite]_97.5%": qs[2],
        "which":"population"
    })
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, f"cond_means_cont_{tag}{suffix}.csv"), index=False)

def cohens_d_from_posterior(mcmc, tag, outdir="results_cont", suffix=""):
    post  = mcmc.get_samples()
    beta  = post["beta"].numpy()
    sigma = post["sigma"].numpy().reshape(-1)
    def summarize(arr):
        lo, med, hi = np.quantile(arr, [0.025, 0.5, 0.975]); pgt0 = (arr > 0).mean()
        return float(lo), float(med), float(hi), float(pgt0)
    d_avatar      = beta[:, 1] / sigma
    d_valence     = beta[:, 2] / sigma
    d_interaction = beta[:, 3] / sigma
    rows = []
    for name, arr in [("avatar_c", d_avatar), ("type_c", d_valence), ("avatar_c:type_c", d_interaction)]:
        lo, med, hi, pgt0 = summarize(arr)
        rows.append({"param": name, "d_2.5%": lo, "d_50%":  med, "d_97.5%": hi, "Pr(d>0)": pgt0})
    df = pd.DataFrame(rows)
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, f"effect_size_d_{tag}{suffix}.csv"), index=False)
    print(f"✓ saved {os.path.join(outdir, f'effect_size_d_{tag}{suffix}.csv')}")

def info_gain_random_effects(mcmc, pid_codes, tag_out, outdir="results_cont", suffix=""):
    post = mcmc.get_samples()
    b    = post["b"].numpy()
    sd_re= post["sd_re"].numpy()
    prior_sd = sd_re.mean(axis=0)
    re_names = ["Intercept","avatar_c","type_c"]
    rows = []
    for j, pid in enumerate(pid_codes):
        for r, name in enumerate(re_names):
            draws = b[:, j, r]
            mu_q  = float(draws.mean()); sd_q  = float(draws.std(ddof=1))
            sd_p  = float(max(prior_sd[r], EPS))
            kl = np.log(sd_p/sd_q) + (sd_q**2 + mu_q**2)/(2*sd_p**2) - 0.5
            rows.append({"pid": pid, "re_param": name,
                         "post_mean": mu_q, "post_sd": sd_q, "prior_sd": sd_p,
                         "KL_nats": kl, "KL_bits": kl/np.log(2.0), "shrinkage": 1.0 - sd_q/sd_p})
    df = pd.DataFrame(rows)
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, f"info_gain_cont_random_{tag_out}{suffix}_by_pid.csv"), index=False)
    (df.groupby("re_param")
       .agg(KL_bits_mean=("KL_bits","mean"), KL_bits_median=("KL_bits","median"),
            KL_bits_sum=("KL_bits","sum"), shrinkage_mean=("shrinkage","mean"),
            n=("pid","nunique"))
       ).reset_index().to_csv(os.path.join(outdir, f"info_gain_cont_random_{tag_out}{suffix}_by_param.csv"), index=False)

def run(tag, outcome_col, item_cols, prior_kind="study2"):
    with open("study2_hyperpriors.json","r") as f:
        hyper = json.load(f)

    df3 = load_study3_continuous("study3_long.csv",
                                 realism_items=item_cols["realism"],
                                 quality_items=item_cols["quality"])

    X, Z, pid_idx, y, pid_codes = build_design(df3, outcome_col)
    mcmc = fit_t_lmm(X, Z, pid_idx, y, hyper, resp_tag=tag, prior_kind=prior_kind)

    idata = az.from_pyro(mcmc)
    os.makedirs("results_cont", exist_ok=True)
    idata.to_netcdf(f"results_cont/study3_cont_{tag}{'' if prior_kind=='study2' else '_weak'}_idata.nc")

    suffix = "" if prior_kind == "study2" else "_weak"
    effects_table(mcmc, tag, suffix=suffix)
    cond_means(mcmc, tag, suffix=suffix)
    cohens_d_from_posterior(mcmc, tag, suffix=suffix)
    info_gain_random_effects(mcmc, pid_codes, tag_out=tag, suffix=suffix)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", choices=["study2","weak"], default="study2")
    args = ap.parse_args()

    item_cols = {
        "realism": ["real_person_rating", "facial_realism_rating", "body_realism_rating"],
        "quality": ["enjoyment_rating", "comfort_rating", "pleasantness_rating"],
    }
    run(tag="realism", outcome_col="realism3", item_cols=item_cols, prior_kind=args.prior)
    run(tag="quality", outcome_col="quality3", item_cols=item_cols, prior_kind=args.prior)
