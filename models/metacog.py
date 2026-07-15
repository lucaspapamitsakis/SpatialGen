#!/usr/bin/env python3
"""
models/metacog.py
------------------
First-pass MetaCOG-style generative model: GLOBAL (non-spatial) false-positive
and false-negative rate parameters, per patient, with the true bone mask B
analytically marginalized out (no C-VAE / no per-pixel discrete latent yet --
see PROJECT_HANDOFF.md section 10 for the eventual localized-patch plan).

Sensor model (per pixel i, for one patient):
    H ~ Beta(a_H, b_H)      global false-positive rate: P(U-Net says bone | true=0)
    M ~ Beta(a_M, b_M)      global false-negative rate: P(U-Net says no-bone | true=1)
    p_i = atlas prior P(true bone = 1) at pixel i  (fixed, from 07_build_bone_atlas.py)
    w_i = p_i * (1 - M) + (1 - p_i) * H            marginal P(U-Net says bone=1)
    obs_i ~ Bernoulli(w_i)                          observed = U-Net's binary mask

B is not sampled as a latent site: it's marginalized in closed form into w_i, so
NUTS only needs to explore the 2-dimensional (H, M) space, which mixes fast even
though a patient may contribute hundreds of thousands of pixels. Given posterior
samples of (H, M), the corrected per-pixel posterior P(B_i=1 | obs_i, H, M) is
then computed exactly via Bayes' rule (`posterior_correct` below) and Monte-Carlo
averaged over the (H, M) posterior.

Ground truth is NEVER passed into this module's inference functions -- only the
atlas prior (built from training-split ground truth, offline) and the observed
U-Net binary prediction.
"""
from __future__ import annotations

import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS


def load_atlas(path) -> dict:
    d = np.load(path)
    return {"atlas": d["atlas"], "bin_edges": d["bin_edges"], "bin_counts": d["bin_counts"]}


def atlas_prior_for_slices(atlas_dict: dict, s_norm: np.ndarray) -> np.ndarray:
    """Look up the atlas bin for each slice's s_norm and return (S, H, W) prior."""
    atlas = atlas_dict["atlas"]
    edges = atlas_dict["bin_edges"]
    n_bins = atlas.shape[0]
    bin_idx = np.clip(np.digitize(s_norm, edges[1:-1]), 0, n_bins - 1)
    return atlas[bin_idx]   # (S, H, W), fancy-indexed


def sensor_model(prior_p: torch.Tensor, obs: torch.Tensor | None,
                 h_prior: tuple[float, float] = (1.0, 1.0),
                 m_prior: tuple[float, float] = (1.0, 1.0)) -> None:
    """Pyro model. prior_p, obs: flat (N,) tensors of atlas prior / U-Net binary mask."""
    H = pyro.sample("H", dist.Beta(*h_prior))
    M = pyro.sample("M", dist.Beta(*m_prior))
    w = prior_p * (1.0 - M) + (1.0 - prior_p) * H
    with pyro.plate("pixels", prior_p.shape[0]):
        pyro.sample("obs", dist.Bernoulli(w.clamp(1e-6, 1 - 1e-6)), obs=obs)


def run_nuts_chain(prior_p: torch.Tensor, obs: torch.Tensor, *,
                   num_samples: int, warmup_steps: int,
                   h_prior: tuple[float, float], m_prior: tuple[float, float],
                   seed: int, disable_progbar: bool = True) -> dict[str, torch.Tensor]:
    """Run one NUTS chain (num_chains=1) with a given seed; returns raw samples dict."""
    pyro.set_rng_seed(seed)
    kernel = NUTS(sensor_model)
    mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps,
               num_chains=1, disable_progbar=disable_progbar)
    mcmc.run(prior_p, obs, h_prior, m_prior)
    return {k: v.detach().clone() for k, v in mcmc.get_samples().items()}


def run_nuts(prior_p: torch.Tensor, obs: torch.Tensor, *,
            num_samples: int = 1000, warmup_steps: int = 500,
            num_chains: int = 2, h_prior: tuple[float, float] = (1.0, 1.0),
            m_prior: tuple[float, float] = (1.0, 1.0), seed: int = 0,
            disable_progbar: bool = True) -> dict[str, torch.Tensor]:
    """
    Run `num_chains` independent NUTS chains sequentially (each with num_chains=1
    internally, seeded differently), avoiding Pyro's multiprocess chain runner
    (which can be finicky in sandboxed/cluster environments). Returns pooled
    samples plus a `_per_chain` entry (chains, num_samples) for diagnostics.
    """
    per_chain = {"H": [], "M": []}
    for c in range(num_chains):
        s = run_nuts_chain(prior_p, obs, num_samples=num_samples, warmup_steps=warmup_steps,
                           h_prior=h_prior, m_prior=m_prior, seed=seed + c,
                           disable_progbar=disable_progbar)
        per_chain["H"].append(s["H"])
        per_chain["M"].append(s["M"])

    per_chain = {k: torch.stack(v, dim=0) for k, v in per_chain.items()}  # (C, N)
    pooled = {k: v.reshape(-1) for k, v in per_chain.items()}
    pooled["_per_chain"] = per_chain
    return pooled


def gelman_rubin_rhat(chains: torch.Tensor) -> float:
    """Standard split R-hat. chains: (C, N) samples of one scalar parameter."""
    chains = chains.double()
    C, N = chains.shape
    if C < 2:
        return float("nan")
    chain_means = chains.mean(dim=1)
    chain_vars = chains.var(dim=1, unbiased=True)
    grand_mean = chain_means.mean()
    B = N / (C - 1) * ((chain_means - grand_mean) ** 2).sum()
    W = chain_vars.mean()
    var_hat = (N - 1) / N * W + B / N
    return float(torch.sqrt(var_hat / W).item())


def effective_sample_size(chains: torch.Tensor, max_lag: int = 200) -> float:
    """Rough ESS via the initial-monotone-sequence estimator on pooled autocovariance."""
    x = chains.reshape(-1).double()
    n = x.shape[0]
    x = x - x.mean()
    var = (x * x).mean()
    if var <= 0:
        return float(n)
    acf = []
    for lag in range(1, min(max_lag, n - 1)):
        c = (x[:-lag] * x[lag:]).mean() / var
        acf.append(c.item())
        if lag >= 2 and acf[-1] + acf[-2] < 0:
            break
    rho_sum = sum(acf)
    ess = n / (1.0 + 2.0 * rho_sum)
    return float(max(1.0, min(n, ess)))


def posterior_correct(prior_p: torch.Tensor, obs: torch.Tensor,
                      h_samples: torch.Tensor, m_samples: torch.Tensor,
                      chunk: int = 200) -> torch.Tensor:
    """
    Exact closed-form pixel posterior P(B=1 | obs, H, M), Monte-Carlo averaged
    over posterior (H, M) samples. prior_p, obs: (N,). h/m_samples: (S,).
    Returns (N,) posterior mean probability. Processes samples in chunks to
    bound memory (chunk x N intermediate tensors, not S x N).
    """
    n = prior_p.shape[0]
    total = h_samples.shape[0]
    acc = torch.zeros(n, dtype=torch.float64)
    p = prior_p.double()[None, :]
    o = obs.double()[None, :]
    for i in range(0, total, chunk):
        H = h_samples[i:i + chunk].double()[:, None]
        M = m_samples[i:i + chunk].double()[:, None]
        post1 = (p * (1.0 - M)) / (p * (1.0 - M) + (1.0 - p) * H)
        post0 = (p * M) / (p * M + (1.0 - p) * (1.0 - H))
        post = torch.where(o == 1, post1, post0)
        acc += post.sum(dim=0)
    return (acc / total).float()
