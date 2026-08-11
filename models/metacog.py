#!/usr/bin/env python3
"""Deterministic grid inference for the SpatialGen MetaCOG sensor model.

For every observed U-Net pixel ``U_i`` and atlas probability ``p_i``:

    H ~ Beta(a_H, b_H)       false-positive rate
    M ~ Beta(a_M, b_M)       false-negative rate
    w_i = p_i(1-M) + (1-p_i)H
    U_i ~ Bernoulli(w_i)

The unknown binary bone pixel is marginalized analytically. Only H and M need
numerical integration. We integrate on an adaptive grid in logit space, which
resolves posterior mass very close to 0 or 1 without random sampling or Pyro.

Ground truth is never accepted by any inference function in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import betaln, expit, logsumexp


def load_atlas(path: str | Path) -> dict[str, np.ndarray]:
    d = np.load(path)
    return {
        "atlas": d["atlas"].astype(np.float32),
        "bin_edges": d["bin_edges"].astype(np.float32),
        "bin_counts": d["bin_counts"],
    }


def atlas_bin_indices(atlas_dict: dict, s_norm: np.ndarray) -> np.ndarray:
    edges = atlas_dict["bin_edges"]
    return np.clip(
        np.digitize(s_norm, edges[1:-1]), 0, atlas_dict["atlas"].shape[0] - 1
    ).astype(np.int16)


def atlas_prior_for_slices(atlas_dict: dict, s_norm: np.ndarray) -> np.ndarray:
    """Return the atlas prior stack corresponding to each slice's s_norm."""
    return atlas_dict["atlas"][atlas_bin_indices(atlas_dict, s_norm)]


@dataclass
class GridPosterior:
    """A normalized posterior mass function on a rectangular H/M grid."""

    h: np.ndarray
    m: np.ndarray
    weights: np.ndarray  # shape (len(m), len(h))
    h_mean: float
    m_mean: float
    h_ci: tuple[float, float]
    m_ci: tuple[float, float]
    h_sd: float
    m_sd: float
    refinements: int
    convergence_delta: float
    edge_mass: float
    converged: bool
    n_pixels: int
    n_positive: int


def _weighted_quantile(values: np.ndarray, weights: np.ndarray,
                       q: float | np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return np.interp(q, cdf, values)


def _histogram_observations(prior_p: np.ndarray, obs: np.ndarray,
                            bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compress pixels into a fine histogram of atlas probabilities."""
    p = np.asarray(prior_p, dtype=np.float64).reshape(-1)
    u = np.asarray(obs, dtype=np.uint8).reshape(-1)
    if p.shape != u.shape or p.size == 0:
        raise ValueError("prior_p and obs must be non-empty arrays with equal shape")
    if np.any((p < 0) | (p > 1)) or np.any((u != 0) & (u != 1)):
        raise ValueError("prior_p must be in [0,1] and obs must be binary")

    edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n1 = np.histogram(p[u == 1], bins=edges)[0].astype(np.float64)
    n0 = np.histogram(p[u == 0], bins=edges)[0].astype(np.float64)
    used = (n1 + n0) > 0
    return centers[used], n1[used], n0[used]


def _log_posterior(
    x_h: np.ndarray,
    x_m: np.ndarray,
    p: np.ndarray,
    n1: np.ndarray,
    n0: np.ndarray,
    h_prior: tuple[float, float],
    m_prior: tuple[float, float],
    p_chunk: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the unnormalized posterior in logit coordinates."""
    h = expit(x_h)
    m = expit(x_m)
    H, M = np.meshgrid(h, m)
    hf, mf = H.reshape(-1), M.reshape(-1)
    ll = np.zeros(hf.size, dtype=np.float64)

    for start in range(0, p.size, p_chunk):
        ps = p[start:start + p_chunk]
        w = hf[:, None] + ps[None, :] * (1.0 - mf[:, None] - hf[:, None])
        w = np.clip(w, 1e-15, 1.0 - 1e-15)
        ll += np.log(w) @ n1[start:start + p_chunk]
        ll += np.log1p(-w) @ n0[start:start + p_chunk]

    # Beta density times the logit-transform Jacobian:
    # h^(a-1)(1-h)^(b-1) * h(1-h) = h^a(1-h)^b.
    ah, bh = h_prior
    am, bm = m_prior
    log_h = ah * np.log(h) + bh * np.log1p(-h) - betaln(ah, bh)
    log_m = am * np.log(m) + bm * np.log1p(-m) - betaln(am, bm)
    log_post = ll.reshape(len(m), len(h)) + log_m[:, None] + log_h[None, :]
    return h, m, log_post


def _summarize(h: np.ndarray, m: np.ndarray, log_post: np.ndarray,
               refinements: int, delta: float, n_pixels: int,
               n_positive: int, tolerance: float) -> GridPosterior:
    weights = np.exp(log_post - logsumexp(log_post))
    h_margin = weights.sum(axis=0)
    m_margin = weights.sum(axis=1)
    h_mean = float(h_margin @ h)
    m_mean = float(m_margin @ m)
    h_sd = float(np.sqrt(h_margin @ ((h - h_mean) ** 2)))
    m_sd = float(np.sqrt(m_margin @ ((m - m_mean) ** 2)))
    h_ci = tuple(float(x) for x in _weighted_quantile(h, h_margin, [0.025, 0.975]))
    m_ci = tuple(float(x) for x in _weighted_quantile(m, m_margin, [0.025, 0.975]))
    edge_mass = float(
        weights[0, :].sum() + weights[-1, :].sum()
        + weights[:, 0].sum() + weights[:, -1].sum()
        - weights[0, 0] - weights[0, -1] - weights[-1, 0] - weights[-1, -1]
    )
    return GridPosterior(
        h=h.astype(np.float32),
        m=m.astype(np.float32),
        weights=weights.astype(np.float32),
        h_mean=h_mean,
        m_mean=m_mean,
        h_ci=h_ci,
        m_ci=m_ci,
        h_sd=h_sd,
        m_sd=m_sd,
        refinements=refinements,
        convergence_delta=float(delta),
        edge_mass=edge_mass,
        # A boundary line may contain about 1/fine_size of a smooth marginal.
        # Requiring <0.2% keeps omitted tails negligible without falsely
        # rejecting a stable posterior concentrated near a physical rate of 0.
        converged=bool(delta <= tolerance and edge_mass <= 2e-3),
        n_pixels=int(n_pixels),
        n_positive=int(n_positive),
    )


def _refinement_bounds(axis_x: np.ndarray, marginal: np.ndarray,
                       x_limit: float) -> tuple[float, float]:
    lo, hi = _weighted_quantile(axis_x, marginal, [0.0001, 0.9999])
    lo, hi = float(lo) - 1.25, float(hi) + 1.25
    if hi - lo < 2.5:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 1.25, mid + 1.25
    return max(-x_limit, lo), min(x_limit, hi)


def integrate_rates(
    prior_p: np.ndarray,
    obs: np.ndarray,
    *,
    h_prior: tuple[float, float] = (1.0, 1.0),
    m_prior: tuple[float, float] = (1.0, 1.0),
    coarse_size: int = 101,
    fine_size: int = 161,
    histogram_bins: int = 2048,
    max_refinements: int = 3,
    tolerance: float = 5e-4,
    x_limit: float = 16.0,
) -> GridPosterior:
    """Numerically integrate the two-rate posterior on an adaptive logit grid."""
    if coarse_size < 21 or fine_size < 41:
        raise ValueError("grid sizes are too small for reliable integration")
    if any(v <= 0 for v in (*h_prior, *m_prior)):
        raise ValueError("Beta prior parameters must be positive")

    p, n1, n0 = _histogram_observations(prior_p, obs, histogram_bins)
    x_h = np.linspace(-x_limit, x_limit, coarse_size)
    x_m = np.linspace(-x_limit, x_limit, coarse_size)
    h, m, logp = _log_posterior(x_h, x_m, p, n1, n0, h_prior, m_prior)
    current = _summarize(
        h, m, logp, 0, np.inf, np.asarray(obs).size,
        int(np.asarray(obs).sum()), tolerance,
    )

    for refinement in range(1, max_refinements + 1):
        weights = np.exp(logp - logsumexp(logp))
        h_bounds = _refinement_bounds(x_h, weights.sum(axis=0), x_limit)
        m_bounds = _refinement_bounds(x_m, weights.sum(axis=1), x_limit)
        x_h = np.linspace(*h_bounds, fine_size)
        x_m = np.linspace(*m_bounds, fine_size)
        h, m, logp = _log_posterior(x_h, x_m, p, n1, n0, h_prior, m_prior)
        delta = max(
            abs(float(np.exp(logp - logsumexp(logp)).sum(axis=0) @ h) - current.h_mean),
            abs(float(np.exp(logp - logsumexp(logp)).sum(axis=1) @ m) - current.m_mean),
        )
        current = _summarize(
            h, m, logp, refinement, delta, np.asarray(obs).size,
            int(np.asarray(obs).sum()), tolerance,
        )
        if current.converged:
            break
    return current


def posterior_correct_grid(
    prior_p: np.ndarray,
    obs: np.ndarray,
    posterior: GridPosterior,
    *,
    probability_bins: int = 2048,
    node_chunk: int = 2048,
) -> np.ndarray:
    """Integrate P(B=1 | U,H,M,p) over a grid posterior."""
    p_arr = np.asarray(prior_p, dtype=np.float64)
    u = np.asarray(obs, dtype=np.uint8)
    if p_arr.shape != u.shape:
        raise ValueError("prior_p and obs must have the same shape")

    indices = np.minimum((p_arr * probability_bins).astype(int), probability_bins - 1)
    used = np.unique(indices)
    p = (used.astype(np.float64) + 0.5) / probability_bins
    lookup = np.full(probability_bins, -1, dtype=np.int32)
    lookup[used] = np.arange(used.size)

    H, M = np.meshgrid(posterior.h.astype(np.float64),
                       posterior.m.astype(np.float64))
    hf, mf = H.reshape(-1), M.reshape(-1)
    wf = posterior.weights.astype(np.float64).reshape(-1)
    q1 = np.zeros(p.size, dtype=np.float64)
    q0 = np.zeros(p.size, dtype=np.float64)

    for start in range(0, wf.size, node_chunk):
        hs = hf[start:start + node_chunk, None]
        ms = mf[start:start + node_chunk, None]
        ws = wf[start:start + node_chunk]
        den1 = p[None, :] * (1.0 - ms) + (1.0 - p[None, :]) * hs
        den0 = p[None, :] * ms + (1.0 - p[None, :]) * (1.0 - hs)
        post1 = p[None, :] * (1.0 - ms) / np.clip(den1, 1e-15, None)
        post0 = p[None, :] * ms / np.clip(den0, 1e-15, None)
        q1 += ws @ post1
        q0 += ws @ post0

    pos = lookup[indices]
    return np.where(u == 1, q1[pos], q0[pos]).astype(np.float32)


def posterior_correct_fixed(prior_p: np.ndarray, obs: np.ndarray,
                            h: float, m: float) -> np.ndarray:
    """Closed-form correction for a fixed (lesioned) rate pair."""
    p = np.asarray(prior_p, dtype=np.float64)
    u = np.asarray(obs, dtype=np.uint8)
    den1 = p * (1.0 - m) + (1.0 - p) * h
    den0 = p * m + (1.0 - p) * (1.0 - h)
    q1 = p * (1.0 - m) / np.clip(den1, 1e-15, None)
    q0 = p * m / np.clip(den0, 1e-15, None)
    return np.where(u == 1, q1, q0).astype(np.float32)
