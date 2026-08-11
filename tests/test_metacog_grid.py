"""Numerical and scientific sanity tests for MetaCOG grid inference."""
from __future__ import annotations

import unittest

import numpy as np

from models.metacog import (
    integrate_rates,
    posterior_correct_fixed,
    posterior_correct_grid,
)


class GridInferenceTests(unittest.TestCase):
    def synthetic_case(self, seed=5, n=20_000, h=0.06, m=0.18):
        rng = np.random.default_rng(seed)
        p = rng.uniform(0.02, 0.95, n).astype(np.float32)
        bone = rng.binomial(1, p).astype(np.uint8)
        obs = bone.copy()
        obs[(bone == 0) & (rng.random(n) < h)] = 1
        obs[(bone == 1) & (rng.random(n) < m)] = 0
        return p, bone, obs

    def test_weights_normalize_and_rates_recover(self):
        p, _, obs = self.synthetic_case()
        result = integrate_rates(
            p, obs, coarse_size=81, fine_size=121, histogram_bins=1024
        )
        self.assertAlmostEqual(float(result.weights.sum()), 1.0, places=6)
        self.assertTrue(result.converged)
        self.assertLess(abs(result.h_mean - 0.06), 0.02)
        self.assertLess(abs(result.m_mean - 0.18), 0.02)
        self.assertLess(result.h_ci[0], result.h_mean)
        self.assertGreater(result.h_ci[1], result.h_mean)

    def test_correction_is_bounded_and_shaped(self):
        p, _, obs = self.synthetic_case(n=5000)
        result = integrate_rates(
            p, obs, coarse_size=61, fine_size=101, histogram_bins=512
        )
        q = posterior_correct_grid(p, obs, result, probability_bins=512)
        self.assertEqual(q.shape, p.shape)
        self.assertTrue(np.isfinite(q).all())
        self.assertGreaterEqual(float(q.min()), 0.0)
        self.assertLessEqual(float(q.max()), 1.0)

    def test_fixed_zero_rates_copy_observation(self):
        p = np.array([0.01, 0.2, 0.8, 0.99], dtype=np.float32)
        obs = np.array([0, 1, 0, 1], dtype=np.uint8)
        q = posterior_correct_fixed(p, obs, 0.0, 0.0)
        np.testing.assert_array_equal((q > 0.5).astype(np.uint8), obs)

    def test_histogram_resolution_is_stable(self):
        p, _, obs = self.synthetic_case(n=12_000)
        low = integrate_rates(
            p, obs, coarse_size=81, fine_size=121, histogram_bins=1024
        )
        high = integrate_rates(
            p, obs, coarse_size=81, fine_size=121, histogram_bins=4096
        )
        self.assertLess(abs(low.h_mean - high.h_mean), 5e-4)
        self.assertLess(abs(low.m_mean - high.m_mean), 5e-4)

    def test_boundary_posterior_near_zero(self):
        rng = np.random.default_rng(9)
        p = rng.uniform(0.02, 0.98, 15_000).astype(np.float32)
        bone = rng.binomial(1, p).astype(np.uint8)
        result = integrate_rates(
            p, bone, coarse_size=81, fine_size=121, histogram_bins=1024
        )
        self.assertTrue(result.converged)
        self.assertLess(result.h_mean, 0.01)
        self.assertLess(result.m_mean, 0.01)


if __name__ == "__main__":
    unittest.main()
