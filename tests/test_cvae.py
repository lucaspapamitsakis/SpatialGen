"""Shape, KL, and sampling sanity tests for the s_norm-conditioned C-VAE."""
from __future__ import annotations

import unittest

import torch

from models.cvae import beta_for_epoch, build_cvae, kl_divergence


class CVAETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = build_cvae(latent_dim=8, channels=(16, 32, 64, 128))
        self.b = 4
        self.bone = torch.zeros(self.b, 1, 64, 64)
        # thin ring-ish strip so the mask is non-trivial
        self.bone[:, :, 20:44, 20:22] = 1.0
        self.bone[:, :, 20:22, 20:44] = 1.0
        self.s_norm = torch.linspace(0.1, 0.9, self.b)

    def test_forward_shapes(self):
        logits, mu, logvar = self.model(self.bone, self.s_norm)
        self.assertEqual(tuple(logits.shape), (self.b, 1, 64, 64))
        self.assertEqual(tuple(mu.shape), (self.b, 8))
        self.assertEqual(tuple(logvar.shape), (self.b, 8))

    def test_kl_nonnegative_and_zero_at_prior(self):
        mu = torch.zeros(3, 5)
        logvar = torch.zeros(3, 5)
        self.assertAlmostEqual(float(kl_divergence(mu, logvar)), 0.0, places=6)
        mu2 = torch.ones(3, 5)
        self.assertGreater(float(kl_divergence(mu2, logvar)), 0.0)

    def test_sample_prior_bounds(self):
        prior = self.model.sample_prior(self.s_norm, n_samples=4, clip_eps=1e-3)
        self.assertEqual(tuple(prior.shape), (self.b, 1, 64, 64))
        self.assertGreaterEqual(float(prior.min()), 1e-3 - 1e-8)
        self.assertLessEqual(float(prior.max()), 1.0 - 1e-3 + 1e-8)

    def test_reconstruct_matches_batch(self):
        probs = self.model.reconstruct(self.bone, self.s_norm, use_mean=True)
        self.assertEqual(tuple(probs.shape), (self.b, 1, 64, 64))
        self.assertTrue(torch.isfinite(probs).all())

    def test_beta_anneal(self):
        self.assertEqual(beta_for_epoch(0, 0.1, 10), 0.0)
        self.assertAlmostEqual(beta_for_epoch(5, 0.1, 10), 0.05)
        self.assertEqual(beta_for_epoch(20, 0.1, 10), 0.1)
        self.assertEqual(beta_for_epoch(3, 0.2, 0), 0.2)


if __name__ == "__main__":
    unittest.main()
