#!/usr/bin/env python3
"""
models/cvae.py
--------------
s_norm-conditioned convolutional VAE for soft skull-bone shape priors.

The encoder sees the bone mask plus a constant s_norm channel. The decoder
sees latent z concatenated with the scalar s_norm. Outputs are raw logits;
callers apply sigmoid for soft P(bone) maps used as MetaCOG priors p_i.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _snorm_channel(bone: torch.Tensor, s_norm: torch.Tensor) -> torch.Tensor:
    """Broadcast per-slice s_norm to a (B,1,H,W) constant channel."""
    if s_norm.ndim == 1:
        s_norm = s_norm[:, None]
    if s_norm.ndim == 2 and s_norm.shape[-1] == 1:
        pass
    elif s_norm.ndim != 2:
        raise ValueError(f"s_norm must be (B,) or (B,1); got {tuple(s_norm.shape)}")
    return s_norm[:, :, None, None].expand(-1, 1, bone.shape[-2], bone.shape[-1])


class ConditionalVAE(nn.Module):
    """2D C-VAE: bone mask + s_norm -> soft bone logits."""

    def __init__(
        self,
        latent_dim: int = 16,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        image_size: int = 64,
    ):
        super().__init__()
        if len(channels) != 4:
            raise ValueError("Expected 4 encoder/decoder channel stages for 64x64")
        if image_size != 64:
            raise ValueError("This architecture is fixed for 64x64 inputs")

        self.latent_dim = int(latent_dim)
        self.channels = tuple(int(c) for c in channels)
        self.image_size = int(image_size)
        self.bottleneck = image_size // (2 ** len(channels))  # 4 for 64
        bot_ch = self.channels[-1]
        flat = bot_ch * self.bottleneck * self.bottleneck

        enc: list[nn.Module] = []
        in_ch = 2  # bone + s_norm channel
        for out_ch in self.channels:
            enc += [
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
                nn.SiLU(inplace=True),
            ]
            in_ch = out_ch
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(flat, self.latent_dim)
        self.fc_logvar = nn.Linear(flat, self.latent_dim)

        self.fc_dec = nn.Linear(self.latent_dim + 1, flat)
        dec: list[nn.Module] = []
        rev = list(reversed(self.channels))
        for i, out_ch in enumerate(rev[1:] + [self.channels[0]]):
            in_ch = rev[i]
            dec += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
                nn.SiLU(inplace=True),
            ]
        self.decoder = nn.Sequential(*dec)
        self.out_conv = nn.Conv2d(self.channels[0], 1, kernel_size=3, padding=1)

    def encode(self, bone: torch.Tensor, s_norm: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([bone, _snorm_channel(bone, s_norm)], dim=1)
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, s_norm: torch.Tensor) -> torch.Tensor:
        if s_norm.ndim == 1:
            s_norm = s_norm[:, None]
        h = self.fc_dec(torch.cat([z, s_norm], dim=1))
        h = h.view(-1, self.channels[-1], self.bottleneck, self.bottleneck)
        h = self.decoder(h)
        return self.out_conv(h)

    def forward(self, bone: torch.Tensor, s_norm: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(bone, s_norm)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, s_norm)
        return logits, mu, logvar

    @torch.no_grad()
    def reconstruct(self, bone: torch.Tensor, s_norm: torch.Tensor,
                    use_mean: bool = True) -> torch.Tensor:
        """Return soft P(bone) reconstructions (B,1,H,W)."""
        self.eval()
        mu, logvar = self.encode(bone, s_norm)
        z = mu if use_mean else self.reparameterize(mu, logvar)
        return torch.sigmoid(self.decode(z, s_norm))

    @torch.no_grad()
    def sample_prior(
        self,
        s_norm: torch.Tensor,
        n_samples: int = 16,
        clip_eps: float = 1e-3,
    ) -> torch.Tensor:
        """Monte Carlo soft prior maps for MetaCOG.

        Parameters
        ----------
        s_norm : (B,) or (B,1)
        n_samples : number of z ~ N(0,I) draws to average
        clip_eps : clip probabilities to [eps, 1-eps]

        Returns
        -------
        prior : (B, 1, H, W) soft P(bone)
        """
        self.eval()
        if s_norm.ndim == 2 and s_norm.shape[-1] == 1:
            s_vec = s_norm.squeeze(-1)
        else:
            s_vec = s_norm
        b = s_vec.shape[0]
        device = s_vec.device
        dtype = s_vec.dtype
        acc = torch.zeros(
            b, 1, self.image_size, self.image_size, device=device, dtype=dtype
        )
        for _ in range(n_samples):
            z = torch.randn(b, self.latent_dim, device=device, dtype=dtype)
            acc = acc + torch.sigmoid(self.decode(z, s_vec))
        prior = acc / float(n_samples)
        if clip_eps > 0:
            prior = prior.clamp(clip_eps, 1.0 - clip_eps)
        return prior


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Mean KL(q(z|x) || N(0,I)) over the batch (nats)."""
    # sum over latent, mean over batch
    return -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def build_cvae(
    latent_dim: int = 16,
    channels: tuple[int, ...] = (32, 64, 128, 256),
    image_size: int = 64,
) -> ConditionalVAE:
    return ConditionalVAE(
        latent_dim=latent_dim, channels=channels, image_size=image_size
    )


def load_cvae(
    checkpoint: str | Path,
    device: torch.device | str = "cpu",
    eval_mode: bool = True,
) -> tuple[ConditionalVAE, dict[str, Any]]:
    """Load a trained C-VAE checkpoint written by scripts/11_train_cvae.py."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = build_cvae(
        latent_dim=int(cfg.get("latent_dim", 16)),
        channels=tuple(cfg.get("channels", (32, 64, 128, 256))),
        image_size=int(cfg.get("image_size", 64)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    if eval_mode:
        model.eval()
    return model, ckpt


def beta_for_epoch(epoch: int, beta_max: float, anneal_epochs: int) -> float:
    """Linear warm-up of the KL weight from 0 to beta_max."""
    if beta_max <= 0:
        return 0.0
    if anneal_epochs <= 0:
        return float(beta_max)
    return float(beta_max) * min(1.0, float(epoch) / float(anneal_epochs))
