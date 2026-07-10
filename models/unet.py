#!/usr/bin/env python3
"""
models/unet.py
--------------
MONAI Attention U-Net for the MR -> skull-bone segmentation baseline.

This is the deterministic baseline whose test-set masks (and Dice) the MetaCOG
generative-inference method will later be compared against. It is a plain 2D
segmentation net: single-channel MR slice in, single-channel bone logit out.
`s_norm` conditioning is intentionally NOT used here -- that belongs to the
C-VAE shape prior, not to the discriminative baseline.
"""
from __future__ import annotations

import torch.nn as nn
from monai.networks.nets import AttentionUnet


def build_attention_unet(
    in_channels: int = 1,
    out_channels: int = 1,
    channels: tuple[int, ...] = (32, 64, 128, 256),
    strides: tuple[int, ...] = (2, 2, 2),
    dropout: float = 0.0,
) -> nn.Module:
    """Attention U-Net returning raw logits (no final activation).

    Defaults suit 64x64 inputs: 3 downsamplings -> 8x8 bottleneck.
    Loss layers apply the sigmoid, so the model emits logits.
    """
    return AttentionUnet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        dropout=dropout,
    )
