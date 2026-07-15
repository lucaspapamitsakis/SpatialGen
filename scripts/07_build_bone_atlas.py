#!/usr/bin/env python3
"""
07_build_bone_atlas.py
-----------------------
Build an empirical, non-learned prior P(bone) over the 64x64 vault-crop grid,
stratified by through-plane position `s_norm`. This stands in for the (not yet
implemented) C-VAE shape prior in the first-pass MetaCOG inference model:

  atlas[bin][y, x] = fraction of TRAINING-split slices with s_norm in that bin
                      that have bone at pixel (y, x)

Only the frozen train split is used, so this never leaks validation/test
ground truth into the prior used at inference time.

`s_norm` in [0, 1] (0 = vault base, 1 = crown) is binned into --n-bins equal
intervals. A patient's own slices are excluded from nothing special here --
the atlas is a population-level prior, shared across all patients, and is
built once from train data only.

Because 64x64 pixel bins can be sparse for extreme s_norm values (near the
crown, where the vault cross-section is small), the atlas is:
  1. averaged within each s_norm bin across all training slices landing there;
  2. lightly Gaussian-blurred spatially (--blur-sigma pixels) for smoothness;
  3. clipped to [--eps, 1 - --eps] so no pixel is ever exactly 0 or 1
     (avoids -inf log-likelihood if the atlas disagrees with an observation).

Output: derivatives/bone_atlas.npz
  atlas       float32 [n_bins, 64, 64]  P(bone) per bin per pixel
  bin_edges   float32 [n_bins + 1]      s_norm bin boundaries
  bin_counts  int64   [n_bins]          number of training slices per bin

Usage:
  .venv/bin/python scripts/07_build_bone_atlas.py
  .venv/bin/python scripts/07_build_bone_atlas.py --n-bins 20 --blur-sigma 1.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
DEF_DATA = ROOT / "derivatives" / "dataset_2d_filtered"
DEF_OUT = ROOT / "derivatives" / "bone_atlas.npz"


def build_atlas(data_dir: Path, pids: list[str], n_bins: int,
                blur_sigma: float, eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    sums = None
    counts = np.zeros(n_bins, dtype=np.int64)

    for pid in pids:
        pack = np.load(data_dir / f"{pid}.npz")
        bone = pack["bone"].astype(np.float32)   # (S, H, W)
        s_norm = pack["s_norm"]                  # (S,)
        if sums is None:
            H, W = bone.shape[1:]
            sums = np.zeros((n_bins, H, W), dtype=np.float64)

        bin_idx = np.clip(np.digitize(s_norm, bin_edges[1:-1]), 0, n_bins - 1)
        for b in range(n_bins):
            sel = bin_idx == b
            if not sel.any():
                continue
            sums[b] += bone[sel].sum(axis=0)
            counts[b] += int(sel.sum())

    atlas = np.zeros_like(sums, dtype=np.float32)
    global_mean = sums.sum(axis=0) / max(counts.sum(), 1)
    for b in range(n_bins):
        if counts[b] > 0:
            atlas[b] = (sums[b] / counts[b]).astype(np.float32)
        else:
            atlas[b] = global_mean.astype(np.float32)   # empty bin -> fall back

    if blur_sigma > 0:
        for b in range(n_bins):
            atlas[b] = gaussian_filter(atlas[b], sigma=blur_sigma)

    atlas = np.clip(atlas, eps, 1.0 - eps).astype(np.float32)
    return atlas, bin_edges.astype(np.float32), counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEF_DATA)
    ap.add_argument("--split", default="train",
                    help="splits.json key to build the atlas from (default: train)")
    ap.add_argument("--n-bins", type=int, default=13,
                    help="number of s_norm bins (default 13, roughly ~5 slices of "
                         "resolution given 55-65 slices/patient)")
    ap.add_argument("--blur-sigma", type=float, default=1.0,
                    help="spatial Gaussian blur sigma in pixels, per bin (0 to disable)")
    ap.add_argument("--eps", type=float, default=1e-3,
                    help="clip atlas probabilities to [eps, 1-eps]")
    ap.add_argument("--out", type=Path, default=DEF_OUT)
    args = ap.parse_args()

    splits = json.loads((args.data_dir / "splits.json").read_text())
    pids = splits[args.split]
    print(f"building atlas from {len(pids)} '{args.split}'-split patients, "
          f"{args.n_bins} s_norm bins, blur_sigma={args.blur_sigma}")

    atlas, bin_edges, counts = build_atlas(args.data_dir, pids, args.n_bins,
                                           args.blur_sigma, args.eps)

    for b in range(args.n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        print(f"  bin {b:2d}  s_norm=[{lo:.2f},{hi:.2f})  n_slices={counts[b]:5d}  "
              f"mean_bone_frac={atlas[b].mean():.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, atlas=atlas, bin_edges=bin_edges, bin_counts=counts)
    print(f"wrote {args.out}  atlas shape={atlas.shape}")


if __name__ == "__main__":
    main()
