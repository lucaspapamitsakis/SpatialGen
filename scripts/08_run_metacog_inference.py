#!/usr/bin/env python3
"""
08_run_metacog_inference.py
-----------------------------
First-pass MetaCOG generative inference: for each patient, jointly infer a
GLOBAL false-positive rate H and false-negative rate M (shared across that
patient's slices) and a corrected bone-mask posterior, conditioned ONLY on
the frozen U-Net's binary predictions (never on ground truth) plus the
empirical, training-only atlas prior from `07_build_bone_atlas.py`.

Per patient:
  1. Look up the atlas prior p_i for every pixel of every slice (via s_norm).
  2. Run NUTS (Hamiltonian Monte Carlo) over the 2D (H, M) posterior, using the
     marginal sensor-model likelihood in models/metacog.py.
  3. Compute the exact closed-form pixel posterior P(bone=1 | obs, H, M),
     Monte-Carlo averaged over the (H, M) posterior samples.
  4. Threshold at 0.5 for a corrected binary mask; compute Dice against ground
     truth for QC only (ground truth is not used as evidence anywhere above).

Requires U-Net predictions from `06_run_unet_inference.py` (mr, bone GT, prob,
mask, s_norm, z_index per patient) and the atlas from `07_build_bone_atlas.py`.

Outputs (default derivatives/metacog_runs/<unet-run-name>/):
  <pid>.npz     mr, bone (GT), unet_mask, prior_p, post_prob, post_mask,
                s_norm, z_index, H_samples, M_samples (pooled across chains)
  summary.csv   per-patient: H/M posterior mean+CI, R-hat, ESS, dice_unet,
                dice_corrected
  summary.json  run config + aggregate stats

Usage:
  # A handful of validation patients (fast iteration)
  .venv/bin/python scripts/08_run_metacog_inference.py \\
      --predictions-dir derivatives/unet_predictions/v1.1-20260715-123154 \\
      --split val --n-patients 5

  # Full test split, once validated
  .venv/bin/python scripts/08_run_metacog_inference.py \\
      --predictions-dir derivatives/unet_predictions/v1.1-20260715-123154 \\
      --split test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.metacog import (
    load_atlas, atlas_prior_for_slices, run_nuts, posterior_correct,
    gelman_rubin_rhat, effective_sample_size,
)

ROOT = Path(__file__).resolve().parents[1]
DEF_DATA = ROOT / "derivatives" / "dataset_2d_filtered"
DEF_ATLAS = ROOT / "derivatives" / "bone_atlas.npz"
DEF_OUT_ROOT = ROOT / "derivatives" / "metacog_runs"


def dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    inter = float((pred * gt).sum())
    denom = float(pred.sum() + gt.sum())
    return (2 * inter + eps) / (denom + eps)


def ci95(vals: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def select_patients(args) -> dict[str, str]:
    if args.patients:
        return {pid: "custom" for pid in args.patients}
    splits = json.loads((args.data_dir / "splits.json").read_text())
    pool = splits[args.split]
    if args.n_patients is not None and args.n_patients < len(pool):
        rng = np.random.default_rng(args.seed)
        pool = sorted(rng.choice(pool, size=args.n_patients, replace=False).tolist())
    return {pid: args.split for pid in pool}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions-dir", type=Path, required=True,
                    help="output dir of 06_run_unet_inference.py (has <pid>.npz with mr/bone/prob/mask)")
    ap.add_argument("--atlas", type=Path, default=DEF_ATLAS)
    ap.add_argument("--data-dir", type=Path, default=DEF_DATA,
                    help="used only to read splits.json for --split selection")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--patients", nargs="+", default=None,
                    help="explicit patient IDs; overrides --split/--n-patients")
    ap.add_argument("--n-patients", type=int, default=None,
                    help="randomly sample this many patients from --split (default: all)")
    ap.add_argument("--num-samples", type=int, default=800)
    ap.add_argument("--warmup-steps", type=int, default=400)
    ap.add_argument("--num-chains", type=int, default=2,
                    help=">=2 needed for R-hat convergence diagnostics")
    ap.add_argument("--h-prior", type=float, nargs=2, default=[1.0, 1.0],
                    help="Beta(a,b) prior on the global false-positive rate H")
    ap.add_argument("--m-prior", type=float, nargs=2, default=[1.0, 1.0],
                    help="Beta(a,b) prior on the global false-negative rate M")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: derivatives/metacog_runs/<predictions-dir-name>")
    args = ap.parse_args()

    atlas_dict = load_atlas(args.atlas)
    pid_split = select_patients(args)
    pids = sorted(pid_split)
    if not pids:
        raise SystemExit("No patients selected")

    out_dir = args.out_dir or (DEF_OUT_ROOT / args.predictions_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"patients={len(pids)}  num_samples={args.num_samples}  "
          f"warmup={args.warmup_steps}  chains={args.num_chains}")
    print(f"H ~ Beta{tuple(args.h_prior)}   M ~ Beta{tuple(args.m_prior)}")

    rows = []
    for i, pid in enumerate(pids, 1):
        t0 = time.time()
        pack = np.load(args.predictions_dir / f"{pid}.npz")
        mr = pack["mr"].astype(np.float32)
        bone = pack["bone"].astype(np.uint8)
        unet_mask = pack["mask"].astype(np.uint8)
        s_norm = pack["s_norm"]
        z_index = pack["z_index"]

        prior_p = atlas_prior_for_slices(atlas_dict, s_norm)   # (S,H,W)
        S, H, W = mr.shape
        prior_flat = torch.from_numpy(prior_p.reshape(-1).astype(np.float32))
        obs_flat = torch.from_numpy(unet_mask.reshape(-1).astype(np.float32))

        samples = run_nuts(prior_flat, obs_flat, num_samples=args.num_samples,
                           warmup_steps=args.warmup_steps, num_chains=args.num_chains,
                           h_prior=tuple(args.h_prior), m_prior=tuple(args.m_prior),
                           seed=args.seed)
        H_s, M_s = samples["H"], samples["M"]
        h_rhat = gelman_rubin_rhat(samples["_per_chain"]["H"])
        m_rhat = gelman_rubin_rhat(samples["_per_chain"]["M"])
        h_ess = effective_sample_size(samples["_per_chain"]["H"])
        m_ess = effective_sample_size(samples["_per_chain"]["M"])

        post_flat = posterior_correct(prior_flat, obs_flat, H_s, M_s)
        post_prob = post_flat.reshape(S, H, W).numpy()
        post_mask = (post_prob > 0.5).astype(np.uint8)

        d_unet = dice(unet_mask, bone)
        d_corr = dice(post_mask, bone)
        h_lo, h_hi = ci95(H_s.numpy())
        m_lo, m_hi = ci95(M_s.numpy())
        dt = time.time() - t0

        np.savez_compressed(
            out_dir / f"{pid}.npz",
            mr=mr, bone=bone, unet_mask=unet_mask, prior_p=prior_p,
            post_prob=post_prob.astype(np.float32), post_mask=post_mask,
            s_norm=s_norm, z_index=z_index,
            H_samples=H_s.numpy().astype(np.float32),
            M_samples=M_s.numpy().astype(np.float32),
        )
        rows.append({
            "patient": pid, "split": pid_split[pid], "n_slices": int(S),
            "H_mean": round(float(H_s.mean()), 5), "H_ci_lo": round(h_lo, 5),
            "H_ci_hi": round(h_hi, 5), "H_rhat": round(h_rhat, 4), "H_ess": round(h_ess, 1),
            "M_mean": round(float(M_s.mean()), 5), "M_ci_lo": round(m_lo, 5),
            "M_ci_hi": round(m_hi, 5), "M_rhat": round(m_rhat, 4), "M_ess": round(m_ess, 1),
            "dice_unet": round(d_unet, 4), "dice_corrected": round(d_corr, 4),
            "dice_delta": round(d_corr - d_unet, 4),
        })
        print(f"[{i:3d}/{len(pids)}] {pid:8s} split={pid_split[pid]:6s} "
              f"H={H_s.mean():.4f} [{h_lo:.4f},{h_hi:.4f}] rhat={h_rhat:.3f}  "
              f"M={M_s.mean():.4f} [{m_lo:.4f},{m_hi:.4f}] rhat={m_rhat:.3f}  "
              f"dice: unet={d_unet:.4f} -> corrected={d_corr:.4f}  ({dt:.1f}s)")

    fields = ["patient", "split", "n_slices", "H_mean", "H_ci_lo", "H_ci_hi", "H_rhat", "H_ess",
              "M_mean", "M_ci_lo", "M_ci_hi", "M_rhat", "M_ess",
              "dice_unet", "dice_corrected", "dice_delta"]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    d_unet_all = np.array([r["dice_unet"] for r in rows])
    d_corr_all = np.array([r["dice_corrected"] for r in rows])
    summary = {
        "predictions_dir": str(args.predictions_dir),
        "atlas": str(args.atlas),
        "n_patients": len(rows),
        "num_samples": args.num_samples,
        "warmup_steps": args.warmup_steps,
        "num_chains": args.num_chains,
        "h_prior": args.h_prior,
        "m_prior": args.m_prior,
        "seed": args.seed,
        "dice_unet_mean": float(d_unet_all.mean()),
        "dice_corrected_mean": float(d_corr_all.mean()),
        "dice_delta_mean": float((d_corr_all - d_unet_all).mean()),
        "max_rhat": float(max(max(r["H_rhat"], r["M_rhat"]) for r in rows)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("-" * 60)
    print(f"patients={len(rows)}  dice: unet={summary['dice_unet_mean']:.4f} -> "
          f"corrected={summary['dice_corrected_mean']:.4f}  "
          f"(delta={summary['dice_delta_mean']:+.4f})")
    print(f"max R-hat across all patients/params = {summary['max_rhat']:.3f} "
          f"({'OK' if summary['max_rhat'] < 1.05 else 'CHECK CONVERGENCE'})")
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
