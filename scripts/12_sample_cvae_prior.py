#!/usr/bin/env python3
"""
12_sample_cvae_prior.py
-----------------------
Export Monte Carlo soft bone priors from a trained s_norm-conditioned C-VAE.

For each patient in a split, average K decoder samples z ~ N(0, I) at that
slice's s_norm to produce p_i ∈ (0,1)^(S×H×W). These maps plug into MetaCOG
via scripts/08_run_metacog_inference.py --prior-dir.

Ground-truth bone is never used at sampling time (only s_norm). Optional QC
mosaics compare prior mean / samples against GT for visual inspection only.

Examples:
  .venv/bin/python scripts/12_sample_cvae_prior.py \
      --checkpoint derivatives/cvae_runs/<stamp>/best.pt --split val

  .venv/bin/python scripts/12_sample_cvae_prior.py \
      --checkpoint derivatives/cvae_runs/<stamp>/best.pt --split test \
      --n-samples 32 --save-qc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.cvae import load_cvae  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEF_DATA = ROOT / "derivatives" / "dataset_2d_filtered"


def pick_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_splits(data_dir: Path) -> dict:
    return json.loads((data_dir / "splits.json").read_text())


@torch.no_grad()
def prior_for_patient(
    model,
    s_norm: np.ndarray,
    *,
    n_samples: int,
    clip_eps: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    s = torch.from_numpy(s_norm.astype(np.float32))
    outs = []
    for start in range(0, len(s), batch_size):
        chunk = s[start:start + batch_size].to(device)
        prior = model.sample_prior(chunk, n_samples=n_samples, clip_eps=clip_eps)
        outs.append(prior.squeeze(1).cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


def save_qc_panel(
    out_path: Path,
    pid: str,
    bone: np.ndarray,
    prior: np.ndarray,
    s_norm: np.ndarray,
    sample_logits: list[np.ndarray],
    n_show: int = 4,
) -> None:
    """Show GT, MC mean prior, and a few individual samples at evenly spaced heights."""
    s = bone.shape[0]
    idxs = np.linspace(0, s - 1, num=min(n_show, s), dtype=int)
    n_cols = 2 + len(sample_logits)
    fig, axes = plt.subplots(len(idxs), n_cols, figsize=(2.2 * n_cols, 2.2 * len(idxs)))
    if len(idxs) == 1:
        axes = np.asarray([axes])
    headers = ["GT bone", f"MC prior (K)"] + [f"sample {i+1}" for i in range(len(sample_logits))]
    for r, i in enumerate(idxs):
        panels = [bone[i], prior[i]] + [samp[i] for samp in sample_logits]
        for c, img in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(headers[c], fontsize=9)
            if c == 0:
                ax.set_ylabel(f"s={s_norm[i]:.2f}", fontsize=8)
    fig.suptitle(pid, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=DEF_DATA)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--patients", nargs="+", default=None)
    ap.add_argument("--n-patients", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=16,
                    help="number of z~N(0,I) draws averaged per slice")
    ap.add_argument("--clip-eps", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--save-qc", action="store_true",
                    help="write a few patient mosaics under out-dir/qc/")
    ap.add_argument("--qc-patients", type=int, default=4)
    ap.add_argument("--qc-samples", type=int, default=3,
                    help="individual decoder samples shown in QC panels")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    model, ckpt = load_cvae(args.checkpoint, device=device, eval_mode=True)
    splits = load_splits(args.data_dir)
    pids = list(args.patients) if args.patients else list(splits[args.split])
    pids = sorted(pids)
    if args.n_patients is not None:
        pids = pids[: args.n_patients]
    if not pids:
        raise SystemExit("No patients selected")

    run_name = args.checkpoint.parent.name
    out_dir = args.out_dir or (
        ROOT / "derivatives" / "cvae_priors" / run_name / args.split
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n_samples": args.n_samples,
        "clip_eps": args.clip_eps,
        "seed": args.seed,
        "latent_dim": int(ckpt.get("config", {}).get("latent_dim", model.latent_dim)),
        "patients": pids,
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2))

    print(f"device={device}  patients={len(pids)}  K={args.n_samples}  -> {out_dir}")
    qc_done = 0
    for i, pid in enumerate(pids, 1):
        pack = np.load(args.data_dir / f"{pid}.npz")
        s_norm = pack["s_norm"].astype(np.float32)
        prior = prior_for_patient(
            model, s_norm,
            n_samples=args.n_samples,
            clip_eps=args.clip_eps,
            batch_size=args.batch_size,
            device=device,
        )
        np.savez_compressed(
            out_dir / f"{pid}.npz",
            prior=prior.astype(np.float32),
            s_norm=s_norm,
            z_index=pack["z_index"],
        )
        print(f"[{i}/{len(pids)}] {pid}  prior mean={prior.mean():.4f}  "
              f"max={prior.max():.4f}")

        if args.save_qc and qc_done < args.qc_patients:
            bone = pack["bone"].astype(np.float32)
            # Extra individual samples for the QC panel only.
            sample_maps = []
            s_t = torch.from_numpy(s_norm).to(device)
            with torch.no_grad():
                for _ in range(args.qc_samples):
                    z = torch.randn(len(s_norm), model.latent_dim, device=device)
                    samp = torch.sigmoid(model.decode(z, s_t)).squeeze(1).cpu().numpy()
                    sample_maps.append(samp.astype(np.float32))
            save_qc_panel(
                out_dir / "qc" / f"{pid}.png",
                pid, bone, prior, s_norm, sample_maps,
            )
            qc_done += 1

    print(f"wrote {len(pids)} prior stacks -> {out_dir}")
    print("MetaCOG: scripts/08_run_metacog_inference.py --prior-dir", out_dir)


if __name__ == "__main__":
    main()
