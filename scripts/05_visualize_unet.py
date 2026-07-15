#!/usr/bin/env python3
"""
05_visualize_unet.py
---------------------
Qualitative QC: for a handful of MR slices, plot the input MR, the ground
truth bone mask, the U-Net's predicted mask, and an error map (false
positive / false negative pixels) side by side.

Slice selection: by default, --n-patients patients are sampled from --split
(deterministically, via --seed), and --n-slices-per-patient slices are picked
evenly spaced across each patient's *s_norm* range (vault base -> crown), so
the figure always shows a spread of slice positions rather than only the
most bone-rich ones.

Outputs one PNG per patient:
  logs/unet_qc/<run-name>/<pid>.png

Usage:
  # Default: 4 random test-split patients, 5 slices each
  .venv/bin/python scripts/05_visualize_unet.py \\
      --run-dir derivatives/unet_runs/v1.1-20260715-123154

  # A specific patient and specific slice indices (0-based, into the .npz)
  .venv/bin/python scripts/05_visualize_unet.py \\
      --run-dir derivatives/unet_runs/v1.1-20260715-123154 \\
      --patients 1BA005 --slice-idx 10 25 40 55
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.unet import build_attention_unet

ROOT = Path(__file__).resolve().parents[1]
DEF_DATA = ROOT / "derivatives" / "dataset_2d_filtered"
DEF_OUT_ROOT = ROOT / "logs" / "unet_qc"


def pick_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def build_model_from_ckpt(ckpt: dict, device: torch.device) -> torch.nn.Module:
    cfg = ckpt.get("config", {})
    channels = tuple(cfg.get("channels", [32, 64, 128, 256]))
    dropout = float(cfg.get("dropout", 0.0))
    model = build_attention_unet(channels=channels, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def resolve_checkpoint(args) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    if args.run_dir is not None:
        ckpt = args.run_dir / ("last.pt" if args.use_last else "best.pt")
        if not ckpt.exists():
            raise SystemExit(f"No checkpoint at {ckpt}")
        return ckpt
    raise SystemExit("Pass either --run-dir or --checkpoint")


def pick_slice_indices(n_slices: int, k: int) -> list[int]:
    k = min(k, n_slices)
    return sorted(set(int(i) for i in np.linspace(0, n_slices - 1, k)))


@torch.no_grad()
def predict(model, mr: np.ndarray, device, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    chunk = torch.from_numpy(mr)[:, None].to(device)
    logits = model(chunk)
    prob = torch.sigmoid(logits)[:, 0].cpu().numpy()
    mask = (prob > threshold).astype(np.uint8)
    return prob, mask


def dice_1(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    inter = float((pred * gt).sum())
    denom = float(pred.sum() + gt.sum())
    return (2 * inter + eps) / (denom + eps)


def plot_patient(pid: str, idxs: list[int], mr: np.ndarray, bone: np.ndarray,
                 prob: np.ndarray, mask: np.ndarray, s_norm: np.ndarray,
                 out_path: Path, run_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 4
    nrows = len(idxs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes)
    col_titles = ["MR input", "Ground truth", "U-Net prediction", "Error (FP=red, FN=blue)"]

    for r, i in enumerate(idxs):
        m, b, p, msk = mr[i], bone[i], prob[i], mask[i]
        d = dice_1(msk, b)

        ax = axes[r, 0]
        ax.imshow(m.T, cmap="gray", origin="lower")
        ax.set_ylabel(f"slice {i}\ns_norm={s_norm[i]:.2f}", fontsize=9)

        ax = axes[r, 1]
        ax.imshow(m.T, cmap="gray", origin="lower")
        ax.imshow(np.ma.masked_where(b.T == 0, b.T), cmap="autumn", origin="lower", alpha=0.6)

        ax = axes[r, 2]
        ax.imshow(m.T, cmap="gray", origin="lower")
        ax.imshow(np.ma.masked_where(msk.T == 0, msk.T), cmap="winter", origin="lower", alpha=0.6)
        ax.set_title(f"Dice={d:.3f}", fontsize=9)

        fp = ((msk == 1) & (b == 0)).astype(np.float32)
        fn = ((msk == 0) & (b == 1)).astype(np.float32)
        err = np.zeros((*m.shape, 3), dtype=np.float32)
        err[..., 0] = fp.T   # red = false positive
        err[..., 2] = fn.T   # blue = false negative
        ax = axes[r, 3]
        ax.imshow(m.T, cmap="gray", origin="lower")
        ax.imshow(np.clip(err, 0, 1), origin="lower", alpha=0.7)

        for c in range(ncols):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(col_titles[c] if c != 2 else f"{col_titles[c]}\nDice={d:.3f}",
                                     fontsize=10)

    fig.suptitle(f"{pid}  ({run_name})", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"{pid}: wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--use-last", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=DEF_DATA)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--patients", nargs="+", default=None,
                    help="explicit patient IDs; overrides --split/--n-patients")
    ap.add_argument("--n-patients", type=int, default=4)
    ap.add_argument("--n-slices-per-patient", type=int, default=5)
    ap.add_argument("--slice-idx", type=int, nargs="+", default=None,
                    help="explicit 0-based slice indices (applied to every selected patient)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: logs/unet_qc/<run-name>")
    args = ap.parse_args()

    ckpt_path = resolve_checkpoint(args)
    device = pick_device(args.device)
    ckpt = load_checkpoint(ckpt_path, device)
    model = build_model_from_ckpt(ckpt, device)
    print(f"loaded {ckpt_path}  (trained_epoch={ckpt.get('epoch')}  "
          f"val_dice={ckpt.get('val_dice')})  device={device}")

    if args.patients:
        pids = args.patients
    else:
        splits = json.loads((args.data_dir / "splits.json").read_text())
        pool = splits[args.split]
        rng = np.random.default_rng(args.seed)
        pids = sorted(rng.choice(pool, size=min(args.n_patients, len(pool)), replace=False).tolist())

    run_name = ckpt_path.parent.name if args.run_dir or args.checkpoint else "run"
    out_dir = args.out_dir or (DEF_OUT_ROOT / run_name)

    for pid in pids:
        pack = np.load(args.data_dir / f"{pid}.npz")
        mr = pack["mr"].astype(np.float32)
        bone = pack["bone"].astype(np.uint8)
        s_norm = pack["s_norm"]

        idxs = args.slice_idx if args.slice_idx else pick_slice_indices(mr.shape[0], args.n_slices_per_patient)
        idxs = [i for i in idxs if 0 <= i < mr.shape[0]]

        prob, mask = predict(model, mr, device, args.threshold)
        plot_patient(pid, idxs, mr, bone, prob, mask, s_norm,
                     out_dir / f"{pid}.png", run_name)

    print(f"figures -> {out_dir}")


if __name__ == "__main__":
    main()
