#!/usr/bin/env python3
"""
06_run_unet_inference.py
-------------------------
Run a trained Attention U-Net checkpoint on every patient in a 2D dataset
directory and save predicted bone probabilities + binary masks, for use as
the observed evidence in the downstream MetaCOG generative-inference stage.

Ground truth is loaded from the .npz files only for QC (per-patient Dice in
the summary); it is never fed to the model. By default this processes ALL
patients across all three splits (train/val/test), since the generative
stage needs U-Net masks over the full cohort, not just the held-out test set.

Outputs (default derivatives/unet_predictions/<run-name>/):
  <pid>.npz      mr, bone (ground truth), prob, mask, s_norm, z_index
                 (S,H,W) for mr/bone/prob/mask; (S,) for s_norm/z_index
  summary.csv    per-patient split, n_slices, dice_mean, bone_frac (gt/pred)
  summary.json   overall + per-split Dice mean/std/95% CI, run provenance

Usage:
  # All patients, using the checkpoint's own run directory for output naming
  .venv/bin/python scripts/06_run_unet_inference.py \\
      --run-dir derivatives/unet_runs/v1.1-20260715-123154

  # Explicit checkpoint + only the test split
  .venv/bin/python scripts/06_run_unet_inference.py \\
      --checkpoint derivatives/unet_runs/v1.1-20260715-123154/best.pt \\
      --splits test

  # A couple of patients only, for a quick smoke test
  .venv/bin/python scripts/06_run_unet_inference.py \\
      --run-dir derivatives/unet_runs/v1.1-20260715-123154 --patients 1BA005 1BA012
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.unet import build_attention_unet

ROOT = Path(__file__).resolve().parents[1]
DEF_DATA = ROOT / "derivatives" / "dataset_2d_filtered"
DEF_OUT_ROOT = ROOT / "derivatives" / "unet_predictions"


def pick_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: Path, device: torch.device) -> dict:
    # weights_only=False: our checkpoints store a config dict with pathlib.Path
    # objects, which the (default, PyTorch >=2.6) weights-only loader rejects.
    return torch.load(path, map_location=device, weights_only=False)


def build_model_from_ckpt(ckpt: dict, device: torch.device) -> torch.nn.Module:
    cfg = ckpt.get("config", {})
    channels = tuple(cfg.get("channels", [32, 64, 128, 256]))
    dropout = float(cfg.get("dropout", 0.0))
    model = build_attention_unet(channels=channels, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def dice_per_slice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Hard Dice per slice. pred/gt: (S,H,W) binary arrays."""
    dims = (1, 2)
    inter = (pred * gt).sum(axis=dims)
    denom = pred.sum(axis=dims) + gt.sum(axis=dims)
    return (2 * inter + eps) / (denom + eps)


def ci95(vals: np.ndarray) -> float:
    if vals.size < 2:
        return float("nan")
    return float(1.96 * vals.std(ddof=1) / np.sqrt(vals.size))


@torch.no_grad()
def run_patient(model, mr: np.ndarray, device, threshold: float, batch_size: int = 64):
    """mr: (S,H,W) float32 -> (prob, mask) each (S,H,W)."""
    probs = np.empty_like(mr, dtype=np.float32)
    for i in range(0, mr.shape[0], batch_size):
        chunk = torch.from_numpy(mr[i:i + batch_size])[:, None].to(device)
        logits = model(chunk)
        probs[i:i + batch_size] = torch.sigmoid(logits)[:, 0].cpu().numpy()
    mask = (probs > threshold).astype(np.uint8)
    return probs, mask


def resolve_checkpoint(args) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    if args.run_dir is not None:
        ckpt = args.run_dir / ("last.pt" if args.use_last else "best.pt")
        if not ckpt.exists():
            raise SystemExit(f"No checkpoint at {ckpt}")
        return ckpt
    raise SystemExit("Pass either --run-dir or --checkpoint")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="unet_runs/<name> dir; uses <run-dir>/best.pt by default")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="explicit checkpoint path, overrides --run-dir")
    ap.add_argument("--use-last", action="store_true",
                    help="use last.pt instead of best.pt when --run-dir is given")
    ap.add_argument("--data-dir", type=Path, default=DEF_DATA)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="which splits.json splits to include (default: all three)")
    ap.add_argument("--patients", nargs="+", default=None,
                    help="explicit patient IDs; overrides --splits")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: derivatives/unet_predictions/<run-name>")
    args = ap.parse_args()

    ckpt_path = resolve_checkpoint(args)
    device = pick_device(args.device)
    ckpt = load_checkpoint(ckpt_path, device)
    model = build_model_from_ckpt(ckpt, device)
    print(f"loaded {ckpt_path}  (trained_epoch={ckpt.get('epoch')}  "
          f"val_dice={ckpt.get('val_dice')})  device={device}")

    if args.patients:
        pid_split = {pid: "custom" for pid in args.patients}
    else:
        splits = json.loads((args.data_dir / "splits.json").read_text())
        pid_split = {pid: sp for sp in args.splits for pid in splits.get(sp, [])}
    pids = sorted(pid_split)
    if not pids:
        raise SystemExit("No patients selected")

    run_name = ckpt_path.parent.name if args.run_dir or args.checkpoint else "run"
    out_dir = args.out_dir or (DEF_OUT_ROOT / run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, pid in enumerate(pids, 1):
        pack = np.load(args.data_dir / f"{pid}.npz")
        mr = pack["mr"].astype(np.float32)
        bone = pack["bone"].astype(np.uint8)
        s_norm = pack["s_norm"]
        z_index = pack["z_index"]

        prob, mask = run_patient(model, mr, device, args.threshold)
        dice = dice_per_slice(mask, bone)

        np.savez_compressed(
            out_dir / f"{pid}.npz",
            mr=mr, bone=bone, prob=prob, mask=mask,
            s_norm=s_norm, z_index=z_index,
        )
        rows.append({
            "patient": pid, "split": pid_split[pid], "n_slices": int(mr.shape[0]),
            "dice_mean": round(float(dice.mean()), 4),
            "bone_frac_gt": round(float(bone.mean()), 5),
            "bone_frac_pred": round(float(mask.mean()), 5),
        })
        print(f"[{i:3d}/{len(pids)}] {pid:8s} split={pid_split[pid]:6s} "
              f"S={mr.shape[0]:3d}  dice={dice.mean():.4f}")

    fields = ["patient", "split", "n_slices", "dice_mean", "bone_frac_gt", "bone_frac_pred"]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    overall = np.array([r["dice_mean"] for r in rows])
    by_split = {}
    for sp in sorted(set(pid_split.values())):
        vals = np.array([r["dice_mean"] for r in rows if r["split"] == sp])
        if vals.size == 0:
            continue
        by_split[sp] = {
            "n_patients": int(vals.size),
            "dice_mean": float(vals.mean()),
            "dice_std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "dice_ci95": ci95(vals),
        }

    summary = {
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_dice": ckpt.get("val_dice"),
        "data_dir": str(args.data_dir),
        "threshold": args.threshold,
        "n_patients": len(rows),
        "overall": {
            "dice_mean": float(overall.mean()),
            "dice_std": float(overall.std(ddof=1)) if overall.size > 1 else 0.0,
            "dice_ci95": ci95(overall),
        },
        "by_split": by_split,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("-" * 60)
    print(f"patients={len(rows)}  overall patient-mean Dice="
          f"{summary['overall']['dice_mean']:.4f} +/- {summary['overall']['dice_ci95']:.4f} (95% CI)")
    for sp, s in by_split.items():
        print(f"  {sp:6s} n={s['n_patients']:3d}  dice={s['dice_mean']:.4f} +/- {s['dice_ci95']:.4f}")
    print(f"predictions -> {out_dir}")


if __name__ == "__main__":
    main()
