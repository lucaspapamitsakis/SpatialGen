#!/usr/bin/env python3
"""
04_train_unet.py
----------------
Train the MONAI Attention U-Net baseline: MR slice -> skull-bone mask.

Loss = weighted BCE (pos_weight for class imbalance) + soft Dice.
Logging = Weights & Biases (optional; disabled by default so smoke tests need
no login). Data = the filtered Stage-2 dataset + frozen patient-level splits, so
the test-set Dice reported here is exactly what the generative method is later
compared against.

Key flags:
  --data-dir      dataset dir (default: derivatives/dataset_2d_filtered)
  --epochs        training epochs (default 60)
  --batch-size    default 32
  --lr            Adam LR (default 1e-3)
  --pos-weight    BCE positive-class weight (default: auto from train set)
  --bce-weight    weight on BCE term (default 1.0)
  --dice-weight   weight on Dice term (default 1.0)
  --augment       enable light flip/rotate augmentation
  --wandb         enable W&B logging (else offline/no-op)
  --wandb-project W&B project (default spatialgen-unet)
  --overfit N     smoke test: train+val on the same N slices, no aug (expect Dice->~1)
  --device        cuda|mps|cpu (default: auto)
  --out-dir       checkpoint dir (default: derivatives/unet_runs/<timestamp>)

Examples:
  # Local smoke test: confirm the model can memorize a few slices
  .venv/bin/python scripts/04_train_unet.py --overfit 16 --epochs 60 --device cpu

  # Full local/cluster run with W&B
  .venv/bin/python scripts/04_train_unet.py --epochs 80 --augment --wandb
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.unet import build_attention_unet

ROOT = Path(__file__).resolve().parents[1]
DEF_DATA = ROOT / "derivatives" / "dataset_2d_filtered"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class SliceDataset(Dataset):
    """All 2D slices from the patients in one split, held in memory.

    Returns (mr[1,H,W] float32, bone[1,H,W] float32). Optional light augmentation
    (horizontal flip + 90-degree rotations) applied identically to MR and mask.
    """

    def __init__(self, data_dir: Path, pids: list[str], augment: bool = False):
        self.augment = augment
        mrs, bones = [], []
        for pid in pids:
            pack = np.load(data_dir / f"{pid}.npz")
            mrs.append(pack["mr"].astype(np.float32))
            bones.append(pack["bone"].astype(np.float32))
        self.mr = np.concatenate(mrs, axis=0)      # (N, H, W)
        self.bone = np.concatenate(bones, axis=0)  # (N, H, W)

    def __len__(self) -> int:
        return self.mr.shape[0]

    def __getitem__(self, i: int):
        mr = self.mr[i]
        bone = self.bone[i]
        if self.augment:
            if np.random.rand() < 0.5:
                mr = np.flip(mr, axis=1); bone = np.flip(bone, axis=1)
            k = np.random.randint(0, 4)
            if k:
                mr = np.rot90(mr, k); bone = np.rot90(bone, k)
        mr = torch.from_numpy(np.ascontiguousarray(mr))[None]
        bone = torch.from_numpy(np.ascontiguousarray(bone))[None]
        return mr, bone


def load_splits(data_dir: Path) -> dict:
    return json.loads((data_dir / "splits.json").read_text())


def compute_pos_weight(data_dir: Path, pids: list[str]) -> float:
    pos = tot = 0
    for pid in pids:
        b = np.load(data_dir / f"{pid}.npz")["bone"]
        pos += int(b.sum()); tot += int(b.size)
    frac = pos / tot
    return (1.0 - frac) / frac


# --------------------------------------------------------------------------- #
# Loss & metric
# --------------------------------------------------------------------------- #
class BCEDiceLoss(nn.Module):
    """Weighted BCE-with-logits + soft Dice on the sigmoid probabilities."""

    def __init__(self, pos_weight: float, bce_w: float = 1.0, dice_w: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.bce_w, self.dice_w = bce_w, dice_w

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dims = (2, 3)
        inter = (probs * target).sum(dims)
        denom = probs.sum(dims) + target.sum(dims)
        dice = (2 * inter + 1.0) / (denom + 1.0)
        dice_loss = 1.0 - dice.mean()
        return self.bce_w * bce + self.dice_w * dice_loss, bce.detach(), dice_loss.detach()


@torch.no_grad()
def dice_score(logits, target, thr: float = 0.5, eps: float = 1e-6) -> float:
    """Hard Dice at a probability threshold, averaged over the batch."""
    pred = (torch.sigmoid(logits) > thr).float()
    dims = (2, 3)
    inter = (pred * target).sum(dims)
    denom = pred.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return float(dice.mean().item())


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def pick_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> dict:
    model.eval()
    tot_loss = tot_dice = n = 0
    for mr, bone in loader:
        mr, bone = mr.to(device), bone.to(device)
        logits = model(mr)
        loss, _, _ = loss_fn(logits, bone)
        bs = mr.size(0)
        tot_loss += float(loss) * bs
        tot_dice += dice_score(logits, bone) * bs
        n += bs
    return {"loss": tot_loss / n, "dice": tot_dice / n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEF_DATA)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=None)
    ap.add_argument("--bce-weight", type=float, default=1.0)
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="spatialgen-unet")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--overfit", type=int, default=0,
                    help="smoke test: use the same N slices for train+val")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    splits = load_splits(args.data_dir)
    train_ds = SliceDataset(args.data_dir, splits["train"], augment=args.augment)
    val_ds = SliceDataset(args.data_dir, splits["val"], augment=False)

    if args.overfit:
        n = args.overfit
        sub = SliceDataset(args.data_dir, splits["train"][:2], augment=False)
        sub.mr, sub.bone = sub.mr[:n], sub.bone[:n]
        train_ds = val_ds = sub
        print(f"[overfit] using {len(sub)} slices for train==val (no augmentation)")

    if args.pos_weight is not None:
        pos_weight = args.pos_weight
    elif args.overfit:
        pos_weight = 1.0
    else:
        pos_weight = compute_pos_weight(args.data_dir, splits["train"])
    print(f"device={device}  train={len(train_ds)}  val={len(val_ds)}  "
          f"pos_weight={pos_weight:.2f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=not args.overfit, num_workers=args.num_workers,
                              drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    model = build_attention_unet(channels=tuple(args.channels),
                                 dropout=args.dropout).to(device)
    loss_fn = BCEDiceLoss(pos_weight, args.bce_weight, args.dice_weight).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or (ROOT / "derivatives" / "unet_runs" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.wandb_name,
                         config={**vars(args), "pos_weight": pos_weight,
                                 "n_train": len(train_ds), "n_val": len(val_ds)})

    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss = run_bce = run_dice = seen = 0
        for mr, bone in train_loader:
            mr, bone = mr.to(device), bone.to(device)
            opt.zero_grad()
            logits = model(mr)
            loss, bce, dloss = loss_fn(logits, bone)
            loss.backward()
            opt.step()
            bs = mr.size(0)
            run_loss += float(loss.detach()) * bs
            run_bce += float(bce) * bs
            run_dice += float(dloss) * bs
            seen += bs
        train_metrics = {"loss": run_loss / seen, "bce": run_bce / seen,
                         "dice_loss": run_dice / seen}
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        dt = time.time() - t0

        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_metrics['loss']:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"val_dice={val_metrics['dice']:.4f}  ({dt:.1f}s)")

        if run is not None:
            run.log({"epoch": epoch,
                     "train/loss": train_metrics["loss"],
                     "train/bce": train_metrics["bce"],
                     "train/dice_loss": train_metrics["dice_loss"],
                     "val/loss": val_metrics["loss"],
                     "val/dice": val_metrics["dice"]})

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_dice": best_dice, "config": vars(args),
                        "pos_weight": pos_weight},
                       out_dir / "best.pt")

    torch.save({"model": model.state_dict(), "epoch": args.epochs,
                "config": vars(args), "pos_weight": pos_weight},
               out_dir / "last.pt")
    print(f"best val Dice = {best_dice:.4f}")
    print(f"checkpoints  -> {out_dir}")
    if run is not None:
        run.summary["best_val_dice"] = best_dice
        run.finish()


if __name__ == "__main__":
    main()
