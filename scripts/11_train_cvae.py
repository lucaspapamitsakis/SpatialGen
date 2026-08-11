#!/usr/bin/env python3
"""
11_train_cvae.py
----------------
Train an s_norm-conditioned convolutional VAE on filtered training bone masks.

The model learns a soft joint shape prior P(bone | z, s_norm) for MetaCOG.
It never sees MR or U-Net outputs. Validation selects the checkpoint by
reconstruction soft Dice (encode→decode with latent mean).

Loss = weighted BCE + soft Dice + β · KL(q(z|B,s) || N(0,I)), with optional
linear β warm-up.

Key flags:
  --data-dir         filtered dataset (default: derivatives/dataset_2d_filtered)
  --latent-dim       default 16
  --beta             final KL weight (default 0.05)
  --beta-anneal      epochs to warm β from 0 → --beta (default 20)
  --augment          enable optional small in-plane rotations (no flips)
  --aug-rotate-deg   max |rotation| in degrees when --augment (default 10)
  --overfit N        smoke test on N train slices
  --wandb            enable W&B logging (curves + reconstruction/generation panels)
  --wandb-image-every  log image panels every N epochs (default 5; also epoch 1 + last)

Examples:
  .venv/bin/python scripts/11_train_cvae.py --overfit 16 --epochs 40 --device cpu
  .venv/bin/python scripts/11_train_cvae.py --epochs 120 --augment --wandb
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import rotate as nd_rotate
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.cvae import (  # noqa: E402
    beta_for_epoch,
    build_cvae,
    kl_divergence,
)

ROOT = Path(__file__).resolve().parents[1]
DEF_DATA = ROOT / "derivatives" / "dataset_2d_filtered"


class BoneSnormDataset(Dataset):
    """Bone masks + s_norm from one patient split, held in memory.

    Optional augmentation is small continuous rotations only (no flips).
    """

    def __init__(
        self,
        data_dir: Path,
        pids: list[str],
        augment: bool = False,
        aug_rotate_deg: float = 10.0,
    ):
        self.augment = bool(augment)
        self.aug_rotate_deg = float(aug_rotate_deg)
        bones, snorms = [], []
        for pid in pids:
            pack = np.load(data_dir / f"{pid}.npz")
            bones.append(pack["bone"].astype(np.float32))
            snorms.append(pack["s_norm"].astype(np.float32))
        self.bone = np.concatenate(bones, axis=0)      # (N, H, W)
        self.s_norm = np.concatenate(snorms, axis=0)   # (N,)

    def __len__(self) -> int:
        return self.bone.shape[0]

    def __getitem__(self, i: int):
        bone = self.bone[i]
        s_norm = float(self.s_norm[i])
        if self.augment and self.aug_rotate_deg > 0:
            angle = float(np.random.uniform(-self.aug_rotate_deg, self.aug_rotate_deg))
            if abs(angle) > 1e-3:
                bone = nd_rotate(
                    bone, angle, reshape=False, order=0, mode="constant", cval=0.0
                )
                bone = (bone > 0.5).astype(np.float32)
        bone_t = torch.from_numpy(np.ascontiguousarray(bone))[None]
        s_t = torch.tensor(s_norm, dtype=torch.float32)
        return bone_t, s_t


def load_splits(data_dir: Path) -> dict:
    return json.loads((data_dir / "splits.json").read_text())


def compute_pos_weight(data_dir: Path, pids: list[str]) -> float:
    pos = tot = 0
    for pid in pids:
        b = np.load(data_dir / f"{pid}.npz")["bone"]
        pos += int(b.sum())
        tot += int(b.size)
    frac = pos / max(tot, 1)
    return (1.0 - frac) / max(frac, 1e-8)


class BCEDiceKLLoss(nn.Module):
    """Weighted BCE-with-logits + soft Dice + β·KL."""

    def __init__(self, pos_weight: float, bce_w: float = 1.0, dice_w: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.bce_w = bce_w
        self.dice_w = dice_w

    def forward(self, logits, target, mu, logvar, beta: float):
        bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        dims = (2, 3)
        inter = (probs * target).sum(dims)
        denom = probs.sum(dims) + target.sum(dims)
        dice = (2 * inter + 1.0) / (denom + 1.0)
        dice_loss = 1.0 - dice.mean()
        kl = kl_divergence(mu, logvar)
        total = self.bce_w * bce + self.dice_w * dice_loss + float(beta) * kl
        return total, bce.detach(), dice_loss.detach(), kl.detach()


@torch.no_grad()
def soft_dice(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    dims = (2, 3)
    inter = (probs * target).sum(dims)
    denom = probs.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return float(dice.mean().item())


@torch.no_grad()
def hard_dice(logits: torch.Tensor, target: torch.Tensor,
              thr: float = 0.5, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(logits) > thr).float()
    dims = (2, 3)
    inter = (pred * target).sum(dims)
    denom = pred.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return float(dice.mean().item())


def pick_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, beta: float) -> dict:
    model.eval()
    tot_loss = tot_soft = tot_hard = tot_kl = n = 0
    tot_mu_norm = tot_logvar = 0.0
    for bone, s_norm in loader:
        bone = bone.to(device)
        s_norm = s_norm.to(device)
        mu, logvar = model.encode(bone, s_norm)
        logits = model.decode(mu, s_norm)  # deterministic recon for metrics
        loss, _, _, kl = loss_fn(logits, bone, mu, logvar, beta)
        probs = torch.sigmoid(logits)
        bs = bone.size(0)
        tot_loss += float(loss) * bs
        tot_soft += soft_dice(probs, bone) * bs
        tot_hard += hard_dice(logits, bone) * bs
        tot_kl += float(kl) * bs
        tot_mu_norm += float(mu.pow(2).mean().sqrt()) * bs
        tot_logvar += float(logvar.mean()) * bs
        n += bs
    return {
        "loss": tot_loss / n,
        "soft_dice": tot_soft / n,
        "hard_dice": tot_hard / n,
        "kl": tot_kl / n,
        "mu_rms": tot_mu_norm / n,
        "logvar_mean": tot_logvar / n,
    }


def _to_hw(img: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = img.detach().cpu().numpy() if torch.is_tensor(img) else np.asarray(img)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"expected HxW image, got shape {arr.shape}")
    return arr.astype(np.float32)


def make_recon_figure(
    bone: torch.Tensor,
    recon: torch.Tensor,
    s_norm: torch.Tensor,
    n_show: int = 6,
) -> plt.Figure:
    """Rows: GT | reconstruction | |error| for up to n_show validation slices."""
    n = min(n_show, bone.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.2 * n))
    if n == 1:
        axes = np.asarray([axes])
    for i in range(n):
        gt = _to_hw(bone[i])
        pr = _to_hw(recon[i])
        err = np.abs(gt - pr)
        for ax, img, title, vmax in (
            (axes[i, 0], gt, "GT bone", 1.0),
            (axes[i, 1], pr, "Reconstruction", 1.0),
            (axes[i, 2], err, "|GT − recon|", max(float(err.max()), 1e-3)),
        ):
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=10)
        axes[i, 0].set_ylabel(f"s={float(s_norm[i]):.2f}", fontsize=9)
    fig.suptitle("Validation reconstructions (encode → decode with μ)", fontsize=11)
    fig.tight_layout()
    return fig


def make_sample_figure(
    samples: torch.Tensor,
    s_levels: torch.Tensor,
    n_per_level: int,
) -> plt.Figure:
    """Grid of prior samples: rows = s_norm levels, cols = independent z draws."""
    n_levels = len(s_levels)
    fig, axes = plt.subplots(
        n_levels, n_per_level, figsize=(2.0 * n_per_level, 2.0 * n_levels)
    )
    if n_levels == 1:
        axes = np.asarray([axes])
    if n_per_level == 1:
        axes = axes[:, None]
    idx = 0
    for r in range(n_levels):
        for c in range(n_per_level):
            ax = axes[r, c]
            ax.imshow(_to_hw(samples[idx]), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"z#{c + 1}", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"s={float(s_levels[r]):.2f}", fontsize=9)
            idx += 1
    fig.suptitle("Prior samples  z ∼ N(0,I),  decode(z, s_norm)", fontsize=11)
    fig.tight_layout()
    return fig


@torch.no_grad()
def collect_wandb_panels(
    model,
    val_ds: BoneSnormDataset,
    device: torch.device,
    *,
    n_recon: int = 6,
    n_sample_levels: int = 4,
    n_samples_per_level: int = 4,
    seed: int = 0,
) -> dict[str, plt.Figure]:
    """Build reconstruction and generation figures for W&B / local QC."""
    model.eval()
    n_recon = min(n_recon, len(val_ds))
    # Prefer a spread of heights when possible.
    order = np.argsort(val_ds.s_norm)
    if len(order) >= n_recon:
        pick = order[np.linspace(0, len(order) - 1, n_recon, dtype=int)]
    else:
        pick = np.arange(len(order))
    bones, snorms = [], []
    for i in pick:
        b, s = val_ds[int(i)]
        bones.append(b)
        snorms.append(s)
    bone = torch.stack(bones, dim=0).to(device)
    s_norm = torch.stack(snorms, dim=0).to(device)
    recon = model.reconstruct(bone, s_norm, use_mean=True)
    recon_fig = make_recon_figure(bone, recon, s_norm, n_show=n_recon)

    s_levels = torch.linspace(0.15, 0.90, n_sample_levels, device=device)
    s_rep = s_levels.repeat_interleave(n_samples_per_level)
    # Fixed CPU RNG so panel z-draws are comparable across devices/epochs.
    g = torch.Generator()
    g.manual_seed(int(seed) + 17)
    z = torch.randn(
        n_sample_levels * n_samples_per_level,
        model.latent_dim,
        generator=g,
    ).to(device)
    samples = torch.sigmoid(model.decode(z, s_rep))
    sample_fig = make_sample_figure(samples, s_levels.cpu(), n_samples_per_level)

    # Monte Carlo mean prior at the same heights (MetaCOG-style p_i).
    mc = model.sample_prior(s_levels, n_samples=16, clip_eps=1e-3)
    mc_fig, axes = plt.subplots(1, n_sample_levels, figsize=(2.2 * n_sample_levels, 2.4))
    if n_sample_levels == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(_to_hw(mc[i]), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"s={float(s_levels[i]):.2f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    mc_fig.suptitle("MC mean prior  E_z[σ(f(z,s))]  (K=16)", fontsize=11)
    mc_fig.tight_layout()

    return {
        "reconstructions": recon_fig,
        "samples": sample_fig,
        "mc_prior": mc_fig,
    }


def should_log_images(epoch: int, total_epochs: int, every: int) -> bool:
    if every <= 0:
        return False
    return epoch == 1 or epoch == total_epochs or (epoch % every == 0)


def init_wandb(args, *, pos_weight: float, n_train: int, n_val: int, out_dir: Path):
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config={
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "pos_weight": pos_weight,
            "n_train": n_train,
            "n_val": n_val,
            "out_dir": str(out_dir),
        },
        dir=str(out_dir),
    )
    # Force all scalar panels to share the epoch x-axis on the website.
    wandb.define_metric("epoch")
    wandb.define_metric("beta", step_metric="epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("latent/*", step_metric="epoch")
    wandb.define_metric("panels/*", step_metric="epoch")
    return run, wandb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEF_DATA)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--pos-weight", type=float, default=None)
    ap.add_argument("--bce-weight", type=float, default=1.0)
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.05,
                    help="final KL weight after annealing")
    ap.add_argument("--beta-anneal", type=int, default=20,
                    help="epochs to warm β from 0 to --beta (0 = constant)")
    ap.add_argument("--augment", action="store_true",
                    help="optional small rotations; off by default; never flips")
    ap.add_argument("--aug-rotate-deg", type=float, default=10.0,
                    help="max absolute rotation degrees when --augment is set")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--wandb", action="store_true",
                    help="log scalars + reconstruction/generation image panels to W&B")
    ap.add_argument("--wandb-project", default="spatial-gen-cvae")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-image-every", type=int, default=5,
                    help="log image panels every N epochs (also epoch 1 and last); "
                         "0 disables image panels")
    ap.add_argument("--wandb-n-recon", type=int, default=6,
                    help="validation slices shown in the reconstruction panel")
    ap.add_argument("--wandb-n-sample-levels", type=int, default=4,
                    help="s_norm levels in the generation panel")
    ap.add_argument("--wandb-n-samples", type=int, default=4,
                    help="independent z draws per s_norm level")
    ap.add_argument("--overfit", type=int, default=0,
                    help="smoke test: use the same N slices for train+val")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    if len(args.channels) != 4:
        raise SystemExit("--channels must have exactly 4 stages for the 64x64 C-VAE")
    if args.augment and args.aug_rotate_deg < 0:
        raise SystemExit("--aug-rotate-deg must be >= 0")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    splits = load_splits(args.data_dir)
    train_ds = BoneSnormDataset(
        args.data_dir, splits["train"],
        augment=args.augment and not args.overfit,
        aug_rotate_deg=args.aug_rotate_deg,
    )
    val_ds = BoneSnormDataset(
        args.data_dir, splits["val"], augment=False
    )

    if args.overfit:
        n = args.overfit
        sub = BoneSnormDataset(args.data_dir, splits["train"][:2], augment=False)
        sub.bone, sub.s_norm = sub.bone[:n], sub.s_norm[:n]
        train_ds = val_ds = sub
        print(f"[overfit] using {len(sub)} slices for train==val (no augmentation)")

    ckpt = None
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        ckpt_cfg = ckpt.get("config", {})
        for k in ("latent_dim", "channels", "image_size"):
            if k in ckpt_cfg and getattr(args, k) != ckpt_cfg[k]:
                print(f"[resume] overriding --{k.replace('_', '-')} "
                      f"{getattr(args, k)} -> {ckpt_cfg[k]} (from checkpoint)")
                setattr(args, k, ckpt_cfg[k])

    if args.pos_weight is not None:
        pos_weight = args.pos_weight
    elif ckpt is not None and ckpt.get("pos_weight") is not None:
        pos_weight = float(ckpt["pos_weight"])
    elif args.overfit:
        pos_weight = 1.0
    else:
        pos_weight = compute_pos_weight(args.data_dir, splits["train"])

    print(f"device={device}  train={len(train_ds)}  val={len(val_ds)}  "
          f"pos_weight={pos_weight:.2f}  latent={args.latent_dim}  "
          f"beta->{args.beta} (anneal {args.beta_anneal})  "
          f"augment={bool(args.augment and not args.overfit)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=not args.overfit,
        num_workers=args.num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_cvae(
        latent_dim=args.latent_dim,
        channels=tuple(args.channels),
        image_size=args.image_size,
    ).to(device)
    loss_fn = BCEDiceKLLoss(pos_weight, args.bce_weight, args.dice_weight).to(device)
    # Move BCE pos_weight buffer to device
    loss_fn.bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device)
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_soft = -1.0
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer") is not None:
            opt.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0))
        best_soft = float(ckpt.get("val_soft_dice", -1.0))
        print(f"[resume] loaded {args.resume}  (trained {start_epoch} epochs, "
              f"prev best val_soft_dice={best_soft:.4f}); training {args.epochs} more")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or (ROOT / "derivatives" / "cvae_runs" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(vars(args), default=str, indent=2)
    )

    run = None
    wandb = None
    if args.wandb:
        run, wandb = init_wandb(
            args,
            pos_weight=pos_weight,
            n_train=len(train_ds),
            n_val=len(val_ds),
            out_dir=out_dir,
        )
        print(f"[wandb] project={args.wandb_project}  run={run.name}  "
              f"images every {args.wandb_image_every} epoch(s)")

    panel_dir = out_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    total_epochs = start_epoch + args.epochs
    for epoch in range(start_epoch + 1, total_epochs + 1):
        beta = beta_for_epoch(epoch, args.beta, args.beta_anneal)
        model.train()
        t0 = time.time()
        run_loss = run_bce = run_dice = run_kl = run_soft = run_hard = seen = 0
        for bone, s_norm in train_loader:
            bone = bone.to(device)
            s_norm = s_norm.to(device)
            opt.zero_grad()
            logits, mu, logvar = model(bone, s_norm)
            loss, bce, dloss, kl = loss_fn(logits, bone, mu, logvar, beta)
            loss.backward()
            opt.step()
            bs = bone.size(0)
            run_loss += float(loss.detach()) * bs
            run_bce += float(bce) * bs
            run_dice += float(dloss) * bs
            run_kl += float(kl) * bs
            with torch.no_grad():
                run_soft += soft_dice(torch.sigmoid(logits), bone) * bs
                run_hard += hard_dice(logits, bone) * bs
            seen += bs
        train_metrics = {
            "loss": run_loss / seen,
            "bce": run_bce / seen,
            "dice_loss": run_dice / seen,
            "kl": run_kl / seen,
            "soft_dice": run_soft / seen,
            "hard_dice": run_hard / seen,
        }
        val_metrics = evaluate(model, val_loader, loss_fn, device, beta)
        dt = time.time() - t0

        print(f"epoch {epoch:3d}/{total_epochs}  beta={beta:.4f}  "
              f"train_loss={train_metrics['loss']:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"val_soft_dice={val_metrics['soft_dice']:.4f}  "
              f"val_hard_dice={val_metrics['hard_dice']:.4f}  "
              f"val_kl={val_metrics['kl']:.4f}  ({dt:.1f}s)")

        log_payload = {
            "epoch": epoch,
            "beta": beta,
            "train/loss": train_metrics["loss"],
            "train/bce": train_metrics["bce"],
            "train/dice_loss": train_metrics["dice_loss"],
            "train/kl": train_metrics["kl"],
            "train/soft_dice": train_metrics["soft_dice"],
            "train/hard_dice": train_metrics["hard_dice"],
            "val/loss": val_metrics["loss"],
            "val/soft_dice": val_metrics["soft_dice"],
            "val/hard_dice": val_metrics["hard_dice"],
            "val/kl": val_metrics["kl"],
            "latent/mu_rms": val_metrics["mu_rms"],
            "latent/logvar_mean": val_metrics["logvar_mean"],
            "time/epoch_sec": dt,
        }

        if should_log_images(epoch, total_epochs, args.wandb_image_every):
            figs = collect_wandb_panels(
                model, val_ds, device,
                n_recon=args.wandb_n_recon,
                n_sample_levels=args.wandb_n_sample_levels,
                n_samples_per_level=args.wandb_n_samples,
                seed=args.seed + epoch,
            )
            for key, fig in figs.items():
                png = panel_dir / f"epoch{epoch:03d}_{key}.png"
                fig.savefig(png, dpi=140)
                if run is not None and wandb is not None:
                    log_payload[f"panels/{key}"] = wandb.Image(
                        str(png), caption=f"epoch {epoch}"
                    )
                plt.close(fig)
            print(f"  panels -> {panel_dir}/epoch{epoch:03d}_*.png")

        if run is not None:
            run.log(log_payload)

        payload = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "val_soft_dice": val_metrics["soft_dice"],
            "val_hard_dice": val_metrics["hard_dice"],
            "val_kl": val_metrics["kl"],
            "beta": beta,
            "config": vars(args),
            "pos_weight": pos_weight,
        }
        if val_metrics["soft_dice"] > best_soft:
            best_soft = val_metrics["soft_dice"]
            torch.save(payload, out_dir / "best.pt")
            if run is not None:
                run.summary["best_epoch"] = epoch

        torch.save(payload, out_dir / "last.pt")

    print(f"best val soft Dice = {best_soft:.4f}")
    print(f"checkpoints  -> {out_dir}")
    if run is not None:
        run.summary["best_val_soft_dice"] = best_soft
        # Final media table so the run page always has a glanceable gallery.
        if args.wandb_image_every > 0 and wandb is not None:
            latest = sorted(panel_dir.glob("epoch*_reconstructions.png"))
            if latest:
                run.log({
                    "panels/final_reconstructions": wandb.Image(str(latest[-1])),
                })
            latest_s = sorted(panel_dir.glob("epoch*_samples.png"))
            if latest_s:
                run.log({
                    "panels/final_samples": wandb.Image(str(latest_s[-1])),
                })
        run.finish()


if __name__ == "__main__":
    main()
