#!/usr/bin/env python3
"""
09_visualize_metacog.py
-------------------------
Visualizations for a `08_run_metacog_inference.py` run:

  1. rate_forest.png       per-patient posterior mean + 95% CI for H and M
                            (two forest plots), sorted by patient ID.
  2. <pid>_posterior.png    posterior histogram + per-chain trace plot for H
                            and M, for a handful of patients (MCMC diagnostics).
  3. <pid>_masks.png        per-slice panel: MR, atlas prior, U-Net mask,
                            corrected posterior probability, corrected binary
                            mask, error map vs ground truth (same style as
                            05_visualize_unet.py, extended with the atlas
                            prior and the corrected outputs).
  4. dice_comparison.png    U-Net vs. Bayes-corrected Dice, paired per patient
                            (scatter against y=x, and a sorted delta bar chart).

Usage:
  .venv/bin/python scripts/09_visualize_metacog.py \\
      --run-dir derivatives/metacog_runs/v1.1-20260715-123154

  # Only a couple of patients' detailed panels
  .venv/bin/python scripts/09_visualize_metacog.py \\
      --run-dir derivatives/metacog_runs/v1.1-20260715-123154 \\
      --patients 1BA014 --n-slices-per-patient 5
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEF_QC_ROOT = ROOT / "logs" / "metacog_qc"


def load_summary(run_dir: Path) -> tuple[list[dict], dict]:
    rows = []
    with open(run_dir / "summary.csv", newline="") as f:
        for row in csv.DictReader(f):
            for k in ("n_slices",):
                row[k] = int(row[k])
            for k in ("H_mean", "H_ci_lo", "H_ci_hi", "H_rhat", "H_ess",
                     "M_mean", "M_ci_lo", "M_ci_hi", "M_rhat", "M_ess",
                     "dice_unet", "dice_corrected", "dice_delta"):
                row[k] = float(row[k])
            rows.append(row)
    config = json.loads((run_dir / "summary.json").read_text())
    return rows, config


def plot_rate_forest(rows: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["patient"])
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10, max(3, 0.35 * len(rows))))

    for ax, key, title, color in [
        (axes[0], "H", "False-positive rate H", "tab:red"),
        (axes[1], "M", "False-negative rate M", "tab:blue"),
    ]:
        means = [r[f"{key}_mean"] for r in rows]
        los = [r[f"{key}_mean"] - r[f"{key}_ci_lo"] for r in rows]
        his = [r[f"{key}_ci_hi"] - r[f"{key}_mean"] for r in rows]
        ax.errorbar(means, y, xerr=[los, his], fmt="o", color=color, ecolor=color,
                   elinewidth=1, capsize=2, markersize=4)
        ax.set_yticks(y)
        ax.set_yticklabels([r["patient"] for r in rows], fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("posterior mean (95% CI)")
        ax.axvline(0, color="gray", lw=0.5, ls="--")
        ax.invert_yaxis()

    fig.suptitle("Per-patient global error-rate posteriors")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_posterior_diagnostics(pid: str, npz_path: Path, num_chains: int,
                               num_samples: int, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from models.metacog import gelman_rubin_rhat, effective_sample_size
    import torch

    pack = np.load(npz_path)
    H = torch.from_numpy(pack["H_samples"]).reshape(num_chains, num_samples)
    M = torch.from_numpy(pack["M_samples"]).reshape(num_chains, num_samples)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for row, (chains, name, color) in enumerate([(H, "H", "tab:red"), (M, "M", "tab:blue")]):
        flat = chains.reshape(-1).numpy()
        rhat = gelman_rubin_rhat(chains)
        ess = effective_sample_size(chains)
        lo, hi = np.percentile(flat, [2.5, 97.5])

        ax = axes[row, 0]
        ax.hist(flat, bins=40, color=color, alpha=0.7, density=True)
        ax.axvline(flat.mean(), color="black", lw=1.5, label=f"mean={flat.mean():.4f}")
        ax.axvspan(lo, hi, color=color, alpha=0.15, label=f"95% CI [{lo:.4f},{hi:.4f}]")
        ax.set_title(f"{name} posterior  (R-hat={rhat:.3f}, ESS={ess:.0f})")
        ax.legend(fontsize=7)

        ax = axes[row, 1]
        for c in range(chains.shape[0]):
            ax.plot(chains[c].numpy(), lw=0.6, alpha=0.8, label=f"chain {c}")
        ax.set_title(f"{name} trace ({chains.shape[0]} chains)")
        ax.set_xlabel("sample")
        ax.legend(fontsize=7)

    fig.suptitle(f"{pid}: MCMC posterior diagnostics")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def pick_slice_indices(n_slices: int, k: int) -> list[int]:
    k = min(k, n_slices)
    return sorted(set(int(i) for i in np.linspace(0, n_slices - 1, k)))


def plot_mask_panels(pid: str, npz_path: Path, idxs: list[int], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pack = np.load(npz_path)
    mr, bone, unet_mask = pack["mr"], pack["bone"], pack["unet_mask"]
    prior_p, post_prob, post_mask = pack["prior_p"], pack["post_prob"], pack["post_mask"]
    s_norm = pack["s_norm"]

    ncols = 5
    nrows = len(idxs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows))
    axes = np.atleast_2d(axes)
    titles = ["MR + ground truth", "atlas prior P(bone)",
             "U-Net mask", "corrected posterior P(bone)", "corrected vs GT error"]

    def dice1(pred, gt, eps=1e-6):
        inter = float((pred * gt).sum())
        denom = float(pred.sum() + gt.sum())
        return (2 * inter + eps) / (denom + eps)

    for r, i in enumerate(idxs):
        m, b, um = mr[i], bone[i], unet_mask[i]
        p, pp, pm = prior_p[i], post_prob[i], post_mask[i]
        d_unet, d_corr = dice1(um, b), dice1(pm, b)

        ax = axes[r, 0]
        ax.imshow(m.T, cmap="gray", origin="lower")
        ax.imshow(np.ma.masked_where(b.T == 0, b.T), cmap="autumn", origin="lower", alpha=0.5)
        ax.set_ylabel(f"slice {i}\ns_norm={s_norm[i]:.2f}", fontsize=9)

        ax = axes[r, 1]
        ax.imshow(p.T, origin="lower", cmap="viridis", vmin=0, vmax=1)

        ax = axes[r, 2]
        ax.imshow(m.T, cmap="gray", origin="lower")
        ax.imshow(np.ma.masked_where(um.T == 0, um.T), cmap="winter", origin="lower", alpha=0.6)
        ax.set_title(f"Dice={d_unet:.3f}", fontsize=9)

        ax = axes[r, 3]
        ax.imshow(pp.T, origin="lower", cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"Dice={d_corr:.3f}", fontsize=9)

        fp = ((pm == 1) & (b == 0)).astype(np.float32)
        fn = ((pm == 0) & (b == 1)).astype(np.float32)
        err = np.zeros((*m.shape, 3), dtype=np.float32)
        err[..., 0] = fp.T
        err[..., 2] = fn.T
        ax = axes[r, 4]
        ax.imshow(m.T, cmap="gray", origin="lower")
        ax.imshow(np.clip(err, 0, 1), origin="lower", alpha=0.7)

        for c in range(ncols):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(titles[c] if c not in (2, 3) else axes[r, c].get_title() + f"\n{titles[c]}",
                                     fontsize=10)

    fig.suptitle(f"{pid}: atlas-prior MetaCOG correction (global H, M)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_dice_comparison(rows: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d_unet = np.array([r["dice_unet"] for r in rows])
    d_corr = np.array([r["dice_corrected"] for r in rows])
    pids = [r["patient"] for r in rows]
    order = np.argsort(d_corr - d_unet)

    fig, axes = plt.subplots(1, 2, figsize=(11, max(4, 0.3 * len(rows))))

    ax = axes[0]
    lo = min(d_unet.min(), d_corr.min()) - 0.01
    hi = max(d_unet.max(), d_corr.max()) + 0.01
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x (no change)")
    ax.scatter(d_unet, d_corr, s=25, alpha=0.8)
    ax.set_xlabel("U-Net Dice")
    ax.set_ylabel("Bayes-corrected Dice")
    ax.set_title("Per-patient Dice: corrected vs. raw U-Net")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    ax = axes[1]
    delta = (d_corr - d_unet)[order]
    ax.barh(np.arange(len(rows)), delta,
           color=["tab:green" if v > 0 else "tab:red" for v in delta])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([pids[i] for i in order], fontsize=6)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Dice(corrected) - Dice(U-Net)")
    ax.set_title("Per-patient Dice delta")

    fig.suptitle(f"n={len(rows)}  mean Dice: U-Net={d_unet.mean():.4f} -> "
                f"corrected={d_corr.mean():.4f}  (delta={d_corr.mean()-d_unet.mean():+.4f})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--patients", nargs="+", default=None,
                    help="patients to make detailed panels for (default: up to --n-panel-patients)")
    ap.add_argument("--n-panel-patients", type=int, default=3)
    ap.add_argument("--n-slices-per-patient", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: logs/metacog_qc/<run-name>")
    args = ap.parse_args()

    rows, config = load_summary(args.run_dir)
    out_dir = args.out_dir or (DEF_QC_ROOT / args.run_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_rate_forest(rows, out_dir / "rate_forest.png")
    plot_dice_comparison(rows, out_dir / "dice_comparison.png")

    if args.patients:
        panel_pids = args.patients
    else:
        rng = np.random.default_rng(args.seed)
        all_pids = [r["patient"] for r in rows]
        panel_pids = sorted(rng.choice(all_pids, size=min(args.n_panel_patients, len(all_pids)),
                                       replace=False).tolist())

    for pid in panel_pids:
        npz_path = args.run_dir / f"{pid}.npz"
        plot_posterior_diagnostics(pid, npz_path, config["num_chains"], config["num_samples"],
                                   out_dir / f"{pid}_posterior.png")
        pack = np.load(npz_path)
        idxs = pick_slice_indices(pack["mr"].shape[0], args.n_slices_per_patient)
        plot_mask_panels(pid, npz_path, idxs, out_dir / f"{pid}_masks.png")

    print(f"figures -> {out_dir}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
