#!/usr/bin/env python3
"""Create focused visual diagnostics for any grid-based MetaCOG run.

Outputs vary by experiment but always include:
  paired_metrics.png       patient Dice and surface-Dice before/after
  rate_recovery.png        inferred versus empirical H/M
  dice_by_height.png       U-Net and correction performance by s_norm
  <pid>_masks.png          MR/target/prior/U-Net/posterior/changed pixels

Global runs additionally receive posterior H/M heatmaps. Patch runs receive
8x8 rate maps. Height-stratified runs receive rate-versus-height curves.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEF_QC_ROOT = ROOT / "logs" / "metacog_qc"


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def plot_paired_metrics(rows: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, raw_key, corrected_key, title in [
        (axes[0], "dice_unet", "dice_corrected", "Patient-volume Dice"),
        (axes[1], "surface_dice_unet", "surface_dice_corrected", "Surface Dice at 3 mm"),
    ]:
        raw = np.array([f(r, raw_key) for r in rows])
        corrected = np.array([f(r, corrected_key) for r in rows])
        lo = float(np.nanmin([raw.min(), corrected.min()])) - 0.01
        hi = float(np.nanmax([raw.max(), corrected.max()])) + 0.01
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="no change")
        ax.scatter(raw, corrected, s=28, alpha=0.8)
        ax.set(xlabel="Frozen U-Net", ylabel="MetaCOG-corrected",
               title=f"{title}\nmean change = {np.mean(corrected-raw):+.4f}",
               xlim=(lo, hi), ylim=(lo, hi))
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_rate_recovery(rows: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, rate, support_key, color in [
        (axes[0], "H", "n_background", "tab:red"),
        (axes[1], "M", "n_bone", "tab:blue"),
    ]:
        grouped: dict[str, list[tuple[float, float, float]]] = {}
        for row in rows:
            support = f(row, support_key)
            inferred = f(row, f"{rate}_inferred")
            empirical = f(row, f"{rate}_empirical")
            if support > 0 and np.isfinite(inferred) and np.isfinite(empirical):
                grouped.setdefault(row["patient"], []).append(
                    (inferred, empirical, support)
                )
        inferred, empirical = [], []
        for patient_rows in grouped.values():
            weights = np.array([r[2] for r in patient_rows])
            inferred.append(np.average([r[0] for r in patient_rows], weights=weights))
            empirical.append(np.average([r[1] for r in patient_rows], weights=weights))
        inferred, empirical = np.asarray(inferred), np.asarray(empirical)
        valid = np.isfinite(inferred) & np.isfinite(empirical)
        if valid.any():
            lo = min(float(inferred[valid].min()), float(empirical[valid].min()), 0.0)
            hi = max(float(inferred[valid].max()), float(empirical[valid].max()), 0.01)
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.scatter(empirical[valid], inferred[valid], s=24, alpha=0.7, color=color)
            mae = float(np.mean(np.abs(inferred[valid] - empirical[valid])))
            r = correlation(empirical[valid], inferred[valid])
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_title(f"{rate}: n={valid.sum()}, r={r:.3f}, MAE={mae:.3f}")
        else:
            ax.text(0.5, 0.5, "No sufficiently supported regions",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(f"Empirical {rate} from CT-derived target")
        ax.set_ylabel(f"Inferred {rate}")
    fig.suptitle("Do patient-level inferred rates track observed U-Net errors?")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_dice_by_height(rows: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    s = np.array([f(r, "s_norm") for r in rows])
    raw = np.array([f(r, "dice_unet") for r in rows])
    corrected = np.array([f(r, "dice_corrected") for r in rows])
    edges = np.linspace(0, 1, 14)
    centers = 0.5 * (edges[:-1] + edges[1:])
    raw_mean, corrected_mean, counts = [], [], []
    for i in range(13):
        sel = (s >= edges[i]) & (s < edges[i + 1] if i < 12 else s <= edges[i + 1])
        counts.append(int(sel.sum()))
        raw_mean.append(float(np.mean(raw[sel])) if sel.any() else np.nan)
        corrected_mean.append(float(np.mean(corrected[sel])) if sel.any() else np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(centers, raw_mean, "o-", label="U-Net")
    axes[0].plot(centers, corrected_mean, "o-", label="corrected")
    axes[0].set_ylabel("Mean slice Dice")
    axes[0].legend()
    delta = np.array(corrected_mean) - np.array(raw_mean)
    axes[1].bar(centers, delta, width=0.065,
                color=np.where(delta >= 0, "tab:green", "tab:red"))
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(xlabel="s_norm (0 = vault base, 1 = MR-derived vertex)",
                ylabel="Corrected − U-Net Dice")
    for x, y, n in zip(centers, delta, counts):
        if np.isfinite(y):
            axes[1].text(x, y, str(n), ha="center",
                         va="bottom" if y >= 0 else "top", fontsize=6)
    fig.suptitle("Performance by physical height (numbers are retained slices)")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_global_posterior(pid: str, pack, out: Path) -> None:
    import matplotlib.pyplot as plt

    if "grid_weights" not in pack.files:
        return
    h, m, weights = pack["grid_h"], pack["grid_m"], pack["grid_weights"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.pcolormesh(h, m, weights, shading="auto", cmap="viridis")
    ax.plot(float(pack["H_mean"].reshape(-1)[0]),
            float(pack["M_mean"].reshape(-1)[0]), "r+", ms=10, mew=2)
    ax.set(xlabel="H (false-positive rate)", ylabel="M (false-negative rate)",
           title=f"{pid}: deterministic H/M posterior")
    fig.colorbar(image, ax=ax, label="posterior grid mass")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_patch_rates(pid: str, pack, out: Path) -> None:
    import matplotlib.pyplot as plt

    h, m = pack["H_mean"], pack["M_mean"]
    if h.ndim != 2:
        return
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for ax, values, title in [
        (axes[0, 0], h, "Inferred H"),
        (axes[0, 1], pack["H_empirical"], "Empirical H"),
        (axes[1, 0], m, "Inferred M"),
        (axes[1, 1], pack["M_empirical"], "Empirical M"),
    ]:
        image = ax.imshow(values.T, origin="lower",
                          vmin=0, vmax=max(0.01, np.nanmax(values)),
                          cmap="magma")
        ax.set_title(title)
        ax.set(xlabel="patch x", ylabel="patch y")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(f"{pid}: 8×8-pixel patch error rates")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_height_rates(rows: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    regions = np.array([int(r["region"]) for r in rows])
    centers = (np.arange(13) + 0.5) / 13
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
    for ax, rate, support in [
        (axes[0], "H", "n_background"),
        (axes[1], "M", "n_bone"),
    ]:
        inferred, empirical = [], []
        for region in range(13):
            sel = (regions == region) & (
                np.array([f(r, support) for r in rows]) >= 100
            )
            inferred.append(
                np.mean([f(r, f"{rate}_inferred") for r, ok in zip(rows, sel) if ok])
                if sel.any() else np.nan
            )
            empirical.append(
                np.mean([f(r, f"{rate}_empirical") for r, ok in zip(rows, sel) if ok])
                if sel.any() else np.nan
            )
        ax.plot(centers, inferred, "o-", label="inferred")
        ax.plot(centers, empirical, "o-", label="empirical")
        ax.set(xlabel="s_norm bin centre", ylabel=rate,
               title=f"{rate} by physical height")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def pick_slices(pack, n: int) -> list[int]:
    changes = (pack["post_mask"] != pack["unet_mask"]).sum(axis=(1, 2))
    if changes.max() > 0:
        return sorted(np.argsort(changes)[-min(n, len(changes)):].tolist())
    return sorted(set(np.linspace(0, len(changes) - 1, min(n, len(changes))).astype(int)))


def plot_mask_panels(pid: str, pack, indices: list[int], out: Path) -> None:
    import matplotlib.pyplot as plt

    nrows, ncols = len(indices), 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 2.7 * nrows))
    axes = np.atleast_2d(axes)
    titles = ["MR + target", "Atlas prior", "U-Net errors",
              "Corrected probability", "Changed pixels", "Corrected errors"]
    for row, s in enumerate(indices):
        mr, gt = pack["mr"][s], pack["bone"][s]
        unet, corrected = pack["unet_mask"][s], pack["post_mask"][s]

        axes[row, 0].imshow(mr.T, cmap="gray", origin="lower")
        axes[row, 0].imshow(np.ma.masked_where(gt.T == 0, gt.T),
                            cmap="autumn", alpha=0.55, origin="lower")
        axes[row, 1].imshow(pack["prior_p"][s].T, vmin=0, vmax=1,
                            cmap="viridis", origin="lower")

        for ax, pred in [(axes[row, 2], unet), (axes[row, 5], corrected)]:
            err = np.zeros((*gt.shape, 3), dtype=np.float32)
            err[..., 0] = ((pred == 1) & (gt == 0)).T
            err[..., 2] = ((pred == 0) & (gt == 1)).T
            ax.imshow(mr.T, cmap="gray", origin="lower")
            ax.imshow(err, alpha=0.75, origin="lower")

        axes[row, 3].imshow(pack["post_prob"][s].T, vmin=0, vmax=1,
                            cmap="viridis", origin="lower")
        change = np.zeros((*gt.shape, 3), dtype=np.float32)
        change[..., 1] = ((corrected == 1) & (unet == 0)).T
        change[..., 0] = ((corrected == 0) & (unet == 1)).T
        axes[row, 4].imshow(mr.T, cmap="gray", origin="lower")
        axes[row, 4].imshow(change, alpha=0.8, origin="lower")

        axes[row, 0].set_ylabel(
            f"slice {s}\ns={float(pack['s_norm'][s]):.3f}\n"
            f"d={float(pack['d_mm'][s]):.0f} mm", fontsize=8
        )
        for col in range(ncols):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=9)
    fig.suptitle(
        f"{pid}: red=false positive/removal, blue=false negative, green=addition"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--patients", nargs="+", default=None)
    ap.add_argument("--n-panel-patients", type=int, default=3)
    ap.add_argument("--n-slices-per-patient", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    config = json.loads((args.run_dir / "summary.json").read_text())
    summary_rows = read_csv(args.run_dir / "summary.csv")
    rate_rows = read_csv(args.run_dir / "rate_metrics.csv")
    slice_rows = read_csv(args.run_dir / "slice_metrics.csv")
    out_dir = args.out_dir or (
        DEF_QC_ROOT / args.run_dir.parent.parent.name / config["variant"] / config["split"]
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")

    plot_paired_metrics(summary_rows, out_dir / "paired_metrics.png")
    plot_rate_recovery(rate_rows, out_dir / "rate_recovery.png")
    plot_dice_by_height(slice_rows, out_dir / "dice_by_height.png")
    if config["locality"] == "snorm":
        plot_height_rates(rate_rows, out_dir / "rates_by_height.png")

    all_pids = [r["patient"] for r in summary_rows]
    if args.patients:
        pids = args.patients
    else:
        # Include lowest, median, and highest Dice changes rather than random cases.
        ordered = sorted(summary_rows, key=lambda r: f(r, "dice_delta"))
        positions = np.linspace(0, len(ordered) - 1,
                                min(args.n_panel_patients, len(ordered))).astype(int)
        pids = [ordered[i]["patient"] for i in positions]

    for pid in pids:
        pack = np.load(args.run_dir / f"{pid}.npz")
        plot_mask_panels(
            pid, pack, pick_slices(pack, args.n_slices_per_patient),
            out_dir / f"{pid}_masks.png",
        )
        if config["locality"] == "global" and config["lesion"] != "rates":
            plot_global_posterior(pid, pack, out_dir / f"{pid}_posterior.png")
        if config["locality"] == "patch":
            plot_patch_rates(pid, pack, out_dir / f"{pid}_patch_rates.png")

    print(f"figures -> {out_dir}")


if __name__ == "__main__":
    main()
