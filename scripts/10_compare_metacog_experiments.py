#!/usr/bin/env python3
"""Compare multiple MetaCOG variants on exactly the same patients.

The script discovers `<root>/<variant>/<split>/summary.csv`, verifies that all
variants use the same patients and frozen U-Net scores, and writes a compact
cross-experiment CSV/JSON plus a poster-ready comparison figure.

Usage:
  .venv/bin/python scripts/10_compare_metacog_experiments.py \
      --root derivatives/metacog_runs/v1.1-20260715-123154 --split test
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 20_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for start in range(0, n_boot, 1000):
        n = min(1000, n_boot - start)
        idx = rng.integers(0, len(values), size=(n, len(values)))
        means[start:start + n] = values[idx].mean(axis=1)
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def discover(root: Path, split: str) -> dict[str, Path]:
    found = {}
    for path in root.glob(f"*/{split}/summary.csv"):
        found[path.parent.parent.name] = path.parent
    return dict(sorted(found.items()))


def rate_recovery(run_dir: Path, rate: str) -> tuple[float, float, float, int]:
    rows = read_csv(run_dir / "rate_metrics.csv")
    support_key = "n_background" if rate == "H" else "n_bone"
    by_patient: dict[str, list[tuple[float, float, int]]] = {}
    for row in rows:
        support = int(row[support_key])
        if support <= 0:
            continue
        x, y = float(row[f"{rate}_inferred"]), float(row[f"{rate}_empirical"])
        if np.isfinite(x) and np.isfinite(y):
            by_patient.setdefault(row["patient"], []).append((x, y, support))
    inferred, empirical = [], []
    for patient_rows in by_patient.values():
        weights = np.array([r[2] for r in patient_rows], dtype=np.float64)
        inferred.append(float(np.average([r[0] for r in patient_rows], weights=weights)))
        empirical.append(float(np.average([r[1] for r in patient_rows], weights=weights)))
    x, y = np.asarray(inferred), np.asarray(empirical)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        pearson = spearman = float("nan")
    else:
        pearson = float(np.corrcoef(x, y)[0, 1])
        spearman = float(spearmanr(x, y).statistic)
    mae = float(np.mean(np.abs(x - y))) if len(x) else float("nan")
    return pearson, spearman, mae, len(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    runs = discover(args.root, args.split)
    if args.variants:
        missing = set(args.variants) - set(runs)
        if missing:
            raise SystemExit(f"Missing {args.split} runs: {sorted(missing)}")
        runs = {v: runs[v] for v in args.variants}
    if len(runs) < 2:
        raise SystemExit(f"Need at least two completed runs under {args.root}")

    rows_by_variant: dict[str, dict[str, dict]] = {}
    patient_set = None
    frozen_scores = None
    for variant, run_dir in runs.items():
        rows = {r["patient"]: r for r in read_csv(run_dir / "summary.csv")}
        current = set(rows)
        if patient_set is None:
            patient_set = current
            frozen_scores = {pid: float(rows[pid]["dice_unet"]) for pid in rows}
        elif current != patient_set:
            raise SystemExit(
                f"{variant} patient set differs: missing={sorted(patient_set-current)}, "
                f"extra={sorted(current-patient_set)}"
            )
        for pid in rows:
            if not np.isclose(float(rows[pid]["dice_unet"]), frozen_scores[pid], atol=1e-10):
                raise SystemExit(f"{variant}/{pid}: frozen U-Net Dice changed")
        rows_by_variant[variant] = rows

    pids = sorted(patient_set)
    comparison: list[dict] = []
    deltas: dict[str, np.ndarray] = {}
    for variant, rows in rows_by_variant.items():
        raw = np.array([float(rows[p]["dice_unet"]) for p in pids])
        corrected = np.array([float(rows[p]["dice_corrected"]) for p in pids])
        delta = corrected - raw
        deltas[variant] = delta
        ci = bootstrap_ci(delta, args.seed)
        surface_delta = np.array([
            float(rows[p]["surface_dice_corrected"]) - float(rows[p]["surface_dice_unet"])
            for p in pids
        ])
        surface_ci = bootstrap_ci(surface_delta, args.seed)
        h_pearson, h_spearman, h_mae, h_n = rate_recovery(runs[variant], "H")
        m_pearson, m_spearman, m_mae, m_n = rate_recovery(runs[variant], "M")
        comparison.append({
            "variant": variant,
            "split": args.split,
            "n_patients": len(pids),
            "dice_unet_mean": float(raw.mean()),
            "dice_corrected_mean": float(corrected.mean()),
            "dice_delta_mean": float(delta.mean()),
            "dice_delta_ci_lo": ci[0],
            "dice_delta_ci_hi": ci[1],
            "patients_improved": int((delta > 0).sum()),
            "patients_worsened": int((delta < 0).sum()),
            "surface_dice_delta_mean": float(surface_delta.mean()),
            "surface_delta_ci_lo": surface_ci[0],
            "surface_delta_ci_hi": surface_ci[1],
            "hd95_delta_mean_mm": float(np.nanmean([
                float(rows[p]["hd95_delta_mm"]) for p in pids
            ])),
            "brier_corrected_mean": float(np.mean([
                float(rows[p]["brier_corrected"]) for p in pids
            ])),
            "frac_changed_mean": float(np.mean([
                float(rows[p]["frac_changed"]) for p in pids
            ])),
            "H_rate_pearson": h_pearson,
            "H_rate_spearman": h_spearman,
            "H_rate_mae": h_mae,
            "H_rate_n_patients": h_n,
            "M_rate_pearson": m_pearson,
            "M_rate_spearman": m_spearman,
            "M_rate_mae": m_mae,
            "M_rate_n_patients": m_n,
        })

    out_dir = args.out_dir or (args.root / "comparison" / args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    (out_dir / "comparison.json").write_text(json.dumps({
        "root": str(args.root),
        "split": args.split,
        "variants": list(runs),
        "patients": pids,
        "results": comparison,
    }, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(runs)
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 1.3 * len(names)), 8))
    rng = np.random.default_rng(args.seed)
    for x, name in enumerate(names):
        jitter = rng.normal(0, 0.055, len(pids))
        axes[0].scatter(np.full(len(pids), x) + jitter, deltas[name],
                        s=18, alpha=0.55)
        axes[0].plot([x - 0.22, x + 0.22], [deltas[name].mean()] * 2,
                     color="black", lw=2)
    axes[0].axhline(0, color="black", ls="--", lw=0.8)
    axes[0].set_xticks(range(len(names)), names, rotation=20, ha="right")
    axes[0].set_ylabel("Patient Dice change")
    axes[0].set_title("Each point is one held-out patient; black line is the mean")

    means = np.array([r["dice_delta_mean"] for r in comparison])
    lo = means - np.array([r["dice_delta_ci_lo"] for r in comparison])
    hi = np.array([r["dice_delta_ci_hi"] for r in comparison]) - means
    axes[1].bar(range(len(names)), means,
                color=np.where(means >= 0, "tab:green", "tab:red"), alpha=0.8)
    axes[1].errorbar(range(len(names)), means, yerr=[lo, hi], fmt="none",
                     color="black", capsize=4)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(range(len(names)), names, rotation=20, ha="right")
    axes[1].set_ylabel("Mean Dice change (95% patient bootstrap CI)")
    axes[1].set_title(f"MetaCOG variants on the frozen {args.split} split")
    fig.tight_layout()
    fig.savefig(out_dir / "experiment_comparison.png", dpi=160)
    plt.close(fig)

    print(f"compared {len(names)} variants on {len(pids)} patients")
    for row in comparison:
        print(
            f"  {row['variant']:16s} Dice delta={row['dice_delta_mean']:+.4f} "
            f"[{row['dice_delta_ci_lo']:+.4f},{row['dice_delta_ci_hi']:+.4f}]  "
            f"improved={row['patients_improved']}/{len(pids)}"
        )
    print(f"comparison -> {out_dir}")


if __name__ == "__main__":
    main()
