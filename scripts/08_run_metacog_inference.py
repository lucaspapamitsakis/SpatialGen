#!/usr/bin/env python3
"""Run deterministic MetaCOG inference and controlled lesion experiments.

All variants share the same atlas sensor model and adaptive H/M grid engine:

  global       one H/M pair per patient
  patch        one pair per 8x8-pixel spatial block (64 pairs for 64x64 images)
  snorm        one pair per atlas s_norm bin

Two one-factor lesions are supported:

  rates        keep the atlas, but freeze H/M to cohort-level values
  prior        keep each patient's rates from the full global model, but replace
               the atlas by the scalar training-set bone prevalence

Ground truth is loaded only after inference to compute evaluation metrics. It is
never passed to the rate integration or mask-correction functions.

Examples
--------
Validation baselines:
  .venv/bin/python scripts/08_run_metacog_inference.py \
      --predictions-dir derivatives/unet_predictions/v1.1-20260715-123154 \
      --split val --locality global
  .venv/bin/python scripts/08_run_metacog_inference.py \
      --predictions-dir derivatives/unet_predictions/v1.1-20260715-123154 \
      --split val --locality patch --patch-size 8
  .venv/bin/python scripts/08_run_metacog_inference.py \
      --predictions-dir derivatives/unet_predictions/v1.1-20260715-123154 \
      --split val --locality snorm

Lesions on test:
  .venv/bin/python scripts/08_run_metacog_inference.py \
      --predictions-dir derivatives/unet_predictions/v1.1-20260715-123154 \
      --split test --lesion rates \
      --fixed-rates-from derivatives/metacog_runs/v1.1-20260715-123154/global/val/summary.csv
  .venv/bin/python scripts/08_run_metacog_inference.py \
      --predictions-dir derivatives/unet_predictions/v1.1-20260715-123154 \
      --split test --lesion prior

C-VAE soft prior (from scripts/12_sample_cvae_prior.py):
  .venv/bin/python scripts/08_run_metacog_inference.py \
      --predictions-dir derivatives/unet_predictions/v1.1-20260715-123154 \
      --split val --locality global \
      --prior-dir derivatives/cvae_priors/<run>/val
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.metacog import (  # noqa: E402
    GridPosterior,
    atlas_bin_indices,
    atlas_prior_for_slices,
    integrate_rates,
    load_atlas,
    posterior_correct_fixed,
    posterior_correct_grid,
)

ROOT = Path(__file__).resolve().parents[1]
DEF_DATA = ROOT / "derivatives" / "dataset_2d_filtered"
DEF_ATLAS = ROOT / "derivatives" / "bone_atlas.npz"
DEF_OUT_ROOT = ROOT / "derivatives" / "metacog_runs"
INPLANE_MM = 180.0 / 64.0
SPACING_S_Y_X = (1.0, INPLANE_MM, INPLANE_MM)


def dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    inter = float((pred * gt).sum())
    return (2.0 * inter + eps) / (float(pred.sum() + gt.sum()) + eps)


def dice_per_slice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    inter = (pred * gt).sum(axis=(1, 2), dtype=np.float64)
    denom = pred.sum(axis=(1, 2), dtype=np.float64) + gt.sum(axis=(1, 2), dtype=np.float64)
    return (2.0 * inter + eps) / (denom + eps)


def empirical_rates(obs: np.ndarray, gt: np.ndarray) -> tuple[float, float, int, int]:
    """Ground-truth scoring only; never call this before inference is complete."""
    neg = int((gt == 0).sum())
    pos = int((gt == 1).sum())
    h = float(((obs == 1) & (gt == 0)).sum() / neg) if neg else float("nan")
    m = float(((obs == 0) & (gt == 1)).sum() / pos) if pos else float("nan")
    return h, m, neg, pos


def largest_component_fraction(mask: np.ndarray) -> float:
    total = int(mask.sum())
    if total == 0:
        return 1.0
    structure = ndimage.generate_binary_structure(mask.ndim, 2)
    labels, n = ndimage.label(mask.astype(bool), structure=structure)
    sizes = np.bincount(labels.reshape(-1))[1:]
    return float(sizes.max() / total) if n else 0.0


def surface_metrics(pred: np.ndarray, gt: np.ndarray,
                    tolerance_mm: float = 3.0) -> tuple[float, float]:
    """Symmetric surface Dice and HD95 in physical millimetres."""
    structure = ndimage.generate_binary_structure(3, 1)
    ps = pred.astype(bool) & ~ndimage.binary_erosion(pred.astype(bool), structure)
    gs = gt.astype(bool) & ~ndimage.binary_erosion(gt.astype(bool), structure)
    if not ps.any() and not gs.any():
        return 1.0, 0.0
    if not ps.any() or not gs.any():
        return 0.0, float("nan")
    to_gt = ndimage.distance_transform_edt(~gs, sampling=SPACING_S_Y_X)[ps]
    to_pred = ndimage.distance_transform_edt(~ps, sampling=SPACING_S_Y_X)[gs]
    surface_dice = float(
        ((to_gt <= tolerance_mm).sum() + (to_pred <= tolerance_mm).sum())
        / (to_gt.size + to_pred.size)
    )
    hd95 = float(np.percentile(np.concatenate([to_gt, to_pred]), 95))
    return surface_dice, hd95


def bootstrap_mean_ci(values: np.ndarray, seed: int = 0,
                      n_boot: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, 1000):
        n = min(1000, n_boot - start)
        idx = rng.integers(0, values.size, size=(n, values.size))
        means[start:start + n] = values[idx].mean(axis=1)
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def select_patients(args) -> dict[str, str]:
    if args.patients:
        return {pid: "custom" for pid in args.patients}
    splits = json.loads((args.data_dir / "splits.json").read_text())
    pool = list(splits[args.split])
    if args.n_patients is not None and args.n_patients < len(pool):
        rng = np.random.default_rng(args.seed)
        pool = sorted(rng.choice(pool, size=args.n_patients, replace=False).tolist())
    return {pid: args.split for pid in pool}


def training_bone_prevalence(data_dir: Path) -> float:
    splits = json.loads((data_dir / "splits.json").read_text())
    positive = total = 0
    for pid in splits["train"]:
        bone = np.load(data_dir / f"{pid}.npz")["bone"]
        positive += int(bone.sum())
        total += int(bone.size)
    return positive / total


def fixed_rates_from_summary(path: Path) -> tuple[float, float]:
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        raise ValueError(f"No patients in {path}")
    # Equal patient weighting. No ground-truth columns are read.
    h = float(np.mean([float(r["H_inferred_mean"]) for r in rows]))
    m = float(np.mean([float(r["M_inferred_mean"]) for r in rows]))
    return h, m


def patient_rates_from_summary(path: Path) -> dict[str, tuple[float, float]]:
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        raise ValueError(f"No patients in {path}")
    return {
        row["patient"]: (
            float(row["H_inferred_mean"]),
            float(row["M_inferred_mean"]),
        )
        for row in rows
    }


def variant_name(args) -> str:
    if args.lesion == "rates":
        return "lesion-rates"
    if args.lesion == "prior":
        return "lesion-prior"
    prior_tag = ""
    if getattr(args, "prior_dir", None) is not None:
        prior_tag = f"-cvae-{Path(args.prior_dir).parent.name}"
    if args.locality == "patch":
        return f"patch{args.patch_size}{prior_tag}"
    if args.locality == "snorm":
        return f"snorm13{prior_tag}"
    return f"global{prior_tag}"


def region_labels(shape: tuple[int, int, int], locality: str, *,
                  patch_size: int, slice_bins: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    s, h, w = shape
    if locality == "global":
        return np.zeros(shape, dtype=np.int16), (1,)
    if locality == "patch":
        if h % patch_size or w % patch_size:
            raise ValueError(f"Image shape {(h, w)} is not divisible by patch size {patch_size}")
        y, x = np.indices((h, w))
        n_x = w // patch_size
        labels_2d = (y // patch_size) * n_x + (x // patch_size)
        return np.broadcast_to(labels_2d, shape).astype(np.int16), (h // patch_size, n_x)
    if locality == "snorm":
        return np.broadcast_to(slice_bins[:, None, None], shape).astype(np.int16), (13,)
    raise ValueError(locality)


def empty_rate_arrays(rate_shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    return {
        "H_mean": np.full(rate_shape, np.nan, dtype=np.float32),
        "M_mean": np.full(rate_shape, np.nan, dtype=np.float32),
        "H_ci_lo": np.full(rate_shape, np.nan, dtype=np.float32),
        "H_ci_hi": np.full(rate_shape, np.nan, dtype=np.float32),
        "M_ci_lo": np.full(rate_shape, np.nan, dtype=np.float32),
        "M_ci_hi": np.full(rate_shape, np.nan, dtype=np.float32),
        "H_empirical": np.full(rate_shape, np.nan, dtype=np.float32),
        "M_empirical": np.full(rate_shape, np.nan, dtype=np.float32),
        "n_pixels": np.zeros(rate_shape, dtype=np.int64),
        "n_bone": np.zeros(rate_shape, dtype=np.int64),
        "n_background": np.zeros(rate_shape, dtype=np.int64),
        "convergence_delta": np.full(rate_shape, np.nan, dtype=np.float32),
        "edge_mass": np.full(rate_shape, np.nan, dtype=np.float32),
        "converged": np.zeros(rate_shape, dtype=np.uint8),
    }


def assign_rate_value(array: np.ndarray, region: int, value: float) -> None:
    array.reshape(-1)[region] = value


def infer_patient(
    prior_p: np.ndarray,
    obs: np.ndarray,
    labels: np.ndarray,
    rate_shape: tuple[int, ...],
    args,
    fixed_rates: tuple[float, float] | None,
) -> tuple[np.ndarray, dict[str, np.ndarray], GridPosterior | None]:
    """Infer rates and corrected probabilities without accepting ground truth."""
    post_prob = np.zeros_like(prior_p, dtype=np.float32)
    rates = empty_rate_arrays(rate_shape)
    global_grid: GridPosterior | None = None

    for region in sorted(int(x) for x in np.unique(labels)):
        sel = labels == region
        p_region = prior_p[sel]
        u_region = obs[sel]

        if fixed_rates is None:
            grid = integrate_rates(
                p_region,
                u_region,
                h_prior=tuple(args.h_prior),
                m_prior=tuple(args.m_prior),
                coarse_size=args.effective_coarse_grid,
                fine_size=args.effective_fine_grid,
                histogram_bins=args.histogram_bins,
                tolerance=args.grid_tolerance,
            )
            q_region = posterior_correct_grid(
                p_region, u_region, grid, probability_bins=args.histogram_bins
            )
            h_mean, m_mean = grid.h_mean, grid.m_mean
            h_ci, m_ci = grid.h_ci, grid.m_ci
            assign_rate_value(rates["convergence_delta"], region, grid.convergence_delta)
            assign_rate_value(rates["edge_mass"], region, grid.edge_mass)
            assign_rate_value(rates["converged"], region, int(grid.converged))
            if rate_shape == (1,):
                global_grid = grid
        else:
            h_mean, m_mean = fixed_rates
            h_ci, m_ci = (h_mean, h_mean), (m_mean, m_mean)
            q_region = posterior_correct_fixed(p_region, u_region, h_mean, m_mean)
            assign_rate_value(rates["convergence_delta"], region, 0.0)
            assign_rate_value(rates["edge_mass"], region, 0.0)
            assign_rate_value(rates["converged"], region, 1)

        post_prob[sel] = q_region
        assign_rate_value(rates["H_mean"], region, h_mean)
        assign_rate_value(rates["M_mean"], region, m_mean)
        assign_rate_value(rates["H_ci_lo"], region, h_ci[0])
        assign_rate_value(rates["H_ci_hi"], region, h_ci[1])
        assign_rate_value(rates["M_ci_lo"], region, m_ci[0])
        assign_rate_value(rates["M_ci_hi"], region, m_ci[1])
        assign_rate_value(rates["n_pixels"], region, int(sel.sum()))

    return post_prob, rates, global_grid


def score_rate_regions(
    obs: np.ndarray,
    gt: np.ndarray,
    labels: np.ndarray,
    rates: dict[str, np.ndarray],
) -> list[dict]:
    """Attach empirical error rates after inference has fully completed."""
    rate_rows: list[dict] = []
    for region in sorted(int(x) for x in np.unique(labels)):
        sel = labels == region
        h_true, m_true, n_bg, n_bone = empirical_rates(obs[sel], gt[sel])
        assign_rate_value(rates["H_empirical"], region, h_true)
        assign_rate_value(rates["M_empirical"], region, m_true)
        assign_rate_value(rates["n_background"], region, n_bg)
        assign_rate_value(rates["n_bone"], region, n_bone)
        rate_rows.append({
            "region": region,
            "H_inferred": float(rates["H_mean"].reshape(-1)[region]),
            "M_inferred": float(rates["M_mean"].reshape(-1)[region]),
            "H_ci_lo": float(rates["H_ci_lo"].reshape(-1)[region]),
            "H_ci_hi": float(rates["H_ci_hi"].reshape(-1)[region]),
            "M_ci_lo": float(rates["M_ci_lo"].reshape(-1)[region]),
            "M_ci_hi": float(rates["M_ci_hi"].reshape(-1)[region]),
            "H_empirical": h_true,
            "M_empirical": m_true,
            "n_pixels": int(sel.sum()),
            "n_background": n_bg,
            "n_bone": n_bone,
            "converged": int(rates["converged"].reshape(-1)[region]),
        })
    return rate_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions-dir", type=Path, required=True)
    ap.add_argument("--atlas", type=Path, default=DEF_ATLAS)
    ap.add_argument("--prior-dir", type=Path, default=None,
                    help="optional dir of <pid>.npz with key 'prior' (S,H,W); "
                         "overrides the empirical atlas soft maps (atlas bin edges "
                         "are still used for --locality snorm)")
    ap.add_argument("--data-dir", type=Path, default=DEF_DATA)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--patients", nargs="+", default=None)
    ap.add_argument("--n-patients", type=int, default=None)
    ap.add_argument("--locality", choices=["global", "patch", "snorm"], default="global")
    ap.add_argument("--patch-size", type=int, default=8)
    ap.add_argument("--lesion", choices=["none", "rates", "prior"], default="none")
    ap.add_argument("--fixed-rates", type=float, nargs=2, default=None, metavar=("H", "M"))
    ap.add_argument("--fixed-rates-from", type=Path, default=None)
    ap.add_argument("--patient-rates-from", type=Path, default=None,
                    help="global summary.csv supplying per-patient H/M for prior lesion")
    ap.add_argument("--h-prior", type=float, nargs=2, default=[1.0, 1.0])
    ap.add_argument("--m-prior", type=float, nargs=2, default=[1.0, 1.0])
    ap.add_argument("--coarse-grid", type=int, default=101)
    ap.add_argument("--fine-grid", type=int, default=161)
    ap.add_argument("--histogram-bins", type=int, default=2048)
    ap.add_argument("--grid-tolerance", type=float, default=5e-4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.lesion != "none" and args.locality != "global":
        raise SystemExit("Lesions are one-factor controls and require --locality global")
    if args.lesion != "rates" and (args.fixed_rates or args.fixed_rates_from):
        raise SystemExit("--fixed-rates* is only valid with --lesion rates")
    if args.lesion == "rates" and bool(args.fixed_rates) == bool(args.fixed_rates_from):
        raise SystemExit("Rate lesion requires exactly one of --fixed-rates or --fixed-rates-from")
    if args.lesion != "prior" and args.patient_rates_from:
        raise SystemExit("--patient-rates-from is only valid with --lesion prior")
    if args.prior_dir is not None and args.lesion == "prior":
        raise SystemExit("--prior-dir cannot be combined with --lesion prior "
                         "(that lesion forces a uniform prevalence prior)")

    args.effective_coarse_grid = args.coarse_grid
    args.effective_fine_grid = args.fine_grid

    fixed_rates = None
    fixed_source = None
    if args.lesion == "rates":
        if args.fixed_rates:
            fixed_rates = tuple(args.fixed_rates)
            fixed_source = "command_line"
        else:
            fixed_rates = fixed_rates_from_summary(args.fixed_rates_from)
            fixed_source = str(args.fixed_rates_from)
        if any(x < 0 or x > 1 for x in fixed_rates):
            raise SystemExit(f"Fixed rates must be in [0,1], got {fixed_rates}")

    atlas_dict = load_atlas(args.atlas)
    # Training targets are needed only to define the deliberately lesioned
    # spatially uniform prior. Ordinary inference requires only atlas + U-Net.
    prior_prevalence = (
        training_bone_prevalence(args.data_dir) if args.lesion == "prior" else None
    )
    pid_split = select_patients(args)
    pids = sorted(pid_split)
    if not pids:
        raise SystemExit("No patients selected")
    variant = variant_name(args)
    out_dir = args.out_dir or (
        DEF_OUT_ROOT / args.predictions_dir.name / variant / args.split
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    patient_fixed_rates = None
    patient_rates_source = None
    if args.lesion == "prior":
        patient_rates_source = args.patient_rates_from or (
            DEF_OUT_ROOT / args.predictions_dir.name / "global" / args.split / "summary.csv"
        )
        if not patient_rates_source.exists():
            raise SystemExit(
                f"Prior lesion needs the matching adaptive global run first: "
                f"{patient_rates_source}"
            )
        patient_fixed_rates = patient_rates_from_summary(patient_rates_source)
        missing = set(pids) - set(patient_fixed_rates)
        if missing:
            raise SystemExit(f"Patient rates missing for: {sorted(missing)}")

    print(f"variant={variant}  split={args.split}  patients={len(pids)}")
    print(f"H ~ Beta{tuple(args.h_prior)}  M ~ Beta{tuple(args.m_prior)}")
    if args.prior_dir is not None:
        print(f"soft prior maps from {args.prior_dir} (atlas bins still used for snorm)")
    if fixed_rates:
        print(f"fixed rates H={fixed_rates[0]:.6f}, M={fixed_rates[1]:.6f}")
    if args.lesion == "prior":
        print(f"uniform prior p={prior_prevalence:.6f} (training bone prevalence)")
        print(f"keeping patient rates from {patient_rates_source}")

    rows: list[dict] = []
    all_rate_rows: list[dict] = []
    all_slice_rows: list[dict] = []

    for index, pid in enumerate(pids, 1):
        t0 = time.time()
        pack = np.load(args.predictions_dir / f"{pid}.npz")
        mr = pack["mr"].astype(np.float32)
        bone = pack["bone"].astype(np.uint8)
        unet_prob = pack["prob"].astype(np.float32)
        unet_mask = pack["mask"].astype(np.uint8)
        s_norm = pack["s_norm"].astype(np.float32)
        z_index = pack["z_index"]
        d_mm = pack["d_mm"].astype(np.float32) if "d_mm" in pack.files else 64.0 * (1.0 - s_norm)

        slice_bins = atlas_bin_indices(atlas_dict, s_norm)
        if args.prior_dir is not None:
            prior_pack = np.load(args.prior_dir / f"{pid}.npz")
            if "prior" not in prior_pack.files:
                raise SystemExit(f"{pid}: {args.prior_dir / f'{pid}.npz'} missing 'prior'")
            prior_p = prior_pack["prior"].astype(np.float32)
            if prior_p.shape != unet_mask.shape:
                raise SystemExit(
                    f"{pid}: prior shape {prior_p.shape} != mask shape {unet_mask.shape}"
                )
        else:
            prior_p = atlas_prior_for_slices(atlas_dict, s_norm).astype(np.float32)
        if args.lesion == "prior":
            prior_p = np.full_like(prior_p, prior_prevalence)
        labels, rate_shape = region_labels(
            mr.shape, args.locality, patch_size=args.patch_size, slice_bins=slice_bins
        )

        rates_for_patient = (
            patient_fixed_rates[pid] if patient_fixed_rates is not None else fixed_rates
        )
        post_prob, rates, global_grid = infer_patient(
            prior_p, unet_mask, labels, rate_shape, args, rates_for_patient
        )
        post_mask = (post_prob > args.threshold).astype(np.uint8)

        # Ground truth enters only after rates and corrected probabilities exist.
        rate_rows = score_rate_regions(unet_mask, bone, labels, rates)
        d_unet = dice(unet_mask, bone)
        d_corr = dice(post_mask, bone)
        sd_unet, hd_unet = surface_metrics(unet_mask, bone)
        sd_corr, hd_corr = surface_metrics(post_mask, bone)
        h_true, m_true, _, _ = empirical_rates(unet_mask, bone)
        h_support = rates["n_background"].astype(np.float64)
        m_support = rates["n_bone"].astype(np.float64)
        h_valid = h_support > 0
        m_valid = m_support > 0
        h_inferred = float(np.average(
            rates["H_mean"][h_valid], weights=h_support[h_valid]
        ))
        m_inferred = float(np.average(
            rates["M_mean"][m_valid], weights=m_support[m_valid]
        ))
        valid = rates["n_pixels"] > 0
        changed = int((post_mask != unet_mask).sum())
        elapsed = time.time() - t0

        payload = dict(
            mr=mr,
            bone=bone,
            unet_prob=unet_prob,
            unet_mask=unet_mask,
            prior_p=prior_p,
            post_prob=post_prob,
            post_mask=post_mask,
            s_norm=s_norm,
            d_mm=d_mm,
            z_index=z_index,
            region_labels=labels,
            **rates,
        )
        if "vertex_z" in pack.files:
            payload["vertex_z"] = np.asarray(pack["vertex_z"]).astype(np.int16)
        if global_grid is not None:
            payload.update(
                grid_h=global_grid.h,
                grid_m=global_grid.m,
                grid_weights=global_grid.weights,
            )
        np.savez_compressed(out_dir / f"{pid}.npz", **payload)

        row = {
            "patient": pid,
            "split": pid_split[pid],
            "variant": variant,
            "n_slices": int(mr.shape[0]),
            "H_inferred_mean": h_inferred,
            "M_inferred_mean": m_inferred,
            "H_empirical": h_true,
            "M_empirical": m_true,
            "dice_unet": d_unet,
            "dice_corrected": d_corr,
            "dice_delta": d_corr - d_unet,
            "surface_dice_unet": sd_unet,
            "surface_dice_corrected": sd_corr,
            "surface_dice_delta": sd_corr - sd_unet,
            "hd95_unet_mm": hd_unet,
            "hd95_corrected_mm": hd_corr,
            "hd95_delta_mm": hd_corr - hd_unet,
            "largest_component_unet": largest_component_fraction(unet_mask),
            "largest_component_corrected": largest_component_fraction(post_mask),
            "brier_unet": float(np.mean((unet_prob - bone) ** 2)),
            "brier_corrected": float(np.mean((post_prob - bone) ** 2)),
            "n_changed": changed,
            "frac_changed": changed / bone.size,
            "all_regions_converged": int(np.all(rates["converged"][valid])),
            "max_convergence_delta": float(np.nanmax(rates["convergence_delta"][valid])),
            "runtime_seconds": elapsed,
        }
        rows.append(row)

        for rr in rate_rows:
            rr.update(patient=pid, split=pid_split[pid], variant=variant)
            all_rate_rows.append(rr)

        du = dice_per_slice(unet_mask, bone)
        dc = dice_per_slice(post_mask, bone)
        for s in range(len(s_norm)):
            all_slice_rows.append({
                "patient": pid,
                "split": pid_split[pid],
                "variant": variant,
                "slice_idx": s,
                "s_norm": float(s_norm[s]),
                "d_mm": float(d_mm[s]),
                "atlas_bin": int(slice_bins[s]),
                "dice_unet": float(du[s]),
                "dice_corrected": float(dc[s]),
                "dice_delta": float(dc[s] - du[s]),
                "bone_pixels": int(bone[s].sum()),
                "changed_pixels": int((post_mask[s] != unet_mask[s]).sum()),
            })

        print(
            f"[{index:2d}/{len(pids)}] {pid} H={h_inferred:.4f} M={m_inferred:.4f} "
            f"Dice {d_unet:.4f}->{d_corr:.4f} ({d_corr-d_unet:+.4f}); "
            f"changed={100*row['frac_changed']:.3f}%  {elapsed:.1f}s"
        )

    def write_csv(path: Path, data: list[dict]) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    write_csv(out_dir / "summary.csv", rows)
    write_csv(out_dir / "rate_metrics.csv", all_rate_rows)
    write_csv(out_dir / "slice_metrics.csv", all_slice_rows)

    delta = np.array([r["dice_delta"] for r in rows])
    surface_delta = np.array([r["surface_dice_delta"] for r in rows])
    dice_ci = bootstrap_mean_ci(delta, args.seed)
    surface_ci = bootstrap_mean_ci(surface_delta, args.seed)
    summary = {
        "variant": variant,
        "locality": args.locality,
        "lesion": args.lesion,
        "split": args.split,
        "predictions_dir": str(args.predictions_dir),
        "atlas": str(args.atlas),
        "prior_dir": str(args.prior_dir) if args.prior_dir is not None else None,
        "data_dir": str(args.data_dir),
        "n_patients": len(rows),
        "patch_size": args.patch_size if args.locality == "patch" else None,
        "h_prior": args.h_prior,
        "m_prior": args.m_prior,
        "fixed_rates": fixed_rates,
        "fixed_rates_source": fixed_source,
        "patient_rates_source": str(patient_rates_source) if patient_rates_source else None,
        "uniform_prior_prevalence": prior_prevalence if args.lesion == "prior" else None,
        "coarse_grid": args.effective_coarse_grid,
        "fine_grid": args.effective_fine_grid,
        "histogram_bins": args.histogram_bins,
        "grid_tolerance": args.grid_tolerance,
        "threshold": args.threshold,
        "dice_unet_mean": float(np.mean([r["dice_unet"] for r in rows])),
        "dice_corrected_mean": float(np.mean([r["dice_corrected"] for r in rows])),
        "dice_delta_mean": float(delta.mean()),
        "dice_delta_bootstrap_ci95": dice_ci,
        "surface_dice_delta_mean": float(surface_delta.mean()),
        "surface_dice_delta_bootstrap_ci95": surface_ci,
        "patients_improved": int((delta > 0).sum()),
        "patients_unchanged": int(np.isclose(delta, 0, atol=1e-12).sum()),
        "all_patients_converged": bool(all(r["all_regions_converged"] for r in rows)),
        "runtime_seconds": float(sum(r["runtime_seconds"] for r in rows)),
        "credible_interval_caveat": (
            "H/M intervals condition on the independent-pixel model and are "
            "narrower than real-world uncertainty because nearby pixels correlate."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("-" * 72)
    print(
        f"{variant} / {args.split}: Dice {summary['dice_unet_mean']:.4f} -> "
        f"{summary['dice_corrected_mean']:.4f}; delta={summary['dice_delta_mean']:+.4f} "
        f"95% bootstrap CI [{dice_ci[0]:+.4f}, {dice_ci[1]:+.4f}]"
    )
    print(
        f"improved={summary['patients_improved']}/{len(rows)}  "
        f"all grids converged={summary['all_patients_converged']}  "
        f"runtime={summary['runtime_seconds']:.1f}s"
    )
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
