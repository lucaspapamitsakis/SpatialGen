#!/usr/bin/env python3
"""
03_make_2d_dataset.py
---------------------
Build the shared 2D training/eval dataset for BOTH the Attention U-Net baseline
and the MetaCOG C-VAE / Bayesian inference stage. Everything downstream consumes
the output of THIS script, so the U-Net-vs-generative comparison is apples-to-apples.

Approach: geometry-based cranial-vault crop in native space (no atlas registration).
This is valid here because SynthRAD2023 has already co-registered MR/CT/bone onto an
identical 1 mm isotropic grid per patient (verified), so the only cross-patient
inconsistency is field-of-view / matrix size / head position -- all handled by a
fixed physical crop centred on the head, plus per-subject `s_norm`.

Per patient the pipeline is:
  1. Reorient MR + bone mask to canonical RAS  (+S == superior, guaranteed).
  2. Head mask from MR (Otsu + largest connected component + hole fill).
  3. Find the skull vertex = superior-most axial slice containing head.
  4. Keep the superior `--extent-mm` mm of slices (the ellipsoidal vault, above
     the orbits/sinuses).  This is MR-derived, so it is deployable without CT.
  5. In-plane: centre on the head centroid, crop a fixed `--box-mm` mm box, resize
     to `--size` px (MR: bilinear, bone: nearest).
  6. Intensity-normalise MR (robust z-score inside the head mask).
  7. Compute `s_norm` per slice: crown = 1.0, vault base = 0.0.

Outputs (all under derivatives/, which is gitignored):
  derivatives/dataset_2d/<pid>.npz     mr[S,H,W] f32, bone[S,H,W] u8, s_norm[S] f32,
                                       z_index[S] i16
  derivatives/dataset_2d/splits.json   patient-level train/val/test (frozen)
  logs/dataset_manifest.csv            per-patient QC row
  logs/dataset_thumbnails/<pid>.png    optional mosaic (--save-thumbnails)

Usage:
  .venv/bin/python scripts/03_make_2d_dataset.py                 # all patients
  .venv/bin/python scripts/03_make_2d_dataset.py 1BA001 1BA014   # subset
  .venv/bin/python scripts/03_make_2d_dataset.py --save-thumbnails
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "mr-ct-data"
BONE_DIR = ROOT / "derivatives" / "bone"
OUT_DIR = ROOT / "derivatives" / "dataset_2d"
MANIFEST = ROOT / "logs" / "dataset_manifest.csv"
SPLITS = OUT_DIR / "splits.json"

# Defaults (1 mm isotropic data -> mm == voxels before resize).
DEF_EXTENT_MM = 65     # superior extent kept below the skull vertex
DEF_BOX_MM = 180       # in-plane crop side length, centred on head centroid
DEF_SIZE = 64          # output slice resolution
DEF_MIN_HEAD_FRAC = 0.01   # min head-area fraction for a slice to count as "head"
SPLIT_SEED = 20260710
SPLIT_FRACS = (0.70, 0.15, 0.15)   # train / val / test, by patient


def otsu_threshold(vol: np.ndarray, nbins: int = 256) -> float:
    """Otsu's threshold over finite, positive intensities."""
    v = vol[np.isfinite(vol)]
    v = v[v > 0]
    if v.size == 0:
        return 0.0
    hist, edges = np.histogram(v, bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = hist.astype(np.float64)
    wsum = w.sum()
    if wsum == 0:
        return 0.0
    p = w / wsum
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    sigma_b2[~np.isfinite(sigma_b2)] = 0.0
    return float(centers[int(np.argmax(sigma_b2))])


def head_mask(mr: np.ndarray) -> np.ndarray:
    """Binary head mask: Otsu -> largest connected component -> fill holes."""
    thr = otsu_threshold(mr)
    m = mr > thr
    if not m.any():
        return m
    lbl, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
        m = lbl == (int(np.argmax(sizes)) + 1)
    # Fill holes slice-wise (axial) so interior air/sinus does not punch holes.
    for k in range(m.shape[2]):
        m[:, :, k] = ndimage.binary_fill_holes(m[:, :, k])
    return m


def resize_2d(img: np.ndarray, size: int, order: int) -> np.ndarray:
    """Resize a square-ish 2D array to (size, size) via spline interpolation."""
    zoom = (size / img.shape[0], size / img.shape[1])
    return ndimage.zoom(img, zoom, order=order, mode="nearest")


def crop_box(sl: np.ndarray, cx: int, cy: int, half: int) -> np.ndarray:
    """Crop a (2*half, 2*half) window centred on (cx, cy), zero-padding at edges."""
    H, W = sl.shape
    out = np.zeros((2 * half, 2 * half), dtype=sl.dtype)
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(H, x1), min(W, y1)
    dx0, dy0 = sx0 - x0, sy0 - y0
    out[dx0:dx0 + (sx1 - sx0), dy0:dy0 + (sy1 - sy0)] = sl[sx0:sx1, sy0:sy1]
    return out


def process_patient(pid: str, args) -> dict | None:
    mr_path = DATA_DIR / pid / "mr.nii.gz"
    bone_path = BONE_DIR / f"{pid}_bone.nii.gz"
    if not mr_path.exists() or not bone_path.exists():
        return {"patient": pid, "status": "missing_input", "n_slices": 0}

    mr_img = nib.as_closest_canonical(nib.load(str(mr_path)))
    bone_img = nib.as_closest_canonical(nib.load(str(bone_path)))
    mr = np.asarray(mr_img.dataobj, dtype=np.float32)
    bone = (np.asarray(bone_img.dataobj) > 0).astype(np.uint8)
    if mr.shape != bone.shape:
        return {"patient": pid, "status": "shape_mismatch", "n_slices": 0}

    # mm-per-voxel along each axis after canonical reorient (data is ~1 mm iso).
    zx, zy, zz = mr_img.header.get_zooms()[:3]

    head = head_mask(mr)
    if not head.any():
        return {"patient": pid, "status": "empty_head", "n_slices": 0}

    # Per-axial-slice head area; a slice "has head" if area >= min fraction of the
    # largest slice's area (robust to a few stray voxels at the very top).
    area = head.reshape(-1, head.shape[2]).sum(axis=0).astype(np.float64)
    min_area = args.min_head_frac * area.max()
    has_head = np.where(area >= min_area)[0]
    if has_head.size == 0:
        return {"patient": pid, "status": "empty_head", "n_slices": 0}

    vertex_z = int(has_head.max())                       # superior-most head slice
    n_keep = int(round(args.extent_mm / zz))
    z_base = max(int(has_head.min()), vertex_z - n_keep + 1)
    z_range = np.arange(z_base, vertex_z + 1)            # base .. crown (inclusive)
    z_range = z_range[np.isin(z_range, has_head)]        # drop any gaps
    if z_range.size == 0:
        return {"patient": pid, "status": "no_vault_slices", "n_slices": 0}

    # In-plane centre = head centroid over the retained vault slices.
    vault_head = head[:, :, z_range]
    cx = int(round(ndimage.center_of_mass(vault_head)[0]))
    cy = int(round(ndimage.center_of_mass(vault_head)[1]))
    half_x = int(round(args.box_mm / zx / 2))
    half_y = int(round(args.box_mm / zy / 2))
    half = max(half_x, half_y)

    # Robust MR intensity stats inside the head over the vault (percentile clip + z).
    mr_vault = mr[:, :, z_range]
    vals = mr_vault[vault_head]
    lo, hi = np.percentile(vals, [1, 99])
    mr_clip = np.clip(mr_vault, lo, hi)
    inside = mr_clip[vault_head]
    mu, sd = float(inside.mean()), float(inside.std() + 1e-6)

    S = z_range.size
    mr_out = np.zeros((S, args.size, args.size), dtype=np.float32)
    bone_out = np.zeros((S, args.size, args.size), dtype=np.uint8)
    for i, z in enumerate(z_range):
        m_sl = crop_box(mr_clip[:, :, i], cx, cy, half)
        b_sl = crop_box(bone[:, :, z].astype(np.float32), cx, cy, half)
        mr_out[i] = resize_2d((m_sl - mu) / sd, args.size, order=1)
        bone_out[i] = (resize_2d(b_sl, args.size, order=0) > 0.5).astype(np.uint8)

    # s_norm: crown (superior) = 1.0, vault base = 0.0.
    if S > 1:
        s_norm = (z_range - z_range.min()) / (z_range.max() - z_range.min())
    else:
        s_norm = np.ones(S, dtype=np.float32)
    s_norm = s_norm.astype(np.float32)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / f"{pid}.npz",
        mr=mr_out, bone=bone_out, s_norm=s_norm,
        z_index=z_range.astype(np.int16),
    )

    if args.save_thumbnails:
        _save_thumb(pid, mr_out, bone_out, s_norm)

    return {
        "patient": pid,
        "status": "ok",
        "n_slices": int(S),
        "vertex_z": vertex_z,
        "z_base": int(z_base),
        "extent_mm": round(float(S * zz), 1),
        "box_mm": args.box_mm,
        "size": args.size,
        "center_xy": f"{cx},{cy}",
        "bone_frac": round(float(bone_out.mean()), 4),
        "empty_bone_slices": int((bone_out.reshape(S, -1).sum(axis=1) == 0).sum()),
    }


def _save_thumb(pid, mr, bone, s_norm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = ROOT / "logs" / "dataset_thumbnails"
    out.mkdir(parents=True, exist_ok=True)
    S = mr.shape[0]
    idxs = np.linspace(0, S - 1, min(6, S)).astype(int)
    fig, axes = plt.subplots(2, len(idxs), figsize=(2 * len(idxs), 4))
    axes = np.atleast_2d(axes)
    for j, k in enumerate(idxs):
        axes[0, j].imshow(mr[k].T, cmap="gray", origin="lower")
        axes[0, j].set_title(f"s={s_norm[k]:.2f}", fontsize=8)
        axes[1, j].imshow(mr[k].T, cmap="gray", origin="lower")
        axes[1, j].imshow(np.ma.masked_where(bone[k].T == 0, bone[k].T),
                          cmap="autumn", origin="lower", alpha=0.6)
        for ax in (axes[0, j], axes[1, j]):
            ax.axis("off")
    fig.suptitle(f"{pid}  (S={S})")
    fig.tight_layout()
    fig.savefig(out / f"{pid}.png", dpi=80)
    plt.close(fig)


def make_splits(pids: list[str]) -> dict:
    rng = np.random.default_rng(SPLIT_SEED)
    order = sorted(pids)
    rng.shuffle(order)
    n = len(order)
    n_tr = int(round(SPLIT_FRACS[0] * n))
    n_va = int(round(SPLIT_FRACS[1] * n))
    return {
        "train": sorted(order[:n_tr]),
        "val": sorted(order[n_tr:n_tr + n_va]),
        "test": sorted(order[n_tr + n_va:]),
        "seed": SPLIT_SEED,
        "fracs": list(SPLIT_FRACS),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pids", nargs="*", help="patient IDs (default: all in mr-ct-data/)")
    ap.add_argument("--extent-mm", type=float, default=DEF_EXTENT_MM)
    ap.add_argument("--box-mm", type=float, default=DEF_BOX_MM)
    ap.add_argument("--size", type=int, default=DEF_SIZE)
    ap.add_argument("--min-head-frac", type=float, default=DEF_MIN_HEAD_FRAC)
    ap.add_argument("--save-thumbnails", action="store_true")
    ap.add_argument("--no-splits", action="store_true",
                    help="skip writing splits.json (e.g. when running a subset)")
    args = ap.parse_args()

    if args.pids:
        pids = args.pids
    else:
        pids = sorted(p.name for p in DATA_DIR.iterdir()
                      if p.is_dir() and p.name.startswith("1B"))
    if not pids:
        raise SystemExit(f"No patients found under {DATA_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    rows, ok_pids = [], []
    for i, pid in enumerate(pids, 1):
        row = process_patient(pid, args)
        if row is None:
            continue
        rows.append(row)
        status = row["status"]
        if status == "ok":
            ok_pids.append(pid)
            print(f"[{i:3d}/{len(pids)}] {pid:8s} ok  S={row['n_slices']:3d} "
                  f"bone_frac={row['bone_frac']:.3f} empty={row['empty_bone_slices']}")
        else:
            print(f"[{i:3d}/{len(pids)}] {pid:8s} {status.upper()}")

    fields = ["patient", "status", "n_slices", "vertex_z", "z_base", "extent_mm",
              "box_mm", "size", "center_xy", "bone_frac", "empty_bone_slices"]
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    n_ok = len(ok_pids)
    slices = [r["n_slices"] for r in rows if r["status"] == "ok"]
    print("-" * 48)
    print(f"processed={len(rows)}  ok={n_ok}  failed={len(rows) - n_ok}")
    if slices:
        print(f"slices/patient: mean={np.mean(slices):.1f} "
              f"min={min(slices)} max={max(slices)}  total={sum(slices)}")
    print(f"manifest: {MANIFEST}")

    if not args.no_splits and not args.pids:
        splits = make_splits(ok_pids)
        with open(SPLITS, "w") as f:
            json.dump(splits, f, indent=2)
        print(f"splits:   {SPLITS}  "
              f"(train={len(splits['train'])} val={len(splits['val'])} "
              f"test={len(splits['test'])})")


if __name__ == "__main__":
    main()
