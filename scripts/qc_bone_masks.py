#!/usr/bin/env python3
"""
qc_bone_masks.py
----------------
Quality-control report for the batch bone-segmentation outputs.

For every binary bone mask in derivatives/bone/ this script computes:
  - volume shape and voxel spacing
  - number / fraction of bone voxels (sanity: expect ~3-5% of the full volume)
  - whether geometry (shape + affine) matches the source CT
  - a per-patient axial bone-extent profile (which slices contain bone)

It writes a CSV summary and flags outliers (empty masks, abnormally large/small
bone fractions) that should be inspected visually in the BioImage Suite viewer.

Usage:
  source .venv/bin/activate
  python scripts/qc_bone_masks.py
  python scripts/qc_bone_masks.py --save-thumbnails   # also dump PNG mosaics
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
BONE_DIR = ROOT / "derivatives" / "bone"
DATA_DIR = ROOT / "mr-ct-data"
OUT_CSV = ROOT / "logs" / "qc_bone.csv"

# Sanity bounds for the bone fraction of the *whole* CT volume.
FRAC_LOW, FRAC_HIGH = 0.015, 0.08


def patient_id_from_mask(p: Path) -> str:
    return p.name.replace("_bone.nii.gz", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-thumbnails", action="store_true",
                    help="Save a PNG mosaic of axial slices per patient (needs matplotlib).")
    args = ap.parse_args()

    masks = sorted(BONE_DIR.glob("*_bone.nii.gz"))
    if not masks:
        raise SystemExit(f"No bone masks found in {BONE_DIR}")

    rows = []
    flagged = []
    for m in masks:
        pid = patient_id_from_mask(m)
        img = nib.load(str(m))
        data = np.asarray(img.dataobj)
        spacing = img.header.get_zooms()[:3]
        n_pos = int((data > 0).sum())
        frac = float(n_pos) / data.size

        # geometry check vs source CT
        ct_path = DATA_DIR / pid / "ct.nii.gz"
        geom_ok = ""
        if ct_path.exists():
            ct = nib.load(str(ct_path))
            geom_ok = (ct.shape == img.shape
                       and np.allclose(ct.affine, img.affine, atol=1e-3))

        flag = ""
        if n_pos == 0:
            flag = "EMPTY"
        elif frac < FRAC_LOW:
            flag = "LOW_FRAC"
        elif frac > FRAC_HIGH:
            flag = "HIGH_FRAC"
        if flag:
            flagged.append((pid, flag, round(frac, 4)))

        rows.append({
            "patient": pid,
            "shape": "x".join(map(str, img.shape)),
            "spacing": ",".join(f"{s:.3f}" for s in spacing),
            "bone_voxels": n_pos,
            "bone_frac": round(frac, 5),
            "geom_matches_ct": geom_ok,
            "flag": flag,
        })

        if args.save_thumbnails:
            _save_thumb(pid, data)

    OUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fracs = np.array([r["bone_frac"] for r in rows])
    print(f"Masks analysed : {len(rows)}")
    print(f"Bone fraction  : mean={fracs.mean():.4f}  "
          f"min={fracs.min():.4f}  max={fracs.max():.4f}")
    print(f"CSV written    : {OUT_CSV}")
    if flagged:
        print(f"\nFLAGGED ({len(flagged)}) -- inspect these in the viewer:")
        for pid, flag, frac in flagged:
            print(f"  {pid:10s} {flag:10s} frac={frac}")
    else:
        print("\nNo outliers flagged. All masks within expected bone-fraction range.")


def _save_thumb(pid: str, data: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = ROOT / "logs" / "thumbnails"
    out_dir.mkdir(parents=True, exist_ok=True)
    z = data.shape[2]
    idxs = np.linspace(z * 0.2, z * 0.9, 9).astype(int)
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for ax, k in zip(axes.ravel(), idxs):
        ax.imshow(data[:, :, k].T, cmap="gray", origin="lower")
        ax.set_title(f"z={k}", fontsize=7)
        ax.axis("off")
    fig.suptitle(pid)
    fig.tight_layout()
    fig.savefig(out_dir / f"{pid}.png", dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()
