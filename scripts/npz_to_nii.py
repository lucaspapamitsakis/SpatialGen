#!/usr/bin/env python3
"""
npz_to_nii.py
-------------
Convert the Stage-2 `.npz` training volumes back to NIfTI so you can inspect
where the vault crop sits relative to the original uncropped MR scan.

The `.npz` files are NOT full 3D scans. Each one is a small stack of cropped,
downsampled 2D slices: shape (65, 64, 64) = 65 axial slices at 64×64 pixels.
This script writes two kinds of output:

  A) Overlays on the ORIGINAL mr.nii.gz grid (best for manual QC)
     - {pid}_vault_zmask.nii.gz   : 1 on axial slices kept in the vault range
     - {pid}_vault_roi_mask.nii.gz: 1 inside the 180 mm in-plane crop box
                                    on those slices (full 3D ROI)

  B) Standalone small 3D volumes built directly from the `.npz` contents
     - {pid}_vault_mr_64.nii.gz   : normalized MR used for U-Net training
     - {pid}_vault_bone_64.nii.gz : bone labels used for U-Net training
     - {pid}_vault_snorm.nii.gz   : s_norm value written into each kept slice
                                    (1.0 = crown, 0.0 = vault base)

Usage:
  .venv/bin/python scripts/npz_to_nii.py 1BA001
  .venv/bin/python scripts/npz_to_nii.py 1BA001 1BA014 --out-dir derivatives/dataset_2d/nii
  .venv/bin/python scripts/npz_to_nii.py            # all patients with an .npz
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import apply_orientation, io_orientation, ornt_transform

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "mr-ct-data"
NPZ_DIR = ROOT / "derivatives" / "dataset_2d_filtered"
MANIFEST = ROOT / "logs" / "dataset_manifest.csv"
DEF_OUT = NPZ_DIR / "nii"


def load_manifest() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    with open(MANIFEST, newline="") as f:
        return {row["patient"]: row for row in csv.DictReader(f)}


def to_original_layout(
    can_data: np.ndarray,
    orig_img: nib.Nifti1Image,
    can_img: nib.Nifti1Image,
) -> np.ndarray:
    """Map a voxel array from canonical RAS layout back to the original MR layout."""
    trans = ornt_transform(
        io_orientation(can_img.affine),
        io_orientation(orig_img.affine),
    )
    out = apply_orientation(can_data, trans)
    if out.shape != orig_img.shape:
        raise ValueError(
            f"Reoriented mask shape {out.shape} != original MR shape {orig_img.shape}"
        )
    return out


def build_roi_masks(
    shape: tuple[int, int, int],
    z_index: np.ndarray,
    center_xy: tuple[int, int],
    box_mm: float,
    zooms: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (z-only mask, full 3D ROI mask) in canonical voxel space."""
    zmask = np.zeros(shape, dtype=np.uint8)
    roi = np.zeros(shape, dtype=np.uint8)

    cx, cy = center_xy
    half_x = int(round(box_mm / zooms[0] / 2))
    half_y = int(round(box_mm / zooms[1] / 2))
    half = max(half_x, half_y)

    x0, x1 = max(0, cx - half), min(shape[0], cx + half)
    y0, y1 = max(0, cy - half), min(shape[1], cy + half)

    for z in z_index:
        zi = int(z)
        zmask[:, :, zi] = 1
        roi[x0:x1, y0:y1, zi] = 1
    return zmask, roi


def snorm_volume(
    shape: tuple[int, int, int],
    z_index: np.ndarray,
    s_norm: np.ndarray,
) -> np.ndarray:
    """Write s_norm as slice intensity on the original grid (0 elsewhere)."""
    vol = np.zeros(shape, dtype=np.float32)
    for z, s in zip(z_index, s_norm):
        vol[:, :, int(z)] = float(s)
    return vol


def stack_affine(
    ref_affine: np.ndarray,
    origin_vox: tuple[int, int, int],
    zooms: tuple[float, float, float],
) -> np.ndarray:
    """
    Build an affine for a cropped sub-volume whose first voxel sits at
    origin_vox in the reference image's voxel grid.
    """
    aff = ref_affine.copy()
    origin_world = nib.affines.apply_affine(ref_affine, origin_vox)
    # Columns of the affine encode axis directions; keep them, move origin.
    aff[:3, 3] = origin_world
    # Ensure zooms are reflected (SynthRAD is ~1 mm iso; header zooms are fine).
    for i in range(3):
        col = aff[:3, i]
        norm = np.linalg.norm(col)
        if norm > 0:
            aff[:3, i] = col / norm * zooms[i]
    return aff


def convert_patient(
    pid: str,
    out_dir: Path,
    manifest: dict[str, dict],
) -> None:
    npz_path = NPZ_DIR / f"{pid}.npz"
    mr_path = DATA_DIR / pid / "mr.nii.gz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not mr_path.exists():
        raise FileNotFoundError(mr_path)

    meta = manifest.get(pid, {})
    if "center_xy" not in meta or not meta["center_xy"]:
        raise ValueError(
            f"No center_xy for {pid} in {MANIFEST}. Re-run 03_make_2d_dataset.py."
        )
    cx, cy = (int(v) for v in meta["center_xy"].split(","))
    box_mm = float(meta.get("box_mm", 180))

    pack = np.load(npz_path)
    mr64 = pack["mr"]          # (S, 64, 64)
    bone64 = pack["bone"]      # (S, 64, 64)
    s_norm = pack["s_norm"]    # (S,)
    z_index = pack["z_index"]  # (S,)

    orig_img = nib.load(str(mr_path))
    can_img = nib.as_closest_canonical(orig_img)
    zooms = tuple(float(z) for z in can_img.header.get_zooms()[:3])

    zmask_can, roi_can = build_roi_masks(
        can_img.shape, z_index, (cx, cy), box_mm, zooms
    )
    snorm_can = snorm_volume(can_img.shape, z_index, s_norm)

    zmask_orig = to_original_layout(zmask_can, orig_img, can_img)
    roi_orig = to_original_layout(roi_can, orig_img, can_img)
    snorm_orig = to_original_layout(snorm_can, orig_img, can_img)

    patient_out = out_dir / pid
    patient_out.mkdir(parents=True, exist_ok=True)

    # --- A) overlays on original MR grid ---------------------------------
    nib.save(nib.Nifti1Image(zmask_orig, orig_img.affine, orig_img.header),
             patient_out / f"{pid}_vault_zmask.nii.gz")
    nib.save(nib.Nifti1Image(roi_orig, orig_img.affine, orig_img.header),
             patient_out / f"{pid}_vault_roi_mask.nii.gz")
    nib.save(nib.Nifti1Image(snorm_orig, orig_img.affine, orig_img.header),
             patient_out / f"{pid}_vault_snorm.nii.gz")

    # --- B) standalone stacks from the .npz (training resolution) ------
    # NIfTI stores (X, Y, Z); our npz is (S, H, W) with S = axial slices.
    mr_vol = np.transpose(mr64, (2, 1, 0)).astype(np.float32)      # (64,64,S)
    bone_vol = np.transpose(bone64, (2, 1, 0)).astype(np.uint8)

    half = int(round(box_mm / zooms[0] / 2))
    z0 = int(z_index.min())
    origin_vox_can = (cx - half, cy - half, z0)
    # Map canonical origin voxel to original layout for a physically aligned crop.
    origin_marker = np.zeros(can_img.shape, dtype=np.uint8)
    origin_marker[cx - half, cy - half, z0] = 1
    origin_orig = to_original_layout(origin_marker, orig_img, can_img)
    origin_vox_orig = tuple(int(v) for v in np.argwhere(origin_orig > 0)[0])

    stack_aff = stack_affine(orig_img.affine, origin_vox_orig, zooms)
    nib.save(nib.Nifti1Image(mr_vol, stack_aff),
             patient_out / f"{pid}_vault_mr_64.nii.gz")
    nib.save(nib.Nifti1Image(bone_vol, stack_aff),
             patient_out / f"{pid}_vault_bone_64.nii.gz")

    print(f"{pid}: wrote 5 NIfTIs -> {patient_out}")
    print(f"       z range (canonical indices): {int(z_index.min())}..{int(z_index.max())}"
          f"  ({len(z_index)} slices)  center=({cx},{cy})  box={box_mm}mm")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pids", nargs="*", help="patient IDs (default: all .npz files)")
    ap.add_argument("--out-dir", type=Path, default=DEF_OUT,
                    help=f"output root directory (default: {DEF_OUT})")
    args = ap.parse_args()

    if args.pids:
        pids = args.pids
    else:
        pids = sorted(p.stem for p in NPZ_DIR.glob("*.npz"))
    if not pids:
        raise SystemExit(f"No .npz files found in {NPZ_DIR}")

    manifest = load_manifest()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for pid in pids:
        convert_patient(pid, args.out_dir, manifest)


if __name__ == "__main__":
    main()
