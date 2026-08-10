#!/usr/bin/env python3
"""
ARCHIVED — one-time repair. Do not run as part of the normal pipeline.

The bug this fixed lived in filter_dataset_slices.py (it rescaled s_norm after
dropping slices). That script now keeps the original vertex-anchored s_norm, so
re-filtering from dataset_2d produces correct labels without this repair.

The dataset_2d_filtered/*.npz files on disk already carry the corrected s_norm
(plus d_mm and vertex_z). Kept here only as historical record of the repair.

03b_reanchor_s_norm.py
-----------------------
Repair the through-plane coordinate so it means the same anatomy in every patient.

THE PROBLEM
-----------
`03_make_2d_dataset.py` anchors each patient's stack to the MR-derived skull vertex
and keeps a fixed physical extent (65 mm, i.e. 65 slices at 1 mm isotropic), so its
`s_norm` is a patient-independent physical coordinate:

    s_norm = 1 - (vertex_z - z) / 64          ->  0 = 64 mm below vertex, 1 = vertex

All 180 patients retained exactly 65 slices, so that coordinate is consistent
across the whole cohort.

`filter_dataset_slices.py` then drops low-bone slices and calls `recompute_s_norm`,
which rescales to the *retained* range. Because every dropped slice came off the top
(828 slices dropped, minimum dropped s_norm 0.859, 94.6% above 0.9, zero off the
bottom) and the number dropped varies per patient (1-10, mean 5.0, affecting 165 of
180 patients), the rescaled coordinate no longer means the same thing per patient:

    s_norm   mean mm below vertex   cross-patient spread
    1.00     4.6                   10.0 mm
    0.90     10.5                   9.0 mm
    0.50     34.3                   5.0 mm
    0.00     64.0                   0.0 mm

Only 15 of 180 patients have s_norm = 1.0 actually at their vertex. Since atlas bins
are ~4.6 mm wide, the crown misalignment spans ~2.2 bins - exactly where the vault
cross-section is smallest and the prior is weakest.

THE FIX
-------
Restore the vertex-anchored, fixed-physical-extent coordinate on the filtered stacks.
Filtering then simply *truncates* each patient's coordinate range from the top rather
than rescaling it: a patient who lost k slices has s_norm in [0, 1 - k/64].

Why absolute physical units rather than proportional (head-size) normalization: the
in-plane crop is already a fixed 180 mm box resized to 64x64, so 1 px = 2.8 mm for
every patient and in-plane anatomy is *not* size-normalized. `PROJECT_HANDOFF.md` s2
records preserving real head size as a deliberate decision. Proportional through-plane
normalization would be inconsistent with both.

Anchors are read from the unfiltered `dataset_2d/<pid>.npz` stack, whose `z_index`
min/max are exactly the stage-03 vault base and vertex. Nothing else is touched:
`mr`, `bone` and `z_index` are verified byte-identical before anything is written.

The MR-derived vertex was validated as a genuine anatomical landmark rather than a
field-of-view boundary: across 23 probed patients the topmost head cross-section is
only 1.7-6.2% of the cross-section 30 mm lower (a dome cap, not a clipped ellipse),
and the distance from vertex to first CT bone is 3.7 +/- 2.4 mm, consistent with
scalp thickness.

WHAT THIS DOES NOT AFFECT
-------------------------
The U-Net never reads `s_norm` (`SliceDataset` in `04_train_unet.py` loads only `mr`
and `bone`), so trained checkpoints and their reported Dice remain valid. Only the
atlas prior (`07_build_bone_atlas.py`) and the MetaCOG prior lookup
(`08_run_metacog_inference.py`) consume `s_norm`; both must be re-run afterward.

Predictions already written by `06_run_unet_inference.py` copy `s_norm` through from
the dataset, so re-run `06` too, or pass `--predictions-dir` to patch them in place.

OUTPUT KEYS
-----------
Adds two self-documenting arrays alongside the corrected `s_norm`:

    d_mm      float32 [S]   mm below the MR-derived skull vertex (0 = vertex)
    vertex_z  int16   scalar  canonical-RAS z index of the vertex

Usage:
  # inspect first; writes nothing
  .venv/bin/python scripts/03b_reanchor_s_norm.py --dry-run

  # apply in place (backs up the original s_norm arrays first)
  .venv/bin/python scripts/03b_reanchor_s_norm.py

  # or write a parallel dataset instead of mutating the frozen one
  .venv/bin/python scripts/03b_reanchor_s_norm.py --out-dir derivatives/dataset_2d_vtx

  # also patch s_norm inside an existing predictions directory
  .venv/bin/python scripts/03b_reanchor_s_norm.py \\
      --predictions-dir derivatives/unet_predictions/<run-name>

  # re-verify a previously repaired dataset
  .venv/bin/python scripts/03b_reanchor_s_norm.py --verify-only
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEF_SRC = ROOT / "derivatives" / "dataset_2d"
DEF_FILT = ROOT / "derivatives" / "dataset_2d_filtered"
REPORT = ROOT / "logs" / "s_norm_reanchor_report.csv"

PASSTHROUGH = ("mr", "bone", "z_index")


def anchors(unfiltered: Path) -> tuple[int, int]:
    """(vault_base_z, vertex_z) from the unfiltered stage-03 stack."""
    z = np.load(unfiltered)["z_index"].astype(int)
    return int(z.min()), int(z.max())


def reanchored(z_index: np.ndarray, base_z: int, vertex_z: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertex-anchored s_norm plus mm-below-vertex, on a fixed physical extent."""
    z = z_index.astype(np.int64)
    extent = vertex_z - base_z                      # 64 mm for every patient
    if extent <= 0:
        raise ValueError(f"degenerate extent {extent}")
    d_mm = (vertex_z - z).astype(np.float32)
    s_norm = (1.0 - d_mm / extent).astype(np.float32)
    return s_norm, d_mm


def spread_table(per_patient: list[tuple[np.ndarray, np.ndarray]],
                 key: str) -> list[tuple[float, float, float, float]]:
    """For a few s_norm levels, how many mm below the vertex does each patient sit?"""
    out = []
    for level in (1.0, 0.9, 0.75, 0.5, 0.25, 0.0):
        ds = []
        for s, d in per_patient:
            if s.size == 0:
                continue
            j = int(np.argmin(np.abs(s - level)))
            # only meaningful if the patient actually reaches this level
            if abs(s[j] - level) <= 0.05:
                ds.append(float(d[j]))
        if ds:
            ds = np.array(ds)
            out.append((level, float(ds.mean()), float(ds.min()), float(ds.max())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pids", nargs="*", help="subset of patients (default: all)")
    ap.add_argument("--src-dir", type=Path, default=DEF_SRC,
                    help="unfiltered stage-03 dataset, used only to read anchors")
    ap.add_argument("--filtered-dir", type=Path, default=DEF_FILT)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write here instead of repairing --filtered-dir in place")
    ap.add_argument("--predictions-dir", type=Path, default=None,
                    help="also patch s_norm/d_mm inside this 06_run_unet_inference.py output")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true",
                    help="check an already-repaired dataset and exit")
    args = ap.parse_args()

    pids = args.pids or sorted(p.stem for p in args.filtered_dir.glob("*.npz"))
    if not pids:
        raise SystemExit(f"No .npz found in {args.filtered_dir}")

    in_place = args.out_dir is None
    dst_dir = args.filtered_dir if in_place else args.out_dir

    rows: list[dict] = []
    before: list[tuple[np.ndarray, np.ndarray]] = []
    after: list[tuple[np.ndarray, np.ndarray]] = []
    payloads: dict[str, dict] = {}
    problems: list[str] = []

    for pid in pids:
        fpath = args.filtered_dir / f"{pid}.npz"
        upath = args.src_dir / f"{pid}.npz"
        if not upath.exists():
            problems.append(f"{pid}: missing unfiltered stack {upath.name}")
            continue

        pack = dict(np.load(fpath))
        base_z, vertex_z = anchors(upath)
        n_unfiltered = len(np.load(upath)["z_index"])
        if n_unfiltered != vertex_z - base_z + 1:
            problems.append(f"{pid}: unfiltered stack has gaps; anchors may be unreliable")

        s_old = pack["s_norm"].astype(np.float32)
        s_new, d_mm = reanchored(pack["z_index"], base_z, vertex_z)

        # a patient's own top slice, expressed as mm below the vertex, is the quantity
        # that was previously forced to 0 for everyone by the rescaling
        rows.append({
            "patient": pid,
            "n_slices": int(len(s_new)),
            "n_dropped": int(n_unfiltered - len(s_new)),
            "vertex_z": vertex_z,
            "base_z": base_z,
            "s_norm_max_old": round(float(s_old.max()), 4),
            "s_norm_max_new": round(float(s_new.max()), 4),
            "top_slice_mm_below_vertex": round(float(d_mm.min()), 1),
            "max_abs_shift": round(float(np.abs(s_new - s_old).max()), 4),
        })
        before.append((s_old, d_mm))
        after.append((s_new, d_mm))

        if args.verify_only:
            ok = np.allclose(s_old, s_new, atol=1e-6)
            if not ok:
                problems.append(f"{pid}: s_norm is NOT vertex-anchored "
                                f"(max shift {np.abs(s_new - s_old).max():.4f})")
            continue

        pack["s_norm"] = s_new
        pack["d_mm"] = d_mm
        pack["vertex_z"] = np.int16(vertex_z)
        payloads[pid] = pack

    # ---------------- reporting ----------------
    if args.verify_only:
        if problems:
            print(f"VERIFY FAILED ({len(problems)} issues):")
            for p in problems[:20]:
                print("  " + p)
            raise SystemExit(1)
        print(f"VERIFY OK: all {len(rows)} patients are vertex-anchored.")
        return

    n_dropped = np.array([r["n_dropped"] for r in rows])
    shift = np.array([r["max_abs_shift"] for r in rows])
    top_mm = np.array([r["top_slice_mm_below_vertex"] for r in rows])
    new_max = np.array([r["s_norm_max_new"] for r in rows])

    print(f"patients={len(rows)}   dropped slices/patient: "
          f"mean={n_dropped.mean():.2f} range=[{n_dropped.min()},{n_dropped.max()}]")
    print(f"patients whose s_norm changes at all: {(shift > 1e-6).sum()} / {len(rows)}")
    print(f"largest single-slice s_norm shift: {shift.max():.4f}")
    print(f"top retained slice sits {top_mm.mean():.1f} mm below vertex "
          f"(range {top_mm.min():.0f}-{top_mm.max():.0f} mm)")
    print(f"new per-patient max s_norm: mean={new_max.mean():.4f} "
          f"range=[{new_max.min():.4f},{new_max.max():.4f}]  "
          f"(was 1.0000 for all {len(rows)})")

    print("\nCross-patient anatomical consistency (mm below the MR-derived vertex "
          "at a given s_norm):")
    print(f"  {'s_norm':>7s}  {'BEFORE (mean, range)':>30s}   {'AFTER (mean, range)':>30s}")
    b = {lvl: (mu, lo, hi) for lvl, mu, lo, hi in spread_table(before, "old")}
    a = {lvl: (mu, lo, hi) for lvl, mu, lo, hi in spread_table(after, "new")}
    for lvl in (1.0, 0.9, 0.75, 0.5, 0.25, 0.0):
        if lvl not in b and lvl not in a:
            continue
        bs = (f"{b[lvl][0]:5.1f} mm  [{b[lvl][1]:4.1f},{b[lvl][2]:4.1f}]  "
              f"spread {b[lvl][2]-b[lvl][1]:4.1f}") if lvl in b else "n/a"
        as_ = (f"{a[lvl][0]:5.1f} mm  [{a[lvl][1]:4.1f},{a[lvl][2]:4.1f}]  "
               f"spread {a[lvl][2]-a[lvl][1]:4.1f}") if lvl in a else "n/a"
        print(f"  {lvl:7.2f}  {bs:>30s}   {as_:>30s}")

    if problems:
        print(f"\nwarnings ({len(problems)}):")
        for p in problems[:10]:
            print("  " + p)

    if args.dry_run:
        print(f"\ndry-run: would write {len(payloads)} patients to {dst_dir}"
              f"{' (in place)' if in_place else ''}")
        return

    # ---------------- write ----------------
    dst_dir.mkdir(parents=True, exist_ok=True)
    if in_place:
        bak = args.filtered_dir / ".bak_snorm"
        bak.mkdir(exist_ok=True)

    for pid, pack in payloads.items():
        fpath = args.filtered_dir / f"{pid}.npz"
        if in_place:
            np.savez_compressed(args.filtered_dir / ".bak_snorm" / f"{pid}.npz",
                                s_norm=np.load(fpath)["s_norm"])
        out = dst_dir / f"{pid}.npz"
        np.savez_compressed(out, **pack)

        # guarantee the repair touched nothing but the coordinate
        chk = np.load(out)
        orig = np.load(fpath)
        for k in PASSTHROUGH:
            if not np.array_equal(chk[k], orig[k]):
                raise SystemExit(f"FATAL: {pid} '{k}' changed during repair; "
                                 f"restore from {args.filtered_dir / '.bak_snorm'}")

    splits = args.filtered_dir / "splits.json"
    if not in_place and splits.exists():
        shutil.copy2(splits, dst_dir / "splits.json")

    REPORT.parent.mkdir(exist_ok=True)
    with open(REPORT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {len(payloads)} patients -> {dst_dir}")
    print(f"report -> {REPORT}")
    if in_place:
        print(f"original s_norm arrays backed up in {args.filtered_dir / '.bak_snorm'}")

    # ---------------- optionally patch predictions ----------------
    if args.predictions_dir is not None:
        n = 0
        for pid, pack in payloads.items():
            ppath = args.predictions_dir / f"{pid}.npz"
            if not ppath.exists():
                continue
            pp = dict(np.load(ppath))
            if not np.array_equal(pp["z_index"], pack["z_index"]):
                print(f"  [SKIP] {pid}: predictions z_index differs from dataset")
                continue
            pp["s_norm"] = pack["s_norm"]
            pp["d_mm"] = pack["d_mm"]
            pp["vertex_z"] = pack["vertex_z"]
            np.savez_compressed(ppath, **pp)
            n += 1
        print(f"patched s_norm in {n} prediction files -> {args.predictions_dir}")

    print("\nNext: rebuild the atlas and re-run inference, since both consume s_norm:")
    print("  .venv/bin/python scripts/07_build_bone_atlas.py")


if __name__ == "__main__":
    main()
