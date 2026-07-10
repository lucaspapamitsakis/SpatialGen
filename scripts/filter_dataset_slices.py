#!/usr/bin/env python3
"""
filter_dataset_slices.py
------------------------
Remove low-bone or manually flagged slices from the Stage-2 `.npz` dataset and
write a new filtered bundle (same format as before).

Why re-pack instead of editing by hand?
  Each `.npz` stores aligned arrays (mr, bone, s_norm, z_index). Dropping a
  slice means indexing all four arrays together and **recomputing s_norm** so
  crown = 1.0 and vault base = 0.0 on the retained range.

Two ways to choose slices to drop:

  1) Automatic thresholds (recommended starting point)
     --min-bone-frac 0.01     drop slices with bone fraction below this
     --min-bone-pixels 50      drop slices with fewer than this many bone voxels
     A slice is dropped if it fails *either* test (when that test is enabled).

  2) Manual exclusion list (CSV or JSON)
     --exclude-file path/to/exclusions.csv

     CSV columns (header required):
       patient, slice_idx
     Optional column:
       reason

     slice_idx is the **0-based index inside the .npz** (0 = vault base,
     64 = crown). Use --slice-index-base 1 if your notes use 1-based counting.

     Example exclusions.csv:
       patient,slice_idx,reason
       1BA005,58,empty crown
       1BA005,59,empty crown

Other useful flags:
  --dry-run          print what would be removed, do not write
  --report           write logs/slice_filter_report.csv (always on real runs)
  --out-dir          default: derivatives/dataset_2d_filtered
  --overwrite        write back into derivatives/dataset_2d (backs up to .bak/)

Usage:
  # Preview automatic filter (empty + tiny crown slices)
  .venv/bin/python scripts/filter_dataset_slices.py --dry-run \\
      --min-bone-frac 0.01 --min-bone-pixels 50

  # Apply automatic filter
  .venv/bin/python scripts/filter_dataset_slices.py \\
      --min-bone-frac 0.01 --min-bone-pixels 50

  # Manual exclusions only (no automatic rules)
  .venv/bin/python scripts/filter_dataset_slices.py --exclude-file exclusions.csv

  # Combine automatic + manual
  .venv/bin/python scripts/filter_dataset_slices.py \\
      --min-bone-frac 0.01 --exclude-file exclusions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "derivatives" / "dataset_2d"
DEF_OUT = ROOT / "derivatives" / "dataset_2d_filtered"
REPORT = ROOT / "logs" / "slice_filter_report.csv"


def load_exclusions(path: Path, index_base: int) -> dict[str, set[int]]:
    """Return {patient: {0-based slice indices to DROP}}."""
    excluded: dict[str, set[int]] = {}
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        for pid, idxs in data.items():
            excluded[pid] = {int(i) - index_base for i in idxs}
    else:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                pid = row["patient"].strip()
                idx = int(row["slice_idx"]) - index_base
                excluded.setdefault(pid, set()).add(idx)
    return excluded


def recompute_s_norm(z_index: np.ndarray) -> np.ndarray:
    z_index = z_index.astype(np.int16)
    if len(z_index) <= 1:
        return np.ones(len(z_index), dtype=np.float32)
    lo, hi = int(z_index.min()), int(z_index.max())
    return ((z_index - lo) / (hi - lo)).astype(np.float32)


def decide_drops(
    pid: str,
    bone: np.ndarray,
    manual: set[int] | None,
    min_frac: float | None,
    min_px: int | None,
) -> list[tuple[int, str, float, int]]:
    """Return list of (slice_idx, reason, bone_frac, bone_px) to drop."""
    drops: list[tuple[int, str, float, int]] = []
    S = bone.shape[0]
    manual = manual or set()
    for i in range(S):
        px = int(bone[i].sum())
        frac = px / bone[i].size
        reasons = []
        if i in manual:
            reasons.append("manual")
        if min_frac is not None and frac < min_frac:
            reasons.append(f"frac<{min_frac}")
        if min_px is not None and px < min_px:
            reasons.append(f"px<{min_px}")
        if reasons:
            drops.append((i, "+".join(reasons), frac, px))
    return drops


def filter_patient(
    pid: str,
    src: Path,
    dst: Path,
    drop_indices: set[int],
) -> dict:
    pack = np.load(src)
    mr, bone, s_norm, z_index = (
        pack["mr"], pack["bone"], pack["s_norm"], pack["z_index"]
    )
    keep = np.array([i for i in range(len(mr)) if i not in drop_indices], dtype=int)
    if keep.size == 0:
        return {"patient": pid, "status": "all_dropped", "n_before": len(mr), "n_after": 0}

    z_new = z_index[keep]
    out = {
        "mr": mr[keep],
        "bone": bone[keep],
        "z_index": z_new.astype(np.int16),
        "s_norm": recompute_s_norm(z_new),
    }
    np.savez_compressed(dst, **out)
    return {
        "patient": pid,
        "status": "ok",
        "n_before": int(len(mr)),
        "n_after": int(len(keep)),
        "n_dropped": int(len(drop_indices)),
        "bone_frac_after": round(float(out["bone"].mean()), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pids", nargs="*", help="subset of patients (default: all .npz)")
    ap.add_argument("--in-dir", type=Path, default=SRC_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEF_OUT)
    ap.add_argument("--overwrite", action="store_true",
                    help="write filtered .npz back into --in-dir (creates .bak/ first)")
    ap.add_argument("--exclude-file", type=Path, default=None,
                    help="CSV/JSON of patient, slice_idx pairs to drop")
    ap.add_argument("--slice-index-base", type=int, default=0,
                    help="0 if slice_idx is 0-based (default); 1 if 1-based")
    ap.add_argument("--min-bone-frac", type=float, default=None,
                    help="drop slices with bone fraction below this (e.g. 0.01)")
    ap.add_argument("--min-bone-pixels", type=int, default=None,
                    help="drop slices with fewer bone voxels than this (e.g. 50)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="write per-slice report (default on non-dry-run)")
    args = ap.parse_args()

    if args.min_bone_frac is None and args.min_bone_pixels is None and not args.exclude_file:
        raise SystemExit(
            "Specify at least one filter: --min-bone-frac, --min-bone-pixels, "
            "or --exclude-file"
        )

    manual_all = load_exclusions(args.exclude_file, args.slice_index_base) if args.exclude_file else {}

    if args.pids:
        pids = args.pids
    else:
        pids = sorted(p.stem for p in args.in_dir.glob("*.npz") if p.name != "splits.json")

    out_dir = args.in_dir if args.overwrite else args.out_dir
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            bak = args.in_dir / ".bak"
            bak.mkdir(exist_ok=True)
        REPORT.parent.mkdir(exist_ok=True)

    report_rows: list[dict] = []
    summary: list[dict] = []

    for pid in pids:
        src = args.in_dir / f"{pid}.npz"
        if not src.exists():
            print(f"[SKIP] {pid}: no {src.name}")
            continue
        bone = np.load(src)["bone"]
        drops = decide_drops(
            pid, bone, manual_all.get(pid),
            args.min_bone_frac, args.min_bone_pixels,
        )
        drop_set = {d[0] for d in drops}

        for idx, reason, frac, px in drops:
            sn = float(np.load(src)["s_norm"][idx])
            report_rows.append({
                "patient": pid, "slice_idx": idx, "s_norm": round(sn, 4),
                "bone_frac": round(frac, 5), "bone_pixels": px,
                "reason": reason, "action": "drop",
            })

        if args.dry_run:
            print(f"{pid}: {len(bone)} -> {len(bone) - len(drop_set)} "
                  f"(drop {len(drop_set)})")
            for idx, reason, frac, px in drops:
                print(f"    idx={idx:2d}  frac={frac:.4f}  px={px:4d}  {reason}")
            continue

        if args.overwrite:
            shutil.copy2(src, args.in_dir / ".bak" / f"{pid}.npz")
        dst = out_dir / f"{pid}.npz"
        meta = filter_patient(pid, src, dst, drop_set)
        summary.append(meta)
        print(f"{pid}: {meta['n_before']} -> {meta['n_after']} "
              f"(dropped {meta.get('n_dropped', 0)})")

    if args.dry_run:
        total_drop = len(report_rows)
        print("-" * 48)
        print(f"dry-run: would drop {total_drop} slices across {len(pids)} patients")
        return

    do_report = args.report or True
    if do_report and report_rows:
        fields = ["patient", "slice_idx", "s_norm", "bone_frac", "bone_pixels",
                  "reason", "action"]
        with open(REPORT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(report_rows)
        print(f"report: {REPORT}  ({len(report_rows)} dropped slices logged)")

    if not args.overwrite and not args.pids and (SRC_DIR / "splits.json").exists():
        shutil.copy2(SRC_DIR / "splits.json", out_dir / "splits.json")
        print(f"copied splits.json -> {out_dir / 'splits.json'}")

    if summary:
        n_drop = sum(r.get("n_dropped", 0) for r in summary)
        n_after = sum(r["n_after"] for r in summary if r["status"] == "ok")
        print("-" * 48)
        print(f"done: {len(summary)} patients  dropped={n_drop}  remaining={n_after} slices")
        print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
