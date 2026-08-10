#!/usr/bin/env python3
"""
qc_bone_topology.py
--------------------
Look at the SHAPE quality of the bone masks, not just how many pixels are right.

Dice measures pixel overlap. It is almost blind to whether a mask is one closed
ring or five disconnected specks, yet that difference is what matters for a
downstream MEG forward model (which needs a closed skull surface) and it is
exactly the kind of error an anatomical shape prior can repair.

Per axial slice this reports:

  n_components   how many separate blobs of bone there are.
                 A superior-vault slice should be ~1 (a single ring).
                 More than that usually means speckle.
  n_holes        how many empty pockets are completely surrounded by bone.
                 A clean vault ring has exactly 1 (the inside of the head).
                 More means pores/gaps within the bone itself.
  largest_frac   what fraction of the bone pixels are in the biggest blob.
                 Near 1.0 = one coherent structure; lower = fragmented.
  clean_ring     True when n_components == 1 and n_holes == 1.

Outputs:
  logs/bone_topology.csv                  one row per slice
  logs/bone_topology/worst_<split>.png    contact sheet, worst slices first
  logs/bone_topology/typical_<split>.png  contact sheet of ordinary slices

Each panel shows the MR with every connected component in its own colour, so
fragmentation is visible at a glance.

Usage:
  .venv/bin/python scripts/qc_bone_topology.py
  .venv/bin/python scripts/qc_bone_topology.py --split test --n-worst 12
  .venv/bin/python scripts/qc_bone_topology.py --data-dir derivatives/dataset_2d
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
DEF_DATA = ROOT / "derivatives" / "dataset_2d_filtered"
REPORT = ROOT / "logs" / "bone_topology.csv"
FIGDIR = ROOT / "logs" / "bone_topology"


def slice_topology(sl: np.ndarray) -> dict:
    m = sl.astype(bool)
    lbl, n = ndimage.label(m)
    filled = ndimage.binary_fill_holes(m)
    _, n_holes = ndimage.label(filled & ~m)
    px = int(m.sum())
    if n > 0:
        sizes = ndimage.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
        largest = float(sizes.max()) / max(px, 1)
    else:
        largest = 0.0
    return {"n_components": int(n), "n_holes": int(n_holes),
            "bone_px": px, "largest_frac": round(largest, 4),
            "clean_ring": int(n == 1 and n_holes == 1)}


def collect(data_dir: Path, pids: list[str], split: str) -> list[dict]:
    rows = []
    for pid in pids:
        pack = np.load(data_dir / f"{pid}.npz")
        bone, s_norm = pack["bone"], pack["s_norm"]
        d_mm = pack["d_mm"] if "d_mm" in pack.files else np.full(len(s_norm), np.nan)
        for i, sl in enumerate(bone):
            r = slice_topology(sl)
            r.update({"patient": pid, "split": split, "slice_idx": i,
                      "s_norm": round(float(s_norm[i]), 4),
                      "d_mm": float(d_mm[i])})
            rows.append(r)
    return rows


def contact_sheet(rows: list[dict], data_dir: Path, out_path: Path,
                  title: str, ncols: int = 4) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    if not rows:
        return
    n = len(rows)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    # distinct colours for components; index 0 stays transparent
    rng = np.random.default_rng(0)
    cols = np.vstack([[0, 0, 0, 0], plt.get_cmap("tab20")(np.arange(20))])
    cmap = ListedColormap(cols)

    cache: dict[str, np.lib.npyio.NpzFile] = {}
    for ax, r in zip(axes, rows):
        pid = r["patient"]
        if pid not in cache:
            cache[pid] = np.load(data_dir / f"{pid}.npz")
        pack = cache[pid]
        mr = pack["mr"][r["slice_idx"]]
        bone = pack["bone"][r["slice_idx"]].astype(bool)
        lbl, ncomp = ndimage.label(bone)
        # relabel so colours are stable-ish and within the colormap range
        shown = np.where(lbl > 0, ((lbl - 1) % 20) + 1, 0)

        ax.imshow(mr.T, cmap="gray", origin="lower")
        ax.imshow(np.ma.masked_where(shown.T == 0, shown.T), cmap=cmap,
                  origin="lower", alpha=0.85, vmin=0, vmax=20, interpolation="nearest")
        flag = "CLEAN" if r["clean_ring"] else "DEFECT"
        ax.set_title(f"{pid}  slice {r['slice_idx']}  s={r['s_norm']:.2f}\n"
                     f"{r['n_components']} blob(s), {r['n_holes']} hole(s)  [{flag}]",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEF_DATA)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--n-worst", type=int, default=8)
    ap.add_argument("--n-typical", type=int, default=8)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    splits = json.loads((args.data_dir / "splits.json").read_text())
    keys = ["train", "val", "test"] if args.split == "all" else [args.split]

    rows: list[dict] = []
    for k in keys:
        rows += collect(args.data_dir, splits[k], k)

    nc = np.array([r["n_components"] for r in rows])
    nh = np.array([r["n_holes"] for r in rows])
    clean = np.array([r["clean_ring"] for r in rows])
    lf = np.array([r["largest_frac"] for r in rows])

    print(f"slices={len(rows)}  splits={keys}")
    print(f"  n_components : mean={nc.mean():.2f}  median={np.median(nc):.0f}  "
          f"max={nc.max()}   >1 in {100 * float((nc > 1).mean()):.1f}% of slices")
    print(f"  n_holes      : mean={nh.mean():.2f}  median={np.median(nh):.0f}  "
          f"max={nh.max()}   !=1 in {100 * float((nh != 1).mean()):.1f}% of slices")
    print(f"  largest_frac : mean={lf.mean():.4f}  "
          f"5th pct={np.percentile(lf, 5):.4f}")
    print(f"  clean single-ring slices: {100 * float(clean.mean()):.1f}%")

    # where in the stack do defects concentrate?
    print("\n  defect rate by height above the vault base:")
    for lo, hi, lab in [(0.0, 0.25, "0.00-0.25 (lowest)"), (0.25, 0.5, "0.25-0.50"),
                        (0.5, 0.75, "0.50-0.75"), (0.75, 1.01, "0.75-1.00 (crown)")]:
        sel = [r for r in rows if lo <= r["s_norm"] < hi]
        if sel:
            bad = 100 * (1 - np.mean([r["clean_ring"] for r in sel]))
            mc = np.mean([r["n_components"] for r in sel])
            print(f"    s_norm {lab:20s} n={len(sel):5d}  defective={bad:5.1f}%  "
                  f"mean blobs={mc:.2f}")

    REPORT.parent.mkdir(exist_ok=True)
    with open(REPORT, "w", newline="") as f:
        fields = ["patient", "split", "slice_idx", "s_norm", "d_mm", "bone_px",
                  "n_components", "n_holes", "largest_frac", "clean_ring"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r[k] for k in fields} for r in rows)
    print(f"\nreport -> {REPORT}")

    if args.no_figures:
        return

    tag = args.split
    worst = sorted(rows, key=lambda r: (-r["n_components"], -r["n_holes"]))[:args.n_worst]
    contact_sheet(worst, args.data_dir, FIGDIR / f"worst_{tag}.png",
                  "Most fragmented bone-mask slices (each colour = one separate blob)",
                  ncols=4)

    typ = [r for r in rows if r["clean_ring"]]
    if typ:
        idx = np.linspace(0, len(typ) - 1, min(args.n_typical, len(typ))).astype(int)
        contact_sheet([typ[i] for i in idx], args.data_dir,
                      FIGDIR / f"typical_{tag}.png",
                      "Clean single-ring slices, for comparison", ncols=4)


if __name__ == "__main__":
    main()
