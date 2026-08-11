# SpatialGen — MetaCOG-based Bone-Segmentation Error Correction

A Python-first revamp of my *Bayesian Framework for Segmentation Error
Quantification* project. The goal: predict skull-bone masks from MR scans with an
Attention U-Net, then run **MetaCOG-style deterministic Bayesian inference** that
jointly estimates the corrected true mask plus global or localized false-positive
/ false-negative error rates — using only the U-Net output, never the test ground
truth.

## Pipeline stages

| Stage | Tooling | Script / Module | Output |
|-------|---------|-----------------|--------|
| 0. Bone GT from CT | BioImage Suite Web (`biswebnode`) | `scripts/01_segment_bone.sh` | `derivatives/bone/<pid>_bone.nii.gz` |
| 0b. QC | nibabel/numpy | `scripts/qc_bone_masks.py` | `logs/qc_bone.csv` |
| 1. Atlas registration | ANTsPy | *(optional / deferred; no script yet)* | aligned MR + bone in template space |
| 2. Crop + slice normalize | numpy/scipy/nibabel | `scripts/03_make_2d_dataset.py` | 2D tensors + `s_norm` |
| 2b. Filter low-bone slices | numpy | `scripts/filter_dataset_slices.py` | filtered 2D tensors |
| 2c. NIfTI inspection export | nibabel | `scripts/npz_to_nii.py` | stacks + original-grid ROI overlays |
| 3. Attention U-Net | MONAI/PyTorch | `models/unet.py` + `scripts/04_train_unet.py` | validation-selected checkpoint |
| 3b. Test inference/evaluation | PyTorch | `scripts/06_run_unet_inference.py` | patient-level metrics + frozen predictions |
| 4. Anatomical prior | numpy/scipy | `scripts/07_build_bone_atlas.py` | empirical `s_norm`-binned atlas baseline |
| 4b. Conditional C-VAE prior | PyTorch | `models/cvae.py` + `scripts/11_train_cvae.py` + `scripts/12_sample_cvae_prior.py` | soft `P(bone\|z,s_norm)` maps for MetaCOG |
| 5. MetaCOG grid inference | numpy/scipy | `models/metacog.py` + `scripts/08_run_metacog_inference.py` | global, 8×8-patch, or `s_norm`-stratified rates + corrected masks |
| 5b. MetaCOG QC figures | matplotlib | `scripts/09_visualize_metacog.py` | posterior/rate maps, height curves, mask panels, paired metrics |
| 5c. Cross-experiment comparison | numpy/matplotlib | `scripts/10_compare_metacog_experiments.py` | paired bootstrap summaries + poster figure |

## Data layout

```
mr-ct-data/<patient>/
    ct.nii.gz     # CT volume (source for bone ground truth)
    mr.nii.gz     # MR volume (U-Net input)
    mask.nii.gz   # dataset-provided body mask (IGNORED here)
derivatives/
    segm3d/<pid>_segm3d.nii.gz   # 3-class label map (0=air,1=soft,2=bone)
    bone/<pid>_bone.nii.gz       # binary skull-bone mask {0,1}
```

180 paired patients are available.

## Stage 0: Bone segmentation (BioImage Suite, batched)

The interactive Dual Viewer workflow is reproduced on the command line via the
`biswebnode` package:

1. **`segmentimage --numclasses 3`** — histogram / k-means-style segmentation of
   the CT into air (label 0), soft tissue (label 1) and bone (label 2).
2. **`thresholdimage --low 2 --high 3 --inval 1 --outval 0`** — keep only the
   bone class to produce a binary mask.

This matches the original `*_segm3d` / `*_bone3d` result logs (bone ≈ 3.9% of the
full CT volume). Run it with:

```bash
npm install -g biswebnode        # once
bash scripts/01_segment_bone.sh  # all patients (resumable)
```

## Stage 2: 2D vault dataset (shared by U-Net and C-VAE)

SynthRAD already co-registers MR/CT/bone onto an identical 1 mm isotropic grid per
patient, so a full atlas registration is **not** needed to make the U-Net baseline
comparable to the generative method — both simply consume one frozen dataset.
`03_make_2d_dataset.py` builds it with a geometry-based crop in native space:

1. Reorient MR + bone to canonical RAS.
2. MR head mask (Otsu + largest component + hole fill); find the skull vertex.
3. Keep the superior ~65 mm of slices (ellipsoidal vault, above orbits/sinuses).
4. Centre on the head centroid, crop a fixed 180 mm box, resize to 64×64
   (MR bilinear, bone nearest); robust per-volume MR z-score.
5. Per-slice vertex-anchored `s_norm` (crown = 1.0, 64 mm below crown = 0.0).

```bash
.venv/bin/python scripts/03_make_2d_dataset.py --save-thumbnails
```

Outputs: `derivatives/dataset_2d/<pid>.npz` (`mr`, `bone`, `s_norm`, `z_index`),
`derivatives/dataset_2d/splits.json` (frozen patient-level 70/15/15 split),
`logs/dataset_manifest.csv`, and QC mosaics in `logs/dataset_thumbnails/`.
Atlas registration (`02_register_atlas.py`) is deferred to an optional Stage-4
ablation: *does aligning skulls tighten the C-VAE shape prior enough to help?*

The current training dataset removes empty or near-empty target slices and
preserves the original bundles:

```bash
.venv/bin/python scripts/filter_dataset_slices.py \
  --min-bone-frac 0.01 --min-bone-pixels 50
```

This produces `derivatives/dataset_2d_filtered/`: 10,872 slices from all 180
patients (7,592 train / 1,638 validation / 1,642 test). Because this filtering
uses the target mask, it is a retrospective target-defined ROI and must be
reconsidered or explicitly documented before final deployment claims. See the
handoff for details.

## Stage 3: Attention U-Net baseline (MR -> bone)

`models/unet.py` wraps a MONAI 2D Attention U-Net (logits out).
`scripts/04_train_unet.py` trains it on the filtered dataset with weighted BCE
(`pos_weight` auto-computed from the train set, ~8.1) + soft Dice, logging to W&B.
It uses validation Dice to select `best.pt` and reads the frozen patient-level
`splits.json`.

```bash
# Smoke test (memorize a few slices; expect val Dice -> ~1.0)
.venv/bin/python scripts/04_train_unet.py --overfit 16 --epochs 40 --device cpu

# Full run (use a GPU on Bouchet; add --wandb for logging)
.venv/bin/python scripts/04_train_unet.py --epochs 80 --augment --wandb
```

Checkpoints (`best.pt` by val Dice, `last.pt`) and `config.json` land in
`derivatives/unet_runs/<timestamp>/`. CPU is ~3.5 min/epoch; use CUDA/MPS for
real runs.

The frozen run `v1.1-20260715-123154` selected epoch 194 at validation Dice
0.91265. `06_run_unet_inference.py` saves predictions and both mean-slice and
patient-volume Dice for all frozen splits.

## Stage 4b: s_norm-conditioned C-VAE shape prior

`models/cvae.py` is a 64×64 convolutional VAE conditioned only on vertex-anchored
`s_norm` (no MR, no U-Net). It outputs soft bone logits; MetaCOG uses Monte Carlo
averages of `σ(f_θ(z, s_norm))` with `z ~ N(0,I)` as the anatomical prior `p_i`.

Training uses weighted BCE + soft Dice + β·KL, with linear β warm-up. Augmentation
is **off by default**; `--augment` enables small continuous rotations only (no
flips). Checkpoints land in `derivatives/cvae_runs/<timestamp>/`.

With `--wandb`, every epoch logs scalar curves (loss / Dice / KL / β / latent
stats) against epoch. Every `--wandb-image-every` epochs (default 5; also epoch 1
and the last epoch) uploads reconstruction panels (GT vs recon), prior samples at
several `s_norm` levels, and Monte Carlo mean prior maps. The same PNGs are saved
under `derivatives/cvae_runs/<stamp>/panels/`.

```bash
# Smoke test (expect reconstruction Dice to rise quickly)
.venv/bin/python scripts/11_train_cvae.py --overfit 16 --epochs 40 --device cpu

# Full train on filtered train split (optional small rotations + W&B)
.venv/bin/python scripts/11_train_cvae.py --epochs 120 --augment --wandb \
  --wandb-image-every 5

# Export soft priors for MetaCOG
.venv/bin/python scripts/12_sample_cvae_prior.py \
  --checkpoint derivatives/cvae_runs/<stamp>/best.pt --split val --save-qc

# MetaCOG with C-VAE prior instead of the empirical atlas maps
.venv/bin/python scripts/08_run_metacog_inference.py \
  --predictions-dir derivatives/unet_predictions/v1.1-20260715-123154 \
  --split val --locality global \
  --prior-dir derivatives/cvae_priors/<stamp>/val
```

## Stage 5: MetaCOG grid experiments

The two-dimensional `P(H,M | U,p)` posterior is integrated deterministically on
an adaptive logit-space grid. No MCMC chains or Pyro installation are required.

```bash
PRED=derivatives/unet_predictions/v1.1-20260715-123154

# Develop on validation
.venv/bin/python scripts/08_run_metacog_inference.py \
  --predictions-dir "$PRED" --split val --locality global
.venv/bin/python scripts/08_run_metacog_inference.py \
  --predictions-dir "$PRED" --split val --locality patch --patch-size 8
.venv/bin/python scripts/08_run_metacog_inference.py \
  --predictions-dir "$PRED" --split val --locality snorm

# Visualize one run
.venv/bin/python scripts/09_visualize_metacog.py \
  --run-dir derivatives/metacog_runs/v1.1-20260715-123154/global/val

# Compare completed variants on the same split
.venv/bin/python scripts/10_compare_metacog_experiments.py \
  --root derivatives/metacog_runs/v1.1-20260715-123154 --split test
```

Every run reports patient-volume Dice, 3 mm surface Dice, HD95, largest-component
fraction, probability Brier score, empirical-vs-inferred rate accuracy, performance
by physical height, changed-pixel fraction, deterministic-grid convergence, and
runtime. Ground truth enters only after inference for scoring.

## Current next steps

1. Train the C-VAE on the filtered train split; QC reconstructions and samples.
2. Export Monte Carlo soft priors and compare MetaCOG (`--prior-dir`) against the
   frozen atlas baselines without changing the U-Net, masks, split, or threshold.
3. If MC averaging is still too weak, jointly infer latent `z` with `(H,M)`.

## Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
