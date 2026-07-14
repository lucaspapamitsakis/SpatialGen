# SpatialGen — MetaCOG-based Bone-Segmentation Error Correction

A Python-first revamp of the *Bayesian Framework for Segmentation Error
Quantification* project. The goal: predict skull-bone masks from MR scans with an
Attention U-Net, then run a **MetaCOG-style** localized Bayesian inference loop
(Pyro) that jointly infers the corrected true mask plus localized
false-positive / false-negative error maps — using only the U-Net output, never
the test ground truth.

For scientific rationale, completed-work details, Bouchet/W&B instructions,
known limitations, and next steps, read **[`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md)**.

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
| 3b. Test inference/evaluation | PyTorch | *(todo)* | patient-level metrics + frozen predictions |
| 4. Class-balanced C-VAE | PyTorch | `models/cvae.py` *(todo)* | anatomical shape prior |
| 5. Localized MetaCOG inference | Pyro | `inference/bayesian_engine.py` *(todo)* | corrected masks + error maps |

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
5. Per-slice `s_norm` (crown = 1.0, vault base = 0.0).

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

Important: the training script does **not yet evaluate the test set or save test
predictions**. A separate final inference/evaluation script is the next required
stage before reporting a baseline or starting MetaCOG inference. The current
`--augment` option also uses 90-degree rotations; establish a no-augmentation
baseline or replace these with small-angle rotations for the primary experiment.

## Current next steps

1. Freeze a scientifically defensible slice-retention policy.
2. Run the first no-augmentation U-Net baseline on Bouchet with W&B.
3. Implement patient-level test evaluation and save U-Net predictions.
4. Implement the `s_norm`-conditioned C-VAE.
5. Implement localized Pyro inference and compare it with the frozen baseline.

## Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
