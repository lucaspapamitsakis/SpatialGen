# SegGen-Revamp — MetaCOG-based Bone-Segmentation Error Correction

A Python-first revamp of the *Bayesian Framework for Segmentation Error
Quantification* project. The goal: predict skull-bone masks from MR scans with an
Attention U-Net, then run a **MetaCOG-style** localized Bayesian inference loop
(Pyro) that jointly infers the corrected true mask plus localized
false-positive / false-negative error maps — using only the U-Net output, never
the test ground truth.

## Pipeline stages

| Stage | Tooling | Script / Module | Output |
|-------|---------|-----------------|--------|
| 0. Bone GT from CT | BioImage Suite Web (`biswebnode`) | `scripts/01_segment_bone.sh` | `derivatives/bone/<pid>_bone.nii.gz` |
| 0b. QC | nibabel/numpy | `scripts/qc_bone_masks.py` | `logs/qc_bone.csv` |
| 1. Atlas registration | ANTsPy | `scripts/02_register_atlas.py` *(todo)* | aligned MR + bone in template space |
| 2. Crop + slice normalize | numpy/nibabel | `scripts/03_make_2d_dataset.py` *(todo)* | 2D tensors + `s_norm` |
| 3. Attention U-Net | MONAI/PyTorch | `models/unet.py` *(todo)* | predicted masks |
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

## Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
