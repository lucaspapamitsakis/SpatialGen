# SpatialGen MetaCOG Grid Experiments

Run date: August 10, 2026  
Frozen U-Net: `v1.1-20260715-123154/best.pt`, epoch 194  
Frozen U-Net test patient-volume Dice: **0.9213**

## What was tested

All experiments used the same U-Net predictions, CT-derived evaluation masks,
patient split, vertex-anchored `s_norm`, training-only bone atlas, Beta(1,1)
rate priors, and 0.5 correction threshold.

- **Global:** one false-positive rate `H` and one false-negative rate `M` per patient.
- **Patch8:** one `H/M` pair per 8×8-pixel block (64 spatial blocks).
- **s_norm13:** one pair per physical-height bin (13 bins).
- **Rate lesion:** one cohort-level pair derived from validation replaces
  patient-specific rates.
- **Prior lesion:** each patient's global-model rates are held fixed while the
  spatial atlas is replaced by the training bone prevalence (0.10982).

The two lesions change one model component at a time.

## Inference equation

For pixel `i`, atlas probability `p_i`, and binary U-Net observation `U_i`:

```text
w_i = p_i(1 - M) + (1 - p_i)H
U_i ~ Bernoulli(w_i)
```

`P(H,M | U,p)` was integrated deterministically on an adaptive logit-space
grid. The likelihood was evaluated in log space to avoid numerical underflow.
All held-out test grids passed the numerical convergence check.

Ground truth was not accepted by the grid or mask-correction functions. It was
loaded afterward to score results and compute empirical error rates.

## Held-out test results (27 patients)

| Variant | Corrected Dice | Mean Dice change | 95% patient-bootstrap CI | Patients improved |
|---|---:|---:|---:|---:|
| Frozen U-Net | **0.9213** | — | — | — |
| Global | 0.9067 | **−0.0146** | [−0.0339, −0.0007] | 0/27 |
| Patch8 | 0.8657 | **−0.0556** | [−0.1103, −0.0112] | 1/27 |
| s_norm13 | 0.8869 | **−0.0344** | [−0.0723, −0.0032] | 0/27 |
| Rate lesion | 0.9213 | 0.0000 | [0.0000, 0.0000] | 0/27 |
| Prior lesion | 0.9213 | 0.0000 | [0.0000, 0.0000] | 0/27 |

Surface Dice changed in the same direction:

- Global: −0.0096
- Patch8: −0.0382
- s_norm13: −0.0221
- Both lesions: 0.0000

HD95 (a “worst 5%” surface-distance measure) also worsened by 0.20 mm, 0.90 mm,
and 0.50 mm for global, Patch8, and s_norm13, respectively.

## What the result means

The implemented atlas-based MetaCOG model **does not improve the strong frozen
U-Net**. This is not a sampler failure:

- deterministic grids converged;
- validation and test agree on the direction;
- both Dice and surface metrics worsen;
- inferred rates poorly match empirical U-Net errors.

For the global test model:

- inferred versus empirical `H`: Pearson `r = -0.239`, MAE `0.0148`;
- inferred versus empirical `M`: Pearson `r = 0.371`, MAE `0.0669`.

Patch and height localization give the model more freedom, but that freedom
mostly lets it mistake atlas-versus-patient anatomical differences for U-Net
errors. The effect becomes more damaging as localization increases.

The lesions are also informative:

- Freezing rates at their validation-cohort average changes no pixels.
- Removing atlas spatial structure while holding patient rates fixed also
  changes no pixels.

Together, these show that most patients lie at the “copy the U-Net” solution.
When the atlas model does move away from that solution, it usually makes the
mask worse.

## Why this motivates a C-VAE

The atlas stores a separate average probability for every pixel. It does not
represent the skull as one connected shape. A C-VAE could instead learn joint
rules such as “the vault should form a coherent ring” and adapt its generated
shape to an individual patient.

The C-VAE is therefore not merely a larger atlas. It tests whether a
patient-adaptive **joint shape prior** can supply evidence strong enough to
correct structured U-Net errors without confusing ordinary anatomical
variation for detector failure.

## Outputs

Machine-readable results:

```text
derivatives/metacog_runs/v1.1-20260715-123154/
├── global/{val,test}/
├── patch8/{val,test}/
├── snorm13/{val,test}/
├── lesion-rates/{val,test}/
├── lesion-prior/{val,test}/
└── comparison/{val,test}/
```

Each run has:

- `summary.csv` and `summary.json`: patient and aggregate outcomes;
- `rate_metrics.csv`: inferred and empirical rates by patient/region;
- `slice_metrics.csv`: Dice and changed pixels by physical height;
- one `.npz` per patient with prior, posterior, mask, rates, and diagnostics.

Figures:

```text
logs/metacog_qc/v1.1-20260715-123154/<variant>/<split>/
```

The main cross-experiment poster figure is:

```text
derivatives/metacog_runs/v1.1-20260715-123154/
comparison/test/experiment_comparison.png
```

## Statistical caveat

The H/M credible intervals condition on the model assumption that pixels are
independent. Nearby image pixels are correlated, so those intervals are too
narrow to represent full real-world uncertainty. Outcome confidence intervals
therefore bootstrap whole patients rather than pixels.
