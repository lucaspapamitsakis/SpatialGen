# AGENTS.md

## Cursor Cloud specific instructions

SpatialGen is a Python 3.12 medical-imaging research pipeline (no web/GUI app).
It predicts skull-bone masks from MR with a MONAI Attention U-Net and then runs
deterministic MetaCOG Bayesian grid inference (`models/metacog.py`) plus an
`s_norm`-conditioned C-VAE shape prior (`models/cvae.py`). See `README.md` for the
full stage-by-stage pipeline and per-script commands; this section only records
the non-obvious cloud caveats.

- Dependencies live in a `.venv` at the repo root and are installed from
  `requirements.txt` by the startup update script. Creating the venv needs the
  system package `python3.12-venv` (already present in the base image).
- Run everything from the repo root with `.venv/bin/python ...`. The pipeline
  scripts self-insert the repo root onto `sys.path`, so `.venv/bin/python
  scripts/NN_*.py` works directly; ad-hoc snippets that `import models` must run
  with the repo root as CWD (or `PYTHONPATH=.`).
- Tests (fast, synthetic data, no dataset or GPU needed):
  `.venv/bin/python -m unittest discover -s tests -v`.
- No linter is configured. The closest available syntax check is
  `.venv/bin/python -m compileall models scripts tests`.
- The large medical dataset (`mr-ct-data/`, `derivatives/`) and frozen U-Net
  checkpoints are `.gitignore`d and are NOT present in this environment. The
  data-driven stages (`scripts/03`, `04`, `06`, `08`, `11`, `12`) therefore
  cannot run end-to-end here without first providing that data. The core
  inference/model code is fully exercisable without data via the test suite and
  by importing `models.metacog` / `models.cvae` directly.
- `torch` here is a CPU-only build (no GPU). Training scripts default to CPU and
  expose smoke-test flags (`--overfit N`, `--device cpu`) but still require the
  2D dataset bundles under `derivatives/dataset_2d_filtered/`.
- Experiment tracking (Weights & Biases) is opt-in: training scripts only log to
  W&B when passed `--wandb`, so default/smoke runs need no credentials.
- Stage 0 bone-GT generation uses the Node package `biswebnode`
  (`npm install -g biswebnode`; Node is available) and is only needed to
  regenerate bone masks from CT — not required for the U-Net / MetaCOG / C-VAE
  code paths.
