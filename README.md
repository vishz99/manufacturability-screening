# Manufacturability Screening

Automated machining-feature recognition on CAD parts: given a STEP file, predict which machined feature (pocket, slot, chamfer, etc.) each face belongs to. Track B of a two-part project (Track A: `shape-similarity-search`). Uses a frozen, pretrained BRepNet backbone with a small trained head, on the MFCAD dataset.

See [docs/project-summary.md](docs/project-summary.md) for what was built, the real results (60.35% validation accuracy -- proof-of-concept grade), and the rough edges. [experiments/EXPERIMENT_LOG.md](experiments/EXPERIMENT_LOG.md) has every variant tried.

## Environment

This project uses **one** environment: the `brepnet` conda environment. There is no separate `.venv`. Everything here (`src/`, `scripts/`, `notebooks/`) imports occwl / pythonocc-core / PyTorch, which live only in that environment.

Setup:

1. Create the environment (definition in [environment.yml](environment.yml), captured from the real working environment):

   ```
   conda env create -f environment.yml
   conda activate brepnet
   ```

2. Install `occwl` (not published on PyPI, so it is deliberately not in `environment.yml`). The working environment has occwl 3.0.0 installed from source:

   ```
   pip install ./external/occwl
   ```

   `external/` is gitignored (vendored BRepNet + occwl copies), so on a fresh clone you must first restore it -- or install occwl from https://github.com/AutodeskAILab/occwl instead. The vendored BRepNet copy in `external/BRepNet` also contains patches for modern PyTorch Lightning, so use that copy rather than upstream.

3. To use the notebook, register the environment as a Jupyter kernel and select **Python (brepnet)** when opening `notebooks/inspect_predictions.ipynb`:

   ```
   python -m ipykernel install --user --name brepnet --display-name "Python (brepnet)"
   ```

Not verified: `environment.yml` has been checked to parse and to match the installed package list, but the environment has not been rebuilt from scratch from it.

## Layout

- `scripts/` -- the canonical pipeline that reproduces the accepted result end to end
- `experiments/` -- every other variant, each with its checkpoint and history
- `notebooks/` -- `inspect_predictions.ipynb` (visual inspection) and the `verify_*.py` correctness proofs
- `outputs/final/` -- the accepted checkpoint, its training history, and per-class report
- `src/` -- data loader and model code
