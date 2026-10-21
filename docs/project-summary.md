# Manufacturability Screening — Project Summary

Self-directed technical exploration (LatentView internship, Industrials Offshore), Track B of a two-part project. Track A (`shape-similarity-search`) is a CAD similarity-search proof of concept using the same BRepNet backbone. This track reuses that backbone for a different downstream task: automated machining-feature recognition — given a CAD part, label which faces belong to which machined feature (pocket, hole, slot, chamfer, etc.).

This is a working document, not a polished pitch. It states the real result and its rough edges plainly, including where the approach didn't fully work.

## What was built

An end-to-end pipeline that takes a STEP file and predicts a machining-feature class for every face:

1. **Data**: MFCAD (15,488 machined-part STEP files with ground-truth per-face labels, 16 feature classes). All 15,488 files were validated (count match, correct label alignment — see below) before use; training used a verified random 1000-file subset (800 train / 200 val).
2. **Feature extraction**: BRepNet's own pipeline converts each STEP file into per-face/per-edge UV-grid geometric features.
3. **Model**: BRepNet's pretrained backbone (originally trained on Fusion 360 Gallery's unrelated 8-class segmentation task), entirely frozen, with a new trainable head for MFCAD's 16 classes.
4. **Training**: only the new head is trained (the backbone never updates), with a class-weighted loss to counter MFCAD's imbalance.
5. **Inspection**: `notebooks/inspect_predictions.ipynb` — colored 3D views (predicted vs. true class per face) plus a per-face table, for any validation file.

## Technique

- **Transfer learning, frozen backbone.** The backbone was never fine-tuned; only a small head sits on top of it. This is standard practice when adapting a pretrained model to a related-but-different task with limited labeled data, and it's cheap: training only ever touches a few thousand parameters, not the ~1M+ in the backbone.
- **Head architecture.** A plain linear layer was tried first as a baseline. The accepted result uses a small MLP instead (`84 → 128 → 16`, one ReLU hidden layer), which gave the largest single accuracy gain of anything tried.
- **Class-weighted cross-entropy loss.** MFCAD is heavily imbalanced — the non-feature "stock" class alone is ~30% of all faces. An unweighted loss caused the model to collapse onto predicting only 4-5 of the 16 classes, leaving most classes at 0% recall. Weighting the loss inversely to each class's training frequency fixed this.
- **A real bug found and fixed along the way**: freezing the backbone's *parameters* (`requires_grad=False`) does not freeze its BatchNorm layers' running statistics — those still drift during `.train()`-mode forward passes regardless of gradients. This was silently corrupting "frozen" backbone behavior during training until caught (by a training-log vs. fresh-reload accuracy mismatch) and fixed by keeping the whole model in `.eval()` mode throughout.

## The accepted result — real numbers

- **Checkpoint**: MLP-128 head, class-weighted loss, 1000-file subset. Trained for 40 epochs, but saved at its **best validation epoch (36 of 40)**, not the final one — see instability finding below for why.
- **Validation accuracy: 60.35%** (2,694 / 4,464 faces), independently verified by reloading the saved checkpoint fresh and re-running inference — not just read off the training log.
- All 16 classes get nonzero recall and nonzero prediction share; the earlier majority-class collapse is gone, and the accuracy gain is broad-based across classes, not concentrated in one or two.
- Per-class accuracy still varies widely: from **96.8%** (`rectangular_through_slot`) down to **15.1%** (`rectangular_blind_step`). Full breakdown, confusion patterns, and predicted-class distribution: `outputs/final/per_class_report.txt`.

## What was investigated, stated plainly

This project spent real effort chasing a specific problem — the MLP head's training instability — rather than just accepting the first working result. Full detail and every experiment's numbers: `experiments/EXPERIMENT_LOG.md`. Summary:

- **Class imbalance was the single biggest correctness fix.** Nothing else mattered until this was addressed.
- **More epochs reliably helped a *linear* head**, with no sign of overfitting even out to 60 epochs (44% → 49% → 52% accuracy across 15/40/60 epochs). A linear head is capacity-limited on this task, not time-limited.
- **Adding MLP capacity gave the biggest jump of anything tried** (roughly +7 to +11 percentage points over the best linear-head result), but introduced an instability the linear head never showed: validation accuracy peaks mid-training and then degrades, even while the loss curve looks smooth throughout. This was confirmed on two independent training runs (different random seeds), so it isn't a one-off fluke.
- **Two standard fixes were tried, in isolation, and neither cleanly solved it.** A 10x lower learning rate reduced the wobble substantially (~5x smaller epoch-to-epoch swings) but cost ~15 percentage points of accuracy within the same epoch budget — and that comparison is confounded, since a smaller learning rate also needs more epochs to reach the same point in training, so "more stable" can't be cleanly separated from "hadn't gotten there yet." Dropout (0.3) on the hidden layer cost nothing but did not meaningfully reduce the wobble by the more robust metric (average epoch-to-epoch swing was statistically unchanged from the untreated run).
- **The practical mitigation actually adopted: track validation accuracy every epoch and save the best-performing checkpoint, not the final one.** This is a workaround, not a fix. The underlying cause of the instability — plausibly just the interaction of a ~13K-parameter MLP, class-weighted loss, and this much noisy/imbalanced data in this many epochs — was not fully diagnosed.

## What this is not

- **Not production-ready.** 60% overall accuracy, with several classes still below 50% recall (some as low as 15%), is a proof-of-concept-grade result. It should not be used to make real manufacturability decisions.
- **Not validated beyond MFCAD.** MFCAD is a public academic dataset of relatively simple prismatic machined parts. No testing was done against more complex or client-representative geometry, and the feature taxonomy (16 fixed classes) is MFCAD's own, not derived from any specific manufacturing process or client's conventions.
- **The backbone was never fine-tuned** — only the small head was ever trained. Whether unfreezing some backbone layers would help (and by how much) is unexplored.

## How to reproduce or inspect this

- `scripts/` holds the canonical pipeline that reproduces the accepted result end to end, in order: `build_subset.py` → `extract_features.py` → `cache_labels.py` → `train_head.py --suffix C --epochs 40 --hidden-size 128 --save-best` → `report_per_class_stats.py`.
- `notebooks/inspect_predictions.ipynb` — pick any validation file by index or filename, see its predicted vs. true per-face classes as colored 3D views plus a table, using occwl's built-in per-face-coloring viewer (`occwl.jupyter_viewer.JupyterViewer.display`). Requires the `brepnet` conda kernel.
- `notebooks/verify_*.py` — the correctness proofs this project relied on (label-to-face alignment, extraction pipeline face order, BRepNet's internal batching permutation, backbone freeze verification).
- `experiments/` — every other variant tried, each with its own checkpoint, training curve, and a short verdict, plus `experiments/EXPERIMENT_LOG.md` summarizing all of it in one table.
- `outputs/final/` — the one accepted checkpoint, its full training history, and its per-class report.

## Screening and UI

Built on top of the accepted result: prediction on new STEP files, rule-based manufacturability flags, and a Streamlit app for interactive inspection.

- **New-part prediction** (`src/inference/predict_part.py`): runs the accepted checkpoint on an arbitrary STEP file. A malformed STEP file can crash pythonocc outright, not just raise a Python exception, so extraction, tessellation, and the face-adjacency graph all run in a subprocess (`src/inference/_extract_worker.py`); the main process only reads back plain arrays and a JSON edge list. A new part is standardized with the original training feature statistics, never statistics computed from the part itself.
- **Feature summary** (`src/inference/part_summary.py`): per class, face count, an approximate instance count (same-predicted-class faces that are adjacent are grouped via occwl's face-adjacency graph), mean confidence, and a reliability label (low/medium/high) read from the accepted checkpoint's own per-class validation recall.
- **Rules** (`src/screening/rules.py`): a fixed table covering all 16 MFCAD classes -- blind features flagged for depth control, pockets and blind slots flagged for internal corner radius, triangular/6-sided profiles flagged for acute corners, slanted/2-sided through-steps flagged for multi-axis setup, plain through-features and chamfer marked informational, stock unflagged. A flag fires when at least one face of that class is predicted at or above a confidence threshold (default 0.5, adjustable in the app). Classes the model is known to be unreliable on get their flag worded "possible" rather than stated outright.
- **App** (`app.py`): pick a validation-split part or upload a STEP file, see a colored 3D view (prediction or, for validation parts, ground truth for comparison), the feature summary table, and the flags.

**Verified, not assumed**: the single-file prediction pipeline was checked against the full validation split (`notebooks/check_predict_part_accuracy.py`) to confirm it reproduces the accepted checkpoint's ~0.6035 accuracy -- a gap here would mean the new pipeline's feature standardization or face ordering silently differs from the training pipeline's.

**Limitations, stated plainly:**

- Face-level accuracy is still 60%, the same proof-of-concept-grade result described above -- the screening layer doesn't improve on it, only exposes it as flags.
- Flags are feature-type checks only, with no dimensional or tolerance analysis (no actual tool-radius, depth, or clearance values are computed).
- Instance counts are approximate: faces of the same class are grouped only by geometric adjacency, which can under- or over-count real feature instances.

## Possible next steps (not started)

- Fine-tune some backbone layers now that a working head/loss/data pipeline exists, rather than always freezing the whole backbone.
- Investigate the MLP instability further — gradient clipping, a learning-rate schedule, or a larger validation set to reduce noise in the accuracy metric itself are the most obvious untried levers.
- Test against more complex or client-representative CAD geometry to see how much of this generalizes beyond MFCAD's relatively simple parts.
