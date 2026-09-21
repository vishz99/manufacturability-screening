# Experiment log

Track B (manufacturability-screening) MFCAD segmentation head experiments.
The accepted result lives in `outputs/final/` (checkpoint, training
history, per-class report) and is reproduced end to end by the
`scripts/` pipeline. Everything below is preserved for reference; none
of it is needed to reproduce the accepted result.

| Folder | Scale | Config | Key result (val accuracy) | Verdict |
|---|---|---|---|---|
| `500_unweighted` | 500-file subset | Linear head, plain cross-entropy, 15 epochs | Not recorded (predates per-epoch history saving) | Superseded — collapsed onto majority classes |
| `500_class_weighted` | 500-file subset | Linear head, class-weighted CE (balanced), 15 epochs | 0.4216 overall, 0 zero-recall classes | Class weighting fixes the majority-class collapse; motivated moving to 1000-file scale |
| `1000_variant_A_linear_15ep` | 1000-file subset | Linear head, class-weighted CE, 15 epochs | 0.4406 overall | Baseline for isolating data-vs-epochs-vs-capacity; curve still improving at epoch 15 (underfit, no overfitting signal) |
| `1000_variant_B_linear_40ep` | 1000-file subset | Linear head, class-weighted CE, 40 epochs | 0.4946 overall | Still improving at epoch 40, loss gap stayed small (+0.027) — underfit, not plateaued |
| `1000_variant_B_continued_60ep` | 1000-file subset | Same linear head, continued 40→60 epochs (fresh optimizer state, see script docstring caveat) | 0.5213 final (epoch 60), best 0.5262 (epoch 59) | Confirmed still improving at 60 epochs, no overfitting (loss gap only grew to +0.033) — a linear head genuinely keeps learning slowly on this data |
| `1000_variant_C_mlp128` *(→ promoted to `outputs/final/`)* | 1000-file subset | MLP head (84→128→16, ReLU), class-weighted CE, 40 epochs, best-checkpoint saved | **0.6035 (epoch 36/40) — ACCEPTED RESULT** | Biggest single capacity gain (+7-11pp over B), broad-based across all 16 classes, but showed a peak-then-wobble instability pattern (val_acc regressed after epoch 36) — mitigated by saving the best-epoch checkpoint rather than the final one |
| `1000_variant_D_lowlr` | 1000-file subset | MLP-128, lr=1e-4 (10x lower than C), 40 epochs, best-checkpoint saved | 0.4512 (epoch 38/40) | Wobble reduced ~5x (max swing 0.0072 vs C's 0.0399) but at a severe cost — 15pp below C, hadn't converged in the same epoch budget. Confound: can't cleanly separate "more stable" from "hasn't reached the unstable regime yet" |
| `1000_variant_E_dropout` | 1000-file subset | MLP-128, dropout=0.3 after hidden layer, original lr=1e-3, 40 epochs, best-checkpoint saved | 0.6004 (epoch 40/40) | Performance on par with C, but wobble NOT meaningfully reduced by the robust metric (mean epoch-to-epoch swing 0.0134 vs C's 0.0132 — statistically unchanged). Final-epoch-happens-to-be-best reads as coincidence, not a fix |

## Overall findings, most to least significant

1. **Class weighting** (balanced inverse-frequency CE loss) was necessary to stop the head collapsing onto majority classes (`stock` ~30% of faces) — the single biggest correctness fix in this whole sequence.
2. **Doubling the dataset (500→1000 files)** gave a modest gain alone (+1.9pp at matched 15 epochs).
3. **More epochs** helped substantially for a linear head (15→40→60 epochs: 0.4406→0.4946→0.5213/peak 0.5262), never showing overfitting — the linear head is capacity-limited, not epoch-limited.
4. **MLP capacity (128 hidden units)** gave the single largest jump (B's 0.4946 → C's 0.6035 at best epoch), but introduced training instability (val accuracy peaks mid-run then wobbles) that a linear head never showed.
5. **Neither reduced learning rate nor dropout cleanly fixed the MLP's instability** within a fixed 40-epoch budget. Lower LR reduces the wobble but at a large, likely-confounded performance cost; dropout costs nothing but doesn't meaningfully change the wobble by the more robust metric. This is consistent with the instability being at least partly inherent to fitting a ~13K-parameter MLP on this much noisy, imbalanced data in this many epochs, rather than a simple hyperparameter fix.
6. **Best-checkpoint saving** (track validation accuracy per epoch, save the peak rather than the final epoch) is the practical mitigation actually adopted for the accepted result, verified reproducible via independent fresh reload multiple times throughout this project.

## Notes on what's NOT preserved

- No training-history JSON exists for `500_unweighted` or `500_class_weighted` — both predate the per-epoch history-saving feature added when the parameterized 1000-file training script was written. Only their final checkpoints and this log's summary of their behavior (from the original conversation) remain.
- `500_shared_data_prep/` holds the 500-file-scale subset list, split file, and data-prep scripts shared by both 500-file experiments (they used the same subset/split, differing only in loss weighting).
