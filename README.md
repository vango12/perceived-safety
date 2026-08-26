# Perceived-safety assessment model

This repository contains the final, reproducible perceived-safety model used in the manuscript revision. It models the original integer perceived-safety score on a 1–10 scale with a cumulative ordered-probit model and evaluates generalization to unseen participants.

## Final model

The model uses actual robot motion speed, human–robot distance, a quadratic distance term and their interaction:

\[
\mathbf{x}_{ij}=\left[\omega_{ij},d_{ij},d_{ij}^{2},\omega_{ij}d_{ij}\right]^{\mathsf T},
\qquad \omega=1.5v_s\;\mathrm{rad/s}.
\]

Here, \(v_s\) is the speed-scaling factor used in the source experiment, \(\omega\) is the actual robot motion speed, and \(d\) is the end-effector-to-head distance. The speed-squared term is not included in the final model because it did not improve participant-independent prediction.

For scores \(Y\in\{1,\ldots,10\}\), the cumulative model is

\[
P(Y_{ij}\le k\mid\mathbf{x}_{ij})=\Phi(\tau_k-\eta_{ij}),\quad k=1,\ldots,9,
\]

with \(\eta_{ij}=\mathbf{x}_{ij}^{*\mathsf T}\boldsymbol\beta\). The model returns the probability of every score, the expected score and predictive uncertainty.

## Validation protocol

- Five-fold participant-independent outer cross-validation.
- Each outer fold trains on 24 participants (1,080 records) and tests on six unseen participants (270 records).
- Nine-shot personalization selects one record from each cell of the 3×3 speed–distance grid using speed and distance only.
- The remaining 36 records per held-out participant are used for personalization evaluation.
- Only a participant-specific latent intercept is updated during calibration; fixed effects and ordinal thresholds remain unchanged.

## Data

The public repository intentionally does not include the participant-level workbook. Place a copy of `实验数据记录表.xlsx` in `data/`, or pass its path with `--data`. The expected first five columns are participant ID, experiment group, speed scale factor, human–robot distance (m), and perceived-safety score (1–10). See [`data/README.md`](data/README.md).

## Run the final evaluation

```bash
python scripts/train_and_evaluate.py --data data/experiment_data.xlsx --output results
```

The script writes metrics, fold parameters, and optional prediction files to the selected output directory. Dependencies are listed in [`requirements.txt`](requirements.txt).

## Reproduced reference results

Using the supplied `实验数据记录表.xlsx`:

| Evaluation | Records | MAE | RMSE | RPS | NLL | Quadratic-weighted κ |
|---|---:|---:|---:|---:|---:|---:|
| Population model, participant-independent OOF | 1,350 | 1.1995 | 1.5299 | 0.08894 | 1.6418 | 0.8135 |
| Matched population model after nine-shot split | 1,080 | 1.2134 | 1.5444 | 0.08976 | 1.6486 | 0.8143 |
| Nine-shot personalized model | 1,080 | 0.9808 | 1.2648 | 0.07192 | 1.4827 | 0.8761 |

The speed conversion is a fixed linear rescaling of the original speed-scale variable. After training-data standardization, it leaves the fitted standardized model and validation metrics unchanged.
