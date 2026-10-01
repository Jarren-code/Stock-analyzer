# Project core: volatility forecasting with LSTM + OLS hybrid model

Python implementation of a hybrid LSTM + linear model for 5-day volatility forecasting; trained using RMSE and tested against persistence, HAR-RV, and OLS baselines.

## About project

The model predicts the ratio of forward realized volatility to trailing realized volatility, using log to minimize outliers' effect. A linear path (initialized from OLS coefficients) is combined with an LSTM path via simple addition. Evaluation uses 20-seed ensembling and block bootstrap confidence intervals to test statistical significance against multiple baselines.

## Setup

1. Create and activate a virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Open `model.ipynb` in Jupyter Notebook
4. Run all cells

## Method

- **Task**: 5-day forward realized volatility, expressed as a log-ratio to trailing volatility
- **Model**: hybrid LSTM + linear path (linear path initialized from OLS coefficients, then fine-tuned)
- **Baselines**: persistence (predict ratio = 1), HAR-RV (`rv_20 * sqrt(H) / trailing_vol_H`), OLS
- **Features**: `log_ret`, `rv_20`, `vol_z`, `mom_5`, plus FinBERT sentiment features (sent_mean, sent_std, sent_count)
- **Evaluation**: 20-seed ensembling with block bootstrap confidence intervals on RMSE and quantile-AUC

## Results

### Example run: NTES

### Average RMSE (lower = better)

| Model | RMSE |
|-------|------|
| Model | 0.57346 |
| HAR-RV | 0.57421 |
| OLS | 0.57057 |
| Persistence | 0.65788 |

### Average AUC (higher = better)

| Model | AUC@0.75 | AUC@0.90 |
|-------|----------|----------|
| Model | 0.779 | 0.746 |
| HAR-RV | 0.776 | 0.804 |
| OLS | 0.776 | 0.740 |
| Persistence | 0.500 | 0.500 |

### Significance tests

**vs. persistence baseline:**
- RMSE improvement: **+0.086** (95% CI: [0.045, 0.114]) — significant
- AUC@0.75 improvement: **+0.283** (95% CI: [0.221, 0.335]) — significant
- AUC@0.90 improvement: **+0.249** (95% CI: [0.164, 0.305]) — significant

**vs. HAR-RV baseline:**
- RMSE improvement: +0.002 (95% CI: [-0.050, 0.050]) — not significant
- AUC@0.75 improvement: +0.006 (95% CI: [-0.059, 0.066]) — not significant
- AUC@0.90 improvement: -0.055 (95% CI: [-0.155, 0.031]) — not significant

## Key findings

- **Beats the persistence baseline significantly** on RMSE (+12.8%) and AUC (+0.28 at 75th percentile, +0.25 at 90th percentile)
- **Matches OLS and HAR-RV** on RMSE; differences are not statistically significant

## SHAP Feature Importance

SHAP analysis on the test set (300 windows) shows the model relies primarily on `vol_z`, `mom_5`, and `log_ret`:

| Feature | Mean \|SHAP\| |
|---------|--------------|
| vol_z | 0.02442 |
| mom_5 | 0.02086 |
| log_ret | 0.00881 |
| rv_20 | 0.00090 |
| sent_mean | 0.00000 |
| sent_std | 0.00000 |
| sent_count | 0.00000 |

## Terminologies

- **`hidden_dim`**: size of the LSTM's hidden state (working memory)
- **`num_layers`**: number of stacked LSTM layers
- **`Std`**: standard deviation across 20 random seeds
- **`AUC@0.75`**: area under the ROC curve for identifying the top 25% of volatility days
- **`AUC@0.90`**: area under the ROC curve for identifying the top 10% of volatility days

## Design decisions

- **Convex combination was tested: `α · LSTM + (1-α) · linear` converged to α ≈ 0.49 across all 20 seeds regardless of initialization, and produced worse RMSE and AUC than simple addition. Because of that, I changed the final model to use `LSTM + linear`.
- **Cached parquets are provided** for prices and sentiment to make the repo immediately runnable without hitting Yahoo Finance rate limits. To regenerate, delete `data/*.parquet` and re-run.


## Example output (NTES)

```
Preparing data for NTES (horizon=5 days)
Train windows: 2816, Val windows: 474, Test windows: 808
Training linear-regression baseline (same inputs)


===MULTI-SEED EVALUATION===
  epoch    0  train_loss 0.714992  val_loss 0.950838  (best 0.950838, no_improve 0)
  Early stopping at epoch 29 (no val improvement for 20 epochs)
  seed 0: rmse_improvement=+0.08502  auc@0.75=0.775  auc@0.9=0.739
  ...
  epoch    0  train_loss 0.711876  val_loss 0.987143  (best 0.987143, no_improve 0)
  Early stopping at epoch 31 (no val improvement for 20 epochs)
  seed 19: rmse_improvement=+0.08595  auc@0.75=0.777  auc@0.9=0.741

======================================================================
MULTI-SEED SUMMARY (20 runs)
======================================================================
RMSE improvement over persistence: mean=+0.08442, std=0.00271, positive in 20/20 runs
AUC@0.75: mean=0.779, std=0.003, above 0.5 in 20/20 runs
AUC@0.9: mean=0.746, std=0.006, above 0.5 in 20/20 runs

Consistent across most seeds on at least one metric, more evidence of real skill.

============================================================
Report (20 seeds)
============================================================
Average RMSE:
  Model:       0.57346
  HAR:         0.57421
  Linear:      0.57057
  Persistence: 0.65788

Average AUC for detecting top-quantile volatility regimes:
  Model => 75th quantile: 0.779, 90th quantile: 0.746
  HAR => 75th quantile: 0.776, 90th quantile: 0.804
  Linear => 75th quantile: 0.776, 90th quantile: 0.740
  Persistence => 75th quantile: 0.500, 90th quantile: 0.500

Average prediction variance ratio: 0.315
============================================================

Ensemble prediction stats: mean=0.0293, std=0.3672
=== model vs. HAR baseline ===

======================================================================
Significance Tests (block bootstrap, ensemble) vs. HAR baseline
======================================================================

PRIMARY -- RMSE improvement, model vs. HAR baseline:
  Observed improvement: 0.00243
  95% CI: [-0.05032, 0.05041]  (not significant (CI includes 0))

SECONDARY -- AUC improvement, model vs. HAR baseline (not just vs. chance):
  [quantile 0.75]
 Observed AUC diff: +0.006
 95% CI: [-0.059, +0.066]  (not significant)
  [quantile 0.9]
 Observed AUC diff: -0.055
 95% CI: [-0.155, +0.031]  (not significant)
=== model vs. persistence baseline ===

======================================================================
Significance Tests (block bootstrap, ensemble) vs. persistence baseline
======================================================================

PRIMARY -- RMSE improvement, model vs. persistence baseline:
  Observed improvement: 0.08610
  95% CI: [0.04527, 0.11418]  (significant (CI excludes 0))

SECONDARY -- AUC improvement, model vs. persistence baseline (not just vs. chance):
  [quantile 0.75]
 Observed AUC diff: +0.283
 95% CI: [+0.221, +0.335]  (significant)
  [quantile 0.9]
 Observed AUC diff: +0.249
 95% CI: [+0.164, +0.305]  (significant)

SHAP Global Feature Importance (mean |SHAP|)
  vol_z        0.02442
  mom_5        0.02086
  log_ret      0.00881
  rv_20        0.00090
  sent_mean    0.00000
  sent_std     0.00000
  sent_count   0.00000

Saved shap_importance.png
```