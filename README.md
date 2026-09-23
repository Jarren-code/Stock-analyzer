[Read the Code version of the README, don't look through the Preview]
#Goal of project: Predict stock price trends while explaining the reasoning behind the prediction. 

# Setup
1. Create and activate a virtual environment
2. Install dependencies:
   pip install -r requirements.txt
3. Run the backend:
   python web_backend.py

What the model is currently doing: 

LSTM training & 5-day volatility prediction with evaluation metrics and significance testing


Baselines / tests:
  - Persistence: matched-horizon trailing realized vol.
  - Linear: same inputs as the LSTM, simplest possible model.
  - Regime discrimination: AUC (label = top-quantile realized vol) +
    Spearman, checked at TWO quantiles (0.75 and 0.90) -- a model can be
    bad at general ranking but still good at flagging rare extreme days,
    or vice versa; one threshold can hide that.
  - Significance (block bootstrap, respects autocorrelation from
    overlapping windows):
      PRIMARY:   RMSE improvement, model vs. persistence
      SECONDARY: AUC improvement, model vs. persistence (not just model
                 vs. chance -- beating 0.5 isn't the same as beating a
                 baseline that already has real discrimination power)
  - Multi-seed: distribution of both, across retrainings.


Current results:
| hidden_dim x num_layers| AUC@0.75 | Std | AUC@0.9 | Std | RMSE std |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 32 × 2 | 0.658 | 0.004 | 0.708 | 0.006 | 0.00279 |
| 8 × 1 | 0.700 | 0.029 | 0.723 | 0.031 | 0.01972 |
| 16 × 1 | 0.673 | 0.010 | 0.705 | 0.012 | 0.00549 |
| 24 × 1 | 0.664 | 0.005 | 0.700 | 0.009 | 0.00410 |


**Metrics**

1. hidden_dim = size of the LSTM's working memory (short-term & long-term)
2. num_layers = number of stacked LSTM layers.
3. AUC@0.75 = area under the ROC curve for identifying top 25% of volatility days.
4. AUC@0.90 = area under the ROC curve for identifying the top 10% of volatility days.
5. Std = standard deviation across 20 random seeds.
6. RMSE std = standard deviation of RMSE improvement over the persistence baseline.

