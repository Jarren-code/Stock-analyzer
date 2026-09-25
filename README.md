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


Current results:
| hidden_dim x num_layers| mean AUC@0.75 | Std | mean AUC@0.9 | Std | RMSE std |
| :--- | :--- | :--- | :--- | :--- | :--- |

| 24 × 1 | 0.732 | 0.003 | 0.746 | 0.004 | - |

Linear model (OLS)'s current mean RMSE: 0.7549

[Our model is closing the gap between its mean RMSE and the linear model's RMSE]

**NOTE:**

[Terminologies]

1. hidden_dim = size of the LSTM's working memory (short-term & long-term)
2. num_layers = number of stacked LSTM layers.
3. Std = standard deviation across 20 random seeds.
4. RMSE std = standard deviation of RMSE improvement over the persistence baseline.

[Clarifications]

1. AUC@0.75 = area under the ROC curve for identifying top 25% of volatility days.
2. AUC@0.90 = area under the ROC curve for identifying the top 10% of volatility days.
3. The mean AUC@0.75 and mean AUC@0.9 were taken from a sample size of 20 training runs.



[Significance Test Results]

PRIMARY -- RMSE improvement, model vs. persistence baseline:
  Observed improvement: 0.06434
  95% CI: [-0.00565, 0.12426]  (not significant (CI includes 0))

SECONDARY -- AUC improvement, model vs. persistence baseline (not just vs. chance):
  [quantile 0.75]
    Observed AUC diff: +0.232
    95% CI: [+0.170, +0.322]  (SIGNIFICANT)
  [quantile 0.9]
    Observed AUC diff: +0.246
    95% CI: [+0.168, +0.343]  (SIGNIFICANT)


