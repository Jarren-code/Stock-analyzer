[Read the Code version of the README, don't look through the Preview]
#Goal of project

Predict stock price trends while explaining the reasoning behind the prediction. 

# Setup
1. Create and activate a virtual environment
2. Install dependencies:
   pip install -r requirements.txt
3. Run the backend:
   python web_backend.py
=======
Preparing data for NTES
Train windows: 2124, Test windows: 509
Training LSTM
  epoch    0  loss 1.026225
  epoch   25  loss 0.993016
  epoch   50  loss 0.989793
  epoch   75  loss 0.987511
  epoch  100  loss 0.986953
  epoch  125  loss 0.986254
  epoch  150  loss 0.984566
  epoch  175  loss 0.976828

Evaluating model

======================================================================
Metric                         Model    Baseline    Beats?
----------------------------------------------------------------------
rmse                         0.02494     0.02296        NO
mae                          0.01757     0.01624        NO
r2                          -0.18156     0.00000        NO
directional_accuracy         0.50884     0.51277        NO
======================================================================

Prediction variance / actual variance: 0.156

Beats baseline on 0/4 metrics.
Honest read: little to no evidence of real predictive skill at this horizon. Worth reporting as-is rather than tuning until the numbers look better -- that risks overfitting to this one test split rather than finding real signal.

Up-rate by prediction decile (0=lowest predicted, N=highest):
0    0.470588
1    0.509804
2    0.529412
3    0.470588
4    0.529412
5    0.420000
6    0.588235
7    0.509804
8    0.588235
9    0.509804
dtype: float64
>>>>>>> dae53c5 (model & README updated)
