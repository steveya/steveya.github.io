---
layout: post
title: "Volatility Forecasts (Part 2 - XGBoost-STES)"
date: 2024-07-18
categories: [Quantitative Finance]
tags: [quantitative-finance, volatility-forecast, machine-learning, research]
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

(Code snippets included in this posts can be found in the [Github repository](https://github.com/steveya/volatility-forecast/notebook/random_forest_ewma.ipynb).) 

## Table of Contents

1. [Recap](#recap)
2. [XGBoost-STES](#xgboost-stes)
3. [Results](#results)
4. [Wrapping Up](#wrapping-up)


## Recap
In the [previous post]({% post_url 2024-07-12-volatility-forecasts-1 %}) on volatility forecasting, we replicated the Smooth Transition Exponential Smoothing (STES) model as in [(Taylor 2004)](https://doi.org/10.1016/j.ijforecast.2003.09.010) and [(Liu, Taylor, Choo 2020)](https://doi.org/10.1016/j.econmod.2020.02.021), and observed similar performance gain relative to simple Exponential Smoothing (ES) on the out-sample Root Mean Squared Error (RMSE) on both simulated contaminated GARCH and S&P 500 returns. While our results were consistent with those of the authors in our limited replication study, we were curious if the results can be further improved with different models using the same features $$X_t$$ (lag return, lag squared return, and lag absolute return.) 

One of the reasons we revisit this topic after all this time is the paper from [(Coulombe 2020)](https://arxiv.org/abs/2006.12724) on using random forest to estimate the parameters of a linear regression model. [(Coulombe 2020)](https://arxiv.org/abs/2006.12724) notices that many of the traditional econometric models accommodate regimes or non-linearity using regime shift (Markov switching), time-varying parameters (TVP), or thresholds. The author shows that many of these variants express the $$\beta$$ coefficients as functions of other variables or as some processes and can be adequately captured by decision trees. The author develops a customized tree model that predicts the $$\beta$$ coefficients in the regression:

$$
\begin{equation}
\begin{aligned}
y_t &= X_t \beta_t + \epsilon_t \\
\beta_t &= \mathcal{F}(X_t) \\
\end{aligned}
\end{equation}
$$

where $$\mathcal{F}$$ is the customized random forest that maps the features $$X_t$$ to the coefficients $$\beta_t$$. The customized tree model uses a bagging technique appropriate for time-series data and adds temporal smoothness regularization to the predicted $$\beta_t$$.

## XGBoost-STES

This paper interested me in replacing the linear equation within the sigmoid function in the STES with a tree ensemble model. Recall that the STES has the following form:

$$
\begin{equation}
\begin{aligned}
\sigma_t^2 &= \alpha_t r_{t-1}^2 + (1-\alpha_t)\widehat{\sigma}_{t-1}^2 \\ 
\alpha_t &= \mathrm{sigmoid}\left(X_t^\top\beta\right) \\
\end{aligned}
\end{equation}
$$

By replacing the linear equation $$X^\top\beta$$ with a tree ensemble model, we can perhaps better separate the feature space into regions where recent realized volatility plays a more critical role in forecasting future realized volatility from regions where long-term historical realized volatility is a better forecast. Volatility time series exhibit non-linear behaviour, and several models already use modern machine learning architectures (such as the Neural Network Heterogeneous Autoregressive Model). While we cannot do an exhaustive review, I'd like to know how my change will improve (or otherwise) the out-of-sample performance on the S&P 500 realized variance prediction error.

The XGBoost-STES model is formulated as

$$
\begin{equation}
\begin{aligned}
\sigma_t^2 &= \alpha_t r_{t-1}^2 + (1-\alpha_t)\widehat{\sigma}_{t-1}^2 \\ 
\alpha_t &= \mathrm{sigmoid}\left(\mathcal{F}(X_t)\right) \\
\end{aligned}
\end{equation}
$$

It aims to minimize the mean squared error (MSE) between the forecasted variance $$\sigma_t^2$$ and the realized squared returns.

$$
\mathtt{loss} = \frac{1}{T}\sum_{t=1}^{T}\left(r_t^2 - \widehat{\sigma}_t^2\right)^2
$$

For each time step $$t$$, the forecasted variance $$\sigma_t^2$$ is a weighted combination of the previous period's squared return $$r_{t-1}^2$$ and the last period's forecasted variance $$\widehat{\sigma}_{t-1}^2$$, modulated by $$\alpha_t$$, which is the output of the XGBoost model passed through a sigmoid function to constrain it between 0 and 1.

To keep the model simple, we do not impose any temporal smoothness regularization as (Coulombe 2020) does, nor do we employ random time-series sampling in the training process. The XGBoost model is used for this study instead of a random forest.

### Implementation of XGBoost-STES
As we do not predict the label (1-day ahead realized variance) directly but instead transform the predictions by the sigmoid function to produce $$\alpha$$s, from which the 1-day ahead variance forecasts are recursively computed from contiguous blocks of the feature matrix, we need to implement our own custom objective function and time-series sample generator.

#### Data Preparation
The training data is a Pandas DataFrame with the following columns:

- `returns`: the log-returns of the asset at time `t-1`
- `abs(returns)`: the absolute value of log-returns of the asset at time `t-1`
- `returns^2`: the squared log-returns of the asset at time `t-1`
- `labels`: the 1-day ahead realized variance of the asset at time `t`

The `returns^2` column is used both as a feature in the XGBoost model and in the recursive calculation of the 1-day ahead variance forecasts. Due to the 1-day offset, the `t` index of the `returns^2` column corresponds to the squared return at time `t-1`. As a result, any training data must contain the `returns^2` column. We do not test the case where it is not a feature in the XGBoost model.

#### Objective Function: Computing the Gradient and Hessian for the XGBoost-STES Model
The XGBoost allows the users to pass in their own objective function of the signature `obj: pred, dtrain -> grad, hess`. The gradient and Hessian are crucial components that guide the optimization process during training. The gradient represents the direction and rate of the steepest ascent of the loss function, and the Hessian measures the curvature of the loss function, providing information on how the gradient should be adjusted during optimization.

The gradient $$\text{grad}_t$$ at each time step is computed as 

$$
\begin{equation}
\begin{aligned}
\text{grad}_t &=2 \times (r_t^2 - \widehat{\sigma}_t^2) \times \frac{\partial \widehat{\sigma}_t^2}{\partial \text{preds}_t} \\
&= 2 \times (r_t^2 - \widehat{\sigma}_t^2) \times \left[\left(r_{t-1}^2 - \widehat{\sigma}_{t-1}^2 \right) \times \alpha_{t-1} \times \left(1 - \alpha_{t-1}\right)\right] \\
\end{aligned}
\end{equation}

$$

where $$\partial \widehat{\sigma}_t^2/\partial \text{preds}_t$$ is the derivative of the forecasted variance with respect to the model's raw predictions. The Hessian $$\text{hess}_t$$ is computed as 

$$
\begin{equation}
\begin{aligned}
\text{hess}_t &= 2 \times \left( \frac{\partial \widehat{\sigma}_t^2}{\partial \text{preds}_t} \right)^2 \\
&= 2 \times \left[\left(r_{t-1}^2 - \widehat{\sigma}_{t-1}^2 \right) \times \alpha_{t-1} \times \left(1 - \alpha_{t-1}\right)\right]^2 \\
\end{aligned}
\end{equation}
$$

and is the second-order derivative that provides the necessary information for refining the gradient descent steps. 

The objective function is implemented using the formula derived above:

```python
def stes_variance_objective(self, preds, train):
    labels = dtrain.get_label()
    alphas = expit(preds)
    
    returns2 = self.data['returns^2'].to_numpy()
    
    grads = np.zeros_like(preds)
    hesss = np.zeros_like(preds)
    varhs = np.zeros_like(preds)

    assert len(preds) == len(labels), "Mismatch between lengths of preds, returns, and labels"

    for t in range(len(alphas)):
        if t == 0:
            lvar_f = np.mean(returns2[:500])
            varhs[t] = alphas[t] * returns2[t] + (1 - alphas[t]) * lvar_f
            d_alpha = returns2[t] - lvar_f
        else:
            varhs[t] = alphas[t] * returns2[t] + (1 - alphas[t]) * varhs[t-1]
            d_alpha = returns2[t] - varhs[t-1]
        
    d_pred = -alphas[t] * (1 - alphas[t]) * d_alpha
    grads[t] = 2 * (labels[t] - varhs[t]) * d_pred
    hesss[t] = 2 * d_pred**2

    return grads, hesss
```

Since we are not using customized time-series subsampling during training for this model, it can be trained by calling `xgboost.train` with the `obj` argument set to our objective function above:

```python
def fit(self, X, y):
    assert 'returns^2' in X, "returns^2 must be in the features!"
    
    self.data = X.copy()
    dtrain = xgb.DMatrix(X, label=y)
    
    self.model = xgb.train(
        self.xgb_params, dtrain, 
        num_boost_round=self.num_boost_round, 
        obj=self.stes_variance_objective, 
        evals=[(dtrain, 'train')], verbose_eval=False
    )

    if self.model is None:
        raise RuntimeError("Model training failed and no model was created.")
    
    return self.model
```

We fit the model on the same data as in my (now updated) [last post](https://steveya.github.io/posts/volatility-forecast-1/). The results are presented in the next section.

## Results
### XGBoost-STES vs STES on Simulated Returns
Table 1 shows the out-of-sample RMSE of the XGBoost-STES model on simulated returns compared with simpler models. The model performs poorly on the simulated GARCH data against the STES-E&AE&SE model that has the same set of transition variables. It is not surprising as the GARCH model is a constant parameter model with some added outliers. On simulated data like this, the more complex models (that deviate from the ground truth) cannot outperform the simpler model (ES) that is closer to the ground truth. 

| Model | RMSE | (Liu et al 2020) |
| --- | --- | --- |
| STES-E&AE&SE | 2.80 | N.A |
| STES-SE  | 2.82 | 2.44 |
| ES       | 2.85 | 2.45 |
| STES-AE  | 2.88 | 2.43 |
| XGB-STES | 2.91 | N.A  |
| STES-E&AE | 2.91 | N.A |
| STES-E&SE | 2.93 | N.A |
| STES-AE&SE | 2.94 | N.A |


[Table 1: Comparison of the STES and ES models on simulated data ($$\eta = 4$$).]

### STES vs ES on SPY Returns
On SPY data however, XGBoost-STES outperforms the simpler models by quite a bit. However, depending on how we tune the model, the test results can vary. When untuned (with some default parameters), the XGBoost-STES model outperforms the rest (perhaps by luck.) When tuned with either `RandomizedSearchCV` from `sklearn.model_selection` or the SOTA HPO library `optuna`, XGBoost-STES performs worse than the simple STES model. It highlights the problem as we move from simpler models such as STES to more complex ML models.

| Model | Test RMSE | Train RMSE |
| --- | --- | --- |
| XGB-STES (Untuned)  | 4.37e-04 | 5.01e-04 |
| STES-E&AE&SE | 4.49e-04 | 4.92e-04 |
| STES-AE&SE   | 4.50e-04 | 4.96e-04 |
| STES-E&SE    | 4.50e-04 | 4.93e-04 |
| STES-E&AE    | 4.52e-04 | 4.95e-04 |
| ES           | 4.64e-04 | 4.99e-04 |
| XGB-STES (Sklearn Tuned) | 4.78e-04 | 5.02e-04 |
| XGB-STES (Optuna Tuned)  | 4.86e-04 | 4.88e-04 |

[Table 2: Comparison of the STES and ES models on SPY returns. Since my sample and data differ from the authors, their results are not listed. Train sample: 2000-01-01 - 2015-11-26, Test sample: 2015-11-27 - 2023-12-31.]

## Wrapping Up
We can do other interesting things to diagnose and understand this model, such as doing feature importance analysis. However, given that we have just 3 features, we will leave that for future posts. We have successfully created a simple ML-based STES hybrid model that outperforms the simpler STES and ES models. Both XGBoost-STES and STES models can be extended to include more variables (such as macro variables, calendar variables, and other time-series features) and other model features (adding regularization to the simple STES models as the number of features increases.) These will be the main focus of future posts in this series. In the [next post]({% post_url 2024-10-02-volatility-forecasts-3 %}) we will tie the ES and STES models with more modern neural network models. Stay tuned.

