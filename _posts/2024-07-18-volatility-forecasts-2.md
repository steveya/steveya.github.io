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
4. [Conclusion](#conclusion)


## Recap
In the [previous post](https://steveya.github.io/posts/volatility-forecast-1/) on volatility forecasting, I replicated the Smooth Transition Exponential Smoothing (STES) model as in [(Taylor 2004)](https://doi.org/10.1016/j.ijforecast.2003.09.010) and [(Liu, Taylor, Choo 2020)](https://doi.org/10.1016/j.econmod.2020.02.021), and observed similar performance gain relative to simple Exponential Smoothing (ES) on the out-sample Root Mean Squared Error (RMSE) on both simulated contaminated GARCH and S&P 500 returns. While our results were consistent with those of the authors in our limited replication study, I was curious if I could improve the results further with the same features $$S_t$$ (lag return, lag squared return, and lag absolute return) but with different models. 

One of the reasons I revisit this topic after all this time is the paper from [(Coulombe 2020)](https://arxiv.org/abs/2006.12724) on using random forest to estimate the parameters of a linear regression model. [(Coulombe 2020)](https://arxiv.org/abs/2006.12724) notices that many of the traditional econometric models accomodate regimes or non-linearity using regime shift (Markov switching), time-varying parameters (TVP), or thresholds. The author shows that many of these variants express the $$\beta$$ coefficients as functions of other variables or as some processes, and can be adequately captured by decision trees. The author develops a customized tree model that predicts the $$\beta$$ coefficients in the regression

$$
\begin{equation}
\begin{aligned}
y_t &= X_t \beta_t + \epsilon_t \\
\beta_t &= \mathcal{F}(S_t) \\
\end{aligned}
\end{equation}
$$

where $$\mathcal{F}$$ is the customized random forest that maps the features $$S_t$$ to the coefficients $$\beta_t$$. The customized tree model uses bagging technique appropriate for time-series data, and adds temporal smoothness regulariztion to the predicted $$\beta_t$$.

## XGBoost-STES

This paper got me interested in replacing the linear equation within the sigmoid function in the STES with a tree ensemble model. Recall that the STES has the following form

$$
\begin{equation}
\begin{aligned}
\sigma^2_{t} &= \alpha_t r^2_{t-1} + (1-\alpha_t)\widehat{\sigma}^2_{t-1} \\ 
\alpha_{t} &= g\left(S_t^\top\beta\right) \\
\end{aligned}
\end{equation}
$$

By replace the linear equation $$S^\top\beta$$ with a tree ensemble model, we can perhaps better separate the feature space into regions where recent realized volatility plays a more important role in forecasting future realized volatility from regions where long-term historical realized volatility is a better forecasts. Volatility time series is known to exhibit non-linear behaviour and there are several models out there that models this behaviour using modern machine learning architectures already (such as Neural Network Heterogeneous Autoregressive Model). I cannot do a exhaustive review, just curious about how my change will improve (or otherwise) the out-of-sample performance on the S&P 500 realized variance prediction error.

The XGBoost-STES model is formulated as

$$
\begin{equation}
\begin{aligned}
\sigma^2_{t} &= \alpha_t r^2_{t-1} + (1-\alpha_t)\widehat{\sigma}^2_{t-1} \\ 
\alpha_{t} &= g\left(\mathcal{F}(S_t)\right) \\
\end{aligned}
\end{equation}
$$

It aims to minimize the mean squared error (MSE) between the forecasted variance $$\sigma_t^2$$ and the realized squared returns.

$$
\mathtt{loss} = \frac{1}{T}\sum_{t=1}^{T}\left(r^2_{t} - \widehat{\sigma}^2_{t}\right)^2
$$

For each time step $$t$$, the forecasted variance $$\sigma_t^2$$ is a weighted combination of the previous period's squared return $$r_{t-1}^2$$ and the previous period's forecasted variance $$\widehat{\sigma}_{t-1}^2$$, modulated by $$\alpha_t$$, which is the output of the XGBoost model passed through a sigmoid function to constrain it between 0 and 1.

To keep the model simple, I do not impose any temporal smoothness regularization as (Coulombe 2020) does, nor do I employ random time-series sampling in the training process. The XGBoost model is used for this study instead of a random forest

### Implementation of XGBoost-STES
As we do not predict the label (1-day ahead realized variance) directly, but instead transform the predictions by the sigmoid function to produce $$\alpha$$s, from which the 1-day ahead variance forecasts are recursively computed from contiguous blocks of the feature matrix, we need to implement out own custom objective function and time-series sample generator.

#### Data Preparation
The training data is a Pandas DataFrame with the following columns:

- `returns`: the log returns of the asset on day `t-1`
- `abs(returns)`: the absolute value of log returns of the asset on day `t-1`
- `returns^2`: the squared log returns of the asset on day `t-1`
- `labels`: the 1-day ahead realized variance of the asset on day `t`

The `returns^2` column is used both as a feature to the XGBoost model and in the recursive calculation the 1-day ahead variance forecasts. Due to the 1-day offset, the `t` index of `returns^2` column corresponds to the squared return on day `t-1`. Also, as a result, any training data must contain `returns^2` column. I am not testing the case where it is not a feature to the XGBoost model.

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
def stes_variance_objective(self, preds, dtrain):
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

I fit the model on the same data as in my (now updated) [last post](https://steveya.github.io/posts/volatility-forecast-1/). The results are presented in the next section.

## Results
### XGBoost-STES vs STES on Simulated Returns
Table 1 shows the out-of-sample RMSE of the XGBoost-STES model on simulated returns compared with other simpler models. The model performs poorly on the simulated GARCH data, and this is not surprising as the GARCH model is a constant parameter model with some added outliers. On data like this, the more complex models (that deviate from the ground truth) cannot outperform the simpler model (ES) that is closer to the ground truth. 

| Model | RMSE | (Liu et al 2020) |
| --- | --- | --- |
| STES-AE  | 2.85 | 2.43 |
| ES       | 2.85 | 2.45 |
| STES-SE  | 2.88 | 2.44 |
| XGB-STES | 3.00 | N.A  |

[Table 1: Comparison of the STES model and the ES model on simulated data ($$\eta = 4$$).]

### STES vs ES on SPY Returns
On SPY data however, XGBoost-STES outperforms the simpler models by quite a bit, but with a catch: depending on how one tune the model, the test results can vary greatly. When untuned (with some default parameters) The XGBoost-STES model performs the best (perhaps by luck.) When tuned using `RandomizedSearchCV` from `sklearn.model_selection`, the XGBoost-STES model performs slightly worse than the STES model, but still outperforms the simpler STES models. When I switch to the SOTA HPO library `optuna`, XGBoost-STES performs way worse than the STES model. It is likely my fault but highlight the problem as we move from simpler models such as STES to more complex ML models.

| Model | Test RMSE | Train RMSE |
| --- | --- | --- |
| XGB-STES (Untuned)  | 4.41e-04 | 5.01e-04 |
| XGB-STES (Sklearn Tuned)  | 4.43e-04 | 5.20e-04 |
| STES-AE&SE | 4.49e-04 | 4.96e-04 |
| STES-E&SE  | 4.50e-04 | 4.95e-04 |
| STES-E&AE  | 4.52e-04 | 4.93e-04 |
| ES         | 4.64e-04 | 4.99e-04 |
| XGB-STES (Optuna Tuned)  | 6.92e-04 | 6.93e-04 |

[Table 2: Comparison of the STES model and the ES model on SPY returns. Since the sample and data I use I different from the authors, their results are not listed. Train sample: 2000-01-01 - 2015-11-26, Test sample: 2015-11-27 - 2023-12-31.]

## Wrapping Up
There are other interesting things we can do to diagnose and understand this model, such as doing feature importance analysis. However, given that we have just 3 features at the moment, I will skip that for now. We have successfully created a simple ML-based STES hybrid model that outperforms the simpler STES and ES models. Both XGBoost-STES and STES models can be extended to include more variables (such as macro variables, calendar variables, and other time-series features) as well as other model feautures (adding regularization to the simple STES models as the number of features increases.) These will be the main focus of future posts in this series.

