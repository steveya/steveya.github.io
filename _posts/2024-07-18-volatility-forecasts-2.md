---
layout: post
title: "(WIP) Extending Smooth Transitioning Exponential Smoothing Volatility Forecasts (Part 2 - XGBoost)"
date: 2024-07-18
categories: [Quantitative Finance]
tags: [quantitative-finance, volatility-forecast, machine-learning, research]
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>


## Table of Contents

1. [Recap](#recap)
2. [XGBoost-STES](#xgboost-stes)
3. [Results](#results)
4. [Conclusion](#conclusion)

## Recap
In the [previous post](https://steveya.github.io/posts/volatility-forecast-1/) on volatility forecasting, I replicated the STES and the ES model as in [(Taylor 2004)](https://doi.org/10.1016/j.ijforecast.2003.09.010) and [(Liu, Taylor, Choo 2020)](https://doi.org/10.1016/j.econmod.2020.02.021), and observe similar results regarding the better out-sample RMSE and robustness against outliers for the STES model. While the results were consistent with those of the authors, I was curious if I could improve the results further using the same features $$S_t$$ (lag return, lag squared return, and lag absolute return.) 

One of the reason I revisit this topic after so many years is [(Coulombe 2020)](https://arxiv.org/abs/2006.12724) on using random forest to estimate the parameters of a linear or predictive regression model. It made a point about how many of the traditional econometric models accomodate regimes by incorporating regime shift, time-varying parameters, or thresholds. The author shows can be adequately modeled using decision trees. The author developed a customized tree model that predicts the $$\beta$$ coefficients in the regression

$$
\begin{equation}
\begin{aligned}
y_t &= X_t \beta_t + \epsilon_t \\
\beta_t &= \mathcal{F}(S_t) \\
\end{aligned}
\end{equation}
$$

where $$\mathcal{F}$$ is a random forest that maps the features $$S_t$$ to the coefficients $$\beta_t$$. The customized tree model use bagging technique appropriate for time-series data, and adds temporal smoothness regulariztion to the predicted $$\beta_t$$.

## XGBoost-STES

This got me interested in replacing the linear equation within the sigmoid function in STES with a tree model. Recall that STES has the following form

$$
\begin{equation}
\begin{aligned}
\sigma^2_{t} &= \alpha_t r^2_{t-1} + (1-\alpha_t)\widehat{\sigma}^2_{t-1} \\ 
\alpha_{t} &= g\left(S_t\beta\right) \\
\end{aligned}
\end{equation}
$$

If I replace the linear equation with a tree model, I can perhaps better separate the feature space into regions where recent realized volatility plays a more important role in forecasting future realized volatility from regions where long-term historical realized volatility is a better forecasts.

In particular, I will be testing the following model

$$
\begin{equation}
\begin{aligned}
\sigma^2_{t} &= \alpha_t r^2_{t-1} + (1-\alpha_t)\widehat{\sigma}^2_{t-1} \\ 
\alpha_{t} &= g\left(\mathcal{F}(S_t)\right) \\
\end{aligned}
\end{equation}
$$

where $$\widehat{\sigma}^2_{t-1}$$ is the estimated volatility at time $$t-1$$ and $$r_t$$ is the asset log returns at time $$t$$. The $$S_t$$ still contains the three features from the previous returns, and $$\mathcal{F}$$ is a XGBoost model, and $$g$$ is a sigmoid function that restrict the coefficients from the XGBoost model to be between 0 and 1, the proper range for $$\alpha_t$$. The loss function remains the same and is the in-sample 1-day ahead RMSE. 

I decide to keep the first model simple by not imposing any temporal smoothness regularization as (Coulombe 2020) did, but similar to (Coulombe 2020), each tree is split over different contiguous subsamples. The XGBoost model is chosen instead of a random forest for this analysis, if the results are promising, I will maybe test a random forest model as well.

### Implementation of XGBoost-STES

#### Custom Objective Function
The core of the XGBoost-STES model is a custom objective function designed to minimize the root mean squared error (RMSE) between the forecasted variance and the realized squared returns. Since the model is not directly predicting the label (realized variance) but instead predicting $$\alpha_4$$ from which the variance forecast is recursively computed, a custom objective function is needed. 

Furthermore, since we are going to handle generating random contiguous subsamples ourselves, we need to pass the subsample indices to the objective function as well.

```python
def stes_variance_objective(self, preds, dtrain, indices=None):
    labels = dtrain.get_label()
    alphas = expit(preds)
    
    if indices is None:
        returns = self.data['returns'].to_numpy()
    else:
        returns = self.data.iloc[indices]['returns'].to_numpy()

    grads = np.zeros_like(preds)
    hesss = np.zeros_like(preds)
    varhs = np.zeros_like(preds)

    assert len(preds) == len(returns) == len(labels), "Mismatch between lengths of preds, returns, and labels"

    for t in range(len(returns)):
        if t == 0:
            lvar_f = np.var(returns)
            varhs[t] = alphas[t] * returns[t]**2 + (1 - alphas[t]) * lvar_f
            d_alpha = returns[t]**2 - lvar_f
        else:
            varhs[t] = alphas[t] * returns[t]**2 + (1 - alphas[t]) * varhs[t-1]
            d_alpha = returns[t]**2 - varhs[t-1]
        
        d_pred = alphas[t] * (1 - alphas[t]) * d_alpha
        grads[t] = 2 * (varhs[t] - labels[t]) * d_pred
        hesss[t] = 2 * d_pred**2

    return grads, hesss
```



## Results

## Conclusion