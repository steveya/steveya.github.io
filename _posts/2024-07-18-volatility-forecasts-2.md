---
layout: post
title: "WIP - Extending Smooth Transitioning Exponential Smoothing Volatility Forecasts (Part 2 - XGBoost)"
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

Similar to (Coulombe 2020), each tree is split over different contiguous subsamples. However, to keep this first model simple, I do not impose any temporal smoothness regularization as (Coulombe 2020) does, and the XGBoost model is used for this study instead of a random forest.

### Implementation of XGBoost-STES
As we do not predict the label (1-day ahead realized variance) directly, but instead transform the predictions by the sigmoid function to produce $$\alpha$$s, from which the 1-day ahead variance forecasts are recursively computed from contiguous blocks of the feature matrix, we need to implement out own custom objective function and time-series sample generator.

#### Objective Function: Computing the Gradient and Hessian for the XGBoost-STES Model
The XGBoost allows the users to pass in their own objective function of the signature `obj: pred, dtrain -> grad, hess`. The gradient and Hessian are crucial components that guide the optimization process during training. The gradient represents the direction and rate of the steepest ascent of the loss function, and the Hessian measures the curvature of the loss function, providing information on how the gradient should be adjusted during optimization.

The gradient $$\text{grad}_t$$ at each time step is computed as 

$$
\begin{equation}
\begin{aligned}
\text{grad}_t &=2 \times (\widehat{\sigma}_t^2 - \text{labels}_t) \times \frac{\partial \widehat{\sigma}_t^2}{\partial \text{preds}_t} \\
&= 2 \times (\widehat{\sigma}_t^2 - \text{labels}_t) \times \left[\left(\widehat{\sigma}_{t-1}^2 - r_{t-1}^2 \right) \times \alpha_t \times \left(1 - \alpha_t\right)\right] \\
\end{aligned}
\end{equation}

$$

where $$\partial \widehat{\sigma}_t^2/\partial \text{preds}_t$$ is the derivative of the forecasted variance with respect to the model's raw predictions. The Hessian $$\text{hess}_t$$ is computed as 

$$
\begin{equation}
\begin{aligned}
\text{hess}_t &= 2 \times \left( \frac{\partial \widehat{\sigma}_t^2}{\partial \text{preds}_t} \right)^2 \\
&= 2 \times \left[\left(\widehat{\sigma}_{t-1}^2 - r_{t-1}^2 \right) \times \alpha_t \times \left(1 - \alpha_t\right)\right]^2 \\
\end{aligned}
\end{equation}
$$

is the second-order derivative that provides the necessary information for refining the gradient descent steps. 

The default, XGBoost uniformly sample input data in the training process. It does not provide an interface to specify the sampling method. Therefore we need to sample the data for each training loop ourselves and pass in the indices of the samples to the objective function.

The training data is a Pandas DataFrame with the following columns:

- `returns`: the log returns of the asset on day `t-1`
- `abs(returns)`: the absolute value of log returns of the asset on day `t-1`
- `returns^2`: the squared log returns of the asset on day `t-1`
- `labels`: the 1-day ahead realized variance of the asset on day `t`

The `returns^2` column is used both as a feature to the XGBoost model and in the recursive calculation the 1-day ahead variance forecasts. Due to the 1-day offset, the `t` index of `returns^2` column corresponds to the squared return on day `t-1`. The objective function is therefore

```python
def stes_variance_objective(self, preds, dtrain, indices=None):
    labels = dtrain.get_label()
    alphas = expit(preds)
    
    if indices is None:
        returns2 = self.data['returns^2'].to_numpy()
    else:
        returns2 = self.data.iloc[indices]['returns^2'].to_numpy()

    grads = np.zeros_like(preds)
    hesss = np.zeros_like(preds)
    varhs = np.zeros_like(preds)

    assert len(preds) == len(returns2) == len(labels), "Mismatch between lengths of preds, returns2, and labels"

    for t in range(len(returns2)):
        if t == 0:
            lvar_f = np.mean(returns2)
            varhs[t] = alphas[t] * returns[t]**2 + (1 - alphas[t]) * lvar_f
            d_alpha = returns2[t] - lvar_f
        else:
            varhs[t] = alphas[t] * returns2[t] + (1 - alphas[t]) * varhs[t-1]
            d_alpha = returns2[t] - varhs[t-1]
        
        d_pred = alphas[t] * (1 - alphas[t]) * d_alpha
        grads[t] = 2 * (varhs[t] - labels[t]) * d_pred
        hesss[t] = 2 * d_pred**2

    return grads, hesss
```


#### Time Series Subsample Generator
The time series subsample generator is a simple implementation of the `iter` function of the Python iterator protocol. It generates a random contiguous subsample of the feature matrix for each training loop. The subsample size is determined by the `window_size` parameter, which is the number of time steps to include in the subsample. For each iteration, a random number is drawn between 0 and the maximum index of the feature matrix minus the window size to ensure that enough data is left to form a contiguous subsample of size `window_size`.

```python
class TimeSeriesDataIterator:
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self.max_index = len(data) - window_size
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.max_index < 0:
            raise StopIteration
            
        start_index = random.randint(0, self.max_index)
        end_index = start_index + self.window_size
        window_data = self.data[start_index:end_index]
        
        indices = list(range(start_index, end_index))
        return xgb.DMatrix(window_data.iloc[:, :-1], label=window_data.iloc[:, -1]), indices
```

#### Training the XGBoost-STES Model
The XGBoost-STES model is trained using the `xgb.train` function. The `params` argument is a dictionary of parameters to be passed to the XGBoost model. The `obj` argument is our custom objective function. 

We also implement the class according to `sklearn` interface in order to take advantage of the hyperparameter tuning and cross-validation functionalities provided by `sklearn`.

```python
def fit(self, X, y=None):
        assert 'label' in X
        assert 'returns^2' in X
        
        self.data = X.copy()
        dtrain = xgb.DMatrix(X)
        
        iterator = TimeSeriesDataIterator(self.data, int(self.ts_subsample * len(X)))

        self.model = None
        for _ in range(self.num_boost_round):
            dtrain, indices = iterator.next()

            if self.model is None:
                self.model = xgb.train(
                    self.xgb_params, dtrain, 
                    num_boost_round=5, 
                    obj=lambda preds, dtrain: self.stes_variance_objective(preds, dtrain, indices), 
                    evals=[(dtrain, 'train')], verbose_eval=False
                )
            else:
                self.model = xgb.train(
                    self.xgb_params, dtrain, 
                    num_boost_round=5, 
                    obj=lambda preds, dtrain: self.stes_variance_objective(preds, dtrain, indices), 
                    evals=[(dtrain, 'train')], 
                    xgb_model=self.model, verbose_eval=False
                )
        
        if self.model is None:
            raise RuntimeError("Model training failed and no model was created.")
        
        return self.model
```
## Results



## Conclusion