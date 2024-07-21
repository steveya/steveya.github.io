---
layout: post
title: "Extending Smooth Transitioning Exponential Smoothing Volatility Forecasts (Part 1 - Baseline Model)"
date: 2024-07-12
tags: [quantitative-finance, volatility-forecast, machine-learning, research]
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Introduction

(Code snippets included in this posts can be found in the [Github repository](https://github.com/steveya/steveya.github.io/blob/8a20c7552a82a586e9334eb12e2df500c9e95379/content/volatility_forecasts/random_forest_ewma.ipynb).)

The exponential smoothing (ES) model is a popular yet simple volatility forecasting method used in finance and economics. It is formulated as

$$
\begin{equation}\label{eq:expsmooth}
    \hat{\sigma}^2_t = \alpha r^2_{t-1} + (1-\alpha)\hat{\sigma}^2_{t-1}
\end{equation}
$$

where $$\hat{\sigma}_t$$ is the estimated volatility at time $$t$$ and $$r_t$$ is the asset log returns at time $$t$$. It is essentially an exponential-weighted moving average of the squared log returns $$r_t^2$$. While technically it is the second non-central moment of the log-return distribution, in finance it is often treated as a variance estimate, assuming either $$\mathbb{E}\left[r_t\right] = 0$$ or is difficult to estimate precisely.

Exponential-weighted moving average (EWMA) is itself a popular smoorhing and time-series forecasting technique used in many fields, and within finance its application goes beyond volatility forecasting. Treated as smoother, the parameter $$\alpha$$ controls the degree of smoothing. Treated as a moving average, it controls the weight of the most recent observation. If the most recent observation is deems important, then it is given higher weight.


## Smooth Transition Exponential Smoothing
I came across a paper on Smooth Transition Exponential Smoothing (STES) (Taylor, 2004) a few years ago. It started with the empirical observation that past return shocks exhibit asymmetric relationship with future realized volatility, and therefore one should assign more different weights to the recent observation depending on characteristics of the recent shock. The STES uses functions of past asset returns to help determine the value of $$\alpha$$. Specifically, STES is fomulated as 

$$
\begin{equation}\label{eq:stexpsmooth}
    \begin{aligned}
        \alpha_t &= \frac{1}{1+\exp\left(X_t \beta\right)} \\
        \hat{\sigma}^2_t &= \alpha_{t-1} r^2_{t-1} + (1-\alpha_{t-1})\hat{\sigma}^2_{t-1}
    \end{aligned}
\end{equation}
$$

in this formulation, $$\alpha_t$$ is no longer a constant as in the ES model, but is determined by the set of variables $$X_t$$, which includes a constant term and $$\lvert r_t \rvert$$ or $$r_t^2$$ that measures the magnitude of the most recent return, and/or $$r_t$$ which can introduce leverage effects into the model. The auther went on to demonstrate that STES performed competitively against ES and other GARCH models in terms of 1-step forecast error for major equity indices.

Not being on a vol trade desk of any sort, my research interest was not about forecasting realized volatility. I wanted to tackle a different but related problem: how to quantitatively select the $$\alpha$$ parameter of a exponential smoothing model. The technique used in this paper not only addresses this problem in the context of vol forecasting, but also resonate with some other work I have seen before at work. At the end I chose another method that is more suited to the problems I had, but this paper left an impression on me. Years later I saw its follow-up paper (Liu, Taylor, Choo 2020) that extended the model to test if past trading volumn help forecast realized volatility. I also came across other papers that model the regression coefficient as a random forest instead of a constant or a random walk. These together give me the idea of replacing the logistic function with a random forest. In this post I implement it to test how it performs on simulated and market data.

### Replication
First I want to see if I can replicate some of the simulation and market results from their 2020 paper. The authors first fit the ES model and the STES model (and other GARCH models) on a training sample of simulated GARCH time series contaminated by extreme outliers, and measure their performance on the test sample 1-step ahead root mean square forecast error. They found that on this simulated data, the STES can already slightly outperform the ES model in the presence of outliers. Repeating the author's experiement on 1000 runs of simulated contaminated GARCH time series, I found that one of the STES model using $$\lvert r_t \rvert$$ slightlyoutperformed the ES model on average.

Table 1: Comparison of the STES model and the ES model on simulated data.

| Model | RMSE |
| --- | --- |
| STES-AE | 2.845 |
| STES-SE | 2.878 |
| ES      | 2.852 |

Table 2: Comparison of the STES model and the ES model on market data.

| Model | RMSE |
| --- | --- |
| STES-AE&SE | 4.50e-04 |
| STES-E&AE | 4.52e-04 |
| STES-E&SE | 4.50e-04 |
| ES        | 4.64e-04 |

# Conclusion and Next Steps
I have replicated (somewhat) the results of the 2020 paper on STES and the ES model. The results are not exactly the same as the paper, but they are close and lead to the same conclusion about the relative performance of the two models. There are a lot more interesting observations from the STES model, in particular, how the fitted parrameters imply that the realized volatility responds to past shocks differently depending on the sign and the magnitude of the shock. In terms of robustness to outliers, the STES also respond better by downweighting the weights on outliers. The out-of-sample difference in error seem small, not order of magnitude. This is understandable, as we have not structurally changed the volatility forecast model; we only change the how the weights $$\alpha$$ are computed.

In the next post I will implement a STES model using XGBoost. 