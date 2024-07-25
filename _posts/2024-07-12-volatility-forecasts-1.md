---
layout: post
title: "Volatility Forecasts (Part 1 - Baseline Model)"
date: 2021-09-07
categories: [Quantitative Finance]
tags: [quantitative-finance, volatility-forecast, machine-learning, research]
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

#### Updates
(Code snippets included in this posts can be found in the [Github repository](https://github.com/steveya/volatility-forecast/notebook/random_forest_ewma.ipynb).) I have updated the data used in this post to the end of 2023 so that the results can be compared with the new model in [the next post](https://steveya.github.io/posts/volatility-forecast-2/).

## Table of Contents

1. [Introduction](#introduction)
2. [Smooth Transition Exponential Smoothing](#smooth-transition-exponential-smoothing)
3. [Results](#results)
4. [Conclusion and Next Steps](#conclusion-and-next-steps)

## Introduction
The exponential smoothing (ES) model is a popular yet simple volatility forecasting method used in finance and economics. It is formulated as

$$
\begin{equation}\label{eq:expsmooth}
    \hat{\sigma}^2_t = \alpha r^2_{t-1} + (1-\alpha)\hat{\sigma}^2_{t-1}
\end{equation}
$$

where $$\hat{\sigma}_t$$ is the estimated volatility at time $$t$$ and $$r_t$$ is the asset log returns at time $$t$$. It is essentially an exponential-weighted moving average of the squared log returns $$r_t^2$$. While technically it is the second non-central moment of the log-return distribution, in finance it is often treated as a variance estimate, assuming either $$\mathbb{E}\left[r_t\right] = 0$$ or is difficult to estimate precisely.

Exponential-weighted moving average (EWMA) is itself a popular smoorhing and time-series forecasting technique used in many fields, and within finance its application goes beyond volatility forecasting. Treated as smoother, the parameter $$\alpha$$ controls the degree of smoothing. Treated as a moving average, it controls the weight of the most recent observation. If the most recent observation is deems important, then it is given higher weight.

## Smooth Transition Exponential Smoothing
I came across a paper on Smooth Transition Exponential Smoothing (STES) [(Taylor, 2004)](https://doi.org/10.1016/j.ijforecast.2003.09.010) a few years ago. It started with the empirical observation that past return shocks exhibit asymmetric relationship with future realized volatility, and therefore one should assign more different weights to the recent observation depending on characteristics of the recent shock. The STES uses functions of past asset returns to help determine the value of $$\alpha$$. Specifically, STES is fomulated as 

$$
\begin{equation}\label{eq:stexpsmooth}
    \begin{aligned}
        \alpha_t &= \frac{1}{1+\exp\left(X_t \beta\right)} \\
        \hat{\sigma}^2_t &= \alpha_{t-1} r^2_{t-1} + (1-\alpha_{t-1})\hat{\sigma}^2_{t-1}
    \end{aligned}
\end{equation}
$$

in this formulation, $$\alpha_t$$ is no longer a constant as in the ES model, but is determined by the set of **transition variables** $$X_t$$, which can include a constant term and $$\lvert r_t \rvert$$ or $$r_t^2$$ that measures the magnitude of the most recent return, and/or $$r_t$$ which can introduce leverage effects into the model, among others. The auther went on to demonstrate that STES performed competitively against ES and other GARCH models in terms of 1-step forecast error for major equity indices.

Not being on a vol trade desk of any sort, my research interest was not about forecasting realized volatility. I wanted to tackle a different but related problem: how to quantitatively select the $$\alpha$$ parameter of a exponential smoothing model. The technique used in this paper not only addresses this problem in the context of vol forecasting, but also resonate with some other work I have seen before at work. At the end I chose another method that is more suited to the specific problems I had, but this paper left an impression on me. Years later in the present day, I saw its follow-up paper [(Liu, Taylor, Choo 2020)](https://doi.org/10.1016/j.econmod.2020.02.021) that extended the model to test if past trading volumn help forecast realized volatility and analyze the robustness of the STES model to outliers. I decide to see if I can replicate their results and maybe extend their model with other variables in the future.

## Results
### STES vs ES on Simulated Returns
First let us replicate some of their results from the (Liu et al 2020) paper. They horserace a set of volatility forecast models that include the ES model and the STES model by fitting their parameters on a training sample and compare their test-sample metrics that includes RMSE among others. They do this first on simulated GARCH time series contaminated by extreme outliers, and found that on the simulated data, the STES can already slightly outperform the ES model, and can better handle outliers. My result is listed in Table 1 with the their results. While I cannot reproduce their numbers, I actually do not think there is a reason why STES should outperform the ES model on the simulated data, as the groudtruth is a constant parameter GARCH model with some added outliers. I follow the author's model naming convention: STES-AE means that only the aboslute return (AE) is used as the transition variable. Similarly, STES-E&AE&SE contains both returns (E), absolute returns (AE) and squared returns (SE) as transition variables. 

| Model | RMSE | (Liu et al 2020) |
| --- | --- | --- |
| STES-AE | 2.85 | 2.43 |
| STES-SE | 2.88 | 2.44 |
| ES      | 2.85 | 2.45 |

[Table 1: Comparison of the STES model and the ES model on simulated data ($$\eta = 4$$).]

### STES vs ES on SPY Returns
When fitting the model to the SPY returns, STES has the potential to outperform the simple ES model. Table 2 shows the out-of-sample RMSE of the STES model on SPY's realized variance and confirms my prior belief.

| Model | Test RMSE | Train RMSE |
| --- | --- | --- |
| STES-E&AE&SE | 4.48e-04 | 4.98e-04 |
| STES-AE&SE   | 4.49e-04 | 4.96e-04 |
| STES-E&SE    | 4.50e-04 | 4.95e-04 |
| STES-E&AE    | 4.52e-04 | 4.93e-04 |
| ES           | 4.64e-04 | 4.99e-04 |

[Table 2: Comparison of the STES model and the ES model on SPY returns. Since the sample and data I use I different from the authors, their results are not listed. Train sample: 2000-01-01 - 2015-11-26, Test sample: 2015-11-27 - 2023-12-31]

While I cannot exactly replicate the results in Liu et al (2020) on the simulated time series, STES performs better on the SPY's data in terms of the out-of-sample RMSE, consistent with the author's calculations on different data sets. However, the gains  seem small, not order of magnitude better, and its net-of-cost performance as a trading strategy is untested. I will be curious to test this forecast as a vol trading strategy.

## Other Observations
The author observes that since STES is not a statistical model, they cannot conduct significance tests on the parameters. In both (Taylor 2004) and (Liu et al 2020) they draw interesting observations about the parameters of the fitted STES model. In particular, the fitted parrameters imply that the realized volatility responds to past shocks differently depending on the sign and the magnitude of the shock. Moreover, in terms of robustness to outliers, the STES also respond better by downweighting the weights on outliers. 

