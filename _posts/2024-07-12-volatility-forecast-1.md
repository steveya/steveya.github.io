---
layout: post
title: "Extending Smooth Transitioning Exponential Smoothing Volatility Forecasts (WIP)"
date: 2024-07-10
tags: [quantitative-finance, volatility-forecast, machine-learning, research]
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Introduction

The exponential smoothing (ES) model is a popular volatility forecasting method used in finance and economics. It is formulated as

$$
\begin{equation}\label{eq:expsmooth}
    \hat{\sigma}^2_t = \alpha r^2_{t-1} + (1-\alpha)\hat{\sigma}^2_{t-1}
\end{equation}
$$

where $$\hat{\sigma}_t$$ is the estimated volatility at time $$t$$ and $$r_t$$ is the asset log returns at time $$t$$. It is a exponential weighted moving average of the squared log returns $$r_t^2$$. While technically it is the second non-central moment of the log-return distribution, in finance it is often treated as a variance estimate, assuming either $$\mathbb{E}\left[r_t\right] = 0$$ or is difficult to estimate precisely.)

Exponential-weighted moving average (EWMA) is itself a popular smoorhing technique used in many fields, and within finance its application goes beyond volatility forecasting. Treated as smoother, the parameter $$\alpha$$ controls the degree of smoothing. Treated as a moving average, it controls the weight (importance) of the most recent observation. EWMA is a topic with a very large body of academic literature, and I am not enough of an expert go into the details here.

Within this large body of literature, I came across a paper on Smooth Transitioning Exponential Smoothing (STES) (Taylor, 2004) a few years ago. Not being on a vol trade desk of any sort, my aim was not about forecasting realized volatility. Indeed I wanted to research about a different problem: how to quantitatively select the $$\alpha$$ parameter of a exponential smoothing model. The technique used in this paper not only addresses this problem in the context of vol forecasting, but also resonate with some other work I have seen before at work. At the end I chose other ways that is more suited to the problems I had, but it left an impression on me. 

The STES uses functions of past asset returns to help determine the value of $$\alpha$$. Specifically, STES is fomulated as 

$$
\begin{equation}\label{eq:expsmooth}
    \begin{aligned}
        \alpha_t &= \frac{1}{1+\exp\left(X_t \beta\right)} \\
        \hat{\sigma}^2_t &= \alpha_{t-1} r^2_{t-1} + (1-\alpha_{t-1})\hat{\sigma}^2_{t-1}
    \end{aligned}
\end{equation}
$$

in this formulation, $$\alpha_t$$ is no longer a constant, but is dynamically determined using a logistic function by the set of variables $$X_t$$, which includes $$\lvert r_t \rvert$$ and $$r_t^2$$. The auther went on to demonstrate that STES performed competitively against ES and other GARCH models in terms of 1-step forecast error for major equity indices.

Years later I remember this paper when I see its follow-up paper (Liu, Taylor, Choo 2020). It uses the same technique to determine if past volumn can help forecast realized volatility. I also came across a few other papers that gave me the idea of replacing the logistic function with a random forest. I decided to try this out and see how it performs.

