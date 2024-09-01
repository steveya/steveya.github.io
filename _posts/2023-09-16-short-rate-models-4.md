---
layout: post
title: "WIP - Short Rate Models (Part 4: Simulating and Calibrating Vasicek Model)"
date: 2023-07-16
categories: [Quantitative Finance]
tags: [study-notes, quantitative-finance, short-rate-models]
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>


## Table of Contents

1. [Simulating Vasicek Model](#simulating-vasicek-model)
2. [Calibration to Market Observed Short Rates](#calibration-to-market-observed-short-rates)

A good tutorial on simulating and calibrating the Ornstein-Uhlenbeck (OU) process using classic methods (MLE, GMM) is the [Hudson and Thames tutorial on the OU process](https://hudsonthames.org/caveats-in-calibrating-the-ou-process/). In this post we discuss methods to simulate from the Vasicek model and calibrate it to the market observed short rates. Since the Vasicek model assumes the short rates follows an OU process, there is a significant overlap between this post and the Hudson and Thames tutorial. 

They summarize recent results from [Bao et al (2015)] who derive the exact distribution of the maximum likelihood estimators of the OU process with various assumptions. They also cover the topic of estimation bias of $$\kappa$$ in finite sample, which is also important in the context of short-rate models. While they also cover the topic of the two different discretization scheme used for simulating the Vasicek model, I do not agree with their conclusion that 

On my end, I will demonstrate how I calibrate the OU process via particle filtering, and how it compares to the maximum likelihood estimates.

## Simulating Vasicek Model
We introduced the Vasicek model in the [previous post](https://steveya.github.io/posts/short-rate-models-3/), and just as we did in [Part 2](https://steveya.github.io/posts/short-rate-models-2/) of this series, we can simulate the Vasicek short-rate process directly from the Euler-Maruyama discretization of its stochastic differential equation (SDE). However, as we pointed out that this method can introduce time-aggregation error when there is strong mean-reversion, a better approach is apply Euler-Maruyama discretization to the solution of the SDE. More specifically, if we discretize the Vasicek SDE, we get

$$
\begin{equation}
\Delta r_t = \kappa(\theta - r_t) \Delta t + \sigma \sqrt{\Delta t} N(0, 1)
\end{equation}
$$

and if we discretize the solution of the Vasicek SDE, as derived from [Post 2](https://steveya.github.io/posts/short-rate-models-2/)

$$
\begin{equation}
r_s = r_t e^{-\kappa \left(s-t\right)} + \theta \left(1 - e^{-\kappa\left(s-t\right)}\right) + \sigma \int_t^s e^{-\kappa \left(u-t\right)} dW_u
\end{equation}
$$

and substitute $$s = t + \Delta t$$, we get the following discrete SDE:

$$
\begin{equation}
\Delta r_t = r_{t + \Delta t} - r_t = \left(1 - e^{-\kappa \Delta t}\right) \left(\theta - r_t\right)  + \sqrt{\frac{ \sigma^2 \left(1 - e^{-2 \kappa \Delta t} \right) }{ 2 \kappa } } N(0, 1)
\end{equation}
$$

As $$\kappa$$ or $$\Delta t$$ gets larger, the difference between $$\kappa \Delta t$$ and $$\left(1 - e^{-\kappa \Delta t}\right)$$ also grows larger; similarly, the difference between $$\sigma \sqrt{\Delta t}$$ and $$\sqrt{\frac{ \sigma^2 \left(1 - e^{-2 \kappa \Delta t} \right) }{ 2 \kappa } }$$ also grows larger. We can illustrate this by plotting the same short rate path generated from these two methods `simulate_vasicek_short_rates_euler` and `simulate_vasicek_short_rates_exact` in the following figure.


```python
def simulate_vasicek_short_rates_euler(r0, kappa, theta, sigma, t, dt, seed=None):
    """
    Simulates the Vasicek short rates model using the 
    Euler-Maruyama discretization. 
    
    Parameters:
    r0 (float): Initial (current) short rate.
    kappa (float): Speed of mean reversion.
    theta (float): Long-term mean level of the short rate.
    sigma (float): Annualized volatility of the short rate process.
    t (float): Number of years to simulate.
    dt (float): Time step size.
    
    Returns:
    times (np.ndarray): Time steps.
    rates (np.ndarray): Simulated short rates.
    """

    np.random.seed(seed)
    
    nd = int(t / dt)

    times = np.linspace(0, t, nd)

    rates = np.zeros(nd)
    rates[0] = r0

    diffusion = sigma * np.random.normal(0, np.sqrt(dt), nd)

    for t in range(1, nd):
        dr = kappa * (theta - rates[t-1]) * dt + diffusion[t]
        rates[t] = rates[t-1] + dr
        
    return times, rates


def simulate_vasicek_short_rates_exact(r0, kappa, theta, sigma, t, dt, seed=None):
    """
    Simulates the Vasicek short rates model using the 
    Doob (analytical solution)
    
    Parameters:
    r0 (float): Initial (current) short rate.
    kappa (float): Speed of mean reversion.
    theta (float): Long-term mean level of the short rate.
    sigma (float): Annualized volatility of the short rate process.
    t (float): Number of years to simulate.
    dt (float): Time step size.
    
    Returns:
    times (np.ndarray): Time steps.
    rates (np.ndarray): Simulated short rates.
    """

    np.random.seed(seed)

    nd = int(t / dt)

    times = np.linspace(0, t, nd)

    rates = np.zeros(nd)
    rates[0] = r0

    exp_kappa_dt = np.exp(-kappa * dt)
    variance = sigma**2 * (1 - np.exp(-2 * kappa * dt)) / (2 * kappa)
    for t in range(1, nd):
        exp_rate = (theta * (1 - exp_kappa_dt) + rates[t-1] * exp_kappa_dt)
        rates[t] = np.random.normal(exp_rate, np.sqrt(variance))
    
    return times, rates


```



## Calibration to Market Observed Short Rates
