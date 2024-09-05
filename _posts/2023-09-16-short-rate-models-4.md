---
layout: post
title: "Short Rate Models (Part 4: Simulating and Calibrating the Vasicek Model)"
date: 2023-09-16
categories: [Quantitative Finance]
tags: [study-notes, quantitative-finance, short-rate-models]
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>


## Table of Contents

1. [Simulating Vasicek Model](#simulating-vasicek-model)
2. [Calibration to Market Observed Short Rates](#calibration-to-market-observed-short-rates)

We introduced the Vasicek model in the [previous post](https://steveya.github.io/posts/short-rate-models-3/). We now discuss methods to simulate the Vasicek model and estimate the model parameters from the market observed short rates. Since the Vasicek model assumes the short rates follow an Ornstein Uhlenbeck (OU) process, the simulation of short rates and the parameter estimation are the same as those of the OU process.

Given the extensive study of the OU process in finance and physics literature, we will only demonstrate how to simulate it using two different methods and compare their results. Then, we will cover the maximum likelihood estimator for the OU process and show its problems on finite samples. More importantly, we will introduce an alternative method that has shown promising results in our empirical studies.

In [Part 2](https://steveya.github.io/posts/short-rate-models-2/), we mentioned that the Euler-Maruyama discretization is only a (first-order) approximation to the underlying stochastic differential equations (SDE) and how we should discretize the solution of the SDE directly to get a more accurate path. The two methods yield the same results for the Merton model, but they do not for the Vasicek model. We will show they are very close, and the difference between these two methods is bounded.

We then derive the maximum likelihood estimator for the Vasicek model parameters for parameter estimation. We estimate the parameters from our simulated OU processes and show that the $$\kappa$$ parameter estimates are biased and have significant variance on finite samples. We then propose the use of particle filtering that results in both smaller bias and variance for $$\kappa$$

## Simulating Vasicek Model
Just as we did to the Merton model in [Part 2](https://steveya.github.io/posts/short-rate-models-2/) of this series, we can simulate the Vasicek short-rate process directly using the Euler-Maruyama discretization of its stochastic differential equation (SDE). However, this method is a first-order approximation and can introduce time-aggregation error when there is strong mean-reversion, a problem not shared by the Merton model. A better approach is to apply Euler-Maruyama discretization to the solution of the SDE. More specifically, if we discretize the Vasicek SDE, we get

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

As $$\kappa$$ or $$\Delta t$$ gets larger, the difference between $$\kappa \Delta t$$ and $$\left(1 - e^{-\kappa \Delta t}\right)$$ also grows larger; similarly, the difference between $$\sigma \sqrt{\Delta t}$$ and $$\sqrt{\frac{ \sigma^2 \left(1 - e^{-2 \kappa \Delta t} \right) }{ 2 \kappa } }$$ also grows larger.

In practice, these differences are insignificant. More concretely, the difference between the drift terms of the two methods $$\Delta_D$$ is 

$$
\begin{equation}
\begin{aligned}
\Delta_D &= \left(1 - e^{-\kappa \Delta t}\right) \left(\theta - r_t\right) - \left(\kappa \Delta t\right) \left(\theta - r_t\right) \\
&= \left(1 - e^{-\kappa \Delta t} - \kappa \Delta t\right) \left(\theta - r_t\right) \\
&= \left(1 - 1 + \kappa \Delta t - \frac{(\kappa \Delta t)^2}{2} + \frac{(\kappa \Delta t)^3}{6} - \cdots - \kappa \Delta t \right)(\theta - r_t)  \quad \text{by Taylor expansion of } e^{-x} \\
&= \left( - \frac{(\kappa \Delta t)^2}{2} + \frac{(\kappa \Delta t)^3}{6} - \cdots \right)(\theta - r_t) \\
&= - \frac{\kappa^2 \Delta t^2}{2} (\theta - r_t) + \frac{\kappa^3 \Delta t^3}{6} (\theta - r_t) - \cdots
\end{aligned}
\end{equation}
$$

and the magnitude of the difference is approximately

$$
|\Delta_D| \approx \frac{\kappa^2 \Delta t^2}{2} | \theta - r_t |
$$

Similarly, the difference between the diffusion terms of the two methods $$\Delta_S$$ is

$$
\begin{equation}
\begin{aligned}
\Delta_S& = \sqrt{\frac{\sigma^2 (1 - e^{-2\kappa \Delta t})}{2\kappa}} - \sigma \sqrt{\Delta t} \\
&= \sqrt{ \frac{ \sigma^2 \left(2\kappa \Delta t - 2(\kappa \Delta t)^2 + \frac{4}{3}(\kappa \Delta t)^3 - \cdots \right) }{2\kappa} } - \sigma \sqrt{\Delta t} \quad \text{by Taylor expansion of } e^{-x} \\
&= \sqrt{ \sigma^2 \left( \Delta t - (\kappa \Delta t^2) + \frac{2}{3} (\kappa \Delta t)^2 \Delta t - \cdots \right) } - \sigma \sqrt{\Delta t} \\
&= \sigma \sqrt{ \Delta t \left( 1 - \kappa \Delta t + \frac{2}{3} \kappa^2 \Delta t^2 - \cdots \right) } - \sigma \sqrt{\Delta t} \\
&= \sigma \sqrt{\Delta t} \left( 1 - \frac{\kappa \Delta t}{2} - \frac{ (\kappa \Delta t)^2 }{8} + \frac{2}{3} \kappa^2 \Delta t^2 \cdot \frac{1}{2} + \cdots - 1 \right) \quad \text{by Taylor expandion of } \sqrt{1 + x}  \\
&= \sigma \sqrt{\Delta t} \left( - \frac{\kappa \Delta t}{2} + \left( \frac{\kappa^2 \Delta t^2}{3} - \frac{ (\kappa \Delta t)^2 }{8} \right) + \cdots \right) \\
&= \sigma \sqrt{\Delta t} \left( - \frac{\kappa \Delta t}{2} + \frac{ (\kappa \Delta t)^2 }{24 } + \cdots \right) \\
&= - \frac{\sigma \kappa \Delta t^{3/2}}{2} + \frac{ \sigma \kappa^2 \Delta t^{5/2} }{24 } + \cdots
\end{aligned}
\end{equation}
$$

and the magnitude of the difference is approximately

$$
|\Delta_S| \approx \frac{\sigma \kappa \Delta t^{3/2}}{2}
$$


In the following figure, we can illustrate this by plotting the same short rate path generated from these two methods, `simulate_vasicek_short_rates_euler` and `simulate_vasicek_short_rates_exact`.


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
Estimating the Ornstein Uhlenbeck model parameters from finite sample data is challenging because the variance around the maximum likelihood estimates of $$\kappa$$ can be quite large, and the bias of $$\kappa$$ is also high. This section briefly covers the basics of Vasicek model parameter estimation using traditional methods such as the MLE. We will then show the size of the estimation bias and variance through simulations. 

I once was working on calibrating the CEV model to the volatility surfaces. At first, I estimate the two parameters of the model by minimizing the mean squared error (MSE) between the observed and model volatility surfaces. However, this approach makes the estimated parameters unstable because there are generally multiple local minima, and the estimates are sensitive to noise. After some brainstorming, I used the particle filter to estimate the parameters. This allows us to track multiple minimas over time and take their averages. This resulted in much smoother estimates over time.

We have a different problem with the Vasicek model parameter estimation, but I suspect that particle filtering can also solve the problem of large variance.

### Maximum Likelihood Estimation

### Particle Filtering

### Comparison

## Wrapping Up