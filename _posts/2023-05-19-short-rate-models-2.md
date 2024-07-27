---
layout: post
title: "Short Rate Models (Part 2: Simulating and Calibrating Merton's Model)"
date: 2023-05-19
categories: [Quantitative Finance]
tags: [study-notes, quantitative-finance, short-rate-models]
---

<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>



## Table of Contents

1. [Simulating Merton Model](#simulating-mertons-model)
2. [Calibration to Market Observed Short Rates](#calibration-to-market-observed-short-rates)


## Simulating Merton's Model
We first show how to simulate short rates using the Merton model with the Euler-Maruyama discretization, a simple and intuitive way to discretize a continuous-time stochastic process by a sequence of discrete-time processes, each with a small time step. The following code simulates $$T$$ years of short rates, where each year is divided into $$N$$ time steps. ChatGPT writes this function and some functions in the next few posts on other simpler models. They will be our starting points for building a fuller-featured library of short-rate models in (short-rate-models)[https://github.com/steveya/short-rate-models].

```python
# Euler-Maruyama Method to Simulate Merton Model
def merton_simulate(r0, mu, sigma, T=10, N=252):
    """
 Simulates the Merton model using the Euler-Maruyama method.
    
 Parameters:
 r0 (float): Initial short rate.
 mu (float): Annualized drift of the short rate.
 sigma (float): Annualized volatility of the short rate.
 T (int): Number of years to simulate.
 N (int): Number of time steps per year.
    
 Returns:
 t (np.ndarray): Time steps.
 r (np.ndarray): Simulated short rates.
 """
 dt = 1 / N
 t = np.linspace(0, T, (T * N) + 1)
 r = np.zeros((T * N) + 1)
 r[0] = r0
 drift = mu * dt
 diffustion = sigma * np.random.normal(0, np.sqrt(dt), (N))
    for i in range(1, N+1):
 dr = drift + diffustion[i-1]
 r[i] = r[i-1] + dr
    
    return t, r
```

In general, this discretization is only an approximation because it ignores errors from time aggregation, which become more pronounced when there is mean-reversion in the short rates. We will discuss and demonstrate this in more detail in the next post on the Vasicek model. However, as a teaser, a more accurate way to discretize a continuous-time process is first to solve the stochastic differential equation.

$$\int_0^t dr_s = \int_0^t \mu ds + \int_0^t \sigma dW_s$$

then discretize according to this solution. However, for Merton's model, the Euler-Maruyama discretization coincides with the more accurate discretization. 

Next, we will discuss calibrating the model to short rates, which differs from calibrating to the yield curves. The main difference stems from the fact that the short rates are observed under the physical measure, whereas the bond yields are observed under the risk-neutral measure. The parameters we calibrated from the short rates are those under the physical measure and cannot be used to price longer-term bonds.


## Calibration to Market Observed Short Rates
There are at least two ways to calibrate Merton's model to market-observed short rates: maximum likelihood estimation (MLE) and the general method of moments (GMM). There are other ways, but I will cover only the MLE, which is by far the most popular method for estimating the short-rate model parameters from market data.

### Brief Review of the Maximum Likelihood Estimation

#### Principles of Maximum Likelihood Estimation

Maximum Likelihood Estimation (MLE) is a method for estimating the parameters of a statistical model given observations. Its fundamental principle is finding the parameter values that make the observed data most probable.

The key steps in MLE are:

1. Define the likelihood function, the probability of observing the data given the model parameters.
2. Take the logarithm of the likelihood function to obtain the log-likelihood.
3. Find the parameter values that maximize the log-likelihood function.

The likelihood function is often denoted as $$L\left(\theta \vert x\right)$$, where $$\theta$$ represents the model parameters and $$x$$ represents the observed data.

#### Deriving the Likelihood for the Merton Model

To derive the likelihood function for the Merton model, we need to consider the probability distribution of the changes in the short rate. As we derived above, under the Merton model, these changes follow a normal distribution:

$$
\delta r_t = r_\left(t+Δt\right) - r_t \sim N\left(\mu\delta t, \sigma^2\delta t\right)
$$

Given a series of observed short rates $$\left\{r_0, r_1, ..., r_n\right\}$$ at times $$\left\{t_0, t_1, ..., t_n\right\}$$, we can write the likelihood function as:

$$
L\left(\mu, \sigma \vert r_0, ..., r_n\right) = \prod_{i=1}^n f\left(r_i \vert r_{i-1}, \mu, \sigma\right)
$$

Where $$f$$ is the probability density function of the normal distribution:

$$
f\left(r_i \vert r_{i-1}, \mu, \sigma\right) = \frac{1}{\sigma\sqrt{2\pi\delta t_i}} \exp\left(-\frac{(r_i - r_{i-1} - \mu\delta t_i)^2}{2\sigma^2\delta t_i}\right)
$$

Taking the logarithm, we get the log-likelihood function:

$$
ln(L(\mu, \sigma \vert r_0, ..., r_n)) = \sum_{i=1}^n \left[-ln(\sigma) - 0.5ln\left(2\pi\delta t_i\right) - \left(r_i - r_{i-1} - \mu\delta t_i\right)^2 / (2\sigma^2\delta t_i)\right]
$$

#### Maximizing the Log-Likelihood

To find the maximum likelihood estimates, we can find the values of $$\mu$$ and $$\sigma$$ that maximize this log-likelihood function. We take the partial derivatives of the log-likelihood with respect to $$\mu$$ and $$\sigma$$, setting them to zero, and solving the resulting equations.

For the Merton model, we can derive closed-form solutions for the maximum likelihood estimators:

$$
\begin{equation}
\begin{aligned}
\mu_{MLE} &= \frac{1}{T} \sum_{i=1}^{n} (r_i - r_{i-1}) \\
\sigma^2_{MLE} &= \frac{1}{T} \sum_{i=1}^{n} (r_i - r_{i-1} - \mu_{MLE} \delta t_i)^2 \\
\end{aligned}
\end{equation}
$$

where $$T$$ is the number of periods.

Numerical optimization techniques are often used to maximize the log-likelihood function, especially for more complex models where closed-form solutions are unavailable.

### Implementing MLE for the Merton Model

With the above, we can now implement the MLE for the Merton model: 

```python
import numpy as np
from scipy.optimize import minimize

# Log-likelihood function for Gaussian models
def log_likelihood(params, rates, dt):
 mu, sigma = params
 n = len(rates) - 1
 ll = -0.5 * n * np.log(2 * np.pi * sigma**2 * dt)
 ll -= np.sum((rates[1:] - rates[:-1] - mu * dt)**2) / (2 * sigma**2 * dt)
    return -ll  # Minimize negative log-likelihood

# Maximum Likelihood Estimation
def mle_calibration(rates, dt):
 initial_guess = [0.01, 0.01]  # Initial guess for mu and sigma
 result = minimize(log_likelihood, initial_guess, args=(rates, dt), method='L-BFGS-B')
    return result.x
```

### Wrapping Up
We now know the Merton model, a simple one-factor short-rate equilibrium model that is also an affine term-structure model. In the previous [post](https://steveya.github.io/posts/short-rate-models-1/), we derived the price of a ZCB from the short-rate model and wrote a simple Python implementation of the ZCB price and yields.

In this post, We also learn how to simulate short rates with this model and how to calibrate its parameters from market-observed short rates. 

We learn that the ZCB prices and yields are computed under the risk-neutral measure and calibrated to market-observed short rates yield parameters under the physical measure. However, the relationship of the parameters under these two measures still needs to be clarified, which prevents us from calibrating the model to market-observed yield curves. We will cover these topics much later, after introducing another popular short-rate model, the Vasicek model, in the next post.


