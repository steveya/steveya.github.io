---
layout: post
title: "Short Rate Models (Part 2: Simulating and Calibrating the Merton Model)"
date: 2023-05-19
categories: [Quantitative Finance]
tags: [study-notes, quantitative-finance, short-rate-models]
---

<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>



## Table of Contents

1. [Simulating Merton Model](#simulating-mertons-model)
2. [Calibration to Market Observed Short Rates](#calibration-to-market-observed-short-rates)


## Simulating Merton's Model
We introduced the Merton short rate model in the [previous post](https://steveya.github.io/posts/short-rate-models-1/). We now show how to simulate short rates using the Merton model with the Euler-Maruyama discretization, a simple and intuitive way to discretize a continuous-time stochastic process by a sequence of small time steps. It is a first-order numerical method for approximating stochastic differential equations (SDEs). As a recap, consider a general $$d$$-dimensional SDE of the form:

$$
dX_t = a(X_t, t) dt + b(X_t, t) dW_t
$$

where:
- $$X_t \in \mathbb{R}^d$$ is the state vector at time $$t$$,
- $$a(X_t, t)$$ is the drift term,
- $$b(X_t, t)$$ is the diffusion term (a matrix),
- $$W_t$$ is a $$m$$-dimensional Wiener process.

The **Euler-Maruyama approximation** for this SDE, over a small time step $$\Delta t$$, is given by:

$$
X_{t + \Delta t} = X_t + a(X_t, t) \Delta t + b(X_t, t) \sqrt{\Delta t} \cdot Z_t
$$

where $$Z_t \sim \mathcal{N}(0, I_m)$$ is a standard normal random vector of dimension $$m$$.

Applying the above definition to the Merton model and re-arrange we get 

$$
\Delta r_t = r_{t + \Delta t} - r_t = \mu \Delta t + \sigma \sqrt{\Delta t} \cdot \epsilon_t
$$

The following code simulates $$t$$ years of short rates at time steps of size $$dt$$.

```python
def simulate_merton_short_rates(r0, mu, sigma, t, dt, seed=None):
    """
    Simulates the Merton short rates model using the 
    Euler-Maruyama discretization. As the drift does 
    not depend on the short rate, each increment can 
    be simulated independently.
    
    Parameters:
    r0 (float): Initial (current) short rate.
    mu (float): The annualized drift of the short rate process.
    sigma (float): Annualized volatility of the short rate process.
    t (float): Number of years to simulate.
    dt (float): Time step size.
    
    Returns:
    times (np.ndarray): Time steps.
    rates (np.ndarray): Simulated short rates.
    """
    np.random.seed(seed)

    nd = int(t / dt)
    
    drift = mu * dt
    diffusion = sigma * np.random.normal(0, np.sqrt(dt), nd - 1)

    dr = drift + diffusion

    times = np.linspace(0, t, nd)
    rates = np.array(list(accumulate(dr, initial=r0)))
    
    return times, rates
```

The **Euler-Maruyama discretization** is only an approximation because it ignores errors from time aggregation. A more accurate way to discretize a continuous-time process is first to solve the stochastic differential equation

$$\int_0^t dr_s = \int_0^t \mu ds + \int_0^t \sigma dW_s$$

then discretize according to this solution. 

$$\Delta r_t = \int_0^{\Delta t} \mu \Delta t + \int_0^t \sigma \epsilon_t $$

We call this more accurate method the **exact method for diecretization**. For the simple Merton model, the Euler-Maruyama discretization coincides with the more accurate discretization, but for a general stochastic process

$$ dr_t = \mu(t, r_t) dt + \sigma(t, r_t) dW_t $$

The term $$\int_0^{\Delta t} \mu(s, r_s) ds \neq \mu(t, r_t) \Delta t$$. However, the error from time aggregation is "small" for "well-behaved" processes.We will discuss this in greater detail in the [simulation and calibration of the Vasicek model](https://steveya.github.io/posts/short-rate-models-4/). 

Using $$\mu = 0.02$$, $$\sigma = 0.02$$ and $$ r_0 = 0.05$$, Figure 1 shows a simulated path of the short rate process.

![Figure 1. A Simulated Path of the Merton Short Rate Process](/assets/img/post_assets/short-rate-models-2/merton_short_rate_simulation.png)

## Simulating Short Rates, Bond Yields and Term Structure from the Merton Model

In the [previous post](https://steveya.github.io/posts/short-rate-models-1/), we derived the price of a ZCB from the short-rate model in two different ways: 1. directly solve the SDE, and 2. guess the form of the solution as $$A(t, T) \exp (-B(t, T) r_t)$$ and solve for $$A$$ and $$B$$. As it turns out, most common short-rate models have bond price solutions in the form of $$A(t, T) \exp (-B(t, T) r_t)$$, so we will use this fact when we implement our short-rate models. Below is a simple Python implementation of the ZCB price and yields.

```python
def B(t, T):
    return T - t

def A(t, T, mu, sigma):
    tau = T - t
    return np.exp(-mu * tau**2 / 2 + (sigma**2 * tau**3) / 6)

def zero_coupon_bond_price(t, T, r, mu, sigma):
    return A(t, T, mu, sigma) * np.exp(-B(t, T) * r)

def zero_coupon_yield(t, T, r, mu, sigma):
    price = zero_coupon_bond_price(t, T, r, mu, sigma)
    return -np.log(price) / (T - t)

```

We use the above formula to simulate the 5-, 10-, and 30-year yields using the above parameters, and Figure 2 shows the results.

![Figure 2. A Simulated Path of the Merton Long Rate Processes](/assets/img/post_assets/short-rate-models-2/merton_long_rate_simulation.png)

We can see that the three long rates are all parallel shifts of one another and the short rates. The simple Merton model is capable of generating only parallel shifts. We can also generate its term structure with some combination of $$\mu$$ and $$\sigma$$, and Figure 3 shows the results.

![Figure 3. Term Structures from the Merton Model](/assets/img/post_assets/short-rate-models-2/merton_term_structure.png)

Unsurprisingly, when $$\sigma$$ is small relative to $$\mu$$, the constant drift term dominates the term structure and is upward-sloping, almost in a straight line. When $$\sigma$$ is large, the long end of the term structure is dominated by the volatility term and is downward sloping due to the convexity effects.

Next, we will discuss calibrating the model to short rates, which differs from calibrating the yield curves. The main difference stems from the short rates being observed under the physical measure, whereas the bond yields are observed under the risk-neutral measure. The parameters we calibrated from the short rates are those under the physical measure and cannot be used to price longer-term bonds.

## Calibration to Market Observed Short Rates
There are at least two ways to calibrate Merton's model to market-observed short rates: maximum likelihood estimation (MLE) and the general method of moments (GMM). There are other ways, but we will cover only the MLE, which is by far the most popular method for estimating the short-rate model parameters from market data.

### Brief Review of the Maximum Likelihood Estimation

#### Principles of Maximum Likelihood Estimation

Maximum Likelihood Estimation (MLE) is a method for estimating the parameters of a statistical model given observations. Its fundamental principle is finding the parameter values that make the observed data most probable.

The critical steps in MLE are:

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

where $$f$$ is the probability density function of the normal distribution:

$$
f\left(r_i \vert r_{i-1}, \mu, \sigma\right) = \frac{1}{\sigma\sqrt{2\pi\delta t_i}} \exp\left(-\frac{(r_i - r_{i-1} - \mu\delta t_i)^2}{2\sigma^2\delta t_i}\right)
$$

Taking the logarithm, we get the log-likelihood function:

$$
ln(L(\mu, \sigma \vert r_0, ..., r_n)) = \sum_{i=1}^n \left[-ln(\sigma) - 0.5ln\left(2\pi\delta t_i\right) - \left(r_i - r_{i-1} - \mu\delta t_i\right)^2 / (2\sigma^2\delta t_i)\right]
$$

#### Maximizing the Log-Likelihood

To find the maximum likelihood estimates, we can find the values of $$\mu$$ and $$\sigma$$ that maximize this log-likelihood function. We take the partial derivatives of the log-likelihood with respect to $$\mu$$ and $$\sigma$$, setting them to zero, and solving the resulting equations to get an analytical solution. 

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
def merton_log_likelihood(params, rates, dt):
    mu, sigma = params
    n = len(rates) - 1
    ll = -0.5 * n * np.log(2 * np.pi * sigma**2 * dt)
    ll -= np.sum((rates[1:] - rates[:-1] - mu * dt)**2) / (2 * sigma**2 * dt)
    return -ll

def merton_mle_calibration(rates, dt):
    initial_guess = [0.01, 0.01]
    result = minimize(merton_log_likelihood, initial_guess, args=(rates, dt), method='L-BFGS-B')
    return result.x

```

We generate 5000 paths of the short rate process using the same $$\mu = 0.02$$, $$\sigma = 0.02$$ and $$ r_0 = 0.05$$ for $$t=5$$ years, and apply the maximum likelihood to estimate the parameters from each path. Figure 4 shows the distribution of the MLE estimates for $$\mu$$ and $$\sigma$$.

![Figure 4. Distribution of ML estimates of the Merton Model parameters](/assets/img/post_assets/short-rate-models-2/merton_term_structure.png)

We can see immediately that the MLE estimates of $$\mu$$, while unbiased, have a high variance. This is consistent with the observation that the mean of a distribution is a lot harder to estimate precisely than the variance. This problem is even more pronounced for the more complicated processes, such as the Vasicek model. We will discuss this in length in the [next next post](https://steveya.github.io/posts/short-rate-models-4/) on calibrating the Vasicek model. 

### Wrapping Up
We now know the Merton model, a simple one-factor short-rate equilibrium model that is also an affine term-structure model. In the previous [post](https://steveya.github.io/posts/short-rate-models-1/), we derived the price of a ZCB from the short-rate model and wrote a simple Python implementation of the ZCB price and yields.

In this post, We also learn how to simulate short rates with this model and how to calibrate its parameters from market-observed short rates. 

We learn that the ZCB prices and yields are computed under the risk-neutral measure and calibrated to market-observed short rates yield parameters under the physical measure. However, the relationship of the parameters under these two measures still needs to be clarified, which prevents us from calibrating the model to market-observed yield curves. We will cover these topics much later, after introducing another popular short-rate model, the Vasicek model, in the [next post](https://steveya.github.io/posts/short-rate-models-3/).


