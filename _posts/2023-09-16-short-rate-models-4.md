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

We introduced the Vasicek model in the [previous post](https://steveya.github.io/posts/short-rate-models-3/). We talked about how the short rate is modelled as an Onstein Uhlenbeck (OU) process, and derived the solution for the OU process and the bond prices . We now discuss methods to simulate the Vasicek model and estimate the model parameters from the market observed short rates. Since the Vasicek model assumes the short rates follow an Ornstein Uhlenbeck (OU) process, the simulation of short rates and the parameter estimation are the same as those of the OU process.

Given the extensive study of the OU process in finance and physics literature, we will only demonstrate how to simulate it using two different methods and compare their results. Then, we will cover the maximum likelihood estimator for the OU process and show its problems on finite samples. More importantly, we will introduce an alternative method that has shown promising results in our empirical studies.

In [Part 2](https://steveya.github.io/posts/short-rate-models-2/), we mentioned that the Euler-Maruyama discretization is only a (first-order) approximation to the underlying stochastic differential equations (SDE) and how we should discretize the solution of the SDE directly to get a more accurate path. The two methods yield the same results for the Merton model, but they do not for the Vasicek model. We will show numerically that the differences are insignificant for commonly used discretization step sizes.

We then derive the maximum likelihood estimator for the Vasicek model parameters for parameter estimation. We estimate the parameters from our simulated OU processes and show that the $$\kappa$$ parameter estimates are biased and have significant variance on finite samples. We then propose the use of particle filters to estimate the parameters. The resulting $$\kappa$$ estimates have both smaller bias and variance.

In the optional section we will show that mathematically, Euler-Maruyama discretization and the exact method are very "close", and their differences are bounded.

## Simulating Vasicek Model
We can simulate the Vasicek short-rate process directly applying the Euler-Maruyama discretization to its stochastic differential equation (SDE). This method can introduce time-aggregation error when the discretization step $$\Delta t$$ is large. A better approach when $$\Delta t$$ is large is to apply Euler-Maruyama discretization to the solution of the SDE. More specifically, if we discretize the Vasicek SDE, we get

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

As $$\kappa$$ or $$\Delta t$$ gets larger, the difference between $$\kappa \Delta t$$ and $$\left(1 - e^{-\kappa \Delta t}\right)$$ also grows larger; similarly, the difference between $$\sigma \sqrt{\Delta t}$$ and $$\sqrt{\frac{ \sigma^2 \left(1 - e^{-2 \kappa \Delta t} \right) }{ 2 \kappa } }$$ also grows larger. In practice, these differences are insignificant, and we will demonstrate this by comparing the simulated path of the OU process under the Euler-Maruyama method and the exact method. In the optional section below we will show that the differences are bounded by $$\mathcal{O}(\Delta t^2)$$. 

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
Using $$r_0 = 0.05$$, $$\kappa = 0.15$$, $$\theta = 0.03$$, $$\sigma = 0.01$$, $$t = 5$$, $$\Delta t = 1 / 252$$, we plot the simulated short rate paths from the two methods below in Figure 1 and their differences in Figure 2.

![Figure 1. Simulated Paths Using the Euler and the Exact Methods](/assets/img/post_assets/short-rate-models-4/discretization_comp.png)

![Figure 2. Euler Discretizatino Error](/assets/img/post_assets/short-rate-models-4/discretization_error.png)


## Calibration to Market Observed Short Rates
Estimating the Ornstein Uhlenbeck model parameters from finite sample data is challenging because the variance around the maximum likelihood estimates of $$\kappa$$ and $$\theta$$ can be quite large, and their biases are also high. This section briefly covers the basics of Vasicek model parameter estimation using traditional methods such as the MLE. We will then show the size of the estimation bias and variance through simulations. 

### Maximum Likelihood Estimation
Just as we did for the Merton model, we can derive the log-likelihood function for the Vasicek model by noting that conditional on the short rate at time $$t$$, the short rate at time $$t+\Delta t$$ is normally distributed. Using the Euler method to discretize the OU process, the log-likelihood function is given by

$$
\begin{equation}
\begin{aligned}
\ell(\kappa, \theta, \sigma) &= \log \left( p \left( r_1, r_2, \ldots, r_n \mid \kappa, \theta, \sigma \right) \right) \\
&= \log \left( \prod_{i=1}^n p \left( r_i \mid r_{i-1}, \kappa, \theta, \sigma \right) \right) \\
&= \log \left( \prod_{i=1}^n \frac{1}{\sqrt{2\pi \sigma^2 \Delta t}} \exp \left( -\frac{\left(r_i - \left(r_{i-1} + \kappa \left(\theta - r_{i-1}\right) \Delta t \right)\right)^2}{2 \sigma^2 \Delta t} \right) \right) \\
&= \text{constant} - \frac{1}{2} \sum_{i=1}^n \frac{\left(r_i - \left(r_{i-1} + \kappa \left(\theta - r_{i-1}\right) \Delta t \right)\right)^2}{\sigma^2 \Delta t}
\end{aligned}
\end{equation}
$$

Below is the Python implementation of the log-likelihood function and the MLE calibration.

```python
def vasicek_log_likelihood(params, rates, dt):
    kappa, theta, sigma = params
    
    exp_r = rates[:-1] + kappa * (theta - rates[:-1]) * dt
    var_r = sigma**2 * dt
    
    logL = np.sum(
        -0.5 * (
            np.log(2 * np.pi * var_r) + 
            (rates[1:] - exp_r)**2 / var_r
        )
    )
    
    return -logL

def calibrate_vasicek_mle(rates, dt):
    initial_params = [1, np.mean(rates), np.std(np.diff(rates)) * np.sqrt(1/dt)]
    bounds = [(0, None), (None, None), (1e-8, None)]
    
    result = minimize(
        vasicek_log_likelihood, initial_params, 
        args=(rates, dt), bounds=bounds, method='Powell'
    )
    
    kappa, theta, sigma = result.x
    return kappa, theta, sigma

```

We generate 5000 paths of the short rate process using the same $$\kappa = 0.15$$, $$\theta = 0.03$$, $$\sigma = 0.01$$ and $$ r_0 = 0.05$$ for $$t=5$$ years, and apply the maximum likelihood to estimate the parameters from each path. Figure 3 shows the distribution of the MLE estimates for $$\kappa$$, $$\theta$$, and $$\sigma$$ against the true value.

![Figure 3. MLE Estimates for the Vasicek Model Parameters ($$\kappa$$)](/assets/img/post_assets/short-rate-models-4/kappa_mle_bootstrap.png)
![Figure 4. MLE Estimates for the Vasicek Model Parameters ($$\theta$$)](/assets/img/post_assets/short-rate-models-4/theta_mle_bootstrap.png)
![Figure 5. MLE Estimates for the Vasicek Model Parameters ($$\sigma$$)](/assets/img/post_assets/short-rate-models-4/sigma_mle_bootstrap.png)

### Particle Filtering

Even though the MLE is asymptotically consistent and achieves the Cramér-Rao lower bound, it can be far off with small sample sizes as we have seen above. In fact, the bias and variance of the MLE estimates are both quite large. This is not good news for the model calibration. 

I once was working on calibrating the CEV model to the volatility surfaces. At first, I estimate the two parameters of the model by minimizing the mean squared error (MSE) between the observed and model volatility surfaces. However, this approach makes the estimated parameters unstable because there are generally multiple local minima, and the estimates are sensitive to noise. After some brainstorming, I used the particle filter to estimate the parameters. This allows us to track multiple minimas over time and take their averages. This resulted in much smoother estimates over time.

We have a different problem with the Vasicek model parameter estimation, but I suspect that particle filtering can also solve the problem of large variance.

### Comparison

## Wrapping Up



### Optional: Strong Convergence of the Euler-Maruyama Method Applied to the OU process
In this section, we show that the Euler-Maruyama method has strong convergence of order 0.5 to the exact solution of the OU process.

Let's first define what strong convergence of a discretization method means. A numerical method $$M$$ has **strong convergence of order $$\alpha$$** to the exact solution for a stochastic process $$X_t$$ if the expected absolute error satisfies 

$$
\begin{equation}
\mathbb{E} \left[  \underset{0 \leq t \leq T}{\sup} \left| X_t - X_t^M \right| \right] = \mathcal{O}(\Delta t^{\alpha})
\end{equation}
$$

where $$X_T$$ is the exact solution of the SDE and $$X_T^M$$ is the numerical approximation. As we have shown earlier, the Euler Maruyama (EM) discretization of the OU process gives the approximation

$$
\begin{equation}
r_{t + \Delta t}^{EM} = r_t^{EM} + \kappa \left(\theta - r_t^{EM} \right) \Delta t + \sigma \sqrt{\Delta t} \epsilon_t
\end{equation}
$$

whereas the exact solution gives the equation

$$
\begin{equation}
r_{t+\Delta t} = r_t e^{-\kappa \left(\Delta t\right)} + \theta \left(1 - e^{-\kappa\left(\Delta t\right)}\right) + \sigma \int_t^{t + \Delta t} e^{-\kappa \left(u-t\right)} dW_u
\end{equation}
$$

Denote $$r(t_n) = r_{t_n}$$. The error $$e_{n+1} = r(t_n) - r_n^{EM}$$ can be decomposed into three parts:

$$
\begin{equation}
\begin{aligned}
e_{n+1} &= r(t_n) - r_n^{EM} \\
&= \left(r(t_n) e^{-\kappa \Delta t} + \theta \left(1 - e^{-\kappa\left(\Delta t\right)}\right) + \sigma \int_t^{t + \Delta t} e^{-\kappa \left(u-t\right)} dW_u\right) \\
&- \left(r_n^{EM} + \kappa \left(\theta - r_n^{EM}\right) \Delta t + \sigma \sqrt{\Delta t} \epsilon_{n} \right) \\
&= \left(r(t_n) e^{-\kappa \Delta t} - r_n^{EM} \right) \quad \longrightarrow \text{Part (1)} \\
&+ \left(\theta \left(1 - e^{-\kappa\left(\Delta t\right)}\right) - \kappa \left(\theta - r_n^{EM}\right) \Delta t \right) \quad \longrightarrow \text{Part (2)} \\
&+ \left(\sigma \int_t^{t + \Delta t} e^{-\kappa \left(u-t\right)} dW_u - \sigma \sqrt{\Delta t} \epsilon_{n} \right) \quad \longrightarrow \text{Part (3)}
\end{aligned}
\end{equation}
$$


The first part is the error from the drift term involving the short rate $$r(t_n)$$:

$$
\begin{equation}
\begin{aligned}
\text{Part (1)} &= r(t_n)e^{-\kappa \Delta t} - \left(r_n^{EM} + \kappa \left(\theta - r_n^{EM}\right) \Delta t \right) \\
&= r(t_n)\left(1 - \kappa\Delta t + \mathcal{O}(\Delta t^2)\right) - r_n^{EM}\left(1 + \kappa \Delta t + \mathcal{O}(\Delta t^2)\right) \quad \text{(by Taylor expansion)} \\
&= r(t_n) - r_n^{EM} - \kappa \Delta t r(t_n) + \kappa \Delta t r_n^{EM} + \mathcal{O}(\Delta t^2) \\
&= \left(1 - \kappa \Delta t\right) e_n + \mathcal{O}(\Delta t^2)
\end{aligned}
\end{equation}
$$

The second part is the error from the drift term involving the mean reversion $$\theta$$:

$$
\begin{equation}
\begin{aligned}
\text{Part (2)} &= \theta \left(1 - e^{-\kappa \Delta t}\right) - \kappa\theta \Delta t \\
&= \theta \left(1 - 1 + \kappa \Delta t + \mathcal{O}(\Delta t^2)\right) - \kappa\theta \Delta t \quad \text{(by Taylor expansion)} \\
&= \mathcal{O}(\Delta t^2)
\end{aligned}
\end{equation}
$$

The third part is the error from the diffusion term, and has a normal distribution with mean 0 and variance $$\sigma^2 \left(\frac{1 - e^{-2\kappa \Delta t}}{2\kappa} - \Delta t \right)$$. The variance can be simplified as

$$
\begin{equation}
\begin{aligned}
\mathrm{Var}(\text{Part (3)}) &= \sigma^2 \left(\frac{1 - e^{-2\kappa \Delta t}}{2\kappa} - \Delta t\right) \\
&= \sigma^2 \left(\Delta t - \frac{2\kappa \Delta t^2}{3} + \mathcal{O}(\Delta t^3) - \Delta t\right) \quad \text{(by Taylor expansion)} \\
&= \mathcal{O}(\Delta t^2)
\end{aligned}
\end{equation}
$$

Putting these terms together, the error $$e_{n+1}$$ can be written as

$$
\begin{equation}
\begin{aligned}
e_{n+1} &= e_n \left(1 - \kappa \Delta t\right) + \Delta e_n \\
\end{aligned}
\end{equation}
$$

where $$\Delta e_n$$ is the error from the mean-reversion term and the diffusion term and has order $$\mathcal{O}(\Delta t^2)$$.

To show that the Euler-Maruyama method converges to the exact solution of the OU process, we need to show that the expected absolute error satisfies

$$
\begin{equation}
\mathbb{E} \left[  \underset{0 \leq t \leq T}{\sup} \left| r(t) - r_t^{EM} \right| \right] = \mathcal{O}(\Delta t)
\end{equation}
$$

Squaring the error and taking the expected value, we get

$$
\begin{equation}
\begin{aligned}
\mathbb{E} \left[ |e_{n+1}|^2 \right] &\leq \left(1 - \kappa \Delta t\right)^2 \mathbb{E} \left[ |e_n|^2 \right] + C_1 \Delta t^2 \\
&\leq \left(1 - 2 \kappa \Delta t\right) \mathbb{E} \left[ |e_n|^2 \right] + C_1 \Delta t^2 \quad \text{for small } \Delta t \\
\end{aligned}
\end{equation}
$$

Here we introduce the a version of Gronwall's inequality, which is a useful tool to bound the solution of a differential or difference equations. It states that if $$f$$ is a non-negative function and satisfies $$\frac{d}{dt} f(t) \leq a_1 f(t) + a_0$$ for all $$t$$, then

$$
\begin{equation}
f(t) \leq f(0) e^{a_1 t} + \frac{a_0}{a_1} \left(e^{a_1 t} - 1\right)
\end{equation}
$$

and for the difference equation, if $$f$$ satisfies $$f(n+1) - f(n) \leq a_1 f(n) + a_0$$ for all $$n \geq 0$$, then

$$
\begin{equation}
f(n) \leq f(0) e^{a_1 n} + \frac{a_0}{a_1} \left(e^{a_1 n} - 1\right)
\end{equation}
$$

Let $$f(n) = \mathbb{E} \left[ \lvert e_n \rvert^2 \right]$$ and substituting into the inequality, we have

$$
\begin{equation}
\begin{aligned}
\mathbb{E} \left[ |e_{n+1}|^2 \right] &\leq \left(1 - 2 \kappa \Delta t\right) \mathbb{E} \left[ |e_n|^2 \right] + C_1 \Delta t^2 \\
&\Longrightarrow \mathbb{E} \left[ |e_{n+1}|^2  - |e_{n}|^2 \right] \leq \left(- 2 \kappa \Delta t\right) \mathbb{E} \left[ |e_n|^2 \right] + C_1 \Delta t^2 \\
&\Longrightarrow \mathbb{E} \left[ |e_{n}|^2 \right] \leq \mathbb{E} \left[ |e_{0}|^2 \right] e^{-2 \kappa \Delta t n} + \frac{C_1 \Delta t^2}{2 \kappa\Delta t} \left(e^{-2 \kappa \Delta t n} - 1\right) \\
&\Longrightarrow \mathbb{E} \left[ |e_{n}|^2 \right] \leq \mathbb{E} \left[ |e_{0}|^2 \right] e^{-2 \kappa \Delta t n} + C_2 \Delta t \\
\end{aligned}
\end{equation}
$$

Finally, to compute the strong convergence rate, we take the square root of the mean-square error $$\mathbb{E} \left[ \lvert e_n \rvert^2 \right]$$, and the expected absolute error has bound $$\mathbb{E} \left[ \lvert e_n \rvert \right] \leq \sqrt{\mathbb{E} \left[ \lvert e_n \rvert^2 \right]} = \mathcal{O}(\Delta t)$$ by Jensen's inequality.

We have shown that the Euler-Maruyama method has strong convergence of order $$1/2$$ to the exact solution of the OU process.