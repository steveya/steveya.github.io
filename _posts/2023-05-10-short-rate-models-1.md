---
layout: post
title: "Short Rate Models (Part 1: Merton's Model)"
date: 2023-05-10
tags: [study-notes, quantitative-finance, short-rate-models]
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Refresher on Short Rate Models (Part 1: Merton's Model)

## Table of Contents

1. [Introduction](#introduction)
2. [Preliminaries](#preliminaries)
3. [Merson's Model](#mertons-model-1973)

## Introduction
Welcome to my refresher series on the **short-rate models**. It is a type of interest rate term structure models that was commonly used in the industry and acamamia. The refresher is disguised as a step-by-step guide written to help you (and me) understand, formulate, and solve this type of models by deriving and implementing well-known models of varying complexity and generality.

The short-rate models explicitly specify the evolution of the instantaneous spot interest rate, also known as the short rate, at which the market lends to a borrower for an infinitesimal period of time. The short rate is assumed to evolve over time according to a stochastic process that captures the observed or theorized behavior. If the short rate evolves according to the assumed process, then by the no-arbitrage principle, all rates at which the market lends to a borrower over longer periods of time can also be determined by the parameters of the process. These parameters are unknown and need to be calibrated from the market observed yield curves. 

In this first post, I start with some preliminaries and introduce a simple short-rate model. I then go step-by-step to derive the price of a zero-coupon bond and the simulation of the model. It is then followed by a Python implementations of the simulation of the short-rate process and the pricing of the zero coupon bond. 

The calibration of the model parameters is actually a big topic in and of itself, and I will make a post of it later.

## Preliminaries
Zero-coupon bonds form the foundation of term structure modeling. For a zero-coupon bond maturing at time $$T$$, the relationship between its price $$P(t, T)$$ and (continuously compounded) yield-to-maturity (YTM) $$y(t, T)$$ at time $$t$$ is given by

- YTM to Price: 

$$P\left(t, T\right) = \exp\left(-y_t^T \left(T-t\right)\right)$$ 

- Price to YTM: 

$$y\left(t, T\right) = -\frac{\ln\left(P_t^T\right)}{T-t}$$

As we explicitly model the dynamic of the short rates, the YTM is the expected average (under the risk-neutral measure $$Q$$) of the short-rate from $$t$$ to $$T$$.

- Short Rate to YTM: 

$$y\left(t, T\right) = \frac{1}{T-t}\mathbb{E}_t^Q\left[\int_t^T r_s ds\right]$$

- Short Rate to Price: 

$$P\left(t, T\right) = \mathbb{E}_t^Q\left[\exp\left(-\int_t^T r_s ds\right)\right]$$

These fundamental relationships provide the basis for more complex term structure models.

## Merton's Model (1973)
The Merton's model is one of the first short rate models and is arguably the simplest one; it laid the foundation for many subsequent short-rate models. It was introduced in 1973 by the renowned economist Robert C. Merton, who extended the concepts of equity option pricing to the bond market by modeling the dynamics of short-rate as a simple Gaussian process.

### Model Specification
Merton's model is defined by the following stochastic differential equation:

$$dr_t = \mu dt + \sigma dW_t$$

where:
- $$r_t$$: short-term interest rate at time $$t$$
- $$\mu$$: constant drift term
- $$\sigma$$: constant volatility
- $$W_t$$: Wiener process (standard Brownian motion)

### Key Concepts

#### Continuous-Time Modeling
Merton's model was one of the first to apply continuous-time stochastic processes to interest rate modeling, paving the way for more sophisticated models. It is a one-factor model as the only source of randomness is the short rate. This will later be generalized to the multi-factor model, where multiple factors drive the dynamic of the short rate.

#### Gaussian Process
The model assumes that the short rate follows a Gaussian process, allowing for both positive and negative levels. It also assumes that rate volatility is constant and independent of the level of the short rate. This modelling choice is inconsistent with empirical restults. While rates can go negative (though mostly inconceivable at 1973), the short-rate volatility generally do depend on the level of the short rate. Short-rates also exhibit mean-reversion as opposed to having a constant drift. Later generations of short-rate models are invented to address these issues.

### Bond Pricing in Merton's Model
The price of a zero-coupon bond is given by:

$$P(t,T) = \exp\left(-(T-t)r_t - \frac{\mu \left(T-t\right)^2}{2} + \frac{\sigma^2\left(T-t\right)^3}{6}\right)$$

### Derivation of the Bond Price Formula
We will derive the formula for the zero-coupon bond prices two different ways. The first is to solve stochastic integral directly, while the second is to "guess" the form of the solution and solve it by matching the bond price formula to the guessed solution.

#### Direct Solution
1. Start with the general bond pricing equation:
   $$P(t,T) = \mathbb{E}_t^Q\left[\exp\left(-\int_t^T r_s ds\right)\right]$$
   where $$\mathbb{E}_t^Q$$ denotes the expectation under the risk-neutral measure.

2. Since $$r_t$$ follows a Gaussian process, the $$\int_t^T r_s ds$$ is also normally distributed.

3. For a normally distributed variable $$X$$ with mean $$m$$ and variance $$v$$, we have:

   $$\mathbb{E}\left[\exp\left(-X\right)\right] = \exp\left(-m + \frac{1}{2}v\right)$$

   If you forget this identity, you can derive it from the definition of expectation.

   $$\mathbb{E}\left[\exp\left(-X\right)\right] = \frac{1}{\sqrt{2\pi v}}\int_{-\infty}^{\infty}\exp\left(-x\right)\exp\left(-\frac{\left(x - m\right)^2}{2v}\right)dx$$

   or you happen to remember the moment generating function of a normal distribution:

   $$M_X(t) = \mathbb{E}\left[\exp\left(tX\right)\right] = \exp\left(\mu t +\frac{\sigma^2t^2}{2}\right)$$

4. Calculate the mean and variance of $$\int_t^T r_s ds$$, note that 

   $$\int_t^T r_s ds = \int_t^T r_t ds + \int_t^T \mu\left(s-t\right)ds + \sigma\int_t^T \int_t^s dW_u ds $$ 

   The integrals in the last term can be interchanged (why? what condition must be satisfied?) and after integrating all deterministic terms we have 
   
   $$\int_t^T r_s ds = r_t\left(T-t\right) + \mu\frac{\left(T-t\right)^2}{2} + \sigma\int_t^T \left(s-t\right) dW_s$$

   which has mean and variance of
      
   $$m = \mathbb{E}_t^Q\left[\int_t^T r_s ds\right] = \left(T-t\right)r_t + \frac{1}{2}\left(T-t\right)^2\mu$$
   
   and

   $$v = \text{Var}_t^Q\left[\int_t^T r_s ds\right] = \text{E}_t^Q\left[\int_t^T r_s^2 ds\right] =\frac{1}{3}\left(T-t\right)^3\sigma^2$$ 

5. Apply the formula from step 3:
   $$P(t,T) = \exp\left(-(T-t)r_t - \frac{1}{2}(T-t)^2\mu + \frac{1}{6}(T-t)^3\sigma^2\right)$$

#### The Guessed Solution
1. Assume the bond price has the form
   
   $$P\left(t,T\right) = A\left(t,T\right)exp\left(-B\left(t,T\right)r\right)$$

2. Apply Itô's lemma to $$P(t,T)$$ (Try!)
   
   $$\frac{dP}{P} = \left[\frac{\partial A}{\partial t}\frac{1}{A} - \frac{\partial B}{\partial t} r - \mu B + \frac{1}{2} B^2 \sigma^2\right]dt - \sigma B dW_t$$

3. Under the risk-neutral measure, the expected return of the bond $$dP/P$$ is the risk-free rate $r$, Substitute the guessed solution into the equation:

   $$r = \frac{\partial A}{\partial t}\frac{1}{A} - \frac{\partial B}{\partial t} r - \mu B + \frac{1}{2} B^2 \sigma^2$$

4. Equate coefficients of $$r$$ and constant terms, we get two equations:
   
   $$-\frac{\partial B}{\partial t} = 1$$

   $$\frac{\partial A}{\partial t}\frac{1}{A} - \mu B + \frac{1}{2}B^2\sigma^2 = 0$$

5. Solve for $$B\left(t,T\right)$$

   From $$-\frac{\partial B}{\partial t} = 1$$, we get $$B\left(t,T\right) = T - t$$. To arrive at this result, we need the boundary condition for $$B\left(T,T\right)$$, can you deduce what it is from the assumed bond price formula from point 1 above?

6. Solve for $$A\left(t,T\right)$$

   Substitute $$B\left(t,T\right)$$ into the second equation:

   $$\frac{\partial A}{\partial t}\frac{1}{A} - \mu B + \frac{1}{2}B^2\sigma^2 = 0$$

   Solve for the integral, we get

   $$\ln\left(A\left(t,T\right)\right) = -\mu\left(T-t\right)^2/2 + \sigma^2\left(T-t\right)^3/6 + C$$

   Where $$C$$ is a constant of integration.

7. Apply the boundary condition $$P\left(T,T\right) = 1$$

   $$1 = A\left(T,T\right)exp\left(-B\left(T,T\right)r\right) = A\left(T,T\right)$$

   Therefore, $$\ln\left(A\left(T,T\right)\right) = 0 = C$$

8. Write the final expression for $$A(t,T)$$

   $$A\left(t,T\right) = \exp\left(-\mu\left(T-t\right)^2/2 + \sigma^2\left(T-t\right)^3/6\right)$$

9. Apply the formula to the bond price

   $$P\left(t,T\right) = \exp\left(-\mu\left(T-t\right)^2/2 + \sigma^2\left(T-t\right)^3/6 - (T-t)r\right)$$

### Implementation: Simulating Merton's Model
We first show how to simulate the Merton's model using the Euler-Maruyama method. The Euler-Maruyama method is a simple and intuitive way to discretize a continuous-time stochastic process. It is based on the idea that the continuous-time process can be approximated by a sequence of discrete-time processes, each with a small time step. The following code simulate $$T$$ years of short rates, where each year is divided into $$N$$ time steps.

1. Import necessary libraries
2. Define model parameters: $$r_0$$, $$\mu$$, $$\sigma$$
3. Set up time grid: $$t = [0, \Delta t, 2\Delta t, ..., T \times N]$$
4. Implement Euler-Maruyama method for simulation:

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

In general, this discretization is only an approximation because it ignores errors from time aggregation, the problem is more pronounced when there is mean-reversion. This will be discussed in more detail in the next post, but the exact way to discretize a continuous-time process is to first solve the stochastic differential equation.

$$\int_0^t dr_s = \int_0^t \mu ds + \int_0^t \sigma dW_s$$

then discretize the solution. For the Merton's model, this Euler-Maruyama discretization works as intended.

### Implementation: Zero Coupon Bond Price and Yields Merton's Model
Implementing the zero-coupon bond price and yields in Merton's model is straightforward. I first implement the $$A$$ and the $$B$$ function from "guessed solution" and use them to implement the bond price and yield formula. We will later see that there is a special sub-class of short-rate models whose zero-coupon bond yield is an affine function of the short rates, i.e. 

$$y(t, T) = \frac{1}{T-t}\left(\log\left(A\left(t,T\right)\right) + B\left(t,T\right) r_t\right)$$

and we can later create a class that implements this special sub-class of short-rate models by simply overriding the $$A$$ and $$B$$ functions

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

### Dynamic of the zero-coupon bond yield
For all one-factor affine short-rate models, the dynamic of thezero-coupon bond yield $$y(t, T)$$ can be derived by applying the Itô's lemma to the zero-coupon bond yield formula.

$$y_t = -\frac{\log\left(A\left(t,T\right)\right) + B\left(t,T\right) r_t}{T-t}$$

e.g.

$$dy = \frac{\partial y}{\partial t} dt + \frac{\partial y}{\partial r} dr + \frac{1}{2}\frac{\partial^2 y}{\partial r^2} \left(dr\right)^2 $$

where

$$\frac{\partial y}{\partial t} = \frac{-1}{T-t}\left(\frac{dA/dt}{A\left(t, T\right)} + \frac{dB\left(t,T\right)}{dt} r_t + y_t\right)$$

$$\frac{\partial y}{\partial r} = -\frac{B\left(t, T\right)}{T-t}$$

$$\frac{\partial^2 y}{\partial r^2} = 0$$

The resuling dynamic for $$y$$ is a stochastic process that depends on both $$y$$ and $$r$$ in its diffusion term. Let $$\tau = T-t$$ then

$$dy = \left[\frac{y_t - r_t}{\tau} - \frac{1}{2}\tau\sigma^2\right] dt - \sigma dW_t$$

Later when we look at the one-factor Vasicek model, in which the short rate follows a Ornstein-Uhlenbeck process, we will see that the dynamic of the zero-coupon bond yield is also a Ornstein-Uhlenbeck process. However, in the case of the Merton's model, the dynamic of the zero-coupon bond yield no longer resembles another Brownian motion with constant drift. In this case the drift depends on the level of the short and long rate and is not a constant.

What is more important is that the yield volatility is the same regardless of $$\tau$$. This is not consistent with empirical observation: the short rates are much more volatile than long rates in general. For short-rate models that include mean-reversion, such as the Vasicek model, they tend to have the opposite problem where long-rate volatility is too low.

### Wrapping Up
While simplistic in its assumptions, the Merton's model is a good starting point for understanding the short-rate models both in terms of technicality and historical development. It already exhibit some of the key features and limitations of the short-rate models. In the next post, I will discuss a plethora of classic short-rate models developed for the rest of the 1970s, before we enter into the world of the multi-factor short-rate models.