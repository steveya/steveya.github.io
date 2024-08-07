---
layout: post
title: "Short Rate Models (Part 1: Introducing Merton's Model)"
date: 2023-05-10
categories: [Quantitative Finance]
tags: [study-notes, quantitative-finance, short-rate-models]
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>


## Table of Contents

1. [Introduction](#introduction)
2. [Preliminaries](#preliminaries)
3. [Merson's Model](#mertons-model-1973)

## Introduction
Welcome to my refresher series on short-rate models, dear reader! These mathematical marvels are not just academic curiosities, but crucial tools in the world of finance. They are among the main workhorses of interest rate modelling, used both in the ivory towers of academic research and by the fast-paced world of Wall Street desk quants. This refresher series aims to help you understand, formulate, and solve these models by deriving and implementing well-known versions of varying complexity and generality.

Short-rate models, despite their seeming simplicity, are incredibly versatile tools for modelling interest rates. They specify how the instantaneous spot interest rate (our 'short rate') evolves, which is the rate at which you can borrow money for an infinitesimally short period. This versatility makes them a key component of many financial models and strategies. 

The short rate is assumed to evolve over time according to a stochastic process that captures the observed or theorized price of market behaviours. Once we've nailed down this evolution, the no-arbitrage principle can help us determine rates for longer borrowing periods.

In this first post, we start with some preliminaries. These are the foundational concepts and tools that we'll need to understand and implement the short-rate models. We'll then go step-by-step to derive the price of a zero-coupon bond and the simulation of the model. It is then followed by Python implementations of the short-rate process simulation and the zero coupon bond pricing. Code snippets included in these posts (and more) can be found in the [Github repository](https://github.com/steveya/short-rate-models/notebook/merton_model.ipynb). The code library will evolve with the series as we build out more and more types of models that can benefit from more abstraction.


## Preliminaries
Zero-coupon bonds (ZCB) form the foundation of term structure modelling. The price of a $$T$$-year ZCB at time $$t$$ ($$P(t, T)$$) is the risk-adjusted present value of 1-dollar in $$T$$-years. The (continuously compounded) yield-to-maturity (YTM) $$y(t, T)$$ at time $$t$$ is given by

- YTM to Price: 

$$P\left(t, T\right) = \exp\left(-y_t^T \left(T-t\right)\right)$$ 

- Price to YTM: 

$$y\left(t, T\right) = -\frac{\ln\left(P_t^T\right)}{T-t}$$

As we explicitly model the short rate dynamic, the YTM is the expected average (under the risk-neutral measure $$Q$$) of the short rate from $$t$$ to $$T$$.

- Short Rate to YTM: 

$$y\left(t, T\right) = \frac{1}{T-t}\mathbb{E}_t^Q\left[\int_t^T r_s ds\right]$$

- Short Rate to Price: 

$$P\left(t, T\right) = \mathbb{E}_t^Q\left[\exp\left(-\int_t^T r_s ds\right)\right]$$

These fundamental relationships will be used to calculate the price and yield of ZCB from the short-rate process. Readers familiar with asset pricing will notice that the price of a ZCB is calculated with respect to the risk-neutral measure $$Q$$. We will not concern ourselves with the risk-neutral and physical measures just yet. All measures are assumed to be risk-neutral unless otherwise indicated. A fuller treatment will be postponed to a later post on model calibration to the observed yield curve.

## Merton's Model (1973)
Merton's model is one of the first short-rate models and is arguably the simplest one; it laid the foundation for many subsequent short-rate models. It was introduced in 1973 by the renowned economist Robert C. Merton, who extended the concepts of equity option pricing to the bond market by modelling short-rate dynamics as a simple Gaussian process.

### Model Specification
Merton's model is defined by the following stochastic differential equation:

$$dr_t = \mu dt + \sigma dW^Q_t$$

where:
- $$r_t$$: short-term interest rate at time $$t$$
- $$\mu$$: constant drift term
- $$\sigma$$: constant volatility
- $$W^Q_t$$: Wiener process (standard Brownian motion) under the risk-neutral measure $$Q$$.

### Key Concepts

#### Continuous-Time Modeling
Merton's model was one of the first to apply continuous-time stochastic processes to interest rate modelling, paving the way for more sophisticated models. It is a one-factor model, as the only source of randomness is the short rate. This will later be generalized to the multi-factor model, where multiple factors drive the dynamic of the short rate.

#### Gaussian Process
The model assumes that the short rate follows a Gaussian process, allowing for positive and negative levels. It also assumes that rate volatility is constant and independent of the level of the short rate. This modelling choice needs to be consistent with empirical results. While rates can go negative (though mostly inconceivable in 1973), the short-rate volatility generally depends on the short rate's level. Short rates also exhibit mean reversion as opposed to having a constant drift. Later generations of short-rate models are invented to address these issues.

#### Equilibrium Short-Rate Model
Merton's model belongs to a subclass of the short-rate model called the **equilibrium model**. This class of model cannot fit to the initial term structure exactly. This is because the yield curve generated by some short-rate models can assume only certain forms. There is another class of short-rate models called the **arbitrage-free model**. which allows $$\mu = \mu_t$$ to depend on time. For the Merton model, if we let $$mu$$ to a deterministic function of time, then we obtain the **Ho-Lee Model** that can fit the initial term structure exactly by varying $$\mu_t$$ deterministically over time.

### Derivation of the Bond Pricing in Merton's Model
The price of a $$T$$-year ZCB is given by:

$$P(t,T) = \exp\left(-(T-t)r_t - \frac{\mu \left(T-t\right)^2}{2} + \frac{\sigma^2\left(T-t\right)^3}{6}\right)$$

We will derive this formula two different ways. The first is to solve stochastic integral directly and the second is to "guess" the form of the solution and solve it by matching the bond price formula to the guessed solution. Even though I call the second way the "guessed" solution, it is actually based on the observation that the Merton model falls in the category of the affine term-structure model, whose prices are all of the form $$A(t, T)exp(-B(t, T) r_t)$$. This will be covered in later posts.

### Derivation of the Bond Pricing in Merton's Model
The price of a $$T$$-year ZCB is given by:

$$P(t,T) = \exp\left(-(T-t)r_t - \frac{\mu \left(T-t\right)^2}{2} + \frac{\sigma^2\left(T-t\right)^3}{6}\right)$$

We will derive this formula in two different ways. The first is to solve stochastic integral directly, and the second is to "guess" the form of the solution and solve it by matching the bond price formula to the guessed solution. Even though I call the second way the "guessed" solution, it is actually based on the observation that the Merton model falls in the category of the affine term-structure model, whose prices are all of the form $$A(t, T)exp(-B(t, T) r_t)$$. This will be covered in later posts.

#### Direct Solution
1. Start with the general bond pricing equation:
 $$P(t,T) = \mathbb{E}_t^Q\left[\exp\left(-\int_t^T r_s ds\right)\right]$$
 where $$\mathbb{E}_t^Q$$ denotes the expectation under the risk-neutral measure.

2. Since $$r_t$$ follows a Gaussian process, the $$\int_t^T r_s ds$$ is also normally distributed.

3. For a normally distributed variable $$X$$ with mean $$m$$ and variance $$v$$, we have:

 $$\mathbb{E}\left[\exp\left(-X\right)\right] = \exp\left(-m + \frac{1}{2}v\right)$$

 If you forget this identity, you can derive it from the definition of expectation.

 $$\mathbb{E}\left[\exp\left(-X\right)\right] = \frac{1}{\sqrt{2\pi v}}\int_{-\infty}^{\infty}\exp\left(-x\right)\exp\left(-\frac{\left(x - m\right)^2}{2v}\right)dx$$

 or you happen to remember the moment-generating function of a normal distribution:

 $$M_X(t) = \mathbb{E}\left[\exp\left(tX\right)\right] = \exp\left(\mu t +\frac{\sigma^2t^2}{2}\right)$$

4. Calculate the mean and variance of $$\int_t^T r_s ds$$, note that 

 $$\int_t^T r_s ds = \int_t^T r_t ds + \int_t^T \mu\left(s-t\right)ds + \sigma\int_t^T \int_t^s dW_u ds $$ 

 The integrals in the last term can be interchanged (why? what condition must be satisfied?), and after integrating all deterministic terms, we have 
   
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

3. Under the risk-neutral measure, the expected return of the bond $$dP/P$$ is the risk-free rate $r$; substitute the guessed solution into the equation:

 $$r = \frac{\partial A}{\partial t}\frac{1}{A} - \frac{\partial B}{\partial t} r - \mu B + \frac{1}{2} B^2 \sigma^2$$

4. Equate coefficients of $$r$$ and constant terms, we get two equations:
   
 $$-\frac{\partial B}{\partial t} = 1$$

 $$\frac{\partial A}{\partial t}\frac{1}{A} - \mu B + \frac{1}{2}B^2\sigma^2 = 0$$

5. Solve for $$B\left(t,T\right)$$

 From $$-\frac{\partial B}{\partial t} = 1$$, we get $$B\left(t,T\right) = T - t$$. We need the boundary condition for $$B\left(T,T\right)$$ to arrive at this result. Can you deduce what it is from the assumed bond price formula from point 1 above?

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

### Implementation: ZCB Price and Yields Merton's Model
Implementing the ZCB price and yields in Merton's model is straightforward once we have derived its solutions as above. Even though we arrive at the solution using two different approaches, the second solution is derived using our "external" knowledge that the Merton model is an affine term structure model, as mentioned above. Therefore, it would make sense to implement our pricing function with this knowledge in mind, but we can wait to build all the abstractions about the affine term structure. We first implement the $$A$$ and the $$B$$ function and use them in the ZCB price and yield formula. Once we introduce the affine term structure model in more detail, we will implement the affine term structure model class and override the $$A$$ and $$B$$ functions, but there is still some way before we get there.

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
For all one-factor affine short-rate models, the risk-neutral dynamic of the zero-coupon bond yield $$y(t, T)$$ can be derived by applying the Itô's lemma to the zero-coupon bond yield formula. This gives

$$
dy = \frac{\partial y}{\partial t} dt + \frac{\partial y}{\partial r} dr + \frac{1}{2}\frac{\partial^2 y}{\partial r^2} \left(dr\right)^2 
$$

where

$$\frac{\partial y}{\partial t} = \frac{1}{T-t}\left(y_t - \frac{dA/dt}{A\left(t, T\right)} + \frac{dB\left(t,T\right)}{dt} r_t\right)$$

$$\frac{\partial y}{\partial r} = \frac{B\left(t, T\right)}{T-t} = 1$$

$$\frac{\partial^2 y}{\partial r^2} = 0$$

The resulting dynamic for $$y$$ is a stochastic process that depends on both $$y$$ and $$r$$ in its diffusion term. Let $$\tau = T-t$$ then

$$dy = \left[\frac{y_t - r_t}{\tau} - \frac{1}{2}\tau\sigma^2\right] dt + \sigma dW_t$$

The drift term can be interpreted as the sum of the annualized carry/rolldown $$\left(y_t - r_t\right)/\tau$$ and the convexity adjustment term $$\frac{1}{2}\tau\sigma^2$$. The diffusion term is the volatility of the short rate $$\sigma$$, which is the same regardless of $$\tau$$. This is inconsistent with empirical observation: the short rates are much more volatile than long rates in general. For short-rate models that include mean-reversion, such as the Vasicek model, tend to have the opposite problem where long-rate volatility is too low. As we will see when we introduce the Vasicek model.

### Wrapping Up
While simplistic in its assumptions, Merton's model serves as an excellent starting point for understanding short-rate models in terms of technicality and historical development. It already exhibits some of the key features and limitations of short-rate models. In the [next post](https://steveya.github.io/posts/short-rate-models-2/), we'll review how to simulate short rates from $$\mu$$ and $$\sigma$$, as well as how to calibrate Merton's model to market observed short rates.


