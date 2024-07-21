---
layout: post
title: "WIP - Short Rate Models (Part 2: Vasicek Model)"
date: 2023-06-10
tags: [study-notes, quantitative-finance, short-rate-models]
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Short Rate Models (Part 2: Vasicek Model)

## Table of Contents

1. [Introduction](https://steveya.github.io/posts/short-rate-models-1/#introduction)
2. [Preliminaries](https://steveya.github.io/posts/short-rate-models-1/#preliminaries)
3. [Merson's Model](https://steveya.github.io/posts/short-rate-models-1/#mertons-model-1973)
4. [Vasicek Model](#vasicek-model)

## Vasicek Model

The Vasicek model, introduced by Oldrich Vasicek in 1977, is one of the earliest and most influential term structure models after the Merton's model. It relaxes the restriction placed on the short rate process by letting it follow a mean-reverting Ornstein-Uhlenbeck process. This assumption ensures that the short rate does not go unbounded as in the Merton model, and allows for an attachment of economic ideas such as the long-term equilibrium short rate that rates tend to fluctuate around a long-term equilibrium level. Because of this, many of the later extensions of the short-rate models are extensions to the Vasicek model.

### Model Specification

The Vasicek model is defined by the following stochastic differential equation:

$$dr_t = \kappa(\theta - r_t)dt + \sigma dW_t$$

Where:
- $$r_t$$: short-term interest rate at time $$t$$
- $$\kappa$$: speed of mean reversion
- $$\theta$$: long-term mean level
- $$\sigma$$: volatility
- $$W_t$$: Wiener process (standard Brownian motion)

### Key Concepts

#### Mean Reversion to Fixed Long-Term Equilibrium
Mean reversion is the tendency of a process to "pull back" towards its long-term average. In the context of interest rates, this reflects the economic intuition that the short rates tend to fluctuate around a long-term equilibrium level. This level is assumed to be fixed at $$\theta$$ in the Vasicek model. In later posts when I introduce several extensions of the short-rate models, one of them will relax this assumption and allow the long-term equilibrium level to be a mean-reverting process itself.

The speed of mean reversion $$\kappa$$ is a key parameter in the Vasicek model. It not only controls the speed of mean reversion, but also influence the volatility of the short rate on top of the $$\sigma$$ parameter. The higher the $$\kappa$$, the faster the mean-reversion speed, and the lower the volatility of the short rate, and vice versa.

#### Ornstein-Uhlenbeck Process
The model assumes that the short rate follows an Ornstein-Uhlenbeck process, which is a well-established method for representing mean-reverting behavior. This modeling choise is more consistent with empirical restults. Similar to the Merton's model, the short rate can still go negative and the the volatility is still independent of the level of the short rate.

#### Equilibrium Short-Rate Model
Similar to the Merton's model, the Vasicek model also belongs to a subclass of the short-rate model called the *Equilibrium short-rate model*. The Arbitrage-Free counterpart to the Vasicek model is the **Hull-White Model**, which allows $$\theta$$ to be deterministically time-varying.

### Bond Pricing in Vasicek's Model
Similar to the Merton's model, the price of a zero-coupon bond can be derived two different ways that closely follow those derivations for the Merton's model. I derive the solution using the "direct" method again below. (Warning: tedious algebra ahead!) The "guessed" solution is left as an exercise. However, I will go over the the "guessed" solution in more generality in the next post, where I will use it to solve a multi-factor short rate model.

#### Direct Solution
1. Start with the general bond pricing equation:
   $$P(t,T) = \mathbb{E}_t^Q\left[\exp\left(-\int_t^T r_s ds\right)\right]$$
   where $$\mathbb{E}_t^Q$$ denotes the expectation under the risk-neutral measure.

2. Since $$r_t$$ follows a Gaussian process, the $$\int_t^T r_s ds$$ is also normally distributed.

3. For a normally distributed variable $$X$$ with mean $$m$$ and variance $$v$$, we have:

   $$\mathbb{E}\left[\exp\left(-X\right)\right] = \exp\left(-m + \frac{1}{2}v\right)$$

4. Calculate the mean and variance of $$\int_t^T r_s ds$$. Recall that in the derivation of the Merton's model, we first derive an expression for $$r_s$$ for $$ s \ge t$$. To get this expression for the Vasicek model, we frist multiply out the drift term:

   $$dr_t = \kappa\theta dt - \kappa r_t dt + \sigma dW_t$$

   and notice that the second term $$dr_t + \kappa r_t dt = 0$$ is a standard first-order differential equation and can be solved by multiplyuing the integrating factor $$\exp\left(\kappa r_t\right)$$. Once we multiply it to all three terms, we can integrate and rearrange to get 

   $$
   \begin{equation}
   r_s = r_t e^{-\kappa \left(s-t\right)} + \theta \left(1 - e^{-\kappa\left(s-t\right)}\right) + \sigma \int_t^s e^{-\kappa \left(u-t\right)} dW_u
   \end{equation}
   $$

   Now to get $$\mathbb{E}_t^Q\left[\int_t^T r_s ds\right]$$ and $$\text{Var}_t^Q\left[\int_t^T r_s ds\right]$$, we need to again integrate $$\int_t^T r_s ds$$, which turns out to be

   $$
   \begin{equation}
   \begin{aligned}
   \int_{t}^{T} r_{s} du &= r_{t} \int_{t}^{T} e^{-\kappa(s-t)} ds + \theta \int_{t}^{T} \left( 1 - e^{-\kappa(s-t)} \right) ds + \sigma \int_{t}^{T} \int_{t}^{s} e^{-\kappa(s-u)} dW_u ds \\
   &= r_{t} \left( \frac{e^{-\kappa(T-t)} - 1}{-\kappa} \right) + \theta \left[ \int_{t}^{T} du - \int_{t}^{T} e^{-\kappa(s-t)} ds \right] + \sigma \int_{t}^{T} \int_{u}^{T} e^{-\kappa(s-u)} ds dW_u\\
   &= r_{t} \left( \frac{1 - e^{-\kappa(T-t)}}{\kappa} \right) + \theta \left[ (T-t) - \frac{1 - e^{-\kappa(T-t)}}{\kappa} \right] + \frac{\sigma}{\kappa} \int_{t}^{T} \left( 1 - e^{-\kappa(T-u)} \right) dW_u \\
   \end{aligned}
   \end{equation}
   $$ 

   We can compute the $$\mathbb{E}_t^Q\left[\int_t^T r_s ds\right]$$ and $$\text{Var}_t^Q\left[\int_t^T r_s ds\right]$$ by taking expectation of the drift and diffusion terms, respectively:

   $$
   \begin{equation}
   \begin{aligned}
   \mathbb{E}_t^Q\left[\int_t^T r_s ds\right] &= r_t \left(\frac{1 - e^{-\kappa\left(T-t\right)}}{\kappa}\right) + \theta \left(\left(T-t\right) - \frac{1 - e^{-\kappa\left(T-t\right)}}{\kappa}\right) \\
   \text{Var}_t^Q\left[\int_t^T r_s ds\right] &= \text{Var}\left[\frac{\sigma}{\kappa}\int_t^T\left(1-e^{-\kappa\left(T-u\right)}\right)dW_u\right] \\
   &=\frac{\sigma^2}{\kappa^2}\int_t^T\left(1-e^{-\kappa\left(T-u\right)}\right)^2du \quad \text{(by Itô's Isometry)} \\
   &= \frac{\sigma^2}{\kappa^2}\int_t^T\left[1 + e^{-2\kappa\left(T-u\right)} - 2e^{-\kappa\left(T-u\right)}\right]du \\
   &= \frac{\sigma^2\left(T-t\right)}{\kappa^2} + \frac{\sigma^2}{2\kappa^3}\left[1-e^{-2\kappa\left(T-u\right)}\right] - \frac{2\sigma^2}{\kappa^3}\left[1 - e^{-\kappa\left(T-u\right)}\right] \\
   &= \frac{\sigma^2}{2\kappa^3}\left(2\kappa\left(T-t\right) - 3 - e^{2\kappa\left(T-t\right)} + 4e^{-2\kappa\left(T-t\right)}\right) \\
   \end{aligned}
   \end{equation}
   $$

5. Substitute the expression for $$\mathbb{E}_t^Q\left[\int_t^T r_s ds\right]$$ and $$\text{Var}_t^Q\left[\int_t^T r_s ds\right]$$ into the bond price formula, and let $$\tau = T-t$$, and after some tedious algebra, we get

   $$
   \begin{equation}
   \begin{aligned}
   P(t,T) &= \mathbb{E}_t^Q\left[\exp\left(-\int_t^T r_s ds\right)\right] \\
   &= \exp\left[-\mathbb{E}_t^Q\left[\int_t^T r_s ds\right]+\frac{1}{2}\text{Var}_t^Q\left[\int_t^T r_s ds\right]\right] \\
   &= \exp\left[-r_t \left(\frac{1 - e^{-\kappa\tau}}{\kappa}\right) - \theta \left(\tau - \frac{1 - e^{-\kappa\tau}}{\kappa}\right)+ \frac{\sigma^2}{4\kappa^3}\left(2\kappa\tau - 3 - e^{2\kappa\tau} + 4e^{-2\kappa\tau}\right)\right]\\ 
   &= \exp\left[-r_t \left(\frac{1 - e^{-\kappa\tau}}{\kappa}\right) + \left(\theta -\frac{\sigma^2}{\kappa^2}\right)\left[\frac{1-e^{-\kappa\tau}}{\kappa}-\tau\right] - \frac{\sigma^2}{4\kappa}\left[\frac{1 - e^{-\kappa\tau}}{\kappa}\right]^2\right]\\
   \end{aligned}
   \end{equation}
   $$

Notice that the term $$\kappa^{-1}\left(1-\exp\left(-\kappa\tau\right)\right)$$ keeps showing up in the solution to this model, such as the zero-coupon bond prices and yields. Mathematically, it comes from the integral of of the exponential decay function $$e^{-\kappa\tau}$$ over time. It adjusts the average level of the short rate in the presence of mean-reversion. The expected value and the variance of the short rate at time future time are given by

$$
\begin{equation}
\begin{aligned}
\mathbb{E}_t^Q\left[r_s \vert r_t\right] &=r_t e^{-\kappa \left(s-t\right)} + \theta \left(1 - e^{-\kappa\left(s-t\right)}\right) \\
\text{Var}_t^Q\left[r_s \vert r_t\right] &= \text{Var}\left[\sigma \int_t^s e^{-\kappa \left(u-t\right)} dW_u\right] \\
&= \sigma^2 \int_t^s e^{-2\kappa \left(u-t\right)} du \quad \text{(by Itô's Isometry)} \\
&= \frac{\sigma^2}{2\kappa}\left(1 - e^{-2\kappa\left(s-t\right)}\right)
\end{aligned}
\end{equation}
$$

The expected value is an weighted average ot the current short rate and the long rate. The weights is this adjustment term. As $$s-t$$ gets larger, the weight on the current short rate gets exponentially. Also note that in the price formula, there is only one term that is muliplied by the shrot rate $$r_t$$. Therefore the zero coupon price sensitibity to the short rate is determined by this exponential decay adjustment term. The longer the tenor, the less its sensitivity to the short rate. 

Even though I am skipping the "guessed" solution derivation here, from the solution above, one can see that the $$A(t,T)$$ and $$B(t,T)$$ functions are 

$$
\begin{equation}
\begin{aligned}
P(t,T) &= A(t,T)e^{-B(t,T)r_t} \\
A(t,T) &= \exp\left[\left(\theta -\frac{\sigma^2}{\kappa^2}\right)\left[\frac{1-e^{-\kappa\tau}}{\kappa}-\tau\right] - \frac{\sigma^2}{4\kappa}\left[\frac{1 - e^{-\kappa\tau}}{\kappa}\right]^2\right] \\
B(t,T) &= \frac{1-e^{-\kappa\tau}}{\tau}
\end{aligned}
\end{equation}
$$

### Implementation: Simulating the Vasicek Model
As with the Merton's model, I first simulate the Vasicek model using the Euler-Maruyama method. Moreover, as mentioned in the previous post, a direct discretization from the stochastic differential equation ignores errors from time aggregation, especially when mean-reversion is strong. To show this, I also provide a more exact discretization of the Vasicek model, and simulate their path under different mean-reversion speeds and discretization schemes.



### Dynamic of the zero-coupon bond yield



