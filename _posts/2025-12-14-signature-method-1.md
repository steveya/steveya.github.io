---
layout: post
title: Signature Methods (Part 1 - Motivation)
date: 2025-12-14
categories: [Quantitative Finance]
tags: [study-notes, signature-method, quantitative-finance, volatility-forecast, machine-learning, research]
math: true
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

I am starting a series on the **signature method** now that I have some free time on my hand over the 2025 holiday season. In this post I talk about what motivates me to study the signature method, and as this series progresses I will build up some examples and applications to see how well it works in practice. The signature method has deeper connection with rough path theory and lie algebra, neither of which I have studied enough to write about intelligently. Hopefully towards the end of the series I will have accumulated enough knowledge to write about their connections to the signature method.

## Motivation for the Signature Method
In quantitative finance we often deal with trajectories: prices evolving, spreads widening then snapping back, volume or volatility arriving in bursts, and signals activated in sequences. However, many modeling pipelines force trajectories into a small set of hand-crafted summaries such as moving averages and volatility, cross-overs, rolling betas, event counts or regime flags. These features are often effective, but they mostly work by compressing a path into a number, and any information about ordering and interactions across variables (channels) is mostly lost. That loss matters because many market phenomena are genuinely path-dependent.

For example, volatility clustering depends on the recent sequence of shocks, not just the net move. We are often interested in "who moved first" (causal effect) by looking at the lead–lag and cross-impact. Intraday microstructure effects such as bid–ask bounce, spread dynamics etc, depend on ordering as well. These are often difficult or ineffective to be compressed into a number.

The signature method starts from a different premise: instead of choosing a small list of summaries, we build a systematic feature map for paths. The goal is to represent a multivariate time series window as a feature vector that retains time-ordered interactions, so that downstream models can learn path-dependence without requiring us to manually enumerate and craft interaction patterns.

Another useful way to motivate the method before defining it is to view it as a path analogue of polynomial feature maps.

For scalar inputs $$x$$, we can use the polynomial features

$$
\phi(x) = (1, x, x^2, x^3, \dots)
$$

to approximate many nonlinear targets $$f(x)$$ using a linear model in the truncated (up to $$m$$-th order polynomial) feature space:

$$
f(x) \approx \langle w,\ \phi^{\le m}(x)\rangle.
$$

For sequential inputs $$X_{[a, b]} = \left\{X_t \vert t \in [a, b], X_t \in \mathbb{R}^d \right\}$$, we seek an analogous map from a window of a multivariate time series (a path) to a feature vector, such that many continuous, path-dependent targets can be approximated by linear functionals of those features:

$$
F(X_{[a,b]}) \approx \langle \ell,\ S^{\le m}(X_{[a,b]})\rangle.
$$

The signatures provide precisely such features $$S(X_{[a,b]})$$. Note that similar to the polynomial features, the full signature is also infinite dimensional, and the approximation is done using the truncated signatures up to level $$m$$. 

More precisely, the signature has the following universal approximation property:

> Let $$\mathcal{P}$$ be a set of $$d$$-dimensional paths on $$[a,b]$$ (typically of bounded variation), and let $$S(X_{[a,b]})$$ denote the signature of a path $$X$$. Fix a compact subset $$K\subset\mathcal{P}$$ (compact in a topology where the signature map is continuous). For any continuous functional $$F:K\to\mathbb{R}$$ and any $$\varepsilon>0$$, there exist:
> - a truncation depth $$m$$, and
> - a coefficient vector for signature coordinates up to level $$m$$)
>
> such that 
> $$ 
> \sup_{X\in K}\left|F(X)-\langle \ell,\ S^{\le m}(X)\rangle\right|<\varepsilon.
> $$

At first glance, signatures sound like a silver bullet that eliminates the need to hand-craft timeseries-based trading signals, since supposedly they can all be approximated by linear functions of signatures. All we have to do is to dump our panel data into a signature transformer, take the outputs and run a linear regression on them against our target! In practice, we have to make these design choices in order to use signature features effectively:

1. **Path Construction**: Signatures operate on a path of inputs. In order to fully leverag the universality property of signatures, the inputs should contain sufficient amount of information. This is usually done through various augmentations such as time augmentation, lead–lag augmentation, inclusion of macro or microstructure channels)etc. This is where we decide which invariances and which statistics we want to make easy to learn.
2. **Truncation and Rgularization**: The full signature is an infinite collection of features. In practice we truncate at depth $$m$$ and rely on regularization and model selection to control complexity.

In the next post we will define signatures explicitly and provide a few examples.
